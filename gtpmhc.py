"""GTpMHC inference network and audited legacy-checkpoint loading.

The binding-score path retains the original peptide-first fixed token slots.
Only tensors that reach the final binding logit are registered. Training losses,
MAE reconstruction, auxiliary classifiers, and their unused projections are absent.
"""

import math
from collections.abc import Mapping
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GraphNorm, global_mean_pool
from torch_geometric.utils import scatter, softmax, to_dense_batch


class MLP(nn.Module):
    """Layer-normalized MLP; Identity keeps original checkpoint indices."""

    def __init__(self, channels, bias=True, out_norm=False):
        super().__init__()
        modules = []
        for index in range(1, len(channels)):
            modules.append(nn.Linear(channels[index - 1], channels[index], bias=bias))
            if index < len(channels) - 1:
                modules.extend([nn.LayerNorm(channels[index]), nn.ReLU(), nn.Identity()])
        if out_norm:
            modules.append(nn.LayerNorm(channels[-1]))
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)


class GCNCluster(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels[0])
        self.Conv_layers = nn.ModuleList(
            GCNConv(channels[index - 1], channels[index]) for index in range(1, len(channels)))

    def forward(self, x, edge_index):
        x = self.in_ln(x)
        for layer in self.Conv_layers[:-1]:
            x = F.relu(layer(x, edge_index))
        return self.Conv_layers[-1](x, edge_index)


class PeptideProteinConv(nn.Module):
    """Cluster-to-peptide attention, preserving legacy parameter names."""

    def __init__(self, channels, heads, t):
        super().__init__()
        self.heads = heads
        self.head_width = channels // heads
        self.t = t
        self.lin_key = nn.Linear(channels, channels, bias=False)
        self.lin_query = nn.Linear(channels, channels, bias=False)
        self.lin_value = nn.Linear(channels, channels, bias=False)
        self.lin_pep_value = nn.Linear(channels, channels, bias=False)
        self.pep_in_norm = nn.LayerNorm(channels)
        self.residue_in_norm = nn.LayerNorm(channels)
        self.pep_out_norm = nn.LayerNorm(channels)
        self.residue_out_norm = nn.LayerNorm(channels)
        self.pep_mlp = MLP([channels * 2, channels * 2, channels], out_norm=True)
        self.residue_mlp = MLP([channels * 2, channels * 2, channels], out_norm=True)

    def forward(self, pep_pool, pep_x, residue_x, edge_index, pep_batch):
        heads, width = self.heads, self.head_width
        residue_hx = self.residue_in_norm(residue_x)
        query = self.lin_query(pep_pool).view(-1, heads, width)
        key = self.lin_key(residue_hx).view(-1, heads, width)
        value = self.lin_value(residue_hx).view(-1, heads, width)
        source, target = edge_index
        alpha = (query[target] * key[source]).sum(dim=-1)
        alpha = alpha / math.sqrt(width)
        alpha = alpha / self.t
        alpha = softmax(alpha, target, num_nodes=pep_pool.size(0))
        pooled = scatter(value[source] * alpha.unsqueeze(-1), target, dim=0,
                         dim_size=pep_pool.size(0), reduce="sum")
        pooled = self.pep_out_norm(pooled.view(-1, heads * width))
        peptide = self.pep_mlp(torch.cat([pep_x, pooled[pep_batch]], dim=-1))
        pep_value = self.lin_pep_value(self.pep_in_norm(pooled)).view(-1, heads, width)
        out_val = pep_value[target] * alpha.view(-1, heads, 1)
        residue_out = torch.zeros((residue_x.size(0), heads, width), device=residue_x.device)
        residue_out[source] = out_val
        residue_out = self.residue_out_norm(residue_out.view(-1, heads * width))
        residue = self.residue_mlp(torch.cat([residue_out, residue_x], dim=-1))
        return peptide, residue


def _rbf(values, D_min=0.0, D_max=1.0, D_count=8, device="cpu"):
    values = values.clamp(D_min, D_max)
    centers = torch.linspace(D_min, D_max, D_count, device=device).view(1, -1)
    width = (D_max - D_min) / D_count
    return torch.exp(-((values.unsqueeze(-1) - centers) / width) ** 2)


def _cluster_pool(x, assignment, mask):
    # The original MinCut pooling's graph output and regularization terms do not
    # enter the binding logit. Preserve its assignment and node-pooling exactly.
    assignment = torch.softmax(assignment, dim=-1)
    x = x * mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    assignment = assignment * mask.to(device=assignment.device, dtype=assignment.dtype).unsqueeze(-1)
    return assignment, torch.matmul(assignment.transpose(1, 2), x)


class FixedSinusoidalPositionEncoding(nn.Module):
    """Add deterministic Transformer positional encodings to batch-first tensors."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        if d_model < 1 or max_len < 1:
            raise ValueError("d_model and max_len must be positive.")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        odd_width = encoding[:, 1::2].size(1)
        encoding[:, 1::2] = torch.cos(position * div_term[:odd_width])

        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional capacity "
                f"{self.encoding.size(1)}."
            )
        position = self.encoding[:, : x.size(1)].to(device=x.device, dtype=x.dtype)
        return x + position


class GraphSequenceDecoder(nn.Module):
    """Self-attention decoder for fixed-slot peptide/HLA graph token sequences."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        dim_feedforward: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.position = FixedSinusoidalPositionEncoding(d_model, max_len)
        self.self_attention = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Identity(),  # Preserve the checkpoint index of the following linear layer.
            nn.Linear(dim_feedforward, d_model),
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        enc_outputs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if enc_outputs.ndim != 3:
            raise ValueError(
                "GraphSequenceDecoder expects [batch, sequence, hidden] tokens."
            )
        if valid_mask.shape != enc_outputs.shape[:2]:
            raise ValueError("valid_mask must match the first two token dimensions.")
        valid_mask = valid_mask.bool()
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid graph token.")

        dec_inputs = self.position(enc_outputs)
        dec_inputs = dec_inputs * valid_mask.unsqueeze(-1).to(dec_inputs.dtype)
        attention_output, _ = self.self_attention(
            dec_inputs,
            dec_inputs,
            dec_inputs,
            key_padding_mask=~valid_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        dec_outputs = self.attention_norm(dec_inputs + attention_output)
        dec_outputs = self.feed_forward_norm(
            dec_outputs + self.feed_forward(dec_outputs)
        )
        dec_outputs = dec_outputs * valid_mask.unsqueeze(-1).to(dec_outputs.dtype)
        return dec_outputs


class GTpMHC(nn.Module):
    """Single-logit inference model, compatible with peptide-first checkpoints.

    Graph depth is len(K). Dropout is disabled and training-only constructor
    flags are deliberately not accepted. Use load_gtpmhc_state for legacy files.
    """

    def __init__(self, hidden_channels=128, K=(5, 10), heads=4, max_pep_len=25,
                 transformer_layers=1, max_hla_position=256, transpmhc_hla_len=34,
                 pep_evo_channels=1280, prot_evo_channels=1280,
                 pep_raw_channels=33, prot_raw_channels=33, t=1.0):
        super().__init__()
        dimensions = {"hidden_channels": hidden_channels, "heads": heads,
                      "max_pep_len": max_pep_len, "transformer_layers": transformer_layers,
                      "max_hla_position": max_hla_position, "transpmhc_hla_len": transpmhc_hla_len,
                      "pep_evo_channels": pep_evo_channels, "prot_evo_channels": prot_evo_channels,
                      "pep_raw_channels": pep_raw_channels, "prot_raw_channels": prot_raw_channels}
        if any(type(value) is not int or value < 1 for value in dimensions.values()):
            raise ValueError("All model dimensions must be positive integers.")
        if hidden_channels % heads or not isinstance(K, (tuple, list)) or not K:
            raise ValueError("hidden_channels must be divisible by heads and K must be nonempty.")
        if any(type(value) is not int or value < 1 for value in K):
            raise ValueError("K must contain positive integer cluster sizes.")
        if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or t <= 0:
            raise ValueError("t must be a finite positive attention temperature.")
        if min(pep_raw_channels, prot_raw_channels) < 21:
            raise ValueError("Raw features must contain 21 amino-acid identity channels.")
        self.hidden_channels = hidden_channels
        self.num_cluster = tuple(K)
        self.total_layer = len(K)
        self.max_pep_len = max_pep_len
        self.transpmhc_hla_len = transpmhc_hla_len
        self.max_hla_position = max_hla_position
        self.pep_raw_channels = pep_raw_channels
        self.prot_raw_channels = prot_raw_channels
        self.aa_identity_channels = 21
        self.edge_dim = 8
        self.pep_evo = MLP([pep_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_evo = MLP([prot_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.hla_mature_position_embedding = nn.Embedding(max_hla_position + 1, hidden_channels)

        def encoder():
            layer = nn.TransformerEncoderLayer(hidden_channels, heads, hidden_channels * 4,
                                               dropout=0.0, activation="gelu", batch_first=True,
                                               norm_first=True)
            # Pre-LN already disables automatic nested conversion in the legacy
            # network; make that explicit without changing its computations.
            return nn.TransformerEncoder(layer, transformer_layers, norm=nn.LayerNorm(hidden_channels),
                                         enable_nested_tensor=False)

        self.pep_transformer = encoder()
        self.hla_transformer = encoder()
        self.joint_transformer = encoder()
        self.pep_sinusoidal_position = FixedSinusoidalPositionEncoding(hidden_channels, max_pep_len)
        self.hla_sinusoidal_position = FixedSinusoidalPositionEncoding(hidden_channels, transpmhc_hla_len)
        self.joint_sinusoidal_position = FixedSinusoidalPositionEncoding(
            hidden_channels, max_pep_len + transpmhc_hla_len)
        self.pep_aa_embedding = nn.Linear(21, hidden_channels, bias=False)
        self.prot_aa_embedding = nn.Linear(21, hidden_channels, bias=False)
        self.pep_physchem_projection = (nn.Linear(pep_raw_channels - 21, hidden_channels)
                                        if pep_raw_channels > 21 else None)
        self.prot_physchem_projection = (nn.Linear(prot_raw_channels - 21, hidden_channels)
                                         if prot_raw_channels > 21 else None)
        for name, has_physchem in (("pep", self.pep_physchem_projection is not None),
                                   ("prot", self.prot_physchem_projection is not None)):
            width = hidden_channels * (2 + int(has_physchem))
            setattr(self, name + "_sequence_seed", nn.Sequential(nn.LayerNorm(width),
                    nn.Linear(width, hidden_channels), nn.GELU(), nn.LayerNorm(hidden_channels)))
        for name, raw_channels, evo_channels in (("pep", pep_raw_channels, pep_evo_channels),
                                                  ("prot", prot_raw_channels, prot_evo_channels)):
            width = raw_channels + evo_channels + hidden_channels
            setattr(self, name + "_graph_input_stem", nn.Sequential(nn.LayerNorm(width),
                    nn.Linear(width, hidden_channels), nn.GELU(), nn.LayerNorm(hidden_channels), nn.Identity()))
        self.prot_convs = nn.ModuleList()
        self.pep_convs = nn.ModuleList()
        self.prot_conv_norms = nn.ModuleList()
        self.pep_conv_norms = nn.ModuleList()
        self.clusters = nn.ModuleList()
        self.inter_convs_global = nn.ModuleList()
        self.prot_norms = nn.ModuleList()
        self.residue_lins = nn.ModuleList()
        self.c2r_mlps = nn.ModuleList()
        self.c2p_mlps = nn.ModuleList()
        self.pep_gns = nn.ModuleList()
        self.prot_gns = nn.ModuleList()
        for cluster_count in K:
            self.prot_convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads,
                                           concat=False, dropout=0.0, edge_dim=self.edge_dim))
            self.pep_convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads,
                                          concat=False, dropout=0.0, edge_dim=self.edge_dim))
            self.prot_conv_norms.append(nn.LayerNorm(hidden_channels))
            self.pep_conv_norms.append(nn.LayerNorm(hidden_channels))
            self.clusters.append(GCNCluster([hidden_channels, hidden_channels * 2, cluster_count]))
            self.prot_norms.append(nn.LayerNorm(hidden_channels))
            self.inter_convs_global.append(PeptideProteinConv(hidden_channels, heads, t))
            self.residue_lins.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
            self.c2r_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.c2p_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
            self.pep_gns.append(GraphNorm(hidden_channels))
            self.prot_gns.append(GraphNorm(hidden_channels))
        self.graph_sequence_decoder = GraphSequenceDecoder(hidden_channels, heads,
                                      hidden_channels * 4, max_pep_len + transpmhc_hla_len)
        flattened = (max_pep_len + transpmhc_hla_len) * hidden_channels
        self.graph_sequence_projection = nn.Sequential(nn.Linear(flattened, hidden_channels * 2),
            nn.ReLU(), nn.LayerNorm(hidden_channels * 2), nn.Identity(),
            nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1))
        self.eval()

    def _project_input_features(
        self,
        x_pep: torch.Tensor,
        x_pretrain_pep: torch.Tensor,
        x_p: torch.Tensor,
        x_pretrain_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build TranspMHC-like amino-acid tokens enriched by local and ESM inputs."""
        pep_evo = self.pep_evo(x_pretrain_pep)
        prot_evo = self.prot_evo(x_pretrain_p)
        if x_pep.size(-1) != self.pep_raw_channels:
            raise ValueError(
                f"Expected {self.pep_raw_channels} peptide raw channels, "
                f"got {x_pep.size(-1)}."
            )
        if x_p.size(-1) != self.prot_raw_channels:
            raise ValueError(
                f"Expected {self.prot_raw_channels} HLA raw channels, "
                f"got {x_p.size(-1)}."
            )

        x_pep = x_pep.to(dtype=pep_evo.dtype)
        x_p = x_p.to(dtype=prot_evo.dtype)
        pep_parts = [
            self.pep_aa_embedding(x_pep[:, : self.aa_identity_channels]),
            pep_evo,
        ]
        prot_parts = [
            self.prot_aa_embedding(x_p[:, : self.aa_identity_channels]),
            prot_evo,
        ]
        if self.pep_physchem_projection is not None:
            pep_parts.append(
                self.pep_physchem_projection(
                    x_pep[:, self.aa_identity_channels :]
                )
            )
        if self.prot_physchem_projection is not None:
            prot_parts.append(
                self.prot_physchem_projection(
                    x_p[:, self.aa_identity_channels :]
                )
            )
        return (
            self.pep_sequence_seed(torch.cat(pep_parts, dim=-1)),
            self.prot_sequence_seed(torch.cat(prot_parts, dim=-1)),
        )


    @staticmethod
    def _pad_tokens(
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        target_len: int,
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current_len = tokens.size(1)
        if current_len > target_len:
            raise ValueError(
                f"{name} length {current_len} exceeds configured maximum {target_len}."
            )
        tokens = tokens * valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        pad = target_len - current_len
        return (
            F.pad(tokens, (0, 0, 0, pad)),
            F.pad(valid_mask, (0, pad), value=False),
        )


    def _encode_transformer_pair(
        self,
        pep_x: torch.Tensor,
        pep_mask: torch.Tensor,
        hla_x: torch.Tensor,
        hla_mask: torch.Tensor,
        hla_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pep_mask = pep_mask.bool()
        hla_mask = hla_mask.bool()
        if not pep_mask.any(dim=1).all() or not hla_mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one peptide and HLA token.")
        if pep_x.size(1) > self.max_pep_len:
            raise ValueError(
                f"Peptide length {pep_x.size(1)} exceeds max_pep_len={self.max_pep_len}."
            )
        if hla_x.size(1) > self.transpmhc_hla_len:
            raise ValueError(
                f"HLA pseudo-sequence length {hla_x.size(1)} exceeds "
                f"transpmhc_hla_len={self.transpmhc_hla_len}."
            )

        if hla_positions is not None:
            if hla_positions.shape != hla_mask.shape:
                raise ValueError(
                    "hla_positions must match the dense HLA token dimensions."
                )
            hla_positions = hla_positions.to(
                device=hla_x.device, dtype=torch.long
            ).clamp(0, self.max_hla_position)
            hla_x = hla_x + self.hla_mature_position_embedding(hla_positions)

        pep_x = self.pep_sinusoidal_position(pep_x)
        hla_x = self.hla_sinusoidal_position(hla_x)
        pep_x = pep_x * pep_mask.unsqueeze(-1).to(dtype=pep_x.dtype)
        hla_x = hla_x * hla_mask.unsqueeze(-1).to(dtype=hla_x.dtype)

        pep_encoded = self.pep_transformer(
            pep_x, src_key_padding_mask=~pep_mask
        )
        hla_encoded = self.hla_transformer(
            hla_x, src_key_padding_mask=~hla_mask
        )

        pep_len = pep_encoded.size(1)
        hla_len = hla_encoded.size(1)
        pep_slots, pep_slot_mask = self._pad_tokens(
            pep_encoded, pep_mask, self.max_pep_len, "Peptide"
        )
        hla_slots, hla_slot_mask = self._pad_tokens(
            hla_encoded,
            hla_mask,
            self.transpmhc_hla_len,
            "HLA pseudo-sequence",
        )

        # Fixed slots keep the HLA offset independent of the longest peptide in
        # the current mini-batch.
        joint_mask = torch.cat([pep_slot_mask, hla_slot_mask], dim=1)
        joint_tokens = torch.cat([pep_slots, hla_slots], dim=1)
        joint_tokens = self.joint_sinusoidal_position(joint_tokens)
        joint_tokens = self.joint_transformer(
            joint_tokens, src_key_padding_mask=~joint_mask
        )
        joint_tokens = joint_tokens * joint_mask.unsqueeze(-1).to(
            dtype=joint_tokens.dtype
        )

        pep_joint = joint_tokens[:, : self.max_pep_len][:, :pep_len]
        hla_joint = joint_tokens[:, self.max_pep_len :][:, :hla_len]
        return pep_joint, hla_joint


    @staticmethod
    def _masked_mean(
        tokens: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("tokens and valid_mask have incompatible shapes.")
        weights = valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


    @staticmethod
    def _batch_vector(
        batch: Optional[torch.Tensor],
        num_nodes: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if batch is None:
            return torch.zeros(num_nodes, dtype=torch.long, device=device)
        batch = batch.to(device=device, dtype=torch.long)
        if batch.ndim != 1 or batch.numel() != num_nodes:
            raise ValueError(
                f"{name} must be a vector with one entry per node; "
                f"got {tuple(batch.shape)} for {num_nodes} nodes."
            )
        if batch.numel() > 0 and int(batch.min().item()) < 0:
            raise ValueError(f"{name} cannot contain negative graph indices.")
        return batch


    @staticmethod
    def _broadcast_global(
        global_feature: torch.Tensor,
        batch: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if global_feature.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, hidden].")
        if batch.numel() == 0:
            return global_feature.new_empty((0, global_feature.size(-1)))
        max_graph = int(batch.max().item())
        if max_graph >= global_feature.size(0):
            raise ValueError(
                f"{name} has {global_feature.size(0)} rows but batch references "
                f"graph index {max_graph}."
            )
        return global_feature.index_select(0, batch)


    def forward(self, x_pep, x_pretrain_pep, peptide_edge_index, x_p, x_pretrain_p,
                residue_edge_index, residue_edge_weight, peptide_edge_weight,
                prot_node_pos=None, pep_batch=None, prot_batch=None):
        if self.training:
            raise RuntimeError("GTpMHC is inference-only; call model.eval().")
        device = x_pretrain_pep.device
        x_pretrain_p = x_pretrain_p.to(device=device)
        x_pep = x_pep.to(device=device, dtype=x_pretrain_pep.dtype)
        x_p = x_p.to(device=device, dtype=x_pretrain_p.dtype)
        peptide_edge_index = peptide_edge_index.to(device=device, dtype=torch.long)
        residue_edge_index = residue_edge_index.to(device=device, dtype=torch.long)
        peptide_edge_weight = peptide_edge_weight.to(device=device)
        residue_edge_weight = residue_edge_weight.to(device=device)
        pep_batch = self._batch_vector(
            pep_batch, x_pep.size(0), device, "pep_batch"
        )
        prot_batch = self._batch_vector(
            prot_batch, x_p.size(0), device, "prot_batch"
        )

        # Encode raw and ESM residue features before adding sequence context.
        pep_seed, hla_seed = self._project_input_features(
            x_pep, x_pretrain_pep, x_p, x_pretrain_p
        )
        pep_dense, pep_mask = to_dense_batch(pep_seed, pep_batch)
        hla_dense, hla_mask = to_dense_batch(hla_seed, prot_batch)
        if pep_dense.size(0) != hla_dense.size(0):
            raise ValueError(
                "Peptide and HLA batches must contain the same number of graphs."
            )

        if prot_node_pos is not None:
            hla_positions, _ = to_dense_batch(
                prot_node_pos.to(device=device).view(-1, 1),
                prot_batch,
                max_num_nodes=hla_dense.size(1),
            )
            hla_positions = hla_positions.squeeze(-1)
        else:
            hla_positions = None

        pep_encoded, hla_encoded = (
            self._encode_transformer_pair(
                pep_dense,
                pep_mask,
                hla_dense,
                hla_mask,
                hla_positions,
            )
        )
        pep_global = self._masked_mean(pep_encoded, pep_mask)
        hla_global = self._masked_mean(hla_encoded, hla_mask)
        batch_size = pep_global.size(0)

        # Broadcast sequence context into both residue graphs.
        pep_global_nodes = self._broadcast_global(
            pep_global, pep_batch, "pep_global"
        )
        hla_global_nodes = self._broadcast_global(
            hla_global, prot_batch, "hla_global"
        )
        pep_x_raw = torch.cat(
            [x_pep, x_pretrain_pep, pep_global_nodes], dim=-1
        )
        residue_x_raw = torch.cat(
            [x_p, x_pretrain_p, hla_global_nodes], dim=-1
        )
        pep_x_cl = self.pep_graph_input_stem(pep_x_raw)
        residue_x_cl = self.prot_graph_input_stem(residue_x_raw)
        peptide_edge_weight = peptide_edge_weight.to(dtype=pep_x_cl.dtype)
        residue_edge_weight = residue_edge_weight.to(dtype=residue_x_cl.dtype)

        residue_edge_attr = _rbf(
            residue_edge_weight,
            D_max=1.0,
            D_count=self.edge_dim,
            device=device,
        )
        pep_edge_attr = _rbf(
            peptide_edge_weight,
            D_max=1.0,
            D_count=self.edge_dim,
            device=device,
        )
        for idx in range(self.total_layer):
            residue_x_cl_conv = self.prot_convs[idx](
                residue_x_cl,
                residue_edge_index,
                edge_attr=residue_edge_attr,
            )
            residue_x_cl = residue_x_cl + F.relu(
                self.prot_conv_norms[idx](residue_x_cl_conv)
            )
            pep_x_cl_conv = self.pep_convs[idx](
                pep_x_cl,
                peptide_edge_index,
                edge_attr=pep_edge_attr,
            )
            pep_x_cl = pep_x_cl + F.relu(
                self.pep_conv_norms[idx](pep_x_cl_conv)
            )
            assignment = self.clusters[idx](
                residue_x_cl, residue_edge_index
            )
            residue_hx, residue_mask = to_dense_batch(
                residue_x_cl, prot_batch
            )
            assignment_dense, _ = to_dense_batch(assignment, prot_batch)
            assignment_dense, cluster_x = _cluster_pool(residue_hx, assignment_dense, residue_mask)
            cluster_x = self.prot_norms[idx](cluster_x)

            cluster_count = self.num_cluster[idx]
            cluster_x_flat = cluster_x.reshape(
                batch_size * cluster_count, -1
            )
            cluster_residue_batch = torch.arange(
                batch_size, device=device
            ).repeat_interleave(cluster_count)
            pep_pool = global_mean_pool(
                pep_x_cl, pep_batch, size=batch_size
            )
            p2p_edge_index = torch.stack(
                [
                    torch.arange(
                        batch_size * cluster_count, device=device
                    ),
                    torch.arange(
                        batch_size, device=device
                    ).repeat_interleave(cluster_count),
                ]
            )

            pep_x_updated, cluster_x_updated = (
                self.inter_convs_global[idx](
                    pep_pool,
                    pep_x_cl,
                    cluster_x_flat,
                    p2p_edge_index,
                    pep_batch,
                )
            )
            pep_x_cl = pep_x_cl + self.c2p_mlps[idx](pep_x_updated)
            pep_x_local = pep_x_cl
            cluster_hx_updated, _ = to_dense_batch(
                cluster_x_updated, cluster_residue_batch
            )
            residue_update = (
                assignment_dense @ cluster_hx_updated
            )[residue_mask]
            residue_x_cl = residue_x_cl + F.relu(
                self.residue_lins[idx](residue_update)
            )
            residue_x_cl = residue_x_cl + self.c2r_mlps[idx](residue_x_cl)
            pep_x_local = self.pep_gns[idx](pep_x_local, pep_batch)
            residue_x_cl = self.prot_gns[idx](residue_x_cl, prot_batch)

        # Keep every graph-enhanced residue as a decoder token. Fixed peptide/HLA
        # slots reproduce TranspMHC's batch-independent concatenation while the
        # valid mask prevents padding from participating in attention or scoring.
        pep_graph_tokens, pep_graph_mask = to_dense_batch(
            pep_x_local,
            pep_batch,
            max_num_nodes=self.max_pep_len,
        )
        hla_graph_tokens, hla_graph_mask = to_dense_batch(
            residue_x_cl,
            prot_batch,
            max_num_nodes=self.transpmhc_hla_len,
        )
        if not torch.equal(pep_graph_mask.sum(dim=1), pep_mask.sum(dim=1)):
            raise RuntimeError("Peptide graph tokens lost sequence/node alignment.")
        if not torch.equal(hla_graph_mask.sum(dim=1), hla_mask.sum(dim=1)):
            raise RuntimeError("HLA graph tokens lost sequence/node alignment.")

        enc_outputs = torch.cat(
            (pep_graph_tokens, hla_graph_tokens), dim=1
        )
        graph_token_mask = torch.cat(
            (pep_graph_mask, hla_graph_mask), dim=1
        )
        dec_outputs = self.graph_sequence_decoder(
            enc_outputs,
            graph_token_mask,
        )
        dec_flat = dec_outputs.reshape(dec_outputs.shape[0], -1)
        # Decoder output is already normalized token-wise and padded tokens are zero.
        # Avoid flattened LayerNorm, which would turn those zeros into non-zero values.
        return self.graph_sequence_projection(dec_flat)


INFERENCE_MODEL_KEYS = frozenset({
    "hidden_channels", "K", "heads", "max_pep_len", "transformer_layers",
    "max_hla_position", "transpmhc_hla_len", "pep_evo_channels", "prot_evo_channels",
    "pep_raw_channels", "prot_raw_channels", "t",
})
_UNUSED_LEGACY_MODEL_KEYS = frozenset({
    "dropout", "dropout_attn_score", "dropout_cluster_edge", "drop_pep_node",
    "drop_residue", "lambda_cl", "lambda_o", "tau", "tau_lse", "transformer_dropout",
    "window_padding", "device", "total_layer", "use_raw_feature_fusion",
})


def inference_model_kwargs(legacy_kwargs):
    """Select inference arguments from a known legacy model_kwargs dictionary.

    Only explicitly audited, inference-inactive flags may be dropped. A raw
    feature bypass or inconsistent graph depth is not an equivalent model.
    This function does not accept a complete training-config document.
    """
    if not isinstance(legacy_kwargs, Mapping):
        raise ValueError("Model kwargs must be a mapping.")
    unknown = set(legacy_kwargs) - INFERENCE_MODEL_KEYS - _UNUSED_LEGACY_MODEL_KEYS
    if unknown:
        raise ValueError("Unknown model configuration keys: " + ", ".join(sorted(map(str, unknown))))
    if legacy_kwargs.get("use_raw_feature_fusion", True) is not True:
        raise ValueError("GTpMHC requires the original fulltoken raw-feature fusion path.")
    if "total_layer" in legacy_kwargs:
        depth = legacy_kwargs["total_layer"]
        clusters = legacy_kwargs.get("K", (5, 10))
        if type(depth) is not int or depth != len(clusters):
            raise ValueError("Legacy total_layer must equal len(K).")
    return {name: value for name, value in legacy_kwargs.items() if name in INFERENCE_MODEL_KEYS}


def _unused_legacy_shapes(model):
    """Exact 76-key allowlist from the original fulltoken state_dict.

    No wildcard prefixes are accepted: misspelled or unrelated parameters must
    fail. Shapes are checked even for known parameters that are not used.
    """
    hidden = model.hidden_channels
    shapes = {}

    def mlp(prefix, channels, out_norm=False):
        first, middle, last = channels
        dimensions = {"0.weight": (middle, first), "0.bias": (middle,),
                      "1.weight": (middle,), "1.bias": (middle,),
                      "4.weight": (last, middle), "4.bias": (last,)}
        if out_norm:
            dimensions.update({"5.weight": (last,), "5.bias": (last,)})
        shapes.update({prefix + ".mlp." + name: shape for name, shape in dimensions.items()})

    for name in ("cls_out_branch1", "cls_out_branch2"):
        mlp(name, (hidden * 4, hidden, 1))
    mlp("consistency_predictor", (hidden, hidden * 2, hidden))
    mlp("consistency_projector", (hidden * 4, hidden * 2, hidden), out_norm=True)
    for name in ("pep_graph_out", "prot_graph_out"):
        mlp(name, (hidden, hidden * 2, hidden), out_norm=True)
    for chain in ("pep", "prot"):
        raw = getattr(model, chain + "_raw_channels")
        evo = getattr(model, chain + "_evo").mlp[0].in_features
        mlp("mae_" + chain + "_raw_decoder", (hidden, hidden, raw))
        mlp("mae_" + chain + "_evo_decoder", (hidden, hidden * 2, evo))
        shapes["mae_" + chain + "_mask_token"] = (1, hidden)
    flattened = (model.max_pep_len + model.transpmhc_hla_len) * hidden
    projection = {"0.weight": (hidden * 2, flattened), "0.bias": (hidden * 2,),
                  "2.weight": (hidden * 2,), "2.bias": (hidden * 2,),
                  "4.weight": (hidden * 4, hidden * 2), "4.bias": (hidden * 4,),
                  "5.weight": (hidden * 4,), "5.bias": (hidden * 4,)}
    shapes.update({"transpmhc_pair_projection." + name: shape for name, shape in projection.items()})
    return shapes


def load_gtpmhc_state(model, checkpoint):
    """Load a trusted original/compact peptide-first state with strict auditing.

    Accepts a state_dict, {state_dict: ...}, or {model_state_dict: ...}, including
    an all-keys DataParallel ``module.`` prefix. Every retained parameter must
    exist and match shape. Only the exact 76 known unused legacy tensors may be
    ignored, and their shapes must also match. No strict=False load is used.
    ``torch.load`` and the trust decision for local files belong to the caller.
    """
    if not isinstance(model, GTpMHC):
        raise TypeError("load_gtpmhc_state requires a GTpMHC model.")
    if isinstance(checkpoint, Mapping):
        for name in ("model_state_dict", "state_dict"):
            if name in checkpoint:
                checkpoint = checkpoint[name]
                break
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError("Checkpoint must contain a nonempty state_dict.")
    state = dict(checkpoint)
    if any(not isinstance(key, str) for key in state):
        raise ValueError("Checkpoint state_dict keys must be strings.")
    prefixed = [key.startswith("module.") for key in state]
    if all(prefixed):
        state = {key[7:]: value for key, value in state.items()}
    elif any(prefixed):
        raise ValueError("Checkpoint mixes prefixed and unprefixed state_dict keys.")
    expected = model.state_dict()
    unused = _unused_legacy_shapes(model)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected) - set(unused))
    if missing or unexpected:
        details = []
        if missing:
            details.append("Missing retained keys: " + ", ".join(missing))
        if unexpected:
            details.append("Unexpected checkpoint keys: " + ", ".join(unexpected))
        raise ValueError("; ".join(details))
    wrong = []
    for key, value in state.items():
        shape = tuple(expected[key].shape) if key in expected else unused[key]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            actual = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            wrong.append(f"{key}: expected {shape}, got {actual}")
    if wrong:
        raise ValueError("Checkpoint shape/type mismatch: " + "; ".join(wrong))
    dropped = sorted(set(state).intersection(unused))
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    return {"retained_keys": len(expected), "ignored_legacy_keys": dropped,
            "checkpoint_format": "legacy_peptide_first" if dropped else "compact_peptide_first"}


__all__ = ["GTpMHC", "inference_model_kwargs", "load_gtpmhc_state", "INFERENCE_MODEL_KEYS"]

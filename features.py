"""GTpMHC sequence features: raw33, full-chain ESM-2 and induced HLA graphs.

Only local ESM checkpoints are read. Public HLA reference sequences supply
numbering, never precomputed features. Importing this module requires stdlib only.
"""

from collections import OrderedDict
import importlib
import math
from pathlib import Path

FEATURE_SCHEMA = "GTpMHC_raw33_v1"
RESIDUES = "ACDEFGHIKLMNPQRSTVWYX"
AA = frozenset(RESIDUES)
PSEUDO_POS_MATURE = (7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80,
                     81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156,
                     158, 159, 163, 167, 171)
DRA_PSEUDO_POS_MATURE = (9, 11, 22, 24, 31, 52, 53, 58, 59, 61, 65, 66, 68, 72, 73)
DRB1_PSEUDO_POS_MATURE = (9, 11, 13, 26, 28, 30, 47, 57, 67, 70, 71, 74, 77, 78, 81, 85, 86, 89, 90)
MIN_COVERAGE = 0.98
MIN_IDENTITY = 0.70
REFERENCE_DOMAIN_LENGTH = 182

# Public numbering references (I: IPD-IMGT/HLA A*02:01:01:01; II: UniProt DRA/DRB1).
REFERENCE_SEQUENCES = {'I': 'MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQRTDAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGQEQRYTCHVQHEGLPKPLTLRWEPSSQPTIPIVGIIAGLVLFGAVITGAVVAAVMWRRKSSDRKGGSYSQAASSDSAQGSDVSLTACKV',
 'DRA': 'MAISGVPVLGFFIIAVLMSAQESWAIKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITNVPPEVTVLTNSPVELREPNVLICFIDKFTPPVVNVTWLRNGKPVTTGVSETVFLPREDHLFRKFHYLPFLPSTEDVYDCRVEHWGLDEPLLKHWEFDAPSPLPETTENVVCALGLTVGLVGIIIGTIFIIKGLRKSNAAERRGPL',
 'DRB1': 'MVCLKLPGGSCMTALTVTLMVLSSPLALSGDTRPRFLWQPKRECHFFNGTERVRFLDRYFYNQEESVRFDSDVGEFRAVTELGRPDAEYWNSQKDILEQARAAVDTYCRHNYGVVESFTVQRRVQPKVTVYPSKTQPLQHHNLLVCSVSGFYPGSIEVRWFLNGQEEKAGMVSTGLIQNGDWTFQTLVMLETVPRSGEVYTCQVEHPSVTSPLTVEWRARSESAQSKMLSGVGGFVLGLLFLGAGLFIYFRNQKGHSGLQPTGFLS'}
REFERENCE_SOURCES = {'I': 'https://raw.githubusercontent.com/ANHIG/IMGTHLA/d972ae287948dc2e5468a40ff3b3d27227541189/fasta/A_prot.fasta',
 'DRA': 'https://www.uniprot.org/uniprotkb/P01903/entry',
 'DRB1': 'https://www.uniprot.org/uniprotkb/P01911/entry'}


def normalize_sequence(value, label="Sequence"):
    if not isinstance(value, str):
        raise ValueError(f"{label}: amino-acid sequences must be text.")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    sequence = "".join("".join(lines).split()).upper()
    if not sequence or set(sequence).difference(AA):
        raise ValueError(f"{label}: use 20 standard amino acids and X, with at most one leading FASTA header.")
    return sequence


def _sequence(value, label):
    return normalize_sequence(value, label)


def load_reference():
    return REFERENCE_SEQUENCES["I"][24:24 + REFERENCE_DOMAIN_LENGTH]


def load_class_ii_reference():
    return {"dra": REFERENCE_SEQUENCES["DRA"][25:125],
            "drb1": REFERENCE_SEQUENCES["DRB1"][29:129]}

def _argmax(values):
    # Stable tie handling inside a path; distinct optimum endpoints are checked.
    return max(range(3), key=values.__getitem__)

def _align(reference, full):
    """Affine-gap semiglobal alignment: cover reference, free target flanks."""
    n, m = len(reference), len(full)
    neg = -10 ** 9
    width = m + 1
    trace = [bytearray((n + 1) * width) for _ in range(3)]
    previous = [[0] * width, [neg] * width, [0] * width]
    for i in range(1, n + 1):
        current = [[neg] * width for _ in range(3)]
        current[1][0] = -8 - (i - 1)
        trace[1][i * width] = 0 if i == 1 else 1
        for j in range(1, m + 1):
            index = i * width + j
            source = _argmax([previous[state][j - 1] for state in range(3)])
            substitution = 2 if reference[i - 1] == full[j - 1] != "X" else -1
            current[0][j] = previous[source][j - 1] + substitution
            trace[0][index] = source
            choices = [previous[0][j] - 8, previous[1][j] - 1, previous[2][j] - 8]
            source = _argmax(choices)
            current[1][j] = choices[source]
            trace[1][index] = source
            choices = [current[0][j - 1] - 8, current[1][j - 1] - 8, current[2][j - 1] - 1]
            source = _argmax(choices)
            current[2][j] = choices[source]
            trace[2][index] = source
        previous = current
    best = max(max(row) for row in previous)
    endpoints = [(j, state) for j in range(m + 1) for state in range(3)
                 if previous[state][j] == best]
    # Repeated HLA domains can create many optima. They must not select silently.
    if len(endpoints) > 32:
        raise ValueError("Multiple equivalent numbering positions found in the HLA sequence; please provide a single complete HLA-I chain and check for repeated domains.")
    mappings = []
    for endpoint, state in endpoints:
        i, j, mapping = n, endpoint, {}
        while i > 0:
            index = i * width + j
            source = trace[state][index]
            if state == 0:
                if j == 0:
                    break
                mapping[i] = j - 1
                i -= 1
                j -= 1
            elif state == 1:
                i -= 1
            else:
                if j == 0:
                    break
                j -= 1
            state = source
        mappings.append(mapping)
    return mappings

def _alignment_stats(reference, full, mapping):
    matches = sum(reference[position - 1] == full[index] != "X" for position, index in mapping.items())
    return matches / len(reference), len(mapping) / len(reference)

def _map_chain(full, reference, positions, chain_name, min_identity,
               sequence_offset=None, expected_pseudo=None, max_hla_position=256):
    full = _sequence(full, chain_name + " protein")
    if len(full) < max(positions) - 1:
        raise ValueError(f"Please provide a complete {chain_name} chain covering the binding domain; a pseudo-sequence cannot recover full context features.")
    if max_hla_position < max(positions):
        raise ValueError("Model mature-chain position capacity is smaller than the current site definition.")
    if sequence_offset is not None and (type(sequence_offset) is not int or not -1 <= sequence_offset <= 60):
        raise ValueError("sequence_offset must be an integer between -1 and 60.")
    expected = None
    if expected_pseudo is not None:
        expected = _sequence(expected_pseudo, "Expected pseudo-sequence")
        if len(expected) != len(positions):
            raise ValueError(f"Expected {chain_name} pseudo-sequence must be {len(positions)} residues long.")
    candidates = _align(reference, full)
    viable = []
    for mapping in candidates:
        identity, coverage = _alignment_stats(reference, full, mapping)
        if identity < min_identity or coverage < MIN_COVERAGE:
            continue
        if not all(position in mapping for position in positions):
            continue
        # Reject insertions/deletions rather than silently changing the
        # constant-offset numbering used to train this model.
        offsets = {index - (position - 1) for position, index in mapping.items()}
        if len(offsets) != 1:
            continue
        offset = next(iter(offsets))
        indices = [mapping[position] for position in positions]
        pseudo = "".join(full[index] for index in indices)
        if "X" in pseudo:
            continue
        viable.append((tuple(indices), offset, identity, coverage, pseudo))
    # Deduplicate identical traceback outcomes, then handle ambiguity before any
    # requested offset or expected sequence can bypass it.
    distinct = {entry[0]: entry for entry in viable}
    if len(distinct) > 1:
        raise ValueError("Multiple tied matches found for HLA numbering; please check for repeated domains instead of selecting a mapping arbitrarily.")
    if not distinct:
        raise ValueError(f"Unable to reliably identify {chain_name} mature-chain positions: insufficient sequence identity, missing key residues, or internal insertions/deletions. Please verify the complete {chain_name} sequence; arbitrary proteins are not treated as HLA.")
    indices, offset, identity, coverage, pseudo = next(iter(distinct.values()))
    if not -1 <= offset <= 60:
        raise ValueError("HLA mature-chain start position is outside the supported leader-sequence range; please provide a single complete corresponding chain.")
    if sequence_offset is not None and offset != sequence_offset:
        raise ValueError(f"Specified mature-chain offset {sequence_offset} does not match the detected offset {offset}.")
    if expected is not None and expected != pseudo:
        raise ValueError("Expected pseudo-sequence does not match the slice from the input full HLA sequence.")
    return {"full_sequence": full, "pseudo_sequence": pseudo, "indices": list(indices),
            "pseudo_positions_mature": list(positions), "sequence_offset": offset,
            "alignment_identity": round(identity, 6), "alignment_coverage": round(coverage, 6),
            "mapping_method": "hla_i_reference_alignment_constant_offset"}

def map_hla_sequence(full, sequence_offset=None, pseudo_sequence=None, max_hla_position=256):
    return _map_chain(full, load_reference(), PSEUDO_POS_MATURE, "HLA-I", MIN_IDENTITY,
                      sequence_offset=sequence_offset, expected_pseudo=pseudo_sequence,
                      max_hla_position=max_hla_position)

def map_hla_ii_sequences(dra, drb1, dra_sequence_offset=None, sequence_offset=None,
                         pseudo_sequence=None, max_hla_position=256):
    """Number DRA and DRB1 independently; reproduce data_init_II's 15+19 slots."""
    if max_hla_position < 33:
        raise ValueError("Class II model position capacity must cover all 34 input slots (0 to 33).")
    # Mature positions are metadata in the II preprocessing; the model receives
    # concatenated node_pos=0..33, not a merged mature-position embedding.
    mapping_limit = max(max_hla_position, max(DRB1_PSEUDO_POS_MATURE))
    expected_alpha = expected_beta = None
    if pseudo_sequence is not None:
        expected = _sequence(pseudo_sequence, "Expected HLA-DR pseudo-sequence")
        if len(expected) != 34:
            raise ValueError("Expected HLA-DR pseudo-sequence must consist of 15 DRA residues plus 19 DRB1 residues (34 residues total).")
        expected_alpha, expected_beta = expected[:15], expected[15:]
    references = load_class_ii_reference()
    alpha = _map_chain(dra, references["dra"], DRA_PSEUDO_POS_MATURE, "DRA", 0.90,
                       sequence_offset=dra_sequence_offset, expected_pseudo=expected_alpha,
                       max_hla_position=mapping_limit)
    beta = _map_chain(drb1, references["drb1"], DRB1_PSEUDO_POS_MATURE, "DRB1", 0.70,
                      sequence_offset=sequence_offset, expected_pseudo=expected_beta,
                      max_hla_position=mapping_limit)
    alpha["mapping_method"] = beta["mapping_method"] = "hla_dr_reference_alignment_constant_offset"
    return {"mhc_class": "II", "dra_mapping": alpha, "drb1_mapping": beta,
            "pseudo_sequence": alpha["pseudo_sequence"] + beta["pseudo_sequence"],
            "mapping_method": "separate_dra_drb1_reference_alignment"}

# Columns: residue mass, pKa, pKb, pKx, pI, hydrophobicity at pH 2 and pH 7.
_PROPERTY_VALUES = (
    (71.08, 103.15, 115.09, 129.12, 147.18, 57.05, 137.14, 113.16, 128.18, 113.16, 131.2, 114.11, 97.12, 128.13, 156.19, 87.08, 101.11, 99.13, 186.22, 163.18),
    (2.34, 1.96, 1.88, 2.19, 1.83, 2.34, 1.82, 2.36, 2.18, 2.36, 2.28, 2.02, 1.99, 2.17, 2.17, 2.21, 2.09, 2.32, 2.83, 2.32),
    (9.69, 10.28, 9.6, 9.67, 9.13, 9.6, 9.17, 9.6, 8.95, 9.6, 9.21, 8.8, 10.6, 9.13, 9.04, 9.15, 9.1, 9.62, 9.39, 9.62),
    (0.0, 8.18, 3.65, 4.25, 0.0, 0.0, 6.0, 0.0, 10.53, 0.0, 0.0, 0.0, 0.0, 0.0, 12.48, 0.0, 0.0, 0.0, 0.0, 0.0),
    (6.0, 5.07, 2.77, 3.22, 5.48, 5.97, 7.59, 6.02, 9.74, 5.98, 5.74, 5.41, 6.3, 5.65, 10.76, 5.68, 5.6, 5.96, 5.89, 5.96),
    (47, 52, -18, 8, 92, 0, -42, 100, -37, 100, 74, -41, -46, -18, -26, -7, 13, 79, 84, 49),
    (41, 49, -55, -31, 100, 0, 8, 99, -23, 97, 74, -28, -46, -10, -14, -5, 13, 76, 97, 63),
)


def seq_feature(sequence):
    """The training raw33 channels, including the original X=0.5 convention."""
    np = importlib.import_module("numpy")
    sequence = "".join(aa if aa in AA else "X" for aa in str(sequence).upper().strip())
    groups = ("AILMV", "FWY", "CNQST", "DE", "HKR")
    normalized = []
    for values in _PROPERTY_VALUES:
        low, high = min(values), max(values)
        normalized.append([(value - low) / (high - low) for value in values] + [0.5])
    rows = [[float(i == j) for j in range(21)]
            + [float(aa in group) for group in groups]
            + [values[i] for values in normalized]
            for i, aa in enumerate(RESIDUES)]
    return np.asarray([rows[RESIDUES.index(aa)] for aa in sequence],
                      dtype=np.float32).reshape(len(sequence), 33)


def contact_map(probabilities, contact_threshold=0.5):
    """Undirected max-reduced contacts + +/-1,2 edges + unit self-loops."""
    torch = importlib.import_module("torch")
    probabilities = torch.as_tensor(probabilities).float().cpu()
    count = probabilities.shape[0]
    if tuple(probabilities.shape) != (count, count):
        raise ValueError("Contact probabilities must be a square matrix.")
    eligible = probabilities >= contact_threshold
    values = probabilities.masked_fill(~eligible, -torch.inf)
    values = torch.maximum(values, values.t())
    mask = eligible | eligible.t()
    for gap in (1, 2):
        index = torch.arange(max(0, count - gap))
        weight = values[index, index + gap].clamp_min(contact_threshold)
        values[index, index + gap] = values[index + gap, index] = weight
        mask[index, index + gap] = mask[index + gap, index] = True
    mask.fill_diagonal_(False)
    edges = mask.nonzero(as_tuple=False).t().contiguous()
    weights = values[edges[0], edges[1]]
    loops = torch.arange(count).repeat(2, 1)
    return torch.cat([edges, loops], dim=1).long(), torch.cat([weights, torch.ones(count)]).float()


def esm_extract(model, batch_converter, sequence, layer=33, approach="last", dim=1280,
                max_len=1022, window_size=600, interval=200):
    """Full-chain ESM embeddings/contacts; overlap averages match training.

    Long-chain contacts cover within-window pairs only. Logits are unused by
    GTpMHC and are deliberately not accumulated or retained here.
    """
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")
    sequence = normalize_sequence(sequence)
    if approach not in {"last", "sum", "mean"}:
        raise ValueError("ESM approach must be last, sum or mean.")
    layers = [layer] if approach == "last" else list(range(1, layer + 1))
    device = next(model.parameters()).device

    def extract(fragment):
        _, _, tokens = batch_converter([("sequence", fragment)])
        with torch.no_grad():
            result = model(tokens.to(device, non_blocking=True),
                           repr_layers=layers, return_contacts=True)
        representations = torch.stack([result["representations"][i][0, 1:len(fragment) + 1]
                                       for i in layers])
        representation = (representations[-1] if approach == "last" else
                          representations.sum(dim=0) if approach == "sum" else
                          representations.mean(dim=0))
        return representation.detach().float().cpu(), result["contacts"][0].detach().float().cpu()

    length = len(sequence)
    if length <= max_len:
        return extract(sequence)
    embeddings = np.zeros((length, dim), dtype=np.float32)
    token_count = np.zeros((length, 1), dtype=np.float32)
    contacts = np.zeros((length, length), dtype=np.float32)
    contact_count = np.zeros((length, length), dtype=np.float32)
    for step in range(math.ceil((length - window_size) / interval) + 1):
        start = step * interval
        end = min(start + window_size, length)
        if end == length and end - start < window_size and length > window_size:
            start = end - window_size
        local_embeddings, local_contacts = extract(sequence[start:end])
        if local_embeddings.shape != (end - start, dim):
            raise ValueError(f"Expected ESM embedding width {dim}, got {local_embeddings.shape}.")
        embeddings[start:end] += local_embeddings.numpy()
        token_count[start:end] += 1
        contacts[start:end, start:end] += local_contacts.numpy()
        contact_count[start:end, start:end] += 1
    return (torch.from_numpy(embeddings / np.maximum(token_count, 1)).float(),
            torch.from_numpy(contacts / np.maximum(contact_count, 1)).float())


class GTpMHCFeatures:
    """Generate and cache training-compatible features from full input chains."""

    def __init__(self, settings, base_dir, device, max_hla_position=256):
        self.settings, self.base_dir = dict(settings), Path(base_dir)
        self.device, self.max_hla_position = str(device), int(max_hla_position)
        self.mhc_class = self.settings.get("mhc_class", "I")
        if self.mhc_class not in {"I", "II"}:
            raise ValueError("mhc_class must be I or II.")
        if self.settings.get("feature_schema", FEATURE_SCHEMA) != FEATURE_SCHEMA:
            raise ValueError(f"Use feature_schema={FEATURE_SCHEMA}: original training raw33 channels.")
        self.torch = importlib.import_module("torch")
        self._peptide_cache, self._protein_cache, self._chain_cache = (OrderedDict() for _ in range(3))
        self._esm_model = self._batch_converter = None

    def _remember(self, cache, key, item):
        limit = int(self.settings.get("max_cached_sequences", 256))
        if limit > 0:
            cache[key] = item
            cache.move_to_end(key)
            while len(cache) > limit:
                cache.popitem(last=False)

    def _load_esm(self):
        if self._esm_model is not None:
            return
        checkpoints = []
        for key, filename in (("esm_checkpoint", "esm2_t33_650M_UR50D.pt"),
                              ("esm_contact_checkpoint", "esm2_t33_650M_UR50D-contact-regression.pt")):
            path = Path(self.settings.get(key, "weights/" + filename)).expanduser()
            path = path if path.is_absolute() else self.base_dir / path
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"A nonempty local ESM checkpoint is required: {path}")
            checkpoints.append(path)
        torch = self.torch
        device_name = self.settings.get("esm_device", "cpu")
        device = torch.device(self.device if device_name == "auto" else device_name)
        if device.type == "cuda" and (not torch.cuda.is_available()
                                     or (device.index or 0) >= torch.cuda.device_count()):
            raise RuntimeError("The configured ESM CUDA device is unavailable.")
        try:
            esm = importlib.import_module("esm")
        except ImportError as exc:
            raise RuntimeError("Install fair-esm to extract sequence features.") from exc
        data, contacts = [torch.load(path, map_location="cpu", weights_only=False)
                          for path in checkpoints]
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            "esm2_t33_650M_UR50D", data, contacts)
        self._esm_model = model.eval().to(device)
        self._batch_converter = alphabet.get_batch_converter()

    def prepare(self):
        self._load_esm()
        return self

    def _sequence_features(self, sequence, kind):
        self._load_esm()
        torch = self.torch
        representation, contacts = esm_extract(
            self._esm_model, self._batch_converter, sequence,
            layer=int(self.settings.get("esm_layer", 33)),
            approach=self.settings.get("esm_approach", "last"))
        edges, weights = contact_map(contacts, float(self.settings.get("contact_threshold", 0.5)))
        raw = torch.from_numpy(seq_feature(sequence)).float()
        item = {"seq": sequence, "num_nodes": len(sequence), "edge_weight": weights,
                "node_pos": torch.arange(len(sequence)).reshape(-1, 1)}
        if kind == "peptide":
            item.update(x_pep=raw, x_pretrain_pep=representation, peptide_edge_index=edges)
        else:
            item.update(seq_feat=raw, token_representation=representation, edge_index=edges)
        self._validate_item(item, kind)
        return item

    def _validate_item(self, item, kind):
        torch, length = self.torch, len(item["seq"])
        raw_key, esm_key, edge_key = (("x_pep", "x_pretrain_pep", "peptide_edge_index")
                                      if kind == "peptide" else
                                      ("seq_feat", "token_representation", "edge_index"))
        for key, width in ((raw_key, 33), (esm_key, 1280)):
            tensor = torch.as_tensor(item[key])
            if tuple(tensor.shape) != (length, width) or not torch.isfinite(tensor).all():
                raise ValueError(f"{key} must contain finite values of shape ({length}, {width}).")
        edges, weights = torch.as_tensor(item[edge_key]), torch.as_tensor(item["edge_weight"])
        if (edges.ndim != 2 or edges.shape[0] != 2 or weights.shape != (edges.shape[1],)
                or not torch.isfinite(edges).all() or not torch.isfinite(weights).all()
                or not torch.equal(edges, edges.long().to(edges.dtype))):
            raise ValueError("Invalid graph edge indices or weights.")
        if edges.numel() and (edges.min() < 0 or edges.max() >= length):
            raise ValueError("Graph edges refer to residues outside the sequence.")
        if kind == "protein":
            positions = item.get("pseudo_pos_mature", item["node_pos"]).reshape(-1)
            if (length != 34 or len(positions) != 34 or not torch.isfinite(positions).all()
                    or not torch.equal(positions, positions.long().to(positions.dtype))
                    or positions.min() < 0 or positions.max() > self.max_hla_position):
                raise ValueError("The model requires 34 valid HLA residue positions.")

    def peptide_feature(self, sequence):
        sequence = normalize_sequence(sequence)
        if sequence in self._peptide_cache:
            self._peptide_cache.move_to_end(sequence)
            return self._peptide_cache[sequence], "memory_cache"
        item = self._sequence_features(sequence, "peptide")
        self._remember(self._peptide_cache, sequence, item)
        return item, "live_esm"

    def _slice_chain_features(self, mapping):
        key = (mapping["full_sequence"], tuple(mapping["indices"]))
        if key in self._chain_cache:
            self._chain_cache.move_to_end(key)
            return self._chain_cache[key]
        torch = self.torch
        full = mapping["full_sequence"]
        full_item = self._sequence_features(full, "chain")
        selected = torch.tensor(mapping["indices"], dtype=torch.long)
        count = len(selected)
        old_to_new = torch.full((len(full),), -1, dtype=torch.long)
        old_to_new[selected] = torch.arange(count)
        edges = full_item["edge_index"]
        keep = (old_to_new[edges[0]] >= 0) & (old_to_new[edges[1]] >= 0)
        item = {"seq": mapping["pseudo_sequence"], "seq_feat": full_item["seq_feat"][selected],
                "token_representation": full_item["token_representation"][selected],
                "edge_index": old_to_new[edges[:, keep]], "edge_weight": full_item["edge_weight"][keep]}
        self._remember(self._chain_cache, key, item)
        return item

    def _generate_hla(self, mapping):
        torch = self.torch
        item = dict(self._slice_chain_features(mapping))
        metadata = {key: mapping.get(key) for key in (
            "mapping_method", "sequence_offset", "alignment_identity", "alignment_coverage",
            "pseudo_sequence", "pseudo_positions_mature")}
        metadata["mhc_class"] = "I"
        item.update(num_nodes=34, full_seq=mapping["full_sequence"],
                    node_pos=torch.arange(34).reshape(-1, 1),
                    full_node_pos=torch.tensor(mapping["indices"]).reshape(-1, 1),
                    pseudo_pos_mature=torch.tensor(mapping["pseudo_positions_mature"]).reshape(-1, 1),
                    signal_len=mapping["sequence_offset"], pseudo_matches=34,
                    pseudo_source="full_esm_slice_by_sequence_alignment", hla_mapping=metadata)
        self._validate_item(item, "protein")
        return item

    def _generate_hla_ii(self, mapping):
        torch = self.torch
        dra, drb1 = mapping["dra_mapping"], mapping["drb1_mapping"]
        if (len(dra["indices"]) != 15 or len(drb1["indices"]) != 19
                or mapping["pseudo_sequence"] != dra["pseudo_sequence"] + drb1["pseudo_sequence"]):
            raise ValueError("HLA-DR requires 15 DRA slots followed by 19 DRB1 slots.")
        alpha, beta = self._slice_chain_features(dra), self._slice_chain_features(drb1)
        metrics = ("mapping_method", "sequence_offset", "alignment_identity", "alignment_coverage",
                   "pseudo_sequence", "pseudo_positions_mature")
        metadata = {"mhc_class": "II", "mapping_method": mapping["mapping_method"],
                    "pseudo_sequence": mapping["pseudo_sequence"],
                    "dra": {key: dra.get(key) for key in metrics},
                    "drb1": {key: drb1.get(key) for key in metrics}}
        item = {"seq": mapping["pseudo_sequence"], "full_seq": drb1["full_sequence"],
                "dra_full_seq": dra["full_sequence"], "drb1_full_seq": drb1["full_sequence"],
                "num_nodes": 34, "node_pos": torch.arange(34).reshape(-1, 1),
                "chain_id": torch.tensor([0] * 15 + [1] * 19).reshape(-1, 1),
                "chain_full_node_pos": torch.tensor(dra["indices"] + drb1["indices"]).reshape(-1, 1),
                "pseudo_source": "separate_full_dra_drb1_esm_slice", "pseudo_matches": 34,
                "seq_feat": torch.cat([alpha["seq_feat"], beta["seq_feat"]]),
                "token_representation": torch.cat([alpha["token_representation"], beta["token_representation"]]),
                "edge_index": torch.cat([alpha["edge_index"], beta["edge_index"] + 15], dim=1),
                "edge_weight": torch.cat([alpha["edge_weight"], beta["edge_weight"]]),
                "hla_mapping": metadata}
        for name, chain in (("dra", dra), ("drb1", drb1)):
            item[name + "_full_node_pos"] = torch.tensor(chain["indices"]).reshape(-1, 1)
            item[name + "_pseudo_pos_mature"] = torch.tensor(chain["pseudo_positions_mature"]).reshape(-1, 1)
            item[name + "_signal_len"] = chain["sequence_offset"]
        # II training feeds 0..33 node_pos; unified mature/full positions would
        # silently change the model's position embeddings and are not supplied.
        self._validate_item(item, "protein")
        return item

    def protein_feature(self, protein, kind="auto", sequence_offset=None, pseudo_sequence=None,
                        dra_sequence=None, dra_sequence_offset=None):
        if kind not in {"auto", "full_sequence"}:
            raise ValueError("Provide a full HLA chain, not an allele identifier or pseudo-sequence.")
        full = normalize_sequence(protein)
        maximum = int(self.settings.get("max_protein_length", 2000))
        if not 34 < len(full) <= maximum:
            raise ValueError(f"Provide a full HLA chain between 35 and {maximum} residues.")
        if self.mhc_class == "II":
            if not isinstance(dra_sequence, str) or not dra_sequence.strip():
                raise ValueError("HLA-DR prediction requires both full DRA and DRB1 sequences.")
            dra = normalize_sequence(dra_sequence)
            if not 34 < len(dra) <= maximum:
                raise ValueError(f"Provide a full DRA chain between 35 and {maximum} residues.")
            mapping = map_hla_ii_sequences(dra, full, dra_sequence_offset=dra_sequence_offset,
                                           sequence_offset=sequence_offset, pseudo_sequence=pseudo_sequence,
                                           max_hla_position=self.max_hla_position)
            alpha, beta = mapping["dra_mapping"], mapping["drb1_mapping"]
            key = ("II", alpha["full_sequence"], tuple(alpha["indices"]),
                   beta["full_sequence"], tuple(beta["indices"]))
            generate = self._generate_hla_ii
        else:
            if dra_sequence is not None or dra_sequence_offset is not None:
                raise ValueError("DRA inputs are only accepted in MHC-II mode.")
            mapping = map_hla_sequence(full, sequence_offset=sequence_offset, pseudo_sequence=pseudo_sequence,
                                        max_hla_position=self.max_hla_position)
            key = ("I", full, tuple(mapping["indices"]), tuple(mapping["pseudo_positions_mature"]))
            generate = self._generate_hla
        if key in self._protein_cache:
            self._protein_cache.move_to_end(key)
            return full, self._protein_cache[key], "memory_cache"
        item = generate(mapping)
        self._remember(self._protein_cache, key, item)
        return full, item, "live_esm"

import random

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL = ('A', 'T', 'G', 'C')

# Transition substitutions (purine<->purine, pyrimidine<->pyrimidine).
TRANSITION_PAIRS = {
    'A': 'R',  # A/G
    'G': 'R',
    'C': 'Y',  # C/T
    'T': 'Y',
}

# Calibration factor that equalizes the effective injection rate between arms.
EFFECTIVE_SCALE = 0.4674

# Calibration of both arms, based on the Illumina literature review
# (GAII 2008 -> NovaSeq X Plus 2026):
#   - "realistic": error peak around 2% in long GC homopolymers
#     (Minoche et al. 2011; NAR Genomics 2026), dropping to
#     ~0.04-0.1% in the well-covered "middle of the read" (Stoler & Nekrutenko,
#     via NAR Genomics 2026).
#   - "stress": same SHAPE of curve (homopolymer + GC), but scaled
#     to cover 0-100%, for robustness/leakage diagnostics (the same
#     role "Castle" used to play, but without depending on intron/exon).
ILLUMINA_CALIBRATION = {
    "realistic": {
        "base_rate":         0.0004,  # ~0.04%, "middle of the read" (Zhou et al. 2019)
        "hp_rate_per_extra": 0.003,   # increment per extra base of homopolymer
        "hp_min_len":        4,       # short homopolymers (<4) are not counted
        "gc_penalty":        0.01,    # extra penalty for GC homopolymers
    },
    "stress": {
        "base_rate":         0.01,
        "hp_rate_per_extra": 0.15,
        "hp_min_len":        4,
        "gc_penalty":        0.20,
    },
}

CASTLE_RATES = {
    'splice_site': 0.00,
    'exon_pos2': 0.10,
    'exon_pos1': 0.20,
    'exon_pos3': 0.64,
    'intron': 1.00,
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _local_gc_content(
    seq: list[dict],
    i: int,
    window: int = 10
) -> float:
    """
    Calculates the local GC content within a specific window (`± window`) around
    a given position `i`.
    """

    start = max(0, i - window)
    end = min(len(seq), i + window + 1)
    local = seq[start:end]
    if not local:
        return 0.5
    gc = sum(1 for b in local if b in ('G', 'C'))
    return gc / len(local)


def _homopolymer_run_length(
    seq: str,
    i: int
) -> int:
    """
    Length of the homopolymer run containing position i
    """
    base = seq[i]
    start = i
    while start > 0 and seq[start - 1] == base:
        start -= 1
    end = i
    while end < len(seq) - 1 and seq[end + 1] == base:
        end += 1
    return end - start + 1

# ---------------------------------------------------------------------------
# Public injection functions
# ---------------------------------------------------------------------------

def inject_degenerate_nucleotides_illumina(
    seq: str,
    exons_intervals: list[list[int]] | None = None,
    introns_intervals: list[list[int]] | None = None,
    injection_rate: float = 1.0,
    mode: str = "realistic",
    gc_window: int = 10,
    scale: float = EFFECTIVE_SCALE
) -> str:
    """
    Injects degenerate bases following the documented error profile
    """
    cal = ILLUMINA_CALIBRATION[mode]
    out = list(seq)

    for i, base in enumerate(out):
        if base not in CANONICAL:
            continue  # natural degeneracy preserved

        run_len = _homopolymer_run_length(seq, i)
        gc_local = _local_gc_content(seq, i, window=gc_window) # type: ignore

        rate = cal["base_rate"]

        # homopolymer effect: only counts above the minimum, grows linearly
        # with the excess length (simple approximation of the non-linear
        # curve reported in Minoche 2011 / Laehnemann 2016)
        if run_len >= cal["hp_min_len"]:
            excess = run_len - cal["hp_min_len"] + 1
            rate += excess * cal["hp_rate_per_extra"]

            # extra penalty if the homopolymer is G or C
            # (NAR Genomics 2026: error increases more in GC homopolymers
            # of 7-11bp than in AT homopolymers of equivalent length)
            if base in ('G', 'C'):
                rate += cal["gc_penalty"]

        # local GC effect: "U"-shaped -- GC extremes (very low
        # or very high) have more error than the middle (Minoche 2011;
        # Laehnemann 2016, Fig. 5)
        gc_deviation = abs(gc_local - 0.5) * 2  # 0 at center, 1 at extremes
        rate += gc_deviation * cal["base_rate"]

        rate = min(rate, 1.0) * injection_rate * scale

        if random.random() < rate:
            out[i] = TRANSITION_PAIRS.get(base, base)

    return ''.join(out)


def inject_degenerate_nucleotides_uniform(
    seq: str,
    exons_intervals: list[list[int]] | None = None,
    introns_intervals: list[list[int]] | None = None,
    injection_rate: float = 0.0,
    scale: float = EFFECTIVE_SCALE
) -> str:
    """
    Injects degenerate bases with constant probability throughout
    """
    p = injection_rate * scale
    if p <= 0.0:
        return seq

    out = list(seq)
    for i, base in enumerate(out):
        if base not in CANONICAL:
            continue                      # natural degeneracy preserved
        if random.random() < p:
            out[i] = TRANSITION_PAIRS[base]
    return ''.join(out)


def _annotation_zone_rates(
    seq_len: int,
    exons_intervals: list[list[int]],
    introns_intervals: list[list[int]]
) -> list[float]:
    """
    Pre-computes the evolutionary conservation rate for each nucleotide position.
    """
    rates = [0.0] * seq_len

    # 1. Introns
    for start, end in introns_intervals:
        for pos in range(start, min(seq_len, end + 1)):
            rates[pos] = CASTLE_RATES['intron']

    # 2. Exons
    for start, end in exons_intervals:
        for i, pos in enumerate(range(start, min(seq_len, end + 1))):
            codon_pos = i % 3
            if codon_pos == 2:
                rates[pos] = CASTLE_RATES['exon_pos3']
            elif codon_pos == 0:
                rates[pos] = CASTLE_RATES['exon_pos1']
            else:
                rates[pos] = CASTLE_RATES['exon_pos2']

    # 3. Splice sites (overwrites exons/introns)
    for start, end in exons_intervals:
        for i in range(max(0, start-6), min(seq_len, start+6)):
            rates[i] = CASTLE_RATES['splice_site']
        for i in range(max(0, end-6), min(seq_len, end+6)):
            rates[i] = CASTLE_RATES['splice_site']

    return rates

def inject_degenerate_nucleotides(seq: str, exons_intervals: list[list[int]], introns_intervals: list[list[int]], injection_rate: float) -> str:
    """
    Inject degenerate nucleotides into the sequence.
    """

    out = list(seq)
    rates = _annotation_zone_rates(len(seq), exons_intervals, introns_intervals)

    for i, base in enumerate(out):
        if base not in ['A', 'T', 'G', 'C']:
            continue  # already degenerate, skip

        if random.random() < injection_rate * rates[i]:
            out[i] = TRANSITION_PAIRS.get(base, base)

    return ''.join(out)

def inject_degenerate_nucleotides_mixed(
    seq: str,
    exons_intervals: list[list[int]],
    introns_intervals: list[list[int]],
    injection_rate: float = 1.0,
    alpha: float = 0.3,
    illumina_mode: str = "realistic",
    gc_window: int = 10,
    scale: float = EFFECTIVE_SCALE
) -> str:
    """
    Injeta degeneracao combinando dois canais biologicamente/tecnicamente
    """
    CASTLE_RATE = 0.08  # ver aviso acima -- calibrar com o proprio MSA se possivel

    out = list(seq)
    cal = ILLUMINA_CALIBRATION[illumina_mode]

    # --- pre-computa zonas de anotacao ---
    annotation_rates = _annotation_zone_rates(len(seq), exons_intervals, introns_intervals)

    for i, base in enumerate(out):
        if base not in CANONICAL:
            continue  # degeneracao natural preservada

        # ---- taxa do canal Castle (conservacao) ----
        castle_component = annotation_rates[i] * CASTLE_RATE

        # ---- taxa do canal Illumina (composicao de sequencia) ----
        run_len = _homopolymer_run_length(seq, i)
        gc_local = _local_gc_content(seq, i, window=gc_window) # type: ignore

        illumina_component = cal["base_rate"]
        if run_len >= cal["hp_min_len"]:
            excess = run_len - cal["hp_min_len"] + 1
            illumina_component += excess * cal["hp_rate_per_extra"]
            if base in ('G', 'C'):
                illumina_component += cal["gc_penalty"]
        gc_deviation = abs(gc_local - 0.5) * 2
        illumina_component += gc_deviation * cal["base_rate"]
        illumina_component *= injection_rate

        # ---- mistura ----
        rate = alpha * castle_component + (1 - alpha) * illumina_component
        rate = min(rate, 1.0) * scale

        if random.random() < rate:
            out[i] = TRANSITION_PAIRS.get(base, base)

    return ''.join(out)

def degenerate_ratio(
    seq: str
) -> float:
    """
    Percentage of non-canonical bases in the sequence.
    """
    if not seq:
        return 0.0
    n_deg = sum(1 for b in seq if b.upper() not in CANONICAL)
    return 100.0 * n_deg / len(seq)

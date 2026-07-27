"""
Degenerate nucleotide injection module.

Provides functions to inject degenerate (ambiguous) bases into DNA sequences,
simulating the error profiles of Illumina sequencers or using controlled
uniform / annotation-conditioned strategies.

Public API:
    - inject_degenerate_nucleotides_illumina : Illumina error-profile injection
    - inject_degenerate_nucleotides_uniform  : uniform injection (control arm)
    - inject_degenerate_nucleotides          : annotation-conditioned injection
    - degenerate_ratio                       : fraction of non-canonical bases
"""

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

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _local_gc_content(seq, i, window=10):
    """
    Local GC content within a +-window window around position i.
    Used as a proxy for GC-rich/poor regions (Dohm et al. 2008;
    Minoche et al. 2011; NAR Genomics 2026 - AVITI vs NovaSeq X).
    """
    start = max(0, i - window)
    end = min(len(seq), i + window + 1)
    local = seq[start:end]
    if not local:
        return 0.5
    gc = sum(1 for b in local if b in ('G', 'C'))
    return gc / len(local)


def _homopolymer_run_length(seq, i):
    """
    Length of the homopolymer run containing position i
    (e.g.: in 'AAAATG', all A positions have run_length 4).
    Basis of the most documented systematic sequencing error effect
    (Minoche 2011; Laehnemann et al. 2016; NAR Genomics 2026).
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

def inject_degenerate_nucleotides_illumina(seq,
                                            exons_intervals=None,
                                            introns_intervals=None,
                                            injection_rate=1.0,
                                            mode="realistic",
                                            gc_window=10,
                                            scale=EFFECTIVE_SCALE):
    """
    Injects degenerate bases following the documented error profile
    for Illumina sequencers (homopolymer + local GC), instead of
    intron/exon annotation.

    The exons_intervals and introns_intervals parameters are accepted
    only to keep the signature compatible with other injection functions,
    and are deliberately ignored (same rationale as the uniform version:
    do not leak the label).

    - seq: original DNA sequence
    - injection_rate: scale factor in [0, 1] over the chosen calibration
      (allows performing the alpha-sweep scan)
    - mode: "realistic" (rates calibrated from Illumina 2026 literature,
      peak ~2%) or "stress" (same curve shape, scaled 0-100%,
      for robustness/leakage diagnostics)
    - gc_window: window size (+-bp) used to compute local GC content
    """
    cal = ILLUMINA_CALIBRATION[mode]
    out = list(seq)

    for i, base in enumerate(out):
        if base not in CANONICAL:
            continue  # natural degeneracy preserved

        run_len = _homopolymer_run_length(seq, i)
        gc_local = _local_gc_content(seq, i, window=gc_window)

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


def inject_degenerate_nucleotides_uniform(seq,
                                          exons_intervals=None,
                                          introns_intervals=None,
                                          injection_rate=0.0,
                                          scale=EFFECTIVE_SCALE):
    """
    Injects degenerate bases with constant probability throughout
    the entire sequence.

    Methodological control arm. Unlike the conditioned version,
    it does NOT consult intron/exon annotation at any point: every
    canonical position has exactly the same substitution probability.
    Thus, the presence of degenerate bases carries no information
    about the class to be predicted.

    The exons_intervals and introns_intervals parameters are accepted
    only to keep the signature compatible with the conditioned function,
    and are deliberately ignored.

    - seq: original DNA sequence
    - injection_rate: nominal rate for the scenario, in [0, 1]
    - scale: calibration factor that equalizes the effective rate between arms
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


def inject_degenerate_nucleotides(seq, exons_intervals, introns_intervals, injection_rate):
    """
    Inject degenerate nucleotides into the sequence.
    - seq: The original DNA sequence.
    - exons_intervals: A list of tuples representing the start and end positions of exons in the sequence.
    - introns_intervals: A list of tuples representing the start and end positions of introns in the sequence.
    - injection_rate: The percentage of nucleotides to be replaced with degenerate nucleotides.

    Rules:
    - Introns (center): high probability
    - Exon 3rd codon position: medium probability
    - Exon 1st/2nd position: low probability
    - Splice sites (+-6 bp from the boundary): zero probability
    - Start codon: zero probability
    """

    RATES = {
        'splice_site':  0.00,
        'exon_pos2':    0.10,
        'exon_pos1':    0.20,
        'exon_pos3':    0.64,
        'intron':       1.00,
    }

    seq = list(seq)

    # Mark splice sites (+-6 bp from each exon/intron boundary)
    splice_zone = set()
    for start, end in exons_intervals:
        for i in range(max(0, start-6), min(len(seq), start+6)):
            splice_zone.add(i)
        for i in range(max(0, end-6), min(len(seq), end+6)):
            splice_zone.add(i)

    # Exon positions mapped to their codon position (0, 1, 2)
    exon_positions = {}  # pos -> codon_position (0, 1, 2)
    for start, end in exons_intervals:
        for i, pos in enumerate(range(start, end + 1)):
            exon_positions[pos] = i % 3  # 0=1st, 1=2nd, 2=3rd

    intron_positions = set()
    for start, end in introns_intervals:
        for pos in range(start, end + 1):
            intron_positions.add(pos)

    for i, base in enumerate(seq):
        if base not in ['A', 'T', 'G', 'C']:
            continue  # already degenerate, skip

        if i in splice_zone:
            rate = RATES['splice_site']
        elif i in exon_positions:
            codon_pos = exon_positions[i]
            if codon_pos == 2:
                rate = RATES['exon_pos3']
            elif codon_pos == 0:
                rate = RATES['exon_pos1']
            else:
                rate = RATES['exon_pos2']
        elif i in intron_positions:
            rate = RATES['intron']
        else:
            rate = 0.0

        if random.random() < injection_rate * rate:
            seq[i] = TRANSITION_PAIRS.get(base, base)

    return ''.join(seq)

def inject_degenerate_nucleotides_mixed(seq,
                                         exons_intervals,
                                         introns_intervals,
                                         injection_rate=1.0,
                                         alpha=0.3,
                                         illumina_mode="realistic",
                                         gc_window=10,
                                         scale=EFFECTIVE_SCALE):
    """
    Injeta degeneracao combinando dois canais biologicamente/tecnicamente
    independentes, discutidos no projeto:

      - Canal "Castle" (conservado/evolutivo): reflete onde a selecao
        purificadora e mais fraca (introns > 3a posicao do codon >
        1a/2a posicao > splice site). CORRELACIONA com o rotulo
        intron/exon por definicao -- isso NAO e um bug, e o proprio
        fenomeno biologico que ele modela (ver discussao no projeto).

      - Canal "Illumina" (erro tecnico de sequenciamento): reflete
        homopolimero + GC local, independente de anotacao de gene.

    IMPORTANTE: misturar os dois NAO elimina o vazamento do canal
    Castle, apenas o dilui proporcionalmente a `alpha`. Use isso como
    uma tentativa de aproximar a mistura real (que de fato existe em
    dado biologico real), nao como uma "correcao" do vazamento -- se
    quiser vazamento zero, use so o canal illumina (alpha=0) ou o
    uniforme.

    - alpha: peso do canal Castle na mistura, em [0, 1].
        alpha=0   -> 100% Illumina (nenhum vazamento)
        alpha=1   -> 100% Castle (vazamento maximo, igual a versao antiga)
        alpha=0.3 (default) -> mistura moderada, mais perto do que se
        espera de uma sequencia GenBank real (predominantemente ruido
        tecnico, com uma contribuicao menor de variacao evolutiva real)
    - injection_rate: escala global aplicada ao canal Illumina
      (mesma semantica de inject_degenerate_nucleotides_illumina)
    - illumina_mode: "realistic" ou "stress", repassado pro canal Illumina

    ATENCAO SOBRE OS NUMEROS DO CANAL CASTLE:
    Ao contrario do canal Illumina (calibrado com literatura recente e
    especifica), nao existe um valor de dS "correto" universal para
    conservacao evolutiva -- varia muito com quao divergentes sao as
    especies de fungos comparadas no seu dataset. O CASTLE_RATE abaixo
    (0.08) e um chute conservador na faixa de especies proximas
    (dS ~0.1-0.2 entre especies irmas). Recomendado: substituir por um
    valor medido diretamente no seu proprio alinhamento multiplo de
    actina (MSA), calculando a taxa real de variacao por posicao
    (1a/2a/3a do codon, intron) entre as sequencias do seu dataset,
    em vez de usar este valor generico da literatura.
    """
    CASTLE_RATE = 0.08  # ver aviso acima -- calibrar com o proprio MSA se possivel

    RATES = {
        'splice_site': 0.00,
        'exon_pos2':   0.10,
        'exon_pos1':   0.20,
        'exon_pos3':   0.64,
        'intron':      1.00,
    }

    out = list(seq)

    # --- pre-computa zonas de anotacao (igual a versao Castle original) ---
    splice_zone = set()
    for start, end in exons_intervals:
        for i in range(max(0, start - 6), min(len(seq), start + 6)):
            splice_zone.add(i)
        for i in range(max(0, end - 6), min(len(seq), end + 6)):
            splice_zone.add(i)

    exon_positions = {}
    for start, end in exons_intervals:
        for i, pos in enumerate(range(start, end + 1)):
            exon_positions[pos] = i % 3

    intron_positions = set()
    for start, end in introns_intervals:
        for pos in range(start, end + 1):
            intron_positions.add(pos)

    cal = ILLUMINA_CALIBRATION[illumina_mode]

    for i, base in enumerate(out):
        if base not in CANONICAL:
            continue  # degeneracao natural preservada

        # ---- taxa do canal Castle (conservacao) ----
        if i in splice_zone:
            castle_component = RATES['splice_site']
        elif i in exon_positions:
            codon_pos = exon_positions[i]
            if codon_pos == 2:
                castle_component = RATES['exon_pos3']
            elif codon_pos == 0:
                castle_component = RATES['exon_pos1']
            else:
                castle_component = RATES['exon_pos2']
        elif i in intron_positions:
            castle_component = RATES['intron']
        else:
            castle_component = 0.0
        castle_component *= CASTLE_RATE

        # ---- taxa do canal Illumina (composicao de sequencia) ----
        run_len = _homopolymer_run_length(seq, i)
        gc_local = _local_gc_content(seq, i, window=gc_window)

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

def degenerate_ratio(seq):
    """Percentage of non-canonical bases in the sequence."""
    if not seq:
        return 0.0
    n_deg = sum(1 for b in seq if b.upper() not in CANONICAL)
    return 100.0 * n_deg / len(seq)

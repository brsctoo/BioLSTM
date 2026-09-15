from Bio.SeqFeature import CompoundLocation, FeatureLocation


def make_exons_intervals_list(
    location: FeatureLocation | CompoundLocation
) -> list[list[int]]:
    """
    Create a list of exon intervals using BioPython's location object directly.
    """
    exons_intervals = []

    # Multi-part location (join, order)
    if hasattr(location, 'parts') and len(location.parts) > 0:
        for part in location.parts:
            start = int(part.start)  # Already 0-based in BioPython # type: ignore
            end = int(part.end) - 1  # End is exclusive, convert to inclusive # type: ignore
            exons_intervals.append([start, end])
    else:
        # Simple location (single exon)
        start = int(location.start) # type: ignore
        end = int(location.end) - 1 # type: ignore
        exons_intervals.append([start, end])

    return exons_intervals

def make_introns_intervals_list(
    exons_intervals: list[list[int]]
) -> list[list[int]]:
    """
    Create a list of intron sequences from the split sequences.
    """

    introns_intervals = []

    for i in range(len(exons_intervals) - 1):
        intron_start = exons_intervals[i][1] + 1
        intron_end   = exons_intervals[i + 1][0] - 1
        if intron_start <= intron_end:
            introns_intervals.append([intron_start, intron_end])

    return introns_intervals

def make_exons_list(
    exons_intervals: list[list[int]],
    seq: str
) -> list[str]:
    """
    Create a list of exon sequences from the split sequences.
    """

    exons = []
    for exon_interval in exons_intervals:
        exons.append(seq[exon_interval[0]:exon_interval[1]+1])

    return exons

def make_introns_list(
    introns_intervals: list[list[int]],
    seq: str
) -> list[str]:
    """
    Create a list of intron sequences from the split sequences.
    """

    introns = []

    # Verify the intron sequences based on 'GT' and 'AG' rules
    for start_idx, end_idx in introns_intervals:
        # Extract the whole intron sequence
        intron_seq = seq[start_idx : end_idx + 1]

        if not intron_seq:
            continue

        # Check the biological GT-AG rule directly on the extracted string
        start_bases = intron_seq[:2]
        end_bases = intron_seq[-2:]

        # We consider it valid if it matches the rule OR if it was cropped
        is_valid_start = (start_bases == "GT" or start_bases == "")
        is_valid_end = (end_bases == "AG" or end_bases == "")

        if is_valid_start and is_valid_end:
            introns.append(intron_seq)

    return introns

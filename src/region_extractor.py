def make_exons_intervals_list(location):
    """
    Create a list of exon intervals using BioPython's location object directly.
    Works with simple, join, order, and complement locations.

    returns: [[start, end], [start, end], ...]
    """
    exons_intervals = []

    # Multi-part location (join, order)
    if hasattr(location, 'parts') and len(location.parts) > 0:
        for part in location.parts:
            start = int(part.start)  # Already 0-based in BioPython
            end = int(part.end) - 1  # End is exclusive, convert to inclusive
            exons_intervals.append([start, end])
    else:
        # Simple location (single exon)
        start = int(location.start)
        end = int(location.end) - 1
        exons_intervals.append([start, end])

    return exons_intervals

def make_introns_intervals_list(exons_intervals):
    """
    Create a list of intron sequences from the split sequences.

    - exons_intervals: A list of exon intervals.
      ex. exons_intervals: [[133, 164], [344, 400], [541, 572]]
    returns: A list of intron intervals.
      ex. returns: [[0, 132], [165, 343], [401, 540], [573, seq_length-1]]
    """

    introns_intervals = []

    for i in range(len(exons_intervals) - 1):
        intron_start = exons_intervals[i][1] + 1
        intron_end   = exons_intervals[i + 1][0] - 1
        if intron_start <= intron_end:
            introns_intervals.append([intron_start, intron_end])

    return introns_intervals

def make_exons_list(exons_intervals, seq):
    """
    Create a list of exon sequences from the split sequences.

    - exons_intervals: A list of exon intervals.
      ex. exons_intervals: [[133, 164], [344, 400], [541, 572]]
    returns: A list of exon sequences.
      ex. returns: ['ATG...TAA', 'GGC...TGA', 'CCT...TAG']
    """

    exons = []
    for exon_interval in exons_intervals:
        exons.append(seq[exon_interval[0]:exon_interval[1]+1])

    return exons

def make_introns_list(introns_intervals, seq):
    """
    Create a list of intron sequences from the split sequences.
    For verification, we see if it starts with 'GT' and ends with 'AG'.

    - introns_intervals: A list of intron intervals.
      ex. introns_intervals: [[0, 132], [165, 343], [401, 540], [573, seq_length-1]]
    returns: A list of intron sequences.
      ex. returns: ['GTA...CAG', 'TTC...GGA', 'AAG...TTC', 'GGC...AAT']
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

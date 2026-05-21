def make_exons_intervals_list(location):
    """
    Create a list of exon intervals using BioPython's location object directly.
    Works with simple, join, order, and complement locations.

    returns: [[start, end], [start, end], ...]
    """
    exons_intervals = []

    # Location com múltiplas partes (join, order)
    if hasattr(location, 'parts') and len(location.parts) > 0:
        for part in location.parts:
            start = int(part.start)  # já é 0-based no BioPython
            end = int(part.end) - 1  # end é exclusivo, converte para inclusivo
            exons_intervals.append([start, end])
    else:
        # Location simples (éxon único)
        start = int(location.start)
        end = int(location.end) - 1
        exons_intervals.append([start, end])

    return exons_intervals

def make_introns_intervals_list(exons_intervals):
    """
    Create a list of intron sequences from the split sequences.

    - exons: A list of exon sequences.
    ex. exons: [[133, 164], [344, 400], [541, 572]]
    returns: A list of intron sequences.
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
    Create a list of intron sequences from the split sequences.

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
    for intron_interval in introns_intervals:
        start = seq[intron_interval[0]:intron_interval[0]+2]
        end = seq[intron_interval[1]-1:intron_interval[1]+1]

        # Caso padrão GT...AG
        if start == "GT" and end == "AG":
           introns.append(seq[intron_interval[0]:intron_interval[1]+1])
        # Caso especial ...AG
        elif start == "" and end == "AG":
           introns.append(seq[intron_interval[0]:intron_interval[1]+1])
        # Caso especial GT...
        elif start == "GT" and end == "":
           introns.append(seq[intron_interval[0]:intron_interval[1]+1])

    return introns

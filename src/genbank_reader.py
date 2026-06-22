"""
This module reads a GenBank file, extracts the relevant sequences and features, and preprocesses the data by
injecting degenerate nucleotides according to specific rules. The processed data is then separated into training
and testing datasets and saved to files for later use in machine learning models.
"""

from Bio import SeqIO
import pickle # Used for file operations
import random
import region_extractor as re

def inject_degenerate_nucleotides(seq, exons_intervals, introns_intervals, injection_rate):
    """
    Inject degenerate nucleotides into the sequence.
    - seq: The original DNA sequence.
    - exons_intervals: A list of tuples representing the start and end positions of exons in the sequence.
    - introns_intervals: A list of tuples representing the start and end positions of introns in the sequence.
    - injection_rate: The percentage of nucleotides to be replaced with degenerate nucleotides.

    Rules (based on Bush 2011, Zhao 2003):
    - Introns (center): high probability
    - Exon 3rd codon position: medium probability
    - Exon 1st/2nd position: low probability
    - Splice sites (±6 bp from edge): zero probability
    - Start codon: zero probability
    """

    TRANSITION_PAIRS = {
        'A': 'R',  # A/G
        'G': 'R',
        'C': 'Y',  # C/T
        'T': 'Y',
    }

    RATES = {
        'splice_site':  0.00,
        'exon_pos2':    0.10,
        'exon_pos1':    0.20,
        'exon_pos3':    0.64,
        'intron':       1.00,
    }

    seq = list(seq)

    # Marks splice sites (±6 from each exon/intron boundary)
    splice_zone = set()
    for start, end in exons_intervals:
        for i in range(max(0, start-6), min(len(seq), start+6)):
            splice_zone.add(i)
        for i in range(max(0, end-6), min(len(seq), end+6)):
            splice_zone.add(i)

    # Exon positions mapped to their respective codon position (0, 1, 2)
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
            continue  # Already degenerate, skip

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

def validate_register(record):
    cds_features = [f for f in record.features if f.type == "CDS"]
    return len(cds_features) > 0

def preprocess_genbank_file(genbank_input_filepath, INJECTION_RATE):
    """Read the GenBank file and preprocess the sequences."""

    # Variable that contains all of the processed data
    data = []
    num = 0
    seen_sequences = set()

    for register in SeqIO.parse(genbank_input_filepath + ".gb", "genbank"):
        try:
            if not validate_register(register):
                # print("Skipping sequence because it failed verification")
                continue

            # Converts to string AND ensures uppercase to prevent hidden bugs
            seq = str(register.seq).upper()

            if len(seq) > 20000:
                # print(f"Skipping sequence with {len(seq)} bases...")
                continue

            if seq in seen_sequences:
                # print("Skipping sequence because it is a duplicate")
                continue

            # Gets the CDS feature first
            cds_feature = None
            for feature in register.features:
                if feature.type == "CDS":
                    cds_feature = feature
                    break

            if cds_feature is None:
                # print("Skipping sequence because it lacks a CDS")
                continue

            # Looks for the mRNA that contains the CDS
            target_feature = None
            cds_start = int(cds_feature.location.start)
            cds_end   = int(cds_feature.location.end)

            for feature in register.features:
                if feature.type == "mRNA":
                    mrna_start = int(feature.location.start)
                    mrna_end   = int(feature.location.end)
                    if mrna_start <= cds_start and mrna_end >= cds_end:
                        target_feature = feature
                        break

            if target_feature is None:
                target_feature = cds_feature

            mrna_start = int(target_feature.location.start)
            mrna_end   = int(target_feature.location.end)

            # Crops EXACTLY the size of the gene (ignores the thousands of base pairs around it)
            cropped_seq_obj = register.seq[mrna_start:mrna_end]

            # Uses your extractor to get raw coordinates
            exons_intervals_raw = re.make_exons_intervals_list(target_feature.location)

            # Shifts coordinates to the new "Point Zero" (since the start was cropped)
            exons_intervals = [[s - mrna_start, e - mrna_start] for s, e in exons_intervals_raw]

            # 2. REVERSE STRAND
            if target_feature.location.strand == -1:
                # Flips the strand inside out (A becomes T, C becomes G, and reversed)
                seq = str(cropped_seq_obj.reverse_complement()).upper()

                L = len(seq)
                exons_intervals_rev = []

                # Mirrors the coordinates to the new inverted strand
                for s, e in exons_intervals:
                    new_start = L - 1 - e
                    new_end = L - 1 - s
                    exons_intervals_rev.append([new_start, new_end])

                # Since the strand was reversed, the last exons became the first ones.
                # So we reorder the list so Python reads it correctly from left to right.
                exons_intervals = sorted(exons_intervals_rev, key=lambda x: x[0])
            else:
                # If it is a forward strand, just convert to string and proceed
                seq = str(cropped_seq_obj).upper()

            # 3. Single-Exon
            # Now we create the introns using your function
            introns_intervals = re.make_introns_intervals_list(exons_intervals)

            seq = inject_degenerate_nucleotides(seq, exons_intervals, introns_intervals, INJECTION_RATE)

            introns = re.make_introns_list(introns_intervals, seq)
            exons = re.make_exons_list(exons_intervals, seq)

            sample = {
                "sequence": seq,
                "exon_intervals": exons_intervals,
                "exons": exons,
                "intron_intervals": introns_intervals,
                "introns": introns,
            }

            num = num + 1
            seen_sequences.add(seq)
            data.append(sample)

        except Exception as e:
            # print(f"\nError processing record {register.id}: {e}")
            # print("Skipping to the next one...\n")
            continue

    return data

def separate_train_test(data, test_size=0.2):
    """Separate the data into training and testing sets."""

    random.seed(123865) # For reproducibility
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))
    train_data = data[0:split_index]
    test_data = data[split_index:len(data)]
    return train_data, test_data

def save_dataset_to_file(genbank_filepath_output, data):
    """Save the processed data to a file."""

    file = open(genbank_filepath_output, "wb") # open
    pickle.dump(data, file) # write
    file.close() # close

def save_preprocessed_genbank_file(genbank_input_filepath, genbank_filepath_output, INJECTION_RATE):
    """Preprocess the genbankfile."""

    print("Preprocessing GenBank file...")
    data = preprocess_genbank_file(genbank_input_filepath, INJECTION_RATE)
    print("Total samples processed: ", len(data))

    print("\n\n")

    print("Separating train and test datasets...")
    train_data, test_data = separate_train_test(data, test_size=0.2)
    print("Train samples: ", len(train_data))
    print("Test samples: ", len(test_data))


    print("\n\n")

    print("Saving datasets to files...")
    save_dataset_to_file(genbank_filepath_output + "_" + "train.mod1", train_data)
    print("Train dataset saved.")
    save_dataset_to_file(genbank_filepath_output + "_" + "test.mod1", test_data)
    print("Test dataset saved.")
    print("Done.")

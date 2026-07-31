"""
This module reads a GenBank file, extracts the relevant sequences and features, and preprocesses the data by
injecting degenerate nucleotides according to specific rules. The processed data is then separated into training
and testing datasets and saved to files for later use in machine learning models.

Pre-processing is split into three explicit steps so that deduplication happens
on the ORIGINAL sequence, before any noise is injected:

    STEP 1 — read records, crop to the gene span, fix reverse strand, build intervals
    STEP 2 — deduplicate on the original (clean) cropped sequence
    STEP 3 — inject degenerate nucleotides on the already-clean set

Deduplicating after injection cannot work: the noise is stochastic, so two copies
of the same gene come out different and both survive.
"""

from Bio import SeqIO
import pickle # Used for file operations
import random
import region_extractor as re

from noise_injector import (
    inject_degenerate_nucleotides,
    inject_degenerate_nucleotides_illumina,
    inject_degenerate_nucleotides_uniform,
    inject_degenerate_nucleotides_mixed,
)

# Maps the injection mode flag to the corresponding injector function.
INJECTION_MODE_MAP = {
    "conditioned": inject_degenerate_nucleotides,
    "uniform":     inject_degenerate_nucleotides_uniform,
    "illumina":    inject_degenerate_nucleotides_illumina,
    "mixed":       inject_degenerate_nucleotides_mixed,
}
DEFAULT_INJECTION_MODE = "conditioned"

MAX_SEQUENCE_LENGTH = 20000


def validate_register(record):
    cds_features = [f for f in record.features if f.type == "CDS"]
    return len(cds_features) > 0


def read_records(genbank_input_filepath):
    """
    STEP 1 — Read the GenBank file and return the cleaned records, WITHOUT injection.

    For each valid record this resolves the target feature (mRNA containing the CDS,
    falling back to the CDS itself), crops the sequence to the gene span, handles the
    reverse strand, and rebases the exon coordinates to the cropped sequence.

    returns: list of dicts with keys "sequence", "exon_intervals", "intron_intervals"
    """

    raw = []

    for register in SeqIO.parse(genbank_input_filepath + ".gb", "genbank"):
        if not validate_register(register):
            continue

        # Converts to string AND ensures uppercase to prevent hidden bugs
        try:
            full_seq = str(register.seq).upper()
        except Exception:
            # Pula registros do GenBank que não possuem a sequência ATCG (Sequence content is undefined)
            continue

        if len(full_seq) > MAX_SEQUENCE_LENGTH:
            # print(f"Skipping sequence with {len(full_seq)} bases...")
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

        raw.append({
            "sequence": seq,
            "exon_intervals": exons_intervals,
            "intron_intervals": introns_intervals,
        })

    return raw


def remove_duplicates(raw):
    """
    STEP 2 — Remove duplicate records, keyed on the ORIGINAL cropped sequence.

    This MUST run before injection. The injected sequence is not a valid key:
    the substitutions are drawn at random, so two copies of the same gene would
    produce two different strings and both would survive.
    """

    seen = set()
    unique = []

    for item in raw:
        key = item["sequence"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


def apply_injection(unique, injection_rate, injection_mode=DEFAULT_INJECTION_MODE, **injector_kwargs):
    """
    STEP 3 — Inject degenerate nucleotides on the already-deduplicated set.

    - injection_mode: which injection strategy to use.
      Options: "conditioned" (annotation-based), "uniform" (control arm),
      "illumina" (Illumina error-profile), "mixed" (weighted combination).
    - injector_kwargs: strategy-specific extra parameters
      (e.g. alpha and illumina_mode for injection_mode="mixed").
    """

    injector = INJECTION_MODE_MAP.get(injection_mode)
    if injector is None:
        raise ValueError(
            f"Unknown injection_mode '{injection_mode}'. "
            f"Valid options: {list(INJECTION_MODE_MAP.keys())}"
        )

    data = []

    for item in unique:
        exons_intervals   = item["exon_intervals"]
        introns_intervals = item["intron_intervals"]

        seq = injector(
            item["sequence"],
            exons_intervals,
            introns_intervals,
            injection_rate,
            **injector_kwargs,
        )

        data.append({
            "sequence": seq,
            "exon_intervals": exons_intervals,
            "exons": re.make_exons_list(exons_intervals, seq),
            "intron_intervals": introns_intervals,
            "introns": re.make_introns_list(introns_intervals, seq),
        })

    return data


def preprocess_genbank_file(genbank_input_filepath, INJECTION_RATE,
                            injection_mode=DEFAULT_INJECTION_MODE, **injector_kwargs):
    """Read the GenBank file and preprocess the sequences."""

    raw = read_records(genbank_input_filepath)
    print(f"Valid records read: {len(raw)}")

    unique = remove_duplicates(raw)
    print(f"After deduplication: {len(unique)}  ({len(raw) - len(unique)} duplicates removed)")

    data = apply_injection(unique, INJECTION_RATE, injection_mode, **injector_kwargs)

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


def save_preprocessed_genbank_file(genbank_input_filepath, genbank_filepath_output, INJECTION_RATE,
                                   injection_mode=DEFAULT_INJECTION_MODE, **injector_kwargs):
    """Preprocess the genbankfile."""

    print("Preprocessing GenBank file...")
    print(f"Injection mode: {injection_mode} | rate: {INJECTION_RATE}")
    data = preprocess_genbank_file(genbank_input_filepath, INJECTION_RATE, injection_mode, **injector_kwargs)
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

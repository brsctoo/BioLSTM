import pickle  # Used for file operations
import random
from typing import TypedDict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from data import region_extractor as rx
from data.noise_injector import (
    inject_degenerate_nucleotides,
    inject_degenerate_nucleotides_illumina,
    inject_degenerate_nucleotides_mixed,
    inject_degenerate_nucleotides_uniform,
)


class GeneSample(TypedDict):
    sequence: str
    exon_intervals: list[list[int]]
    exons: list[str]
    intron_intervals: list[list[int]]
    introns: list[str]

# Maps the injection mode flag to the corresponding injector function.
INJECTION_MODE_MAP = {
    "conditioned": inject_degenerate_nucleotides,
    "uniform":     inject_degenerate_nucleotides_uniform,
    "illumina":    inject_degenerate_nucleotides_illumina,
    "mixed":       inject_degenerate_nucleotides_mixed,
}
DEFAULT_INJECTION_MODE = "conditioned"

MAX_SEQUENCE_LENGTH = 20000

def validate_register(
    record: SeqRecord
) -> bool:
    """
    Checks if the record contains at least one Coding DNA Sequence (CDS) feature.
    """

    cds_features = []

    for feature in record.features:
        if feature.type == "CDS":
            cds_features.append(feature)
    return len(cds_features) > 0

def read_records(
    genbank_input_filepath: str
) -> list[dict[str, object]]:
    """
    STEP 1 — Read the GenBank file and return the cleaned records
    """

    raw = []

    for register in SeqIO.parse(genbank_input_filepath + ".gb", "genbank"):
        if not validate_register(register):
            continue

        try:
            # Converts to string AND ensures uppercase to prevent hidden bugs
            full_seq = str(register.seq).upper()
        except Exception:  # noqa: BLE001, S112
            # Occurs if 'register' does not have a '.seq' attribute or throws UndefinedSequenceError
            continue

        if len(full_seq) > MAX_SEQUENCE_LENGTH:
            # Skip sequences that are too big
            print(f"Skipping sequence with {len(full_seq)} bases...")
            continue

        # Gets the CDS feature first
        cds_feature = None
        for feature in register.features:
            if feature.type == "CDS":
                cds_feature = feature
                break

        if cds_feature is None:
            continue

        # 1. CDS feature
        target_feature = cds_feature
        cds_start = int(cds_feature.location.start)
        cds_end = int(cds_feature.location.end)

        for feature in register.features:
            if feature.type == "mRNA":
                start = int(feature.location.start)
                end = int(feature.location.end)

                if start <= cds_start and end >= cds_end:
                    target_feature = feature
                    break  # Found it

        if target_feature is None:
            target_feature = cds_feature

        target_start = int(target_feature.location.start)
        target_end = int(target_feature.location.end)

        # Crops the size of the gene
        cropped_seq_obj = register.seq[target_start:target_end]

        exons_intervals_raw = rx.make_exons_intervals_list(target_feature.location)
        exons_intervals = [[s - target_start, e - target_start] for s, e in exons_intervals_raw]

        # 2. Reverse Strand
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

            # Reorder the list so Python reads it correctly from left to right.
            exons_intervals = sorted(exons_intervals_rev, key=lambda x: x[0])
        else:
            # If it is a forward strand, just convert to string and proceed
            seq = str(cropped_seq_obj).upper()

        # 3. Single-Exon
        # Now we create the introns using your function
        introns_intervals = rx.make_introns_intervals_list(exons_intervals)

        raw.append({
            "sequence": seq,
            "exon_intervals": exons_intervals,
            "intron_intervals": introns_intervals,
        })

    return raw


def remove_duplicates(
    raw: list[dict[str, object]]
) -> list[dict[str, object]]:
    """
    STEP 2 — Remove duplicate records, keyed on the ORIGINAL cropped sequence.
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


def apply_injection(
    unique: list[dict[str, object]],
    injection_rate: float,
    injection_mode: str = DEFAULT_INJECTION_MODE,
    **injector_kwargs: float | str
) -> list[GeneSample]:
    """
    STEP 3 — Inject degenerate nucleotides on the already-deduplicated set.
    """

    injector = INJECTION_MODE_MAP.get(injection_mode)
    if injector is None:
        raise ValueError(
            f"Unknown injection_mode '{injection_mode}'. "
            f"Valid options: {list(INJECTION_MODE_MAP.keys())}"
        )

    data: list[GeneSample] = []

    for item in unique:
        exons_intervals = item["exon_intervals"]
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
            "exons": rx.make_exons_list(exons_intervals, seq), # type: ignore
            "intron_intervals": introns_intervals,
            "introns": rx.make_introns_list(introns_intervals, seq), # type: ignore
        })

    return data


def preprocess_genbank_file(
    genbank_input_filepath: str,
    INJECTION_RATE: float,
    injection_mode: str = DEFAULT_INJECTION_MODE,
    **injector_kwargs: float | str
) -> list[GeneSample]:
    """
    Read the GenBank file and preprocess the sequences.
    """

    raw = read_records(genbank_input_filepath)
    print(f"Valid records read: {len(raw)}")

    unique = remove_duplicates(raw)
    print(f"After deduplication: {len(unique)}  ({len(raw) - len(unique)} duplicates removed)")

    data = apply_injection(unique, INJECTION_RATE, injection_mode, **injector_kwargs)

    return data


def separate_train_test(
    data: list[GeneSample],
    test_size: float = 0.2
) -> tuple[list[GeneSample], list[GeneSample]]:
    """
    Separate the data into training and testing sets.
    """

    random.seed(123865) # For reproducibility
    random.shuffle(data)

    split_index = int(len(data) * (1 - test_size))
    train_data = data[0:split_index]
    test_data = data[split_index:len(data)]
    return train_data, test_data


def save_dataset_to_file(
    genbank_filepath_output: str,
    data: list[GeneSample]
) -> None:
    """
    Save the processed data to a file.
    """

    with open(genbank_filepath_output, "wb") as file:
        pickle.dump(data, file)


def save_preprocessed_genbank_file(
    genbank_input_filepath: str,
    genbank_filepath_output: str,
    INJECTION_RATE: float,
    injection_mode: str = DEFAULT_INJECTION_MODE,
    **injector_kwargs: float | str
) -> None:
    """
    Preprocess the genbankfile.
    """

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

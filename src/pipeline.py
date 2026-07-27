"""
Bi-LSTM Pipeline for Intron/Exon Identification.

Orchestrates the full workflow: GenBank search, GenBank pre-processing
(cropping, strand correction, deduplication, degenerate-base injection),
feature extraction (sliding window + one-hot encoding), model training,
and validation.

============================================================
USAGE
============================================================

    python pipeline.py [mode] [--injection-rate R] [--injection-mode M]
                        [--alpha A] [--illumina-mode IM] [--name N] [--seed S]

    If [mode] is omitted, an interactive menu is shown.

============================================================
POSITIONAL ARGUMENTS
============================================================

    mode    Operation mode (optional — interactive menu if omitted):
              search_data                Query NCBI and save a .gb file.
              train                      Pre-process the GenBank file and train the model.
              test                       Validate an already-trained model on the test split.
              full                       Run train followed by test in a single call.
              create_train_test_files    Only run pre-processing + featurization (no training).
              validate_specific_dataset  Validate the named model against an arbitrary .mod1 file.

============================================================
OPTIONAL FLAGS
============================================================

    --injection-rate  R   (float, default: 0.0)
        Scaling factor applied on top of the chosen injection strategy.
        0.0 = no degenerate bases injected.
        1.0 = full rate as defined by the selected mode.
        Note: 100% injection does not mean 100% of bases are replaced —
        see EFFECTIVE_SCALE in noise_injector.py.

    --injection-mode  M   (str, default: conditioned)
        Selects the injection strategy used during pre-processing:
          conditioned   Annotation-based injection. Introns get the highest
                        substitution probability, exon 3rd codon position gets
                        medium probability, and 1st/2nd positions get low
                        probability. Splice sites and start codons are protected
                        (rate = 0). This is the main experimental arm — it
                        deliberately correlates with the intron/exon label.
          uniform       Control arm. Every canonical base has the same
                        constant substitution probability, regardless of its
                        position in the gene structure. Degenerate bases carry
                        no information about intron/exon class. This is the
                        arm that isolates label leakage in 'conditioned'.
          illumina      Illumina error-profile arm. Substitution probability is
                        driven by local GC content and homopolymer run length,
                        mimicking systematic sequencing errors. Not rate-matched
                        against the other arms — use it as a robustness check,
                        not as the causal control.
          mixed         Weighted combination of 'conditioned' and 'illumina',
                        controlled by --alpha. Diluting the leakage does not
                        remove it.

    --alpha  A            (float, default: 0.3)
        Weight of the conditioned channel in 'mixed' mode, in [0, 1].
        0.0 = pure Illumina (no label leakage), 1.0 = pure conditioned
        (full leakage). Ignored for all other modes.

    --illumina-mode  IM    (str, default: realistic)
        Illumina calibration sub-mode, used by 'illumina' and 'mixed':
          realistic   Literature-calibrated rates (peak ~2%).
          stress      Same curve shape, scaled 0-100%, for robustness/
                      leakage diagnostics.

    --name  N             (str, default: actin_fungi)
        Experiment label. Used both to locate the input .gb file
        (../assets/genbank_data/{N}.gb) and to name all output files.
        Use this to run multiple experiments without overwriting results.

    --seed  S             (int, default: 123865)
        Global random seed (Python random, NumPy, TensorFlow). Makes
        injection noise, dataset balancing, and weight initialization
        reproducible. NOTE: the train/test split itself always uses a
        fixed seed (123865) inside genbank_reader.separate_train_test,
        independent of --seed, so different experiments stay comparable
        on the same held-out set.

============================================================
EXAMPLES
============================================================

    # Search NCBI and save actin_fungi.gb
    python pipeline.py search_data

    # Train with no injection (baseline)
    python pipeline.py train

    # Train with 50% conditioned injection (default mode)
    python pipeline.py train --injection-rate 0.5 --name actin_rate50

    # Train with uniform injection at 50% (the causal control for the above)
    python pipeline.py train --injection-rate 0.5 --injection-mode uniform --name actin_uniform50

    # Train with mixed injection (30% conditioned weight)
    python pipeline.py train --injection-rate 1.0 --injection-mode mixed --alpha 0.3 --name actin_mixed

    # Full run (train + validate) with a fixed seed for reproducibility
    python pipeline.py full --injection-rate 0.5 --seed 7 --name actin_rate50_seed7

    # Validate only (uses the model saved under the given name)
    python pipeline.py test --name actin_rate50

    # Interactive menu (no positional argument)
    python pipeline.py

============================================================
"""

import os
import gc
import argparse
from datetime import datetime

import genbank_searcher
import genbank_reader
import modeling
import train_model
import validation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- Genbank Search Configuration -----------
INCLUDE_PLANTAE_IN_TRAINING = True

MAX_RECORDS = 5000
BATCH_SIZE = 50
MAX_PER_SPECIES = 400
MAX_GENERAL = 3000
MAX_HOUSEKEEPING = 1000

_ORGANISM_SCOPE = (
    "(Fungi[Organism] OR Metazoa[Organism]" +
    (" OR Plantae[Organism])" if INCLUDE_PLANTAE_IN_TRAINING else ")")
)

QUERY_GENERAL = (
    f'{_ORGANISM_SCOPE} '
    'AND biomol_genomic[PROP] '
    'AND "complete cds"[Title] '
    'AND 200:15000[Sequence Length] '
    'NOT "whole genome"[Title] NOT chromosome[Title] NOT wgs[Keyword] '
    'NOT "PREDICTED"[Title] '
    'NOT mitochondrion[All Fields] NOT mitochondrial[All Fields] '
    'NOT chloroplast[All Fields] NOT plastid[All Fields]'
)

QUERY_HOUSEKEEPING = (
    '(TEF1-alpha[Gene] OR actin[Gene] OR tubulin[Gene]) '
    'AND Eukaryota[Organism] '
    'AND ("complete cds"[Title] OR "partial cds"[Title]) '
    'NOT WGS[Keyword] '
    'NOT genome[Title] '
    'NOT contig[Title] '
    'NOT scaffold[Title] '
    'NOT mitochondrion[All Fields] '
    'NOT chloroplast[All Fields]'
)

# -------------------------------------------------

DEFAULT_INJECTION_RATE = 0.0
DEFAULT_INJECTION_MODE = "conditioned"  # Options: conditioned | uniform | illumina | mixed
DEFAULT_NAME = "actin_fungi"
DEFAULT_SEED = 123865

GB_FILE_NAME = "all_proteins"
OUTPUT_FILE = os.path.join(BASE_DIR, f"../assets/genbank_data/{GB_FILE_NAME}.gb")


def get_output_paths(name):
    genbank_input = os.path.join(BASE_DIR, f"../assets/genbank_data/{name}")
    mod1 = os.path.join(BASE_DIR, f"../assets/processed_data/mod1/data_{name}")
    mod2 = os.path.join(BASE_DIR, f"../assets/processed_data/mod2/data_XY_{name}.npz")
    result = os.path.join(BASE_DIR, f"../assets/result/model_{name}_onehot.h5")
    return genbank_input, mod1, mod2, result


def log_stage(msg):
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    print(f"{'='*60}\n")


def set_global_seed(seed):
    """
    Seed Python's random, NumPy, and TensorFlow so a run is reproducible
    end-to-end: injection noise, dataset balancing, and weight initialization
    all draw from the same seed.

    The train/test split (genbank_reader.separate_train_test) is deliberately
    NOT covered here — it hardcodes its own fixed seed so different experiments
    in a sweep stay comparable on the same held-out set regardless of --seed.
    """
    import random
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_injector_kwargs(injection_mode, alpha, illumina_mode):
    """Strategy-specific extra parameters, forwarded to noise_injector.py."""
    injector_kwargs = {}
    if injection_mode == "illumina":
        injector_kwargs["mode"] = illumina_mode
    elif injection_mode == "mixed":
        injector_kwargs["alpha"] = alpha
        injector_kwargs["illumina_mode"] = illumina_mode
    return injector_kwargs


def save_run_metadata(name, injection_rate, injection_mode, ratio_degenerate, injector_kwargs):
    """Persist the effective degeneration rate and run hyperparameters."""
    lines = [
        f"RATIO_DEGENERATE_NUCLEOTIDES = {ratio_degenerate}",
        f"injection_mode = {injection_mode}",
        f"injection_rate = {injection_rate}",
    ]
    for key, value in injector_kwargs.items():
        lines.append(f"{key} = {value}")

    txt_filepath = os.path.join(BASE_DIR, f"../assets/RATIO_{name}.txt")
    with open(txt_filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Run metadata saved to: {txt_filepath}")


def search_data_pipeline():
    query_to_print = QUERY_GENERAL.replace(' AND ', '\nAND ')
    query_housekeeping_to_print = QUERY_HOUSEKEEPING.replace(' AND ', '\nAND ')
    log_stage("SEARCH — Querying GenBank")
    print(f"GENERAL QUERY: \n {query_to_print}")
    print(f"HOUSEKEEPING QUERY: \n {query_housekeeping_to_print}")
    genbank_searcher.main(QUERY_GENERAL, QUERY_HOUSEKEEPING, MAX_RECORDS, BATCH_SIZE,
                           MAX_PER_SPECIES, MAX_GENERAL, MAX_HOUSEKEEPING, OUTPUT_FILE)
    log_stage("SEARCH — DONE.")


def create_train_test_files(injection_rate, injection_mode, name, **injector_kwargs):
    genbank_input, mod1, mod2, _ = get_output_paths(name)

    log_stage(f"PRE-PROCESSING  |  rate={injection_rate}  mode={injection_mode}  name={name}")
    print("Input .gb file:", genbank_input + ".gb")
    print("Output mod1   :", mod1)
    genbank_reader.save_preprocessed_genbank_file(
        genbank_input, mod1, injection_rate, injection_mode, **injector_kwargs
    )
    gc.collect()
    log_stage("PRE-PROCESSING — DONE. Memory freed.")

    log_stage("FEATURIZATION  (sliding window + one-hot)")
    print("Input mod1 :", mod1 + "_train.mod1")
    print("Output mod2:", mod2)
    ratio_degenerate = modeling.modeling_train_data(mod1 + "_train.mod1", mod2)
    gc.collect()
    log_stage(f"FEATURIZATION — DONE. Degenerate ratio: {ratio_degenerate:.2f}%. Memory freed.")

    save_run_metadata(name, injection_rate, injection_mode, ratio_degenerate, injector_kwargs)

    return mod2


def train_pipeline(injection_rate, injection_mode, name, **injector_kwargs):
    _, _, mod2, result = get_output_paths(name)
    create_train_test_files(injection_rate, injection_mode, name, **injector_kwargs)

    log_stage("TRAINING  (Bi-LSTM)")
    print("Input mod2  :", mod2)
    print("Output model:", result)
    train_model.train_model(mod2, result)
    gc.collect()
    log_stage("TRAINING — DONE. Model saved. Memory freed.")


def validate_pipeline(name):
    _, mod1, _, result = get_output_paths(name)

    log_stage("VALIDATION — Loading model and test data")
    print("Model    :", result)
    print("Test data:", mod1 + "_test.mod1")
    validation.validate_model(result, mod1 + "_test.mod1")
    log_stage("VALIDATION — DONE.")


def validate_specific_dataset(name):
    _, _, _, result = get_output_paths(name)
    specific_dataset = input("Enter the name of the dataset: ")
    specific_dataset = os.path.join(BASE_DIR, f"../assets/processed_data/mod1/{specific_dataset}")

    log_stage("VALIDATION (custom dataset) — Loading model and test data")
    print("Model    :", result)
    print("Test data:", specific_dataset + ".mod1")
    validation.validate_model(result, specific_dataset + ".mod1")
    log_stage("VALIDATION — DONE.")


def main():
    parser = argparse.ArgumentParser(description="Bi-LSTM Pipeline for Intron/Exon Identification")

    # Positional argument: operation mode. Choices cover every action reachable
    # from the interactive menu, so a full sweep can be scripted without prompts.
    parser.add_argument("mode", nargs='?',
        choices=["search_data", "train", "test", "full",
                 "validate_specific_dataset", "create_train_test_files"],
        help="Operation mode (optional — interactive menu if omitted).")

    parser.add_argument("--injection-rate", type=float, default=DEFAULT_INJECTION_RATE,
        help="Degenerate nucleotide injection rate (e.g. 0.5 for 50%%). Default: 0.0")

    parser.add_argument("--injection-mode", type=str, default=DEFAULT_INJECTION_MODE,
        choices=["conditioned", "uniform", "illumina", "mixed"],
        help="Injection strategy: 'conditioned' (annotation-based, label-correlated), "
             "'uniform' (control arm, no label information), "
             "'illumina' (sequencing error-profile), "
             "'mixed' (weighted blend of conditioned + illumina). "
             f"Default: {DEFAULT_INJECTION_MODE}")

    parser.add_argument("--alpha", type=float, default=0.3,
        help="Weight of the conditioned channel in 'mixed' mode, in [0, 1]. "
             "0.0 = pure Illumina (no leakage), 1.0 = pure conditioned "
             "(full leakage). Ignored for other modes. Default: 0.3")

    parser.add_argument("--illumina-mode", type=str, default="realistic",
        choices=["realistic", "stress"],
        help="Illumina calibration sub-mode: 'realistic' (literature-calibrated "
             "rates, peak ~2%%) or 'stress' (same curve shape, scaled 0-100%%, "
             "for robustness/leakage diagnostics). Used by 'illumina' and "
             "'mixed'. Default: realistic")

    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
        help="Experiment name — used to find the input .gb file and to name "
             f"all output files (e.g. actin_fungi_rate50). Default: {DEFAULT_NAME}")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
        help="Global random seed for reproducibility (Python random, NumPy, "
             f"TensorFlow). Default: {DEFAULT_SEED}")

    args = parser.parse_args()

    injection_rate = args.injection_rate
    injection_mode = args.injection_mode
    name = args.name
    seed = args.seed

    injector_kwargs = build_injector_kwargs(injection_mode, args.alpha, args.illumina_mode)

    set_global_seed(seed)

    # If no mode was provided, show interactive menu
    if args.mode is None:
        print("\n" + "="*50)
        print("Bi-LSTM Pipeline for Intron/Exon Identification")
        print("="*50)
        print("\nSelect an option:")
        print("0 - Search data on GenBank")
        print("1 - Train (Train the model)")
        print("2 - Test (Validate the model)")
        print("3 - Full (Train and validate)")
        print("4 - Validate specific dataset")
        print("5 - Create train and test files")
        print("="*50)
        print(f"\nInjection rate: {injection_rate} (use --injection-rate to change)")
        print(f"Injection mode: {injection_mode} (use --injection-mode to change)")
        print(f"Experiment name: {name} (use --name to change)")
        print(f"Seed: {seed} (use --seed to change)")

        choice = input("\nEnter the option number (0/1/2/3/4/5): ").strip()

        modes_map = {
            "0": "search_data",
            "1": "train",
            "2": "test",
            "3": "full",
            "4": "validate_specific_dataset",
            "5": "create_train_test_files",
        }
        args.mode = modes_map.get(choice)

        if args.mode is None:
            print("\n❌ Invalid option! Please use 0, 1, 2, 3, 4, or 5.")
            return

    if args.mode == "search_data":
        search_data_pipeline()
    elif args.mode == "train":
        train_pipeline(injection_rate, injection_mode, name, **injector_kwargs)
    elif args.mode == "test":
        validate_pipeline(name)
    elif args.mode == "full":
        train_pipeline(injection_rate, injection_mode, name, **injector_kwargs)
        gc.collect()
        log_stage("TRANSITION — Training complete. Freeing memory before validation.")
        validate_pipeline(name)
    elif args.mode == "validate_specific_dataset":
        validate_specific_dataset(name)
    elif args.mode == "create_train_test_files":
        create_train_test_files(injection_rate, injection_mode, name, **injector_kwargs)


if __name__ == "__main__":
    main()

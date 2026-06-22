import os
import genbank_searcher
import genbank_reader
import modeling
import train_model
import validation
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- Genbank Search Configuration -----------

MAX_RECORDS = 5000
BATCH_SIZE = 50
MAX_PER_SPECIES = 400
MAX_GENERAL = 2000
MAX_HOUSEKEEPING = 2000

QUERY_GENERAL = (
    '(Fungi[Organism] OR Metazoa[Organism]) '
    'AND biomol_genomic[PROP] '
    'AND "complete cds"[Title] '
    'AND 200:15000[Sequence Length] '
    'NOT "whole genome"[Title] NOT chromosome[Title] NOT wgs[Keyword] '
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
DEFAULT_NAME = "actin_fungi"

GB_FILE_NAME = "all_proteins"
OUTPUT_FILE = os.path.join(BASE_DIR, f"../assets/genbank_data/{GB_FILE_NAME}.gb")

def get_output_paths(name):
    genbank_input = os.path.join(BASE_DIR, f"../assets/genbank_data/{name}")
    mod1 = os.path.join(BASE_DIR, f"../assets/processed_data/mod1/data_{name}")
    mod2 = os.path.join(BASE_DIR, f"../assets/processed_data/mod2/data_XY_{name}.npz")
    result = os.path.join(BASE_DIR, f"../assets/result/model_{name}_onehot.h5")
    return genbank_input, mod1, mod2, result

def search_data_pipeline():
    QUERY_to_print = QUERY_GENERAL.replace(' AND ', '\nAND ')
    QUERY_housekeeping_to_print = QUERY_HOUSEKEEPING.replace(' AND ', '\nAND ')
    print(f"Searching records in GenBank with GENERAL QUERY: \n {QUERY_to_print}")
    print(f"And searching records in GenBank with HOUSEKEEPING QUERY: \n {QUERY_housekeeping_to_print}")
    genbank_searcher.main(QUERY_GENERAL, QUERY_HOUSEKEEPING, MAX_RECORDS, BATCH_SIZE, MAX_PER_SPECIES, MAX_GENERAL, MAX_HOUSEKEEPING, OUTPUT_FILE)

def create_train_test_files(injection_rate, name):
    genbank_input, mod1, mod2, _ = get_output_paths(name)
    print("Starting the training pipeline...")
    print("Injection rate for degenerate nucleotides: ", injection_rate)
    genbank_reader.save_preprocessed_genbank_file(genbank_input, mod1, injection_rate)
    modeling.modeling_train_data(mod1 + "_train.mod1", mod2)

def train_pipeline(injection_rate, name):
    _, mod1, mod2, result = get_output_paths(name)
    create_train_test_files(injection_rate, name)
    train_model.train_model(mod2, result)

def validate_pipeline(name):
    _, mod1, _, result = get_output_paths(name)
    print(f"Starting the validation pipeline with: \n File {mod1.split('/')[-1]}_test.mod1 \n Model {result.split('/')[-1]}...")
    validation.validate_model(result, mod1 + "_test.mod1")

def validate_specific_dataset(name):
    _, _, _, result = get_output_paths(name)
    specific_dataset = input("Enter the name of the dataset: ")
    specific_dataset = os.path.join(BASE_DIR, f"../assets/processed_data/mod1/{specific_dataset}")
    print(f"Starting the validation pipeline with: \n File {specific_dataset.split('/')[-1]}.mod1 \n Model {result.split('/')[-1]}...")
    validation.validate_model(result, specific_dataset + ".mod1")

def main():
    parser = argparse.ArgumentParser(description="Bi-LSTM Pipeline for Intron/Exon Identification")

    parser.add_argument("mode", nargs='?', choices=["train", "test", "full"],
        help="Choose operation mode: train, test or full")

    parser.add_argument("--injection-rate", type=float, default=DEFAULT_INJECTION_RATE,
        help="Degenerate nucleotide injection rate (e.g. 0.5 for 50%%). Default: 0.0")

    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
        help="Experiment name, used in output files (e.g. actin_fungi_rate50). Default: actin_fungi")

    args = parser.parse_args()
    injection_rate = args.injection_rate
    name = args.name

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
        print(f"Experiment name: {name} (use --name to change)")

        choice = input("\nEnter the option number (0/1/2/3/4/5): ").strip()

        modes_map = {
            "0": "search_data",
            "1": "train",
            "2": "test",
            "3": "full",
            "4": "validate_specific_dataset",
            "5": "create_train_test_files"
        }
        args.mode = modes_map.get(choice)

        if args.mode is None:
            print("\n❌ Invalid option! Please use 0, 1, 2, 3, 4, or 5.")
            return

    if args.mode == "search_data":
        search_data_pipeline()
    elif args.mode == "train":
        train_pipeline(injection_rate, name)
    elif args.mode == "test":
        validate_pipeline(name)
    elif args.mode == "full":
        train_pipeline(injection_rate, name)
        validate_pipeline(name)
    elif args.mode == "validate_specific_dataset":
        validate_specific_dataset(name)
    elif args.mode == "create_train_test_files":
        create_train_test_files(injection_rate, name)

if __name__ == "__main__":
    main()

"""
Script that with only one function, pipeline(), that executes the entire pipeline of data loading, preprocessing and model training.
01 - genbank_reader.py: contains functions to read and preprocess the GenBank file, including validation of records, separation of train and test datasets, and saving the processed data to files.
"""

import os
import numpy as np
import genbank_searcher
import genbank_reader
import modeling
import train_model
import validation
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

GB_FILE_NAME = "all_proteins" # Name of the .gb archive to be created
OUTPUT_FILE = os.path.join(BASE_DIR, f"../assets/genbank_data/{GB_FILE_NAME}.gb")

# For injection rate, 100% is like 33% of the bases being replaced by degenerate nucleotides, 50% is like 16.5%, and so on.
INJECTION_RATE = 0.0

# Name of archive .gb you want to use
NOME_GB = "actin_fungi"

# Name of the archive after processed
NAME_PROCESSED_GB = "data_all_proteins"

NAME_MODEL = "final_model"

genbank_data_filepath_input = os.path.join(BASE_DIR, f"../assets/genbank_data/{NOME_GB}")
mod1_filepath_output = os.path.join(BASE_DIR, f"../assets/processed_data/mod1/{NAME_PROCESSED_GB}")
mod2_filepath_output = os.path.join(BASE_DIR,f"../assets/processed_data/mod2/{NAME_PROCESSED_GB}.npz")
result_filepath_output = os.path.join(BASE_DIR,f"../assets/result/{NAME_MODEL}.h5")

def search_data_pipeline():
    QUERY_to_print = QUERY_GENERAL.replace(' AND ', '\nAND ')
    QUERY_little_to_print = QUERY_HOUSEKEEPING.replace(' AND ', '\nAND ')
    print(f"Procurando arquivos no genbank com a QUERY GERAL: \n {QUERY_to_print}")
    print(f"E procurando arquivos no genbank com a QUERY HOUSEKEEPING: \n {QUERY_little_to_print}")
    genbank_searcher.search_data(QUERY_GENERAL, QUERY_HOUSEKEEPING, OUTPUT_FILE)

def create_train_test_files():
    # Use the genbank_data to create the mod1 data, which is the preprocessed data ready for modeling
    # .gb file → mod1 (train and test)
    print("Starting the training pipeline...")
    print("Injection rate for degenerate nucleotides: ", INJECTION_RATE)
    genbank_reader.save_preprocessed_genbank_file(genbank_data_filepath_input, mod1_filepath_output, INJECTION_RATE)

    # Use the mod1 train data to create the mod2 data, which is the X and y arrays ready for model training
    # .mod1 train → mod2 (X and y)
    modeling.modeling_train_data(mod1_filepath_output + "_train.mod1", mod2_filepath_output)
    return None

def train_pipeline():
    create_train_test_files()

    # Use the mod2 data to train the model and save it for later use in validation.py
    # mod2 (X and y) → h5 model
    train_model.train_model(mod2_filepath_output, result_filepath_output)

    return None

def validate_pipeline():
    # Use the model trained and the test data to validate the model and print the final metrics
    # h5 model + mod1 test → metric
    print(f"Starting the validation pipeline with: \n The archive {mod1_filepath_output.split('/')[-1]}_test.mod1 \n The model {result_filepath_output.split('/')[-1]}...")
    validation.validate_model(result_filepath_output, mod1_filepath_output + "_test.mod1")

def validate_specific_dataset():
    # Use to validate an specific dataset that you want
    specific_dataset = input("Enter the name of the dataset: ")
    specific_dataset = os.path.join(BASE_DIR, f"../assets/processed_data/mod1/{specific_dataset}")
    print(f"Starting the validation pipeline with: \n The archive {specific_dataset.split('/')[-1]}.mod1 \n The model {result_filepath_output.split('/')[-1]}...")
    validation.validate_model(result_filepath_output, specific_dataset + ".mod1")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Bi-LSTM para Identificação de Íntrons/Éxons")

    # Argumento opcional: Modo de operação
    parser.add_argument("mode", nargs='?', choices=["train", "test", "full"],
    help="Escolha o modo: train (treinar), test (validar) ou full (ambos)")

    args = parser.parse_args()

    # Se nenhum modo foi fornecido, exibir menu interativo
    if args.mode is None:
        print("\n" + "="*50)
        print("Pipeline Bi-LSTM para Identificação de Íntrons/Éxons")
        print("="*50)
        print("\nEscolha uma opção:")
        print("0 - Search data on genbank (Procurar os arquivos no genbank)")
        print("1 - train  (Treinar o modelo)")
        print("2 - test   (Validar o modelo)")
        print("3 - full   (Treinar e validar)")
        print("4 - validate specific dataset   (Validar um dataset em específico)")
        print("5 - create train and test files  (Criar os arquivos de teste e treino)")
        print("="*50)

        choice = input("\nDigite o número da opção (0/1/2/3/4): ").strip()

        modes_map = {"0": "search_data", "1": "train", "2": "test", "3": "full", "4": "validate_specific_dataset", "5": "create_train_test_files"}
        args.mode = modes_map.get(choice)

        if args.mode is None:
            print("\n❌ Opção inválida! Use 0, 1, 2 ou 3.")
            return

    if args.mode == "search_data":
        search_data_pipeline()
    elif args.mode == "train":
        train_pipeline()
    elif args.mode == "test":
        validate_pipeline()
    elif args.mode == "full":
        train_pipeline()
        validate_pipeline()
    elif args.mode == "validate_specific_dataset":
        validate_specific_dataset()
    elif args.mode == "create_train_test_files":
        create_train_test_files()

if __name__ == "__main__":
    main()

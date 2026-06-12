"""
Script to fetch a dataset from GenBank with a custom query and preprocess it for testing only.
Usage: python fetch_test_dataset.py --query "your query here" --output "output_name"
"""
import os
import argparse
from Bio import Entrez
import genbank_reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Entrez.email = "bcominscheffel@email.com"

def fetch_genbank(query, output_filepath, max_records=500):
    """Fetch records from GenBank and save as .gb file."""

    print(f"Searching GenBank with the following query:\n{query}\n")

    # Fetch IDs
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records)
    record = Entrez.read(handle)
    handle.close()

    # Cast to standard string list to resolve Pyright's reportIndexIssue on IntegerElement
    # Ensure record behaves as a dictionary and extract IDs safely
    record_dict = dict(record) if isinstance(record, dict) else {}
    ids = [str(id_item) for id_item in record_dict.get("IdList", [])]
    print(f"Found {len(ids)} records.")

    if not ids:
        print("No records found. Please check your query.")
        return False

    # Download records in batch
    print("Downloading records...")
    handle = Entrez.efetch(db="nucleotide", id=ids, rettype="gb", retmode="text")

    gb_path = output_filepath + ".gb"
    with open(gb_path, "w") as f:
        f.write(handle.read())
    handle.close()

    print(f"File saved to: {gb_path}")
    return True

def fetch_and_preprocess(query, output_name, max_records=500, injection_rate=0.0):
    """Fetch from GenBank and preprocess for testing."""

    output_filepath = os.path.join(BASE_DIR, "../assets/genbank_data", output_name)
    mod1_filepath = os.path.join(BASE_DIR, "../assets/processed_data/mod1", output_name)

    # 1. Download from GenBank
    success = fetch_genbank(query, output_filepath, max_records)
    if not success:
        return

    # 2. Preprocess (same as the standard pipeline)
    print("\nPreprocessing...")
    data = genbank_reader.preprocess_genbank_file(output_filepath, injection_rate)
    print(f"Total samples processed: {len(data)}")

    if not data:
        print("No valid samples remaining after preprocessing.")
        return

    # 3. Save everything as "test" — without splitting into train/test
    output_path = mod1_filepath + "_test.mod1"
    genbank_reader.save_dataset_to_file(output_path, data)
    print(f"\nTest dataset saved to: {output_path}")
    print("Done! Use validate_pipeline() pointing to this file.")

def main():
    parser = argparse.ArgumentParser(description="Fetch and preprocess GenBank dataset for testing")
    parser.add_argument("--query",   required=True,  help="GenBank search query")
    parser.add_argument("--output",  required=True,  help="Output file name (without extension)")
    parser.add_argument("--max",     type=int, default=500, help="Maximum number of records to download (default: 500)")
    parser.add_argument("--injection_rate", type=float, default=0.0, help="Degenerate nucleotide injection rate (default: 0.0)")
    args = parser.parse_args()

    fetch_and_preprocess(args.query, args.output, args.max, args.injection_rate)

if __name__ == "__main__":
    main()

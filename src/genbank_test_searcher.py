"""
Script to fetch a dataset from GenBank with a custom query and preprocess it for testing only.
Usage: python fetch_test_dataset.py --query "your query here" --output "output_name"
"""
import os
import argparse
from Bio import Entrez, SeqIO
import genbank_reader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Entrez.email = "bcominscheffel@email.com"

def fetch_genbank(query, output_filepath, max_records=500):
    """Fetch records from GenBank and save as .gb file."""

    print(f"Buscando no GenBank com a query:\n{query}\n")

    # Busca os IDs
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records)
    record = Entrez.read(handle)
    handle.close()

    ids = record["IdList"]
    print(f"Encontrados {len(ids)} registros.")

    if not ids:
        print("Nenhum registro encontrado. Verifique a query.")
        return False

    # Baixa os registros em lote
    print("Baixando registros...")
    handle = Entrez.efetch(db="nucleotide", id=ids, rettype="gb", retmode="text")

    gb_path = output_filepath + ".gb"
    with open(gb_path, "w") as f:
        f.write(handle.read())
    handle.close()

    print(f"Arquivo salvo em: {gb_path}")
    return True

def fetch_and_preprocess(query, output_name, max_records=500, injection_rate=0.0):
    """Fetch from GenBank and preprocess for testing."""

    output_filepath = os.path.join(BASE_DIR, "../assets/genbank_data", output_name)
    mod1_filepath = os.path.join(BASE_DIR, "../assets/processed_data/mod1", output_name)

    # 1. Baixa do GenBank
    success = fetch_genbank(query, output_filepath, max_records)
    if not success:
        return

    # 2. Preprocessa (igual ao pipeline normal)
    print("\nPreprocessando...")
    data = genbank_reader.preprocess_genbank_file(output_filepath, injection_rate)
    print(f"Total de amostras processadas: {len(data)}")

    if not data:
        print("Nenhuma amostra válida após preprocessamento.")
        return

    # 3. Salva tudo como "test" — sem separar train/test
    output_path = mod1_filepath + "_test.mod1"
    genbank_reader.save_dataset_to_file(output_path, data)
    print(f"\nDataset de teste salvo em: {output_path}")
    print("Pronto! Use validate_pipeline() apontando para esse arquivo.")

def main():
    parser = argparse.ArgumentParser(description="Fetch e preprocessa dataset do GenBank para teste")
    parser.add_argument("--query",   required=True,  help="Query do GenBank")
    parser.add_argument("--output",  required=True,  help="Nome do arquivo de saída (sem extensão)")
    parser.add_argument("--max",     type=int, default=500, help="Máximo de registros a baixar (default: 500)")
    parser.add_argument("--injection_rate", type=float, default=0.0, help="Taxa de injeção de bases degeneradas (default: 0.0)")
    args = parser.parse_args()

    fetch_and_preprocess(args.query, args.output, args.max, args.injection_rate)

if __name__ == "__main__":
    main()

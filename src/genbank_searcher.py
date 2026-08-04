"""
Search for highly diverse, random data on GenBank using the NCBI API.
"""

import os
import random
import time
from typing import cast
from Bio import Entrez, SeqIO

Entrez.email = "bcominscheffel@gmail.com"

def main(QUERY_GENERAL, QUERY_HOUSEKEEPING, MAX_RECORDS, BATCH_SIZE, MAX_POR_ESPECIE, MAX_GENERAL, MAX_HOUSEKEEPING, OUTPUT_FILE):
    search_data(QUERY_GENERAL, MAX_RECORDS, BATCH_SIZE, MAX_POR_ESPECIE, MAX_GENERAL, OUTPUT_FILE)

def search_data(QUERY_GENERAL, MAX_RECORDS, BATCH_SIZE, MAX_POR_ESPECIE, MAX_GENERAL, OUTPUT_FILE):
    print("Searching GenBank for a highly diverse random dataset...")

    handle = Entrez.esearch(db="nucleotide", term=QUERY_GENERAL, retmax=0)
    total_count = int(Entrez.read(handle)["Count"]) # type: ignore
    handle.close()

    print(f"Total matching records in GenBank: {total_count}")

    if total_count == 0:
        print("Erro: Nenhuma sequência encontrada. Verifique a query.")
        return

    pool_size = MAX_GENERAL
    max_start = max(0, total_count - pool_size)
    random_start = random.randint(0, max_start)

    print(f"Fetching a random pool of {pool_size} IDs starting from offset {random_start}...")

    handle = Entrez.esearch(db="nucleotide", term=QUERY_GENERAL, retstart=random_start, retmax=pool_size)
    result = cast(dict, Entrez.read(handle))
    ids = result["IdList"]
    handle.close()

    random.shuffle(ids)
    print(f"Total IDs to process: {len(ids)}.")

    print("Passing through the diversity filter...")
    especies_count = {}
    total_saves = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i : i + BATCH_SIZE]

            try:
                fetch_handle = Entrez.efetch(db="nucleotide", id=batch, rettype="gb", retmode="text")
                records = SeqIO.parse(fetch_handle, "genbank")

                for record in records:
                    especie = record.annotations.get("organism", "Desconhecida")

                    if especie not in especies_count:
                        especies_count[especie] = 0

                    if especies_count[especie] < MAX_POR_ESPECIE:
                        SeqIO.write(record, out, "genbank")
                        especies_count[especie] += 1
                        total_saves += 1

                fetch_handle.close()
            except Exception as e:
                print(f"Erro no batch: {e}. Pulando para o próximo...")
                continue
            print(f"Batch processed. Records saved so far: {total_saves}/{MAX_RECORDS}")
            time.sleep(1)

            if total_saves >= MAX_RECORDS:
                print(f"Meta of {MAX_RECORDS} diversified records reached. Stopping download.")
                break

    print("-" * 30)
    print("DOWNLOAD COMPLETE!")
    print(f"Total filtered sequences: {total_saves}")
    print(f"Total unique species in the dataset: {len(especies_count)}")

"""
Search for data on GenBank using the NCBI API

The script combines results from a general query and a housekeeping gene query, ensuring diversity by limiting
the number of records per species. The results are saved in a GenBank format file, and the script includes progress
updates and a safety check to stop after reaching a certain number of diversified records.
"""

import os
import random
import time
from typing import cast

from Bio import Entrez, SeqIO

Entrez.email = "bcominscheffel@gmail.com"

# Configuration of search parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main(QUERY_GENERAL, QUERY_HOUSEKEEPING, MAX_RECORDS, BATCH_SIZE, MAX_POR_ESPECIE, MAX_GENERAL, MAX_HOUSEKEEPING, OUTPUT_FILE):
    search_data(QUERY_GENERAL, QUERY_HOUSEKEEPING, MAX_RECORDS, BATCH_SIZE, MAX_POR_ESPECIE, MAX_GENERAL, MAX_HOUSEKEEPING, OUTPUT_FILE)

def search_data(QUERY_GENERAL, QUERY_HOUSEKEEPING, MAX_RECORDS, BATCH_SIZE, MAX_POR_ESPECIE, MAX_GENERAL, MAX_HOUSEKEEPING, OUTPUT_FILE):
    """Search for data on GenBank using the NCBI API and save results in a GenBank format file."""

    print("Searching GenBank genes...")

    # Search general query
    handle = Entrez.esearch(db="nucleotide", term=QUERY_GENERAL, retmax=MAX_GENERAL)

    result = cast(dict, Entrez.read(handle))  # Read the result of the search query -> Object DictElement
    ids_general = result["IdList"]  # Get the list of IDs from the general query

    handle.close()
    print(f"{len(ids_general)} IDs of general Query.")

    # Search housekeeping gene query
    handle = Entrez.esearch(db="nucleotide", term=QUERY_HOUSEKEEPING, retmax=MAX_HOUSEKEEPING)
    result = cast(dict, Entrez.read(handle))  # Use cast to explicitly tell the type of result, which is a dictionary
    ids_housekeeping = result["IdList"]  # Get the list of IDs from the general query
    handle.close()
    print(f"{len(ids_housekeeping)} IDs of housekeeping query.")

    # Combine and remove duplicates
    seen = set()
    ids = [x for x in ids_general + ids_housekeeping if not (x in seen or seen.add(x))]
    random.shuffle(ids)
    print(f"Total unique after combining: {len(ids)} IDs.")

    print("Passing through the diversity filter...")
    especies_count = {}
    total_saves = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i : i + BATCH_SIZE]

            fetch_handle = Entrez.efetch(db="nucleotide", id=batch, rettype="gb", retmode="text")

            # Read the batch using Bio.SeqIO
            records = SeqIO.parse(fetch_handle, "genbank")

            for record in records:
                # Determine the species of the record (using "organism" annotation, or "Desconhecida" if not available)
                especie = record.annotations.get("organism", "Desconhecida")

                if especie not in especies_count:
                    especies_count[especie] = 0

                if especies_count[especie] < MAX_POR_ESPECIE:
                    SeqIO.write(record, out, "genbank")
                    especies_count[especie] += 1
                    total_saves += 1

            fetch_handle.close()
            print(f"Batch processed. Records saved in the current file: {total_saves}")
            time.sleep(0.5)  # Avoid hitting the server too hard

            # Safety limit: if we already reached MAX_RECORDS diversified records, we can stop to avoid unnecessary downloads
            if total_saves >= MAX_RECORDS:
                print(f"Meta of {MAX_RECORDS} diversified records reached. Stopping download.")
                break

    print("-" * 30)
    print("DOWNLOAD COMPLETE!")
    print(f"Total filtered sequences: {total_saves}")
    print(f"Total unique species in the dataset: {len(especies_count)}")

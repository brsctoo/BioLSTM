"""
Busca dados diretamente do GenBank usando a API do NCBI, com uma query específica para:
- Genes TEF1-alpha em fungos
"""

import random
from Bio import Entrez
from Bio import SeqIO
import os
import time

Entrez.email = "bcominscheffel@gmail.com"

# Query para buscar genes TEF1-alpha em fungos com cds completa ou parcial diretamente do banco de dados GenBank, utilizando a API do NCBI.

# CDS (complete coding sequence) é a sequência de DNA que codifica uma proteína completa, incluindo os códons de início e término.

MAX_RECORDS = 5000
BATCH_SIZE = 50
MAX_POR_ESPECIE = 400
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_GENERAL = 2000  # quantos quer da query geral
MAX_HOUSEKEEPING = 2000     # quantos quer do Housekeeping


def search_data(QUERY, QUERY_TEF1, OUTPUT_FILE):
    print("Buscando genes no GenBank...")

    # Busca query geral
    handle = Entrez.esearch(db="nucleotide", term=QUERY, retmax=MAX_GENERAL)
    result = Entrez.read(handle)
    ids_general = result["IdList"]
    handle.close()
    print(f"{len(ids_general)} IDs da query geral.")

    # Busca Housekeeping
    handle = Entrez.esearch(db="nucleotide", term=QUERY_TEF1, retmax=MAX_HOUSEKEEPING)
    result = Entrez.read(handle)
    ids_housekeeping = result["IdList"]
    handle.close()
    print(f"{len(ids_housekeeping)} IDs da query housekeeping.")

    # Combina e remove duplicatas
    seen = set()
    ids = [x for x in ids_general + ids_housekeeping if not (x in seen or seen.add(x))]
    random.shuffle(ids)
    print(f"Total único após combinação: {len(ids)} IDs.")

    print("Passando pelo Funil de Diversidade...")
    contagem_especies = {}
    salvos_total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i:i + BATCH_SIZE]

            fetch_handle = Entrez.efetch(
                db="nucleotide", id=batch, rettype="gb", retmode="text"
            )

            # Lê o lote usando o Biopython
            records = SeqIO.parse(fetch_handle, "genbank")

            for record in records:
                # Descobre qual é a espécie
                especie = record.annotations.get("organism", "Desconhecida")

                if especie not in contagem_especies:
                    contagem_especies[especie] = 0

                if contagem_especies[especie] < MAX_POR_ESPECIE:
                    SeqIO.write(record, out, "genbank")
                    contagem_especies[especie] += 1
                    salvos_total += 1

            fetch_handle.close()
            print(f"Lote processado. Registros salvos no arquivo atual: {salvos_total}")
            time.sleep(0.5)  # Para evitar sobrecarga no servidor do NCBI

            # Trava de segurança: Se já bateu 2000 genes diversificados, para de baixar
            if salvos_total >= 4000:
                print("Meta de 4000 registros diversificados atingida!")
                break

    print("-" * 30)
    print(f"DOWNLOAD CONCLUÍDO!")
    print(f"Total de sequências filtradas: {salvos_total}")
    print(f"Total de espécies únicas no dataset: {len(contagem_especies)}")

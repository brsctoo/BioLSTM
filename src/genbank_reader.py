"""
Lê a sequência
"""

from Bio import SeqIO
import pickle # Used for file operations
import random
import region_extractor as re

def inject_degenerate_nucleotides(seq, exons_intervals, introns_intervals, injection_rate):
    """
    Inject degenerate nucleotides into the sequence.
    - seq: The original DNA sequence.
    - exons_intervals: A list of tuples representing the start and end positions of exons in the sequence.
    - introns_intervals: A list of tuples representing the start and end positions of introns in the sequence.
    - injection_rate: The percentage of nucleotides to be replaced with degenerate nucleotides.

    Rules (baseadas em Bush 2011, Zhao 2003):
    - Íntrons (centro): probabilidade alta
    - Éxon 3ª posição do códon: probabilidade média
    - Éxon 1ª/2ª posição: probabilidade baixa
    - Splice sites (±6 bp da borda): probabilidade zero
    - Start codon: probabilidade zero
    """

    TRANSITION_PAIRS = {
        'A': 'R',  # A/G
        'G': 'R',
        'C': 'Y',  # C/T
        'T': 'Y',
    }

    RATES = {
        'splice_site':  0.00,
        'exon_pos2':    0.10,
        'exon_pos1':    0.20,
        'exon_pos3':    0.64,
        'intron':       1.00,
    }

    seq = list(seq)

    # Marca splice sites (±6 de cada borda éxon/íntron)
    splice_zone = set()
    for start, end in exons_intervals:
        for i in range(max(0, start-6), min(len(seq), start+6)):
            splice_zone.add(i)
        for i in range(max(0, end-6), min(len(seq), end+6)):
            splice_zone.add(i)

    # Posições de éxon com sua posição no códon (0, 1, 2)
    exon_positions = {}  # pos -> codon_position (0, 1, 2)
    for start, end in exons_intervals:
        for i, pos in enumerate(range(start, end + 1)):
            exon_positions[pos] = i % 3  # 0=1ª, 1=2ª, 2=3ª

    intron_positions = set()
    for start, end in introns_intervals:
        for pos in range(start, end + 1):
            intron_positions.add(pos)

    for i, base in enumerate(seq):
        if base not in ['A', 'T', 'G', 'C']:
            continue  # já é degenerada, pula

        if i in splice_zone:
            rate = RATES['splice_site']
        elif i in exon_positions:
            codon_pos = exon_positions[i]
            if codon_pos == 2:
                rate = RATES['exon_pos3']
            elif codon_pos == 0:
                rate = RATES['exon_pos1']
            else:
                rate = RATES['exon_pos2']
        elif i in intron_positions:
            rate = RATES['intron']
        else:
            rate = 0.0

        if random.random() < injection_rate * rate:
            seq[i] = TRANSITION_PAIRS.get(base, base)

    return ''.join(seq)

def validate_register(record):
    cds_features = [f for f in record.features if f.type == "CDS"]
    return len(cds_features) > 0

def preprocess_genbank_file(genbank_input_filepath, INJECTION_RATE):
    """Read the GenBank file and preprocess the sequences."""

    # Variable that contains all of the processed data
    data = []
    num = 0
    seen_sequences = set()

    for register in SeqIO.parse(genbank_input_filepath + ".gb", "genbank"):
        try:
            if not validate_register(register):
                print(f"Pulando sequência pois não passou na verificação")
                continue

            # Converte para string E garante maiúsculas para evitar erros ocultos
            seq = str(register.seq).upper()

            if len(seq) > 20000:
                print(f"Pulando sequência de {len(seq)} bases...")
                continue

            if seq in seen_sequences:
                print(f"Pulando sequência pois é repetida")
                continue

            # Pega o CDS primeiro
            cds_feature = None
            for feature in register.features:
                if feature.type == "CDS":
                    cds_feature = feature
                    break

            if cds_feature is None:
                print(f"Pulando sequência pois não tem CDS")
                continue

            # Procura o mRNA que contém o CDS
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

            # ... (código anterior que acha o target_feature do mRNA) ...

            if target_feature is None:
                target_feature = cds_feature

            # ======================================================
            # 1. O CORTE CIRÚRGICO (Matando a Armadilha do Promotor)
            # ======================================================
            mrna_start = int(target_feature.location.start)
            mrna_end   = int(target_feature.location.end)

            # Recorta EXATAMENTE o tamanho do gene (ignora os milhares de pares de bases ao redor)
            cropped_seq_obj = register.seq[mrna_start:mrna_end]

            # Usa o seu extractor para pegar as coordenadas brutas
            exons_intervals_raw = re.make_exons_intervals_list(target_feature.location)

            # Arrasta as coordenadas para o novo "Ponto Zero" (já que cortamos o começo)
            exons_intervals = [[s - mrna_start, e - mrna_start] for s, e in exons_intervals_raw]

            # ======================================================
            # 2. A MÁGICA DA FITA REVERSA
            # ======================================================
            if target_feature.location.strand == -1:
                # Vira a fita do avesso (A vira T, C vira G, e de trás pra frente)
                seq = str(cropped_seq_obj.reverse_complement()).upper()

                L = len(seq)
                exons_intervals_rev = []

                # Espelha as coordenadas para a nova fita invertida
                for s, e in exons_intervals:
                    new_start = L - 1 - e
                    new_end = L - 1 - s
                    exons_intervals_rev.append([new_start, new_end])

                # Como a fita virou de trás pra frente, os últimos éxons viraram os primeiros.
                # Então, reordenamos a lista para o Python ler certinho da esquerda pra direita
                exons_intervals = sorted(exons_intervals_rev, key=lambda x: x[0])
            else:
                # Se for fita positiva, é só converter pra string e seguir a vida
                seq = str(cropped_seq_obj).upper()

            # ======================================================
            # 3. O ESCUDO ANTI-ALUCINAÇÃO (Single-Exon)
            # ======================================================
            # Agora criamos os íntrons usando a sua função
            introns_intervals = re.make_introns_intervals_list(exons_intervals)

            if len(introns_intervals) == 0:
                print("Pulando gene sem íntrons...")

            # ======================================================
            # SEGUE O JOGO
            # ======================================================
            seq = inject_degenerate_nucleotides(seq, exons_intervals, introns_intervals, INJECTION_RATE)

            introns = re.make_introns_list(introns_intervals, seq)
            exons = re.make_exons_list(exons_intervals, seq)

            sample = {
                "sequence": seq,
                "exon_intervals": exons_intervals,
                "exons": exons,
                "intron_intervals": introns_intervals,
                "introns": introns,
            }

            num = num + 1
            seen_sequences.add(seq)
            data.append(sample)

        except Exception as e:
            print(f"\nErro ao processar o registro {register.id}: {e}")
            print("Pulando para o próximo...\n")
            continue

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

def save_preprocessed_genbank_file(genbank_input_filepath, genbank_filepath_output, INJECTION_RATE):
    """Preprocess the genbankfile."""

    print("Preprocessing GenBank file...")
    data = preprocess_genbank_file(genbank_input_filepath, INJECTION_RATE)
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

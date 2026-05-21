"""
The script processes genomic sequences with annotated exons and introns to prepare training data.

Steps involved:
1. Load data: Reads a pickled list of sequences and their exon coordinates.
2. Tag positions: Converts each sequence into a list of labels (0 = exon, 1 = intron) of the same length as the sequence.
3. Encode nucleotides: Transforms each base into a numeric representation (A=1, T=2, G=3, C=4, degenerate=5).
4. Create sliding windows: For every position in each sequence, generates a centered window of 60 nucleotides.
5. Pair input with label: Stores each window along with the central position label (0 or 1) in a list XY.
6. Count exon/intron positions: Computes total numbers of introns and exons.
7. Save processed data: Pickles the final XY dataset for later use in model training.

obs.
RYSWKMBDHVN = degenerate bases

sample = {
            "sequence": seq [ACTG...],
            "exon_intervals": exons_intervals [(start1, end1), (start2, end2), ...],
            "exons": exons [ACTG..., ACTG..., ...],
            "intron_intervals": introns_intervals [(start1, end1), (start2, end2), ...],
            "introns": introns [ACTG..., ACTG..., ...],
}

data.append(sample)
"""

import gc
import random
import numpy as np
import os
import pickle

BASE_TO_VECTOR = {
    'A':[1,0,0,0],
    'T':[0,1,0,0],
    'G':[0,0,1,0],
    'C':[0,0,0,1],

    'R':[0.5,0,0.5,0],
    'Y':[0,0.5,0,0.5],
    'S':[0,0,0.5,0.5],
    'W':[0.5,0.5,0,0],
    'K':[0,0.5,0.5,0],
    'M':[0.5,0,0,0.5],

    'B':[0,1/3,1/3,1/3],
    'D':[1/3,1/3,1/3,0],
    'H':[1/3,1/3,0,1/3],
    'V':[1/3,0,1/3,1/3],

    'N':[0.25,0.25,0.25,0.25]
}

quantidade_bases_degeneradas = 0

quantidade_bases = 0

def tag_positions(sample) -> list[int]:
    """Tag each position in the sequence as exon (1) or intron (0)."""

    tag = [-1] * len(sample["sequence"])  # Initialize all positions as intron (0)

    for start, end in sample["intron_intervals"]:
        for i in range(start, end + 1):
            tag[i] = 0  # íntron real

    for start, end in sample["exon_intervals"]:
        for i in range(start, end + 1): # Inclusive end position
            tag[i] = 1  # Mark exon positions as 1

    return tag

def slide_window(sample, window_size=180) -> list[list[int]]:
    """Create a sliding window centered at the given position."""

    half = window_size // 2
    seq = transform_baseSeq_to_onehot(sample["sequence"])

    windows = []

    for k in range(len(seq)):
        window = []

        for offset in range(-half, half):
            pos = k + offset
            if pos < 0 or pos >= len(seq):
                window.append([0,0,0,0])  # Padding
            else:
                window.append(seq[pos])

        windows.append(window)

    return windows

def transform_baseSeq_to_onehot(baseSeq) -> list[int]:
    """Transform nucleotide base sequence to numeric sequence."""

    encoded = []
    global quantidade_bases
    quantidade_bases += len(baseSeq)

    for base in baseSeq.upper():
        if base in BASE_TO_VECTOR:
            if base not in ['A', 'T', 'G', 'C']:
                global quantidade_bases_degeneradas
                quantidade_bases_degeneradas += 1

        encoded.append(
            BASE_TO_VECTOR.get(base, [0,0,0,0])  # unknown → padding
        )

    return np.array(encoded, dtype=np.float16) # Convert to numpy array for better performance in model training

def save_XY_to_file(output_path, X_list, y_list):
    total_samples = len(X_list)
    print(f"Empilhando {total_samples} amostras... (Método Anti-Crash)")

    # 1. Cria a matriz JÁ NO FORMATO LEVE (float16). Isso vai ocupar só ~1.5 GB de RAM.
    X = np.empty((total_samples, 180, 4), dtype=np.float16)

    # 2. Preenche a matriz de forma cirúrgica, linha por linha (não dá pico de RAM)
    for i, x_window in enumerate(X_list):
        X[i] = x_window

    # 3. Mata a lista antiga na mesma hora para desocupar a RAM
    del X_list

    # 4. Converte os labels para o menor tamanho possível
    y = np.array(y_list, dtype=np.int8)
    del y_list

    print(f"Salvando arquivo comprimido em: {output_path}")
    np.savez_compressed(output_path, X=X, y=y)

    # Limpeza final
    del X
    del y

def extract_windows_numpy(seq_onehot, indices, window_size=180):
    """
    Extrai janelas usando slicing numpy com padding vetorizado.
    Evita listas de listas intermediárias.
    """
    half = window_size // 2
    n = len(seq_onehot)

    # Pré-aloca diretamente o array final (sem listas intermediárias)
    X = np.zeros((len(indices), window_size, 4), dtype=np.float16)

    for i, k in enumerate(indices):
        start, end = k - half, k + half

        # Calcula o quanto está dentro dos limites
        src_start = max(0, start)
        src_end   = min(n, end)
        dst_start = src_start - start  # offset no destino (onde o padding termina)
        dst_end   = dst_start + (src_end - src_start)

        X[i, dst_start:dst_end] = seq_onehot[src_start:src_end]
        # Fora do intervalo já é zero (padding) graças ao np.zeros

    return X

def extract_balanced_windows(sample, tagged_seq, window_size=180):
    tagged_arr = np.asarray(tagged_seq)  # evita chamadas .index() repetidas

    indices_intron = np.where(tagged_arr == 0)[0]
    indices_exon = np.where(tagged_arr == 1)[0]

    min_len = min(len(indices_intron), len(indices_exon))
    if min_len == 0:
        return None, None

    # Sorteia sem criar listas Python grandes
    idx_i = np.random.choice(indices_intron, min_len, replace=False)
    idx_e = np.random.choice(indices_exon,   min_len, replace=False)

    indices_finais = np.concatenate([idx_i, idx_e])
    np.random.shuffle(indices_finais)

    # Só converte para one-hot o necessário
    seq_onehot = transform_baseSeq_to_onehot(sample["sequence"])
    seq_onehot = np.asarray(seq_onehot, dtype=np.float16)  # uma conversão só

    X = extract_windows_numpy(seq_onehot, indices_finais, window_size)
    y = tagged_arr[indices_finais].astype(np.int8)

    return X, y


def build_XY_dataset(data):
    print("Contando amostras para pré-alocar memória...")
    tagged_seqs = []

    for sample in data:
        tagged = tag_positions(sample)
        tagged_seqs.append(tagged)

    # --- Passo 1: coleta TUDO sem balancear por gene ---
    window_size = 180
    X_blocos = []
    y_blocos = []

    for i, (sample, tagged_seq) in enumerate(zip(data, tagged_seqs)):
        if i % 50 == 0:
            print(f"Processando gene {i+1}/{len(data)}")

        tagged_arr = np.asarray(tagged_seq)
        indices = np.where(tagged_arr >= 0)[0]  # só posições válidas (não -1)

        if len(indices) == 0:
            continue

        seq_onehot = transform_baseSeq_to_onehot(sample["sequence"])
        seq_onehot = np.asarray(seq_onehot, dtype=np.float16)

        X = extract_windows_numpy(seq_onehot, indices, window_size)
        y = tagged_arr[indices].astype(np.int8)

        X_blocos.append(X)
        y_blocos.append(y)

    print("Concatenando tudo...")
    X_final = np.concatenate(X_blocos, axis=0)
    y_final = np.concatenate(y_blocos, axis=0)

    # --- Passo 2: balanceia GLOBALMENTE ---
    print("Balanceando globalmente...")
    idx_exon   = np.where(y_final == 1)[0]
    idx_intron = np.where(y_final == 0)[0]
    min_len = min(len(idx_exon), len(idx_intron))

    print(f"Total éxons: {len(idx_exon)}, Total íntrons: {len(idx_intron)}, Usando: {min_len} de cada")

    idx_bal = np.concatenate([
        np.random.choice(idx_exon,   min_len, replace=False),
        np.random.choice(idx_intron, min_len, replace=False)
    ])
    np.random.shuffle(idx_bal)

    X_final = X_final[idx_bal]
    y_final = y_final[idx_bal]

    print(f"Formato final de X: {X_final.shape}")
    print(f"Formato final de Y: {y_final.shape}")
    gc.collect()

    return X_final, y_final

"""
def extract_balanced_windows(sample, tagged_seq, window_size=120):
    half = window_size // 2
    seq_onehot = transform_baseSeq_to_onehot(sample["sequence"])

    # 1. Mapeia os índices de Íntrons (0) e Éxons (1)
    indices_intron = [i for i, tag in enumerate(tagged_seq) if tag == 0]
    indices_exon = [i for i, tag in enumerate(tagged_seq) if tag == 1]

    # 2. Descobre quem é a minoria e iguala o jogo (Undersampling)
    min_len = min(len(indices_intron), len(indices_exon))

    if min_len == 0:
        return [], [] # Previne erro se a sequência for anômala (só intron ou só exon)

    indices_intron_balanceados = random.sample(indices_intron, min_len)
    indices_exon_balanceados = random.sample(indices_exon, min_len)

    # Junta os índices sorteados e embaralha
    indices_finais = indices_intron_balanceados + indices_exon_balanceados
    random.shuffle(indices_finais)

    X_balanced = []
    y_balanced = []

    # 3. Cria as janelas SOMENTE para os pontos sorteados
    for k in indices_finais:
        window = []
        for offset in range(-half, half):
            pos = k + offset
            if pos < 0 or pos >= len(seq_onehot):
                window.append([0, 0, 0, 0])  # Padding nas bordas do gene
            else:
                window.append(seq_onehot[pos])

        X_balanced.append(window)
        y_balanced.append(tagged_seq[k])

    return X_balanced, y_balanced

def build_XY_dataset(data):
    X_blocos = []
    y_blocos = []

    for i, sample in enumerate(data):
        if i % 50 == 0:  # A cada 50 genes, imprime o progresso
            print(f"Processando gene {i+1}/{len(data)}")

        tagged_seq = tag_positions(sample)

        # Chama a nova função inteligente
        X_bal, y_bal = extract_balanced_windows(sample, tagged_seq)

        if len(X_bal) > 0:
            X_blocos.append(np.array(X_bal, dtype=np.float16))
            y_blocos.append(np.array(y_bal, dtype=np.int8))

    X_final = np.concatenate(X_blocos, axis=0)
    y_final = np.concatenate(y_blocos, axis=0)

    print(f"Formato final de X: {X_final.shape}")
    print(f"Formato final de Y: {y_final.shape}")

    gc.collect() # Passa a vassoura final

    return X_final, y_final

"""

def modeling_train_data(data_filepath_input, XY_filepath_output):
    data = pickle.load(open(data_filepath_input, "rb"))

    # Captura as listas processadas
    X_list, y_list = build_XY_dataset(data)

    # Salva usando a nova lógica
    save_XY_to_file(XY_filepath_output, X_list, y_list)

    print("Concluído com sucesso!")
    print("Quantidade de bases degeneradas: ", quantidade_bases_degeneradas)
    print("Quantidade de bases: ", quantidade_bases)
    print("Porcentagem de bases degeneradas: ", (quantidade_bases_degeneradas / quantidade_bases) * 100, "%")

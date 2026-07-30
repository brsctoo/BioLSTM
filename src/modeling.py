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

Note:
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
import numpy as np
import pickle

# Central window size — overridden at runtime by pipeline.py via set_window_size()
WINDOW_SIZE = 400

def set_window_size(size: int) -> None:
    """Called by pipeline.py to propagate --window-size into this module."""
    global WINDOW_SIZE
    WINDOW_SIZE = size

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

degenerate_bases_count = 0
total_bases_count = 0

def tag_positions(sample) -> list[int]:
    """Tag each position in the sequence as exon (1) or intron (0)."""

    tag = [-1] * len(sample["sequence"])  # Initialize all positions as intron (0)

    for start, end in sample["intron_intervals"]:
        for i in range(start, end + 1):
            tag[i] = 0  # Real intron

    for start, end in sample["exon_intervals"]:
        for i in range(start, end + 1): # Inclusive end position
            tag[i] = 1  # Mark exon positions as 1

    return tag

def slide_window(sample, window_size=None) -> list[list[int]]:
    """Create a sliding window centered at the given position."""
    if window_size is None: window_size = WINDOW_SIZE

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

# Fixed return type hint from list[int] to np.ndarray
def transform_baseSeq_to_onehot(baseSeq) -> np.ndarray:
    """Transform nucleotide base sequence to numeric sequence."""

    encoded = []
    global total_bases_count
    total_bases_count += len(baseSeq)

    for base in baseSeq.upper():
        if base in BASE_TO_VECTOR:
            if base not in ['A', 'T', 'G', 'C']:
                global degenerate_bases_count
                degenerate_bases_count += 1

        encoded.append(
            BASE_TO_VECTOR.get(base, [0,0,0,0])  # unknown → padding
        )

    return np.array(encoded, dtype=np.float16) # Convert to numpy array for better performance in model training

def save_XY_to_file(output_path, X_list, y_list):
    total_samples = len(X_list)
    print(f"Stacking {total_samples} samples... (Anti-Crash Method)")

    # 1. Pre-allocate matrix DIRECTLY IN LIGHT FORMAT (float16) to limit RAM overhead (~1.5 GB)
    X = np.empty((total_samples, WINDOW_SIZE, 4), dtype=np.float16)

    # 2. Fill the matrix sequentially row by row to prevent spikes in memory consumption
    for i, x_window in enumerate(X_list):
        X[i] = x_window

    # 3. Immediately drop the old list structure to free memory allocation
    del X_list

    # 4. Downcast targets to the lowest valid memory size possible
    y = np.array(y_list, dtype=np.int8)
    del y_list

    print(f"Saving compressed data array to: {output_path}")
    np.savez_compressed(output_path, X=X, y=y)

    # Final sweep
    del X
    del y

def extract_windows_numpy(seq_onehot, indices, window_size=None):
    """
    Extract windows using numpy slicing with vectorized padding.
    Avoids building heavy intermediate Python lists of lists.
    """
    if window_size is None: window_size = WINDOW_SIZE
    half = window_size // 2
    n = len(seq_onehot)

    # Pre-allocate the final array memory region directly
    X = np.zeros((len(indices), window_size, 4), dtype=np.float16)

    for i, k in enumerate(indices):
        start, end = k - half, k + half

        # Calculate coordinates bounding box inside sequence range
        src_start = max(0, start)
        src_end   = min(n, end)
        dst_start = src_start - start  # Destination offset (where boundary padding ends)
        dst_end   = dst_start + (src_end - src_start)

        X[i, dst_start:dst_end] = seq_onehot[src_start:src_end]
        # Any index out of bounds remains zero natively due to np.zeros configuration

    return X

def extract_windows_labels_numpy(tagged_arr, indices, window_size=None):
    if window_size is None: window_size = WINDOW_SIZE
    half = window_size // 2
    n = len(tagged_arr)

    # Preenche com -1 (desconhecido/fora do genoma)
    Y_win = np.full((len(indices), window_size), -1, dtype=np.int8)

    for i, k in enumerate(indices):
        start, end = k - half, k + half
        src_start = max(0, start)
        src_end   = min(n, end)
        dst_start = src_start - start
        dst_end   = dst_start + (src_end - src_start)

        Y_win[i, dst_start:dst_end] = tagged_arr[src_start:src_end]

    return Y_win

def extract_balanced_windows(sample, tagged_seq, window_size=None):
    if window_size is None: window_size = WINDOW_SIZE
    tagged_arr = np.asarray(tagged_seq)  # Prevents repetitive evaluations of .index() loops

    indices_intron = np.where(tagged_arr == 0)[0]
    indices_exon = np.where(tagged_arr == 1)[0]

    min_len = min(len(indices_intron), len(indices_exon))
    if min_len == 0:
        return None, None

    # Sample configurations randomly without initiating major standard Python structures
    idx_i = np.random.choice(indices_intron, min_len, replace=False)
    idx_e = np.random.choice(indices_exon,   min_len, replace=False)

    final_indices = np.concatenate([idx_i, idx_e])
    np.random.shuffle(final_indices)

    # Only cast target elements to one-hot structure conditionally as required
    seq_onehot = transform_baseSeq_to_onehot(sample["sequence"])
    seq_onehot = np.asarray(seq_onehot, dtype=np.float16)  # Execute single pass matrix conversion

    X = extract_windows_numpy(seq_onehot, final_indices, window_size)
    y = tagged_arr[final_indices].astype(np.int8)

    return X, y

def build_XY_from_gene_list(gene_list, window_size=None, stride=2):
    """
    Generate X, y dataset from a list of gene samples WITHOUT global
    undersampling, preserving the natural exon/intron class distribution.
    """
    if window_size is None: window_size = WINDOW_SIZE
    X_blocks = []
    y_blocks = []
    y_window_blocks = []

    for i, sample in enumerate(gene_list):
        if i % 50 == 0:
            print(f"  Processing gene {i+1}/{len(gene_list)}")

        tagged_seq = tag_positions(sample)
        tagged_arr = np.asarray(tagged_seq)

        # Only annotated positions (skip -1 UTR/flanking regions)
        indices = np.where(tagged_arr >= 0)[0]

        # --- A MÁGICA DO STRIDE ENTRA AQUI ---
        # Aplica o fatiamento para pegar 1 posição a cada X nucleotídeos
        indices = indices[::stride]
        # -------------------------------------

        if len(indices) == 0:
            continue

        seq_onehot = transform_baseSeq_to_onehot(sample["sequence"])
        seq_onehot = np.asarray(seq_onehot, dtype=np.float16)

        X = extract_windows_numpy(seq_onehot, indices, window_size)
        y = tagged_arr[indices].astype(np.int8)
        y_win = extract_windows_labels_numpy(tagged_arr, indices, window_size)

        X_blocks.append(X)
        y_blocks.append(y)
        y_window_blocks.append(y_win)

    if not X_blocks:
        raise ValueError("No windows were generated. Check the input data.")

    X_final = np.concatenate(X_blocks, axis=0)
    y_final = np.concatenate(y_blocks, axis=0)
    y_window_final = np.concatenate(y_window_blocks, axis=0)

    # Shuffle within the split to remove sequential ordering bias
    rng_idx = np.random.permutation(len(y_final))
    return X_final[rng_idx], y_final[rng_idx], y_window_final[rng_idx]


def modeling_train_data_gene_split(data_filepath_input, XY_train_output, XY_val_output,
                                   val_gene_fraction=0.2):
    """
    CORRECTED VERSION: splits genes BEFORE generating windows, eliminating
    data leakage caused by overlapping sliding windows across train/val sets.

    Key differences from modeling_train_data:
      - Split unit is the GENE (independent entity), not the window (derived).
      - Windows are generated separately for each split — no cross-contamination.
      - Natural exon/intron distribution is PRESERVED (no 50/50 balancing).
        Class imbalance is handled by BinaryFocalCrossentropy + class_weight
        in train_model_gene_split.

    Saves two separate .npz files:
      XY_train_output : 80% of genes -> training windows
      XY_val_output   : 20% of genes -> validation windows

    Args:
        data_filepath_input : path to the .mod1 pickle file produced by genbank_reader.
        XY_train_output     : output path for the training .npz file.
        XY_val_output       : output path for the validation .npz file.
        val_gene_fraction   : fraction of genes reserved for validation (default: 0.2).

    Returns:
        X_train, y_train, X_val, y_val as numpy arrays.
    """
    data = pickle.load(open(data_filepath_input, "rb"))

    print(f"Total genes loaded: {len(data)}")

    # 1. Split at the GENE level (independent unit)
    n_val   = max(1, int(len(data) * val_gene_fraction))
    n_train = len(data) - n_val

    # Shuffle genes (not windows!)
    gene_indices = np.random.permutation(len(data))
    train_genes = [data[i] for i in gene_indices[:n_train]]
    val_genes   = [data[i] for i in gene_indices[n_train:]]

    print(f"Gene-level split -> Train: {len(train_genes)} genes | Val: {len(val_genes)} genes")

    # 2. Generate windows SEPARATELY for each split (no cross-contamination)
    print("\nGenerating TRAIN windows (natural distribution, no undersampling)...")
    X_train, y_train, y_window_train = build_XY_from_gene_list(train_genes)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Exons: {np.sum(y_train==1):,} | Introns: {np.sum(y_train==0):,}")
    print(f"  Exon proportion: {np.mean(y_train==1)*100:.1f}%")

    print("\nGenerating VALIDATION windows (natural distribution, no undersampling)...")
    X_val, y_val, y_window_val = build_XY_from_gene_list(val_genes)
    print(f"  X_val shape  : {X_val.shape}")
    print(f"  Exons: {np.sum(y_val==1):,}   | Introns: {np.sum(y_val==0):,}")
    print(f"  Exon proportion: {np.mean(y_val==1)*100:.1f}%")

    # 3. Save both splits to separate files
    print(f"\nSaving training split to  : {XY_train_output}")
    np.savez_compressed(XY_train_output, X=X_train, y=y_train, y_window=y_window_train)

    print(f"Saving validation split to: {XY_val_output}")
    np.savez_compressed(XY_val_output, X=X_val, y=y_val, y_window=y_window_val)

    print("\nGene-split featurization complete!")

    # Report degenerate nucleotide stats (accumulated across all processed genes)
    print("Degenerate nucleotides count: ", degenerate_bases_count)
    print("Total sequence bases count  : ", total_bases_count)
    if total_bases_count > 0:
        ratio = (degenerate_bases_count / total_bases_count) * 100
        print(f"Ratio of degenerate nucleotides: {ratio:.4f}%")

    return X_train, y_train, X_val, y_val

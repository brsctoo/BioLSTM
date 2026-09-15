
import pickle

import numpy as np
from numpy._typing import NDArray

# Central window size
WINDOW_SIZE = 400

def set_window_size(
    size: int
) -> None:
    """
    Called by pipeline.py to propagate --window-size into this module.
    """
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

def tag_positions(
    sample: dict[str, object]
) -> list[int]:
    """
    Tag each position in the sequence as exon (1) or intron (0).
    """

    tag = [-1] * len(sample["sequence"])  # Initialize all positions as -1 # type: ignore

    for start, end in sample["intron_intervals"]: # type: ignore
        for i in range(start, end + 1):
            if 0 <= i < len(tag):
                tag[i] = 0  # Real intron

    for start, end in sample["exon_intervals"]: # type: ignore
        for i in range(start, end + 1):
            if 0 <= i < len(tag):
                tag[i] = 1  # Mark exon positions as 1

    return tag

def slide_window(
    sample: dict[str, object],
    window_size: int | None = None
) -> list[list[int]]:
    """
    Create a sliding window centered at the given position.
    """
    if window_size is None:
         window_size = WINDOW_SIZE

    half = window_size // 2
    seq = transform_baseSeq_to_onehot(sample["sequence"]) # type: ignore

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
def transform_baseSeq_to_onehot(
    baseSeq: str | list[str]
) -> np.ndarray:
    """
    Transform nucleotide base sequence to numeric sequence.
    """

    encoded = []
    global total_bases_count
    total_bases_count += len(baseSeq)

    for base in baseSeq.upper(): # type: ignore
        # Degenerance counter
        if base in BASE_TO_VECTOR and base not in ['A', 'T', 'G', 'C']:
            global degenerate_bases_count
            degenerate_bases_count += 1

        encoded.append(
            BASE_TO_VECTOR.get(base, [0,0,0,0]) # Unknown = Padding
        )

    return np.array(encoded, dtype=np.float16) # Convert to numpy array for better performance in model training

def extract_windows_numpy(
    seq_onehot: np.ndarray,
    indices: list[int],
    window_size: int | None = None
) -> np.ndarray:
    """
    Extract windows using numpy slicing with vectorized padding.
    """
    if window_size is None:
         window_size = WINDOW_SIZE

    half = window_size // 2
    num_seq = len(seq_onehot)

    # 1. Create a matrix filled with ZEROS.
    X = np.zeros((len(indices), window_size, 4), dtype=np.float16)

    for i, center_pos in enumerate(indices):
        # 2. Calculate the ideal window boundaries
        ideal_start = center_pos - half
        ideal_end = center_pos + half

        # 3. In the real DNA sequence, slice
        seq_start = max(0, ideal_start)
        seq_end = min(num_seq, ideal_end)

        # 4. Inside the 400-space window, where to "paste" the DNA
        # If ideal_start was -10, seq_start will be 0. So 0 - (-10) = +10.
        # Meaning we start pasting the DNA at position 10 of the window (leaving 10 zeros at the beginning).
        window_start = seq_start - ideal_start

        # The end position in the window is simply the start + the size of the sliced piece
        slice_length = seq_end - seq_start
        window_end = window_start + slice_length

        # 5. Paste the exact DNA slice into the exact spot in the window
        X[i, window_start:window_end] = seq_onehot[seq_start:seq_end]

    return X

def extract_windows_labels_numpy(
    tagged_arr: np.ndarray,
    indices: list[int],
    window_size: int | None = None
) -> np.ndarray:
    """
    Extract the label windows (0 for intron, 1 for exon) using numpy slicing.
    """
    if window_size is None:
        window_size = WINDOW_SIZE

    half = window_size // 2
    num_seq = len(tagged_arr)

    # 1. Create a matrix filled with -1
    Y_win = np.full((len(indices), window_size), -1, dtype=np.int8)

    for i, center_pos in enumerate(indices):
        # 2. Calculate the ideal window boundaries
        ideal_start = center_pos - half
        ideal_end = center_pos + half

        # 3. Slice boundaries within the valid array length
        seq_start = max(0, ideal_start)
        seq_end   = min(num_seq, ideal_end)

        # 4. Where to paste this valid slice inside the window
        # Example: If ideal_start is -20, seq_start is 0. window_start becomes +20.
        window_start = seq_start - ideal_start

        # The end position in the window is simply the start + slice length
        slice_length = seq_end - seq_start
        window_end   = window_start + slice_length

        # 5. Paste the exact labels slice into the target window
        Y_win[i, window_start:window_end] = tagged_arr[seq_start:seq_end]

    return Y_win

def build_XY_from_gene_list(
    gene_list: list[dict[str, object]],
    window_size: int | None = None,
    stride: int = 70
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate X, y dataset from a list of gene samples without global
    """

    if window_size is None:
         window_size = WINDOW_SIZE

    # Lists to store the matrices generated for each individual gene
    X_blocks = []
    y_blocks = []

    for i, sample in enumerate(gene_list):
        # Print progress every 50 genes
        if i % 50 == 0:
            print(f"Processing gene {i+1}/{len(gene_list)}")

        # 1. Tag the DNA sequence (1 for exon, 0 for intron, -1 for unknown)
        tagged_seq = tag_positions(sample)
        tagged_arr = np.asarray(tagged_seq)

        # 2. Find valid center positions
        # Only center windows on annotated positions (skip -1 flanking regions)
        indices = np.where(tagged_arr >= 0)[0]
        indices = indices[::stride]

        # If the gene is too small or has no valid annotations
        if len(indices) == 0:
            continue

        # 3. Convert the raw DNA string into the One-Hot numeric matrix
        seq_onehot = transform_baseSeq_to_onehot(sample["sequence"]) # type: ignore
        seq_onehot = np.asarray(seq_onehot, dtype=np.float16)

        # 4. Extract the DNA features (X) and the labels (y) using numpy tools
        X = extract_windows_numpy(seq_onehot, indices, window_size) # type: ignore
        y = extract_windows_labels_numpy(tagged_arr, indices, window_size) # type: ignore

        # Append the processed matrices of this gene to our blocks
        X_blocks.append(X)
        y_blocks.append(y)

    if not X_blocks:
        raise ValueError("No windows were generated. Check the input data.")

    # 6. Glue all individual gene blocks into one massive array
    X_final = np.concatenate(X_blocks, axis=0)
    y_final = np.concatenate(y_blocks, axis=0)

    # 7. Shuffle the rows so the model doesn't memorize the sequential order of genes
    rng_idx = np.random.permutation(len(y_final))
    return X_final[rng_idx], y_final[rng_idx]


def modeling_train_data_gene_split(
    data_filepath_input: str,
    XY_train_output: str,
    XY_val_output: str,
    val_gene_fraction: float = 0.2
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Splits genes BEFORE generating windows, eliminating
    """
    with open(data_filepath_input, "rb") as f:
        data = pickle.load(f)

    print(f"Total genes loaded: {len(data)}")

    # 1. Split at the GENE level (independent unit)
    n_val = max(1, int(len(data) * val_gene_fraction))
    n_train = len(data) - n_val

    # Shuffle genes (not windows!)
    gene_indices = np.random.permutation(len(data))
    train_genes = [data[i] for i in gene_indices[:n_train]]
    val_genes   = [data[i] for i in gene_indices[n_train:]]

    print(f"Gene-level split -> Train: {len(train_genes)} genes | Val: {len(val_genes)} genes")

    # 2. Generate windows SEPARATELY for each split (no cross-contamination)
    print("\nGenerating TRAIN windows (natural distribution, no undersampling)...")
    X_train, y_train = build_XY_from_gene_list(train_genes)
    print(f"  X_train shape: {X_train.shape}")

    print("\nGenerating VALIDATION windows (natural distribution, no undersampling)...")
    X_val, y_val = build_XY_from_gene_list(val_genes)
    print(f"  X_val shape  : {X_val.shape}")

    # 3. Save both splits to separate files
    print(f"\nSaving training split to  : {XY_train_output}")
    np.savez_compressed(XY_train_output, X=X_train, y=y_train)

    print(f"Saving validation split to: {XY_val_output}")
    np.savez_compressed(XY_val_output, X=X_val, y=y_val)

    print("\nGene-split featurization complete!")



    return X_train, y_train, X_val, y_val

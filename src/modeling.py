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
    X = np.empty((total_samples, 180, 4), dtype=np.float16)

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

def extract_windows_numpy(seq_onehot, indices, window_size=180):
    """
    Extract windows using numpy slicing with vectorized padding.
    Avoids building heavy intermediate Python lists of lists.
    """
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

def extract_balanced_windows(sample, tagged_seq, window_size=180):
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


def build_XY_dataset(data):
    print("Counting target samples to map pre-allocated memory blocks...")
    tagged_seqs = []

    for sample in data:
        tagged = tag_positions(sample)
        tagged_seqs.append(tagged)

    # --- Step 1: Collect ALL items globally without intermediate per-gene matching bounds ---
    window_size = 180
    X_blocks = []
    y_blocks = []

    for i, (sample, tagged_seq) in enumerate(zip(data, tagged_seqs)):
        if i % 50 == 0:
            print(f"Processing gene structure {i+1}/{len(data)}")

        tagged_arr = np.asarray(tagged_seq)
        indices = np.where(tagged_arr >= 0)[0]  # Filter valid target indices exclusively (skip -1)

        if len(indices) == 0:
            continue

        seq_onehot = transform_baseSeq_to_onehot(sample["sequence"])
        seq_onehot = np.asarray(seq_onehot, dtype=np.float16)

        X = extract_windows_numpy(seq_onehot, indices, window_size)
        y = tagged_arr[indices].astype(np.int8)

        X_blocks.append(X)
        y_blocks.append(y)

    print("Concatenating intermediate block arrays...")
    X_final = np.concatenate(X_blocks, axis=0)
    y_final = np.concatenate(y_blocks, axis=0)

    # --- Step 2: Perform GLOBAL Undersampling/Balancing ---
    print("Balancing dataset distribution globally...")
    idx_exon   = np.where(y_final == 1)[0]
    idx_intron = np.where(y_final == 0)[0]
    min_len = min(len(idx_exon), len(idx_intron))

    print(f"Total exons mapped: {len(idx_exon)}, Total introns mapped: {len(idx_intron)}, Slicing subset window size: {min_len} each")

    idx_bal = np.concatenate([
        np.random.choice(idx_exon,   min_len, replace=False),
        np.random.choice(idx_intron, min_len, replace=False)
    ])
    np.random.shuffle(idx_bal)

    X_final = X_final[idx_bal]
    y_final = y_final[idx_bal]

    print(f"Final resolved array shape for matrix X: {X_final.shape}")
    print(f"Final resolved array shape for targets Y: {y_final.shape}")
    gc.collect()

    return X_final, y_final


def modeling_train_data(data_filepath_input, XY_filepath_output):
    data = pickle.load(open(data_filepath_input, "rb"))

    # Extract target array segments safely
    X_list, y_list = build_XY_dataset(data)

    # Export using optimized zero-peak pipeline structure
    save_XY_to_file(XY_filepath_output, X_list, y_list)

    print("Data processing pipeline completed successfully!")
    print("Degenerate nucleotides count: ", degenerate_bases_count)
    print("Total sequence bases count: ", total_bases_count)
    print("Ratio of degenerate nucleotides: ", (degenerate_bases_count / total_bases_count) * 100, "%")

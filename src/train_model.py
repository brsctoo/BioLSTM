"""
Get the data_XY from modeling.py, train the model, and save it for later use in validation.py.

The data_XY is a list of tuples: [(X, Y), ...], where X is the input sequence - window - (list of integers) and Y is the corresponding label (0 or 1).
"""

import numpy as np
from lstm_model import create_model, EPOCHS, BATCH_SIZE

def train_model(XY_filepath_input, result_filepath_output):
    # Fresh model + fresh optimizer state for every call. Importing a
    # module-level instance instead would leak weights and Adam's
    # momentum/variance accumulators across experiments run in the
    # same process.
    lstm_model = create_model()

    # Load dataset (npz: X, y) with allow_pickle=True
    data = np.load(XY_filepath_input, allow_pickle=True)
    X = np.array(data['X'], dtype=np.float32)
    y = np.array(data['y'], dtype=np.int32)

    # 2. RANDOMIZE DATASET (important to avoid bias in training)
    np.random.seed(123865)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # 3. SPLIT DATASET
    split_index = int(len(y) * 0.8)
    X_train = X[:split_index]
    Y_train = y[:split_index]
    X_test = X[split_index:]
    Y_test = y[split_index:]

    total = len(Y_train)

    print("Exons:", np.sum(Y_train == 1))
    print("Introns:", np.sum(Y_train == 0))
    print("Ratio:", np.mean(Y_train))

    print("Training the model...")
    print(f"Total training samples: {total}")

    # --- START OF MANUAL CLASS WEIGHT CALCULATION ---
    total_samples = len(Y_train)

    # Count how many 0s (introns) and 1s (exons) exist in the training set
    count_0 = np.sum(Y_train == 0)
    count_1 = np.sum(Y_train == 1)

    # Apply standard mathematical balancing formula (2 stands for the total number of target classes)
    weight_0 = total_samples / (2.0 * count_0)
    weight_1 = total_samples / (2.0 * count_1)

    class_weights = {0: weight_0, 1: weight_1}

    history = lstm_model.fit(
        X_train,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        class_weight=class_weights,
        verbose=2 # type: ignore
    )

    print("\nModel training completed.\n")
    test_results = lstm_model.evaluate(X_test, Y_test, verbose=2) # type: ignore
    print(f'\nTest results - Loss: {test_results[0]:.4f} - Accuracy: {100*test_results[1]:.2f}%\n')
    lstm_model.save(result_filepath_output)
    print(" ")
    print(" ")
    print(" ")
    print("History:", history)


# ---------------------------------------------------------------------------
# CORRECTED TRAINING FUNCTION — Gene-level split, no data leakage
# ---------------------------------------------------------------------------

def train_model_gene_split(XY_train_filepath, XY_val_filepath, result_filepath_output):
    """
    CORRECTED VERSION: trains using two separate .npz files produced by
    modeling.modeling_train_data_gene_split (gene-level split).

    Key differences from train_model:
      - Accepts pre-split train and validation arrays instead of a single
        shuffled pool, so genes in X_val NEVER appear in X_train.
      - Uses validation_data=(X_val, y_val) — NOT validation_split —
        which would re-contaminate from the same pool.
      - Computes class_weight from the real (unbalanced) distribution so
        BinaryFocalCrossentropy can focus on the minority class without
        discarding majority-class samples.

    Args:
        XY_train_filepath      : path to the training .npz file.
        XY_val_filepath        : path to the validation .npz file.
        result_filepath_output : path where the trained model (.h5) will be saved.

    Returns:
        history : Keras History object from model.fit.
    """
    lstm_model = create_model()

    # Load pre-split datasets
    print("Loading training data...")
    train_data = np.load(XY_train_filepath, allow_pickle=True)
    X_train = np.array(train_data['X'], dtype=np.float32)
    y_train = np.array(train_data['y'], dtype=np.int32)

    print("Loading validation data...")
    val_data = np.load(XY_val_filepath, allow_pickle=True)
    X_val = np.array(val_data['X'], dtype=np.float32)
    y_val = np.array(val_data['y'], dtype=np.int32)

    print(f"\n--- REAL dataset distribution (no artificial balancing) ---")
    print(f"Train -> Exons: {np.sum(y_train==1):,} | Introns: {np.sum(y_train==0):,} | "
          f"Exon proportion: {np.mean(y_train==1)*100:.1f}%")
    print(f"Val   -> Exons: {np.sum(y_val==1):,}   | Introns: {np.sum(y_val==0):,}   | "
          f"Exon proportion: {np.mean(y_val==1)*100:.1f}%")
    print("----------------------------------------------------------\n")

    # Compute class weights from the real training distribution
    # BinaryFocalCrossentropy + class_weight are complementary:
    # focal loss focuses gradient on hard samples; class_weight scales the
    # gradient magnitude to counteract the raw class frequency imbalance.
    total_samples = len(y_train)
    count_0 = np.sum(y_train == 0)  # introns
    count_1 = np.sum(y_train == 1)  # exons
    weight_0 = total_samples / (2.0 * count_0)
    weight_1 = total_samples / (2.0 * count_1)
    class_weights = {0: float(weight_0), 1: float(weight_1)}
    print(f"Class weights -> intron (0): {weight_0:.4f} | exon (1): {weight_1:.4f}\n")

    # Train using validation_data with fully separated gene sets
    # (NOT validation_split, which would re-contaminate from the same pool)
    history = lstm_model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),  # genes completely separated
        class_weight=class_weights,
        verbose=2  # type: ignore
    )

    print("\nModel training completed.\n")
    test_results = lstm_model.evaluate(X_val, y_val, verbose=2)  # type: ignore
    print(f'\nValidation results -> Loss: {test_results[0]:.4f} | '
          f'Accuracy: {100*test_results[1]:.2f}%\n')

    lstm_model.save(result_filepath_output)
    print(f"Model saved to: {result_filepath_output}")
    print(" ")
    print(" ")
    print("History:", history)
    return history

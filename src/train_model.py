"""
Get the data_XY from modeling.py, train the model, and save it for later use in validation.py.

The data_XY is a list of tuples: [(X, Y), ...], where X is the input sequence - window - (list of integers) and Y is the corresponding label (0 or 1).
"""

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from lstm_model import create_model, BATCH_SIZE

def train_model_gene_split(XY_train_filepath, XY_val_filepath, result_filepath_output, epochs=100):
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
        XY_train_filepath    : path to the training .npz file.
        XY_val_filepath      : path to the validation .npz file.
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
    print(f"Val     -> Exons: {np.sum(y_val==1):,}   | Introns: {np.sum(y_val==0):,}   | "
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

    # Configuração do Early Stopping rigoroso para o fluxo gene-split (paciência de 3 épocas)
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Train using validation_data with fully separated gene sets
    # (NOT validation_split, which would re-contaminate from the same pool)
    history = lstm_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),  # genes completely separated
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2  # type: ignore
    )

    print("\nModel training completed.\n")
    test_results = lstm_model.evaluate(X_val, y_val, verbose=2)  # type: ignore
    print(f'\nValidation results -> Loss: {test_results[0]:.4f} | ' # type: ignore
          f'Accuracy: {100*test_results[1]:.2f}%\n') # type: ignore

    lstm_model.save(result_filepath_output)
    print(f"Model saved to: {result_filepath_output}")
    print(" ")
    print(" ")
    print("History:", history)
    return history

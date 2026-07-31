"""
Get the data_XY from modeling.py, train the model, and save it for later use in validation.py.

The data_XY is a list of tuples: [(X, Y), ...], where X is the input sequence - window - (list of integers) and Y is the corresponding label (0 or 1).
"""

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

    print(f"\n--- REAL dataset distribution (Seq2Seq) ---")
    count_0 = np.sum(y_train == 0)  # introns
    count_1 = np.sum(y_train == 1)  # exons
    total_valid = count_0 + count_1

    print(f"Train -> Exons: {count_1:,} | Introns: {count_0:,} | "
          f"Exon proportion: {count_1/total_valid*100:.1f}%")
    print("----------------------------------------------------------\n")

    weight_0 = total_valid / (2.0 * max(1, count_0))
    weight_1 = total_valid / (2.0 * max(1, count_1))

    # Separa o canal do RF (5º canal) do canal de DNA (4 canais)
    if X_train.shape[-1] == 5:
        # A probabilidade RF é a mesma pra janela inteira, pegamos de qualquer ponto
        rf_train = X_train[:, 0, 4:5]
        X_train_dna = X_train[:, :, :4]
    else:
        rf_train = np.zeros((X_train.shape[0], 1), dtype=np.float32)
        X_train_dna = X_train[:, :, :4]

    if X_val.shape[-1] == 5:
        rf_val = X_val[:, 0, 4:5]
        X_val_dna = X_val[:, :, :4]
    else:
        rf_val = np.zeros((X_val.shape[0], 1), dtype=np.float32)
        X_val_dna = X_val[:, :, :4]

    train_inputs = {'dna_input': X_train_dna, 'rf_input': rf_train}
    val_inputs = {'dna_input': X_val_dna, 'rf_input': rf_val}

    # Criar sample weights temporais (N, WINDOW_SIZE)
    sample_weights_arr = np.zeros_like(y_train, dtype=np.float32)
    sample_weights_arr[y_train == 0] = weight_0
    sample_weights_arr[y_train == 1] = weight_1
    # y_train == -1 fica com peso 0 (ignorado)

    val_sample_weights_arr = np.zeros_like(y_val, dtype=np.float32)
    val_sample_weights_arr[y_val == 0] = weight_0
    val_sample_weights_arr[y_val == 1] = weight_1
    # y_val == -1 fica com peso 0 (ignorado) na validação

    # Evitar que a loss quebre com valores -1
    y_train_clean = np.where(y_train == -1, 0, y_train)
    y_val_clean = np.where(y_val == -1, 0, y_val)

    # Adicionar dimensão final (N, WINDOW_SIZE, 1)
    y_train_clean = np.expand_dims(y_train_clean, -1)
    y_val_clean = np.expand_dims(y_val_clean, -1)

    train_sample_weights = {'final_out': sample_weights_arr, 'aux_lstm_out': sample_weights_arr}
    val_sample_weights = {'final_out': val_sample_weights_arr, 'aux_lstm_out': val_sample_weights_arr}

    train_targets = {'final_out': y_train_clean, 'aux_lstm_out': y_train_clean}
    val_targets = {'final_out': y_val_clean, 'aux_lstm_out': y_val_clean}

    # Configuração de Callbacks: ReduceLROnPlateau + EarlyStopping (monitorando val_auc)
    # Obs: como a métrica é um dicionário no compile com o nome final_out, o Keras
    # vai registrá-la como 'val_final_out_auc' durante o fit, então usaremos esse nome.
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_final_out_auc',
        mode='max',
        factor=0.5,
        patience=6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_final_out_auc',
        mode='max',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # Train using validation_data with fully separated gene set
    history = lstm_model.fit(
        train_inputs,
        train_targets,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(val_inputs, val_targets, val_sample_weights), # type: ignore
        sample_weight=train_sample_weights,
        callbacks=[lr_scheduler, early_stop],
        verbose=2  # type: ignore
    )

    print("\nModel training completed.\n")
    test_results = lstm_model.evaluate(val_inputs, val_targets, verbose=2, return_dict=True)  # type: ignore

    if isinstance(test_results, dict):
        loss_val = test_results.get('loss', 0.0)
        acc_val = test_results.get('final_out_accuracy', test_results.get('accuracy', 0.0))
    else:
        loss_val = test_results[0]
        acc_val = test_results[3]

    print(f'\nValidation results -> Total Loss: {loss_val:.4f} | '
          f'Final Output Accuracy: {100*acc_val:.2f}%\n')

    lstm_model.save(result_filepath_output)
    print(f"Model saved to: {result_filepath_output}")
    print(" ")
    print(" ")
    print("History:", history)
    return history

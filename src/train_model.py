"""
Get the data_XY from modeling.py, train the model, and save it for later use in validation.py.

The data_XY is a list of tuples: [(X, Y), ...], where X is the input sequence - window - (list of integers) and Y is the corresponding label (0 or 1).
"""

import numpy as np
from lstm_model import lstm_model, EPOCHS, BATCH_SIZE
import os
from keras.models import load_model
from keras.callbacks import EarlyStopping # Important to speed up training

def train_model(XY_filepath_input, result_filepath_output):
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

    # --- INÍCIO DO CÁLCULO MANUAL DE PESOS ---
    total_samples = len(Y_train)
    
    # Contamos quantos 0s (introns) e 1s (exons) existem no treino
    count_0 = np.sum(Y_train == 0)
    count_1 = np.sum(Y_train == 1)
    
    # Aplicamos a fórmula matemática de balanceamento (2 é o número de classes)
    weight_0 = total_samples / (2.0 * count_0)
    weight_1 = total_samples / (2.0 * count_1)
    
    pesos_dit = {0: weight_0, 1: weight_1}

    history = lstm_model.fit(
        X_train,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        class_weight=pesos_dit, 
        verbose=1
    )

    print("\nModel training completed.\n")
    test_results = lstm_model.evaluate(X_test, Y_test, verbose=1)
    print(f'\nTest results - Loss: {test_results[0]:.4f} - Accuracy: {100*test_results[1]:.2f}%\n') 
    lstm_model.save(result_filepath_output)
    print(" ") 
    print(" ")
    print(" ")
    print("Histórico:", history)
"""
Creates the deep hybrid CNN + Stacked Bi-LSTM model for biological sequence classification.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Add, Concatenate,
    Bidirectional, LSTM, Dense, Dropout, Input, Attention, GlobalMaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy

# Define model parameters
SEQUENCE_LENGTH = 60
NUM_DISTINCT_WORDS = 5
EMBEDDING_DIM = 60

EPOCHS = 10
BATCH_SIZE = 100
LSTM_UNITS = 60

# Reduzido para 1e-4 para evitar que o modelo decore o treino muito rápido
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
WINDOWS_SIZE = 400  # overridden by pipeline.py via set_window_size()

def set_window_size(size: int) -> None:
    """Called by pipeline.py to propagate --window-size into this module."""
    global WINDOWS_SIZE
    WINDOWS_SIZE = size

def residual_block(x, filters, kernel_size):
    """Bloco residual padrão (foco local, sem dilatação)."""
    shortcut = x
    fx = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
    fx = BatchNormalization()(fx)
    fx = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(fx)
    fx = BatchNormalization()(fx)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)

    return Add()([shortcut, fx])

def create_model():
    """
    Constrói a rede Seq2Seq: CNN (Local) + Bi-LSTM (Médio).
    O modelo recebe janelas de DNA e prediz Íntron/Éxon para CADA nucleotídeo (N, WINDOW_SIZE, 1).
    """
    inp_dna = Input(shape=(WINDOWS_SIZE, 4), name="dna_input")

    # --- 1. Frontend de CNN (Contexto Local) ---
    x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inp_dna)
    x = BatchNormalization()(x)

    x = residual_block(x, filters=64, kernel_size=5)
    x = residual_block(x, filters=128, kernel_size=5)

    x = Dropout(0.3)(x)

    # --- 2. Bi-LSTM Seq2Seq ---
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)

    # --- 3. Classificador Final ---
    # Aplica Dense em cada step da sequência independentemente (TimeDistributed nativo)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    out_final = Dense(1, activation='sigmoid', name='final_out')(x)

    model = Model(inputs={'dna_input': inp_dna}, outputs={'final_out': out_final})

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={'final_out': BinaryFocalCrossentropy(gamma=2.0, alpha=0.50)},
        sample_weight_mode='temporal', # Crucial para ignorar os -1 (UTR)
        metrics={
            'final_out': [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        }
    )

    return model

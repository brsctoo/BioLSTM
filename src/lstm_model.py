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
    Constrói a rede híbrida Seq2Seq: CNN (Local) + Bi-LSTM (Temporal) + RF (Global fusionada no tempo).
    """
    inp_dna = Input(shape=(WINDOWS_SIZE, 4), name="dna_input")
    inp_rf = Input(shape=(1,), name="rf_input")

    # --- 1. Frontend de CNN (Contexto Local) ---
    x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inp_dna)
    x = BatchNormalization()(x)

    x = residual_block(x, filters=64, kernel_size=5)
    x = residual_block(x, filters=128, kernel_size=5)

    x = Dropout(0.3)(x)

    # --- 2. Bi-LSTM mantendo a sequência (Seq2Seq: return_sequences=True) ---
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)

    # A camada de Atenção preserva a sequência quando recebe os dois inputs iguais.
    # O holofote agora atua olhando para os pontos do tempo sem achatar a sequência inteira.
    attention_out = Attention()([lstm_out, lstm_out]) # Type: ignore
    x = attention_out

    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # --- 3.5 Saída Auxiliar da LSTM (Seq2Seq) ---
    out_lstm = Dense(1, activation='sigmoid', name='aux_lstm_out')(x)

    # --- 4. Fusão Híbrida (Late Fusion adaptado para Seq2Seq) ---
    # Precisamos "esticar" a feature única do RF (tamanho 1) para todos os timesteps (tamanho WINDOWS_SIZE)
    from tensorflow.keras.layers import RepeatVector
    rf_repeated = RepeatVector(WINDOWS_SIZE)(inp_rf)

    # Junta a inteligência temporal (LSTM Seq2Seq) com o palpite estatístico (RF repetido no tempo)
    merged = Concatenate(axis=-1)([x, rf_repeated])

    # --- 5. Classificador Final ---
    x_dense = Dense(32, activation='relu')(merged)
    x_dense = Dropout(0.3)(x_dense)
    out_final = Dense(1, activation='sigmoid', name='final_out')(x_dense)

    model = Model(
        inputs={'dna_input': inp_dna, 'rf_input': inp_rf},
        outputs={'aux_lstm_out': out_lstm, 'final_out': out_final}
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            'aux_lstm_out': BinaryFocalCrossentropy(gamma=2.0, alpha=0.25),
            'final_out': BinaryFocalCrossentropy(gamma=2.0, alpha=0.25)
        },
        loss_weights={
            'aux_lstm_out': 0.25, # Regularizador
            'final_out': 1.0 # Alvo principal
        },
        metrics={
            'final_out': [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        }
    )

    return model

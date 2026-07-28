"""
Creates the deep hybrid CNN + Stacked Bi-LSTM model for biological sequence classification.
"""

import tensorflow as tf
from keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    Bidirectional, LSTM, Dense, Dropout, Input
)
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryFocalCrossentropy

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

# Alpha adicionado! Exons (1) recebem mais "peso" de atenção da rede do que Introns (0)
LOSS_FUNCTION = BinaryFocalCrossentropy(gamma=2.0, alpha=0.75)
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.AUC(name='auc')
]

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization, Add,
    Bidirectional, LSTM, Dense, Dropout, Input, Attention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy

def residual_block(x, filters, kernel_size, dilation_rate):
    """Bloco residual com dilatação para ampliar o campo de visão da CNN."""
    shortcut = x
    fx = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', dilation_rate=dilation_rate)(x)
    fx = BatchNormalization()(fx)
    fx = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', dilation_rate=dilation_rate)(fx)
    fx = BatchNormalization()(fx)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)

    return Add()([shortcut, fx])

def create_model():
    """
    Constrói a rede híbrida: CNN Dilatada + Bi-LSTM + Mecanismo de Atenção.
    Mantém a LSTM com capacidade máxima de focar nas partes críticas da janela de 400.
    """
    inp = Input(shape=(WINDOWS_SIZE, 5))  # 4 canais One-Hot + 1 canal P(Éxon) do RF

    # --- 1. Frontend de CNN Dilatada (Contexto Amplo) ---
    x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu', dilation_rate=1)(inp)
    x = BatchNormalization()(x)

    x = residual_block(x, filters=64, kernel_size=5, dilation_rate=2)
    x = residual_block(x, filters=128, kernel_size=5, dilation_rate=4)

    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # --- 2. Bi-LSTM mantendo a sequência (return_sequences=True) ---
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)

    # --- 3. Camada de Atenção (O "Holofote") ---
    attention_out = Attention()([lstm_out, lstm_out]) #Type: ignore

    x = GlobalAveragePooling1D()(attention_out) #Type: ignore
    x = Dropout(0.4)(x)

    # --- 4. Classificador Final ---
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=BinaryFocalCrossentropy(gamma=2.0, alpha=0.75),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

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
LEARNING_RATE = 5e-4
VALIDATION_SPLIT = 0.2
WINDOWS_SIZE = 120

LOSS_FUNCTION = BinaryFocalCrossentropy(gamma=2.0)
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

def create_model():
    """
    Constrói uma arquitetura híbrida otimizada (CNN + Deep Stacked Bi-LSTM) para sequências biológicas.
    """
    inp = Input(shape=(WINDOWS_SIZE, 4))

    # --- Block 0: very short motifs (codon start, GT-AG variations) ---
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)

    # --- Block 1: short motifs (codons, GT-AG) ---
    x = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # --- Block 2: medium motifs (polypyrimidine tract, branch point) ---
    x = Conv1D(filters=128, kernel_size=16, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # --- Deep Bi-LSTM: Stacked layers for long directional context ---
    # 1ª Camada: Retorna sequências para alimentar a próxima camada recorrente
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)

    # 2ª Camada: Processa o contexto aprofundado e compacta a saída
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(x)
    x = Dropout(0.4)(x)

    # --- Classifier ---
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )

    return model

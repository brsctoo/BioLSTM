"""
The script creates the LSTM model for sequence classification.

It will use expecific parameters for the model architecture.
"""
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Conv1D, MaxPooling1D, BatchNormalization,
    Bidirectional, LSTM, Dense, Dropout, Input
)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import BinaryFocalCrossentropy  # trocado


# Define model parameters
SEQUENCE_LENGTH = 60
NUM_DISTINCT_WORDS = 5
EMBEDDING_DIM = 60

EPOCHS = 10
BATCH_SIZE = 100
LSTM_UNITS = 60
LEARNING_RATE = 5e-4
VALIDATION_SPLIT = 0.2
WINDOWS_SIZE = 180

LOSS_FUNCTION = BinaryFocalCrossentropy(gamma=2.0)  # trocado
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

"""
def create_model():
    lstm_model = Sequential([
        keras.Input(shape=(60,4)), # 60 linhas (tamanho da sequência) e 4 colunas (one-hot encoding de A,T,G,C)
        Bidirectional(LSTM(LSTM_UNITS)),

        Dense(32, activation="relu"),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    lstm_model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )

    return lstm_model
"""

def create_model():
    inp = Input(shape=(WINDOWS_SIZE, 4))

    # --- Bloco 0: motivos muito curtos (início de códon, variações GT-AG) ---
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)

    # --- Bloco 1: motivos curtos (codons, GT-AG) ---
    x = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # --- Bloco 2: motivos médios (polypyrimidine tract, branch point) ---
    x = Conv1D(filters=128, kernel_size=16, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # --- BiLSTM: contexto direcional longo ---
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)

    # --- Classificador ---
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=LOSS_FUNCTION,  # agora usa a variável, consistente com o resto
        metrics=METRICS
    )

    return model

lstm_model = create_model()

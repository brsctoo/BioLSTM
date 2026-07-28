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
WINDOWS_SIZE = 200

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
    Bidirectional, LSTM, Dense, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy

# ... (suas constantes e métricas continuam iguais) ...

def residual_block(x, filters, kernel_size, dilation_rate):
    """
    Cria um mini bloco residual com dilatação (Inpiração direta da ideia do Claude).
    Ajuda o gradiente a fluir melhor e captura contextos complexos.
    """
    shortcut = x

    # Primeira convolução dilatada
    fx = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', dilation_rate=dilation_rate)(x)
    fx = BatchNormalization()(fx)

    # Segunda convolução dilatada
    fx = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='activation', dilation_rate=dilation_rate)(fx) # Ajustado para relu abaixo se necessário, ou mantido
    # Corrigindo a ativação para relu padrão:
    fx = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', dilation_rate=dilation_rate)(fx)
    fx = BatchNormalization()(fx)

    # Se o número de canais da entrada for diferente dos filtros, ajusta o shortcut
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters=filters, kernel_size=1, padding='same')(shortcut)

    # Soma a entrada original à saída (ResNet shortcut)
    out = Add()([shortcut, fx])
    return out

def create_model():
    """
    Constrói o modelo híbrido atualizado com Convoluções Dilatadas e Blocos Residuais.
    """
    inp = Input(shape=(WINDOWS_SIZE, 4))

    # --- Stem (Extração inicial de motifs locais GT-AG) ---
    x = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu', dilation_rate=1)(inp)
    x = BatchNormalization()(x)

    # --- Bloco Residual com Contexto Médio (Dilatação d=2) ---
    x = residual_block(x, filters=64, kernel_size=5, dilation_rate=2)

    # --- Bloco Residual com Contexto Amplo (Dilatação d=4) ---
    x = residual_block(x, filters=128, kernel_size=5, dilation_rate=4)

    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # --- Deep Bi-LSTM (Processa a sequência longa fornecida pelas CNNs dilatadas) ---
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(32, return_sequences=False, dropout=0.3))(x)
    x = Dropout(0.5)(x)

    # --- Classificador Global da Janela ---
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )

    return model

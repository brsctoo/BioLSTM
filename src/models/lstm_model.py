import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Attention,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define model parameters
SEQUENCE_LENGTH = 60
NUM_DISTINCT_WORDS = 5
EMBEDDING_DIM = 60

EPOCHS = 10
BATCH_SIZE = 100
LSTM_UNITS = 60

# Reduced to 1e-4 to prevent the model from memorizing the training data too quickly
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
WINDOWS_SIZE = 400  # overridden by pipeline.py via set_window_size()

def set_window_size(
    size: int
) -> None:
    """
    Called by pipeline.py to propagate --window-size into this module.
    """
    global WINDOWS_SIZE
    WINDOWS_SIZE = size

def extract_6mers(
    x_tensor: object
) -> object:
    import tensorflow as tf
    # Transforms One-Hot (batch, W, 4) into Indices (batch, W)
    indices = tf.argmax(x_tensor, axis=-1, output_type=tf.int32) # type: ignore

    # Pad to keep the exact WINDOW_SIZE length.
    # To read 6 letters, we place a margin of 2 on the left and 3 on the right.
    paddings = tf.constant([[0, 0], [2, 3]])
    padded = tf.pad(indices, paddings, mode='CONSTANT', constant_values=0)

    # Extract the 6 nucleotides from the window
    pos1 = padded[:, :-5]
    pos2 = padded[:, 1:-4]
    pos3 = padded[:, 2:-3]
    pos4 = padded[:, 3:-2]
    pos5 = padded[:, 4:-1]
    pos6 = padded[:, 5:]

    # Calculate Hexamer ID (4^6 = 4096 possible hexamers)
    kmer_id = pos1 * 1024 + pos2 * 256 + pos3 * 64 + pos4 * 16 + pos5 * 4 + pos6
    return kmer_id


def _hexamer_frontend(inp_dna: object) -> object:
    from tensorflow.keras.layers import Embedding, Lambda  # type: ignore
    kmers = Lambda(extract_6mers)(inp_dna)
    x = Embedding(input_dim=4096, output_dim=64, name="hexamer_embedding")(kmers)
    return Dropout(0.2)(x)


def _conv_frontend(inp_dna: object) -> object:
    from tensorflow.keras.layers import Conv1D
    x = Conv1D(filters=64, kernel_size=6, padding="same", activation="relu",
               name="motif_conv")(inp_dna)
    x = BatchNormalization()(x)
    return Dropout(0.2)(x)


_FRONTENDS = {"hexamer": _hexamer_frontend, "conv": _conv_frontend}


def create_model(frontend: str = "hexamer") -> object:
    """
    Seq2Seq: [frontend] (Local) + Bi-LSTM (Temporal) + RF (Global fusionada no tempo).
    frontend: "hexamer" ou "conv".
    """
    inp_dna = Input(shape=(WINDOWS_SIZE, 4), name="dna_input")
    inp_rf = Input(shape=(1,), name="rf_input")

    # --- 1. Frontend de motivo local ---
    x = _FRONTENDS[frontend](inp_dna)

    # --- 2. Deep Bi-LSTM mantendo a sequência (Seq2Seq: return_sequences=True) ---
    # Primeira camada: atua como extratora de motivos locais (substituindo a CNN)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    # Segunda camada: integra o contexto global ao longo dos 200bp
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

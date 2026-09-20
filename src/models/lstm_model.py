import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Attention,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Multiply,
    RepeatVector,
)
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 100
WINDOWS_SIZE = 400  # overridden by pipeline.py via set_window_size()


def set_window_size(
    size: int
) -> None:
    """
    Called by pipeline.py to propagate --window-size into this module.
    """
    global WINDOWS_SIZE
    WINDOWS_SIZE = size


def create_model() -> object:
    inp_dna = Input(shape=(WINDOWS_SIZE, 4), name="dna_input")
    inp_rf = Input(shape=(1,), name="rf_input")

    # --- 1. Frontend de motivo local ---
    x = Conv1D(filters=64, kernel_size=6, padding="same", activation="relu", name="motif_conv")(inp_dna)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # --- 2. Deep Bi-LSTM mantendo a sequência (Seq2Seq: return_sequences=True) ---
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)

    attention_out = Attention()([lstm_out, lstm_out]) # Type: ignore
    x = attention_out

    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    out_lstm = Dense(1, activation='sigmoid', name='aux_lstm_out')(x)

    # --- 3. Fusão com gate aprendido ---
    rf_repeated = RepeatVector(WINDOWS_SIZE)(inp_rf)
    gate_input = Concatenate(axis=-1)([x, rf_repeated])
    gate = Dense(1, activation='sigmoid', name='rf_trust_gate')(gate_input)
    rf_weighted = Multiply(name='rf_weighted')([rf_repeated, gate])
    merged = Concatenate(axis=-1)([x, rf_weighted])

    # --- 4. Classificador Final ---
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
            'aux_lstm_out': 0.25,
            'final_out': 1.0
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

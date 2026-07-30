"""
Validation module to assess the trained Bi-LSTM model performance
on the test dataset using alignment metrics and positional metrics.
"""

import rf_model as rf_module
import modeling
import numpy as np
import pickle
import keras
from minineedle import needle # type: ignore

def smooth_predict(predict_raw, window_size=20):
    half = window_size // 2
    smoothed = []

    for i in range(len(predict_raw)):
        start = max(0, i - half)
        end = min(len(predict_raw), i + half + 1)

        window = predict_raw[start:end]

        if sum(window) >= len(window) / 2:
            smoothed.append(1)
        else:
            smoothed.append(0)

    return smoothed

def print_positional_prediction_ratio(sample_index, y_true, y_pred_raw, y_pred_smooth):
    """
    EXPLICIT DEBUG:
    Shows the percentage of positions marked as exon (class 1)
    to compare true label vs raw prediction vs smoothed prediction.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_raw_arr = np.asarray(y_pred_raw).reshape(-1)
    y_pred_smooth_arr = np.asarray(y_pred_smooth).reshape(-1)

    real_exon_pct = 100.0 * np.mean(y_true_arr == 1)
    pred_raw_exon_pct = 100.0 * np.mean(y_pred_raw_arr == 1)
    pred_smooth_exon_pct = 100.0 * np.mean(y_pred_smooth_arr == 1)

    print("----- DEBUG % PER POSITION (EXON=1) -----")
    print(f"Sample {sample_index}:")
    print(f"  Real (Y==1)             : {real_exon_pct:.2f}%")
    print(f"  Predicted raw (raw==1)  : {pred_raw_exon_pct:.2f}%")
    print(f"  Predicted smooth (==1) : {pred_smooth_exon_pct:.2f}%")
    print("----------------------------------------")

def validate_model(model_path, data_test, rf=None, threshold=0.50):
    """
    Valida o modelo Bi-LSTM no conjunto de teste.

    Parâmetros
    ----------
    model_path : str
        Caminho do modelo .h5 treinado.
    data_test : str
        Caminho do arquivo .mod1 de teste.
    rf : RandomForestClassifier | None
        Se fornecido, injeta P(Éxon) como 5º canal antes de cada predict,
        igualando o shape ao que o modelo foi treinado (400, 5).
        Se None, assume modelo antigo com entrada (400, 4).
    """
    class SafeAttention(keras.layers.Attention):
        def __init__(self, **kwargs):
            if 'score_mode' in kwargs and callable(kwargs['score_mode']):
                kwargs['score_mode'] = 'dot'
            super().__init__(**kwargs)

    try:
        keras.config.enable_unsafe_deserialization()
    except AttributeError:
        pass

    loaded_model = keras.models.load_model(model_path, custom_objects={'Attention': SafeAttention})

    # Assert model is not None to resolve Pyright's attribute inference warning
    if loaded_model is None:
        raise ValueError(f"Failed to load Keras model from path: {model_path}")

    final_metrics = []

    raw_test_data = pickle.load(open(data_test, "rb"))

    data_test = []
    for sample in raw_test_data:
        # If it does NOT contain the blind gap, add it to the official test list
        if 'NNNNNNNNNN' not in sample["sequence"].upper():
            data_test.append(sample)

    print("-" * 50)
    print(f"Total samples in file      : {len(raw_test_data)}")
    print(f"Clean samples for testing  : {len(data_test)}")
    print(f"Discarded (with NNNNNNNNN) : {len(raw_test_data) - len(data_test)}")
    print("-" * 50)

    all_y_true, all_y_pred, all_y_prob = [], [], []

    count = 0
    for sample in data_test:
        count += 1
        print("Sequence:", count, "Progress:", 100 * count / len(data_test), "%")

        tagged_sequence = modeling.tag_positions(sample) # returns something like: [0,0,1,0,0...]

        windows = modeling.slide_window(sample)  # usa modeling.WINDOW_SIZE setado pelo pipeline

        X = []
        Y = []

        for j in range(len(windows)):
            X.append(windows[j])
            Y.append(tagged_sequence[j])

        X = np.array(X)  # (N, W, 4)
        Y = np.array(Y)

        # --- VERIFICAÇÃO DE ARQUITETURA ---
        # Checa se o modelo usa Late Fusion (tem múltiplas entradas)
        if hasattr(loaded_model, 'inputs') and len(loaded_model.inputs) >= 2:
            X_dna = X
            if rf is not None:
                # Reaproveita a função de injeção, mas pega só a coluna de probabilidade
                X_aug = rf_module.inject_rf_proba(rf, X)  # (N, W, 5)
                X_rf = X_aug[:, 0, 4:]                    # (N, 1)
            else:
                X_rf = np.zeros((X.shape[0], 1))
            
            X_inputs = {'dna_input': X_dna, 'rf_input': X_rf}
            
            if len(loaded_model.inputs) == 3:
                pos = np.arange(len(windows), dtype=np.float32) / max(1, len(sample["sequence"]) - 1)
                pos = np.expand_dims(pos, axis=-1)
                X_inputs['pos_input'] = pos
        
        else:
            # Modelo antigo (Early Fusion ou Baseline sem RF)
            expected_channels = loaded_model.input_shape[-1]
            if rf is not None and expected_channels == 5:
                X_inputs = rf_module.inject_rf_proba(rf, X)  # (N, W, 5)
            elif expected_channels == 4:
                X_inputs = X                                 # (N, W, 4)
            else:
                raise ValueError(f"Modelo antigo espera {expected_channels} canais (não suportado).")

        predictions = loaded_model.predict(X_inputs)
        if isinstance(predictions, list):
            predictions = predictions[-1] # Pega o final_out
        elif isinstance(predictions, dict):
            predictions = predictions['final_out']
            
        # Se o modelo for Seq2Seq (3D output: N, WINDOW_SIZE, 1),
        # pegamos apenas a predição central (nucleotídeo alvo).
        if predictions.ndim == 3:
            mid = predictions.shape[1] // 2
            predictions = predictions[:, mid, :]
            
        THRESHOLD     = threshold   
        SMOOTH_WINDOW = 20

        prob = np.asarray(predictions).flatten()             
        predict_raw = (prob > THRESHOLD).astype("int32")
        predict_smoothed = np.array(smooth_predict(predict_raw, window_size=SMOOTH_WINDOW))

        # ===== REQUESTED DEBUG BLOCK: % per position =====
        print_positional_prediction_ratio(count, Y, predict_raw, predict_smoothed)

        mask = Y >= 0                                      
        all_y_true.extend(Y[mask].tolist())
        all_y_pred.extend(predict_smoothed[mask].tolist())
        all_y_prob.extend(prob[mask].tolist())                 

        # Final sequence of predicted exons and introns
        final_seq = [sample["sequence"][i] for i in range(len(predict_smoothed))
                     if mask[i] and predict_smoothed[i] == 1]

        # Sequence of true exons and introns
        true_final_seq = [sample["sequence"][i] for i in range(len(Y))
                          if mask[i] and Y[i] == 1]

        if len(final_seq) != 0 and len(true_final_seq) != 0:
            alignment = needle.NeedlemanWunsch(final_seq, true_final_seq)
            alignment.align()

            # Get the aligned sequences (with inserted gaps)
            seq1_aligned, seq2_aligned = alignment.get_aligned_sequences()

            # Calculate Identity (Matches / Total Alignment Length)
            matches = sum(1 for a, b in zip(seq1_aligned, seq2_aligned) if a == b)
            identity = matches / len(seq1_aligned) # Returns a clean value between 0.0 and 1.0

            print(f"Alignment Identity: {identity:.4f}")
            final_metrics.append([identity])

    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    total = 0
    for metric in final_metrics:
        # Removed abs(). The value is now the real Identity (e.g., 0.85 for an 85% match)
        identity_value = metric[0]
        total += identity_value

        # Splitting intervals based on alignment scores
        if identity_value <= 0.1:
            q0 += 1
        elif identity_value <= 0.2:
            q1 += 1
        elif identity_value <= 0.3:
            q2 += 1
        elif identity_value <= 0.4:
            q3 += 1
        elif identity_value <= 0.5:
            q4 += 1
        elif identity_value <= 0.6:
            q5 += 1
        elif identity_value <= 0.7:
            q6 += 1
        elif identity_value <= 0.8:
            q7 += 1
        elif identity_value <= 0.9:
            q8 += 1
        else:
            q9 += 1

    mean_identity = total / len(final_metrics)

    print("\n--- GENERAL RESULTS ---")
    print("Total samples: ", len(final_metrics))
    print("MEAN ALIGNMENT IDENTITY: {:.2f}%".format(mean_identity * 100))
    print(" 0-10%:   ", q0)
    print(" 10-20%:  ", q1)
    print(" 20-30%:  ", q2)
    print(" 30-40%:  ", q3)
    print(" 40-50%:  ", q4)
    print(" 50-60%:  ", q5)
    print(" 60-70%:  ", q6)
    print(" 70-80%:  ", q7)
    print(" 80-90%:  ", q8)
    print(" 90-100%: ", q9)
    print("\n")

    # ── Positional Metric via Keras ───────────────────────────────
    y_true_t = np.array(all_y_true, dtype=np.float32)
    y_pred_t = np.array(all_y_pred, dtype=np.float32)
    y_prob_t = np.array(all_y_prob, dtype=np.float32)

    auc_metric = keras.metrics.AUC()
    auc_metric.update_state(y_true_t, y_prob_t)

    keras_metrics = {
        "Accuracy" : keras.metrics.BinaryAccuracy(),
        "Precision": keras.metrics.Precision(),
        "Recall"   : keras.metrics.Recall(),
        "AUC"      : auc_metric,
        "TP"       : keras.metrics.TruePositives(),
        "TN"       : keras.metrics.TrueNegatives(),
        "FP"       : keras.metrics.FalsePositives(),
        "FN"       : keras.metrics.FalseNegatives(),
    }

    for name, metric in keras_metrics.items():
        if name != "AUC":
            metric.update_state(y_true_t, y_pred_t)

    tp = keras_metrics["TP"].result().numpy()
    fp = keras_metrics["FP"].result().numpy()
    fn = keras_metrics["FN"].result().numpy()

    print("═" * 50)
    print("POSITIONAL METRIC (splice site boundaries)")
    print("═" * 50)
    for name, metric in keras_metrics.items():
        value = metric.result().numpy()
        suffix = "%" if name not in ("TP", "TN", "FP", "FN") else ""
        factor = 100 if name not in ("TP", "TN", "FP", "FN") else 1
        print(f"  {name:10s}: {value*factor:.2f}{suffix}")

    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f"  {'F1':10s}: {f1*100:.2f}%")

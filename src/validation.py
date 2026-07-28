"""
Validation module to assess the trained Bi-LSTM model performance
on the test dataset using alignment metrics and positional metrics.
"""

import modeling
import numpy as np
import pickle
import keras
from minineedle import needle

def smooth_predict(predict_raw, window_size=6):
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

def validate_model(model_path, data_test):
    loaded_model = keras.models.load_model(model_path)

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

    all_y_true = []
    all_y_pred = []

    count = 0
    for sample in data_test:
        count += 1
        print("Sequence:", count, "Progress:", 100 * count / len(data_test), "%")

        tagged_sequence = modeling.tag_positions(sample) # returns something like: [0,0,1,0,0...]

        windows = modeling.slide_window(sample, window_size=400) # sliding window -> list of windows

        X = []
        Y = []

        for j in range(len(windows)):
            X.append(windows[j])
            Y.append(tagged_sequence[j])

        X = np.array(X)
        Y = np.array(Y)

        predict_raw = (loaded_model.predict(X) > 0.5).astype("int32") # type: ignore
        predict_smoothed = smooth_predict(predict_raw, window_size=6)

        # ===== REQUESTED DEBUG BLOCK: % per position =====
        print_positional_prediction_ratio(count, Y, predict_raw, predict_smoothed)

        # Appending data for global verification metrics
        all_y_true.extend(Y.tolist())
        all_y_pred.extend(predict_smoothed)

        # Final sequence of predicted exons and introns
        final_seq = []
        for i in range(len(predict_smoothed)):
            if predict_smoothed[i] == 1:
                final_seq.append(sample["sequence"][i])

        # Sequence of true exons and introns
        true_final_seq = []
        for i in range(len(Y)):
            if Y[i] == 1:
                true_final_seq.append(sample["sequence"][i])

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

            # --- ERROR INSPECTION X-RAY ---
            # If identity is below 60%, print visual x-ray details
            # if identity < 0.60:
                # print(f"\n[ERROR ALERT] Investigating failure (Identity: {identity*100:.1f}%)")
                # print("Original Sequence : ", sample["sequence"])
                # print("Predicted Sequence: ", "".join(final_seq))
                # print("Real Sequence     : ", "".join(true_final_seq))
            # --- END OF X-RAY ---

    # print(final_metrics)

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

    keras_metrics = {
        "Accuracy" : keras.metrics.BinaryAccuracy(),
        "Precision": keras.metrics.Precision(),
        "Recall"   : keras.metrics.Recall(),
        "AUC"      : keras.metrics.AUC(),
        "TP"       : keras.metrics.TruePositives(),
        "TN"       : keras.metrics.TrueNegatives(),
        "FP"       : keras.metrics.FalsePositives(),
        "FN"       : keras.metrics.FalseNegatives(),
    }

    for name, metric in keras_metrics.items():
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

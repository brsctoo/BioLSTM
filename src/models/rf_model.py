
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Feature schema version. Saved along with the model.
FEATURE_SCHEMA_VERSION = 2

# =============================================================================
# 1. FEATURE ENGINEERING
# =============================================================================

def _valid_mask(
    one_hot: NDArray
) -> NDArray:
    """
    Real positions vs padding. `extract_windows_numpy` pads the sequence
    """
    return one_hot.sum(axis=2) > 1e-6            # (B, W)


def compute_kmer_frequencies(
    one_hot: NDArray,
    k: int = 3
) -> NDArray:
    """
    Normalized frequencies of the 4^k k-mers per window.
    """
    n_kmers = 4 ** k
    W = one_hot.shape[1]
    if W < k:
        return np.zeros((one_hot.shape[0], n_kmers), dtype=np.float32)

    seq_indices = np.argmax(one_hot, axis=2)               # (B, W)
    valid = _valid_mask(one_hot)                           # (B, W)

    kmer_windows = sliding_window_view(seq_indices, window_shape=k, axis=1)
    valid_windows = sliding_window_view(valid, window_shape=k, axis=1)
    kmer_ok = valid_windows.all(axis=2)                    # (B, n_pos) without padding

    powers = 4 ** np.arange(k - 1, -1, -1)
    kmer_indices = (kmer_windows * powers).sum(axis=2)     # (B, n_pos)

    onehot_k = (kmer_indices[:, :, None] == np.arange(n_kmers))
    onehot_k = onehot_k & kmer_ok[:, :, None]              # discards padding
    counts = onehot_k.sum(axis=1).astype(np.float32)       # (B, n_kmers)

    total = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, total, out=np.zeros_like(counts),
                     where=total > 0)


def compute_max_orf_length(
    one_hot: NDArray
) -> NDArray:
    """
    Longest continuous stretch without a stop codon (TAA, TAG, TGA), evaluating all 3
    """
    W = one_hot.shape[1]
    B = one_hot.shape[0]
    if W < 3:
        return np.zeros((B, 1), dtype=np.float32)

    seq = np.argmax(one_hot, axis=-1)
    valid = _valid_mask(one_hot)

    kmers = sliding_window_view(seq, window_shape=3, axis=1)          # (B, W-2, 3)
    kvalid = sliding_window_view(valid, window_shape=3, axis=1).all(axis=2)

    # A=0, T=1, G=2, C=3  ->  TAA=(1,0,0)  TAG=(1,0,2)  TGA=(1,2,0)
    c0, c1, c2 = kmers[:, :, 0], kmers[:, :, 1], kmers[:, :, 2]
    is_taa = (c0 == 1) & (c1 == 0) & (c2 == 0)
    is_tag = (c0 == 1) & (c1 == 0) & (c2 == 2)
    is_tga = (c0 == 1) & (c1 == 2) & (c2 == 0)
    is_stop = is_taa | is_tag | is_tga

    # padding also interrupts the ORF
    is_break = is_stop | (~kvalid)

    max_orf = np.zeros(B, dtype=np.int32)
    for frame in range(3):
        breaks_f = is_break[:, frame::3]
        cur = np.zeros(B, dtype=np.int32)
        best = np.zeros(B, dtype=np.int32)
        for i in range(breaks_f.shape[1]):
            cur = (cur + 1) * (~breaks_f[:, i])
            best = np.maximum(best, cur)
        max_orf = np.maximum(max_orf, best)

    return (max_orf * 3).astype(np.float32).reshape(-1, 1)             # (B, 1)


def compute_fourier_period_3(
    one_hot: NDArray
) -> NDArray:
    """
    Voss method (1992): FFT over the 4 binary channels to detect the
    """
    W = one_hot.shape[1]
    if W < 6:
        return np.zeros((one_hot.shape[0], 1), dtype=np.float32)

    X = np.fft.fft(one_hot, axis=1)
    S = np.sum(np.abs(X) ** 2, axis=2)            # (B, W)

    idx = round(W / 3.0)
    idx = min(max(idx, 1), W - 1)

    s_peak = S[:, idx]
    s_mean = np.mean(S[:, 1:], axis=1)            # <- without DC
    score = s_peak / (s_mean + 1e-8)
    return score.astype(np.float32).reshape(-1, 1)



def _single_scale(
    one_hot: NDArray,
    k: int,
    use_features: dict[str, bool] | None = None
) -> NDArray:
    """
    Feature block for ONE window.
    """
    if use_features is None:
        use_features = {"kmer": True, "orf": True, "fourier": True}

    blocks = []
    if use_features.get("kmer", True):
        blocks.append(compute_kmer_frequencies(one_hot, k=k))
    if use_features.get("orf", True):
        blocks.append(compute_max_orf_length(one_hot))
    if use_features.get("fourier", True):
        blocks.append(compute_fourier_period_3(one_hot))

    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_feature_matrix(
    one_hot: NDArray,
    k: int = 3,
    use_features: dict[str, bool] | None = None
) -> NDArray:
    """
    Converts the 3D One-Hot tensor into a 2D tabular matrix for the Random Forest.
    """
    return _single_scale(one_hot, k, use_features)

# =============================================================================
# 2. TRAINING
# =============================================================================

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    k: int = 3,
    use_features: dict[str, bool] | None = None,
) -> RandomForestClassifier:
    """
    Trains the RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,          # required for injection without leakage
        random_state=random_state,
        n_jobs=-1,
    )

    print(f"  [RF] Training RandomForest ({n_estimators} trees, "
          f"max_depth={max_depth}, features={X_train.shape[1]})...")
    model.fit(X_train, y_train)
    print(f"  [RF] Training complete. OOB Score: {model.oob_score_:.4f}")

    # --- feature schema metadata ---
    model.feature_k = k                                    # type: ignore[attr-defined]
    model.feature_schema_version = FEATURE_SCHEMA_VERSION   # type: ignore[attr-defined]
    if use_features is None:
        use_features = {"kmer": True, "orf": True, "fourier": True}
    model.use_features = use_features                      # type: ignore[attr-defined]

    return model


def get_feature_config(
    rf: RandomForestClassifier
) -> tuple[int, dict[str, bool]]:
    """
    Retrieves (k, use_features) from the model.
    """
    default_features = {"kmer": True, "orf": True, "fourier": True}
    if getattr(rf, "feature_schema_version", None) == FEATURE_SCHEMA_VERSION:
        use_features = getattr(rf, "use_features", default_features)
        return rf.feature_k, use_features # type: ignore

    n = getattr(rf, "n_features_in_", 17)
    k = 3 if n >= 65 else 2
    print(f"  [RF] Warning: model without schema metadata ({n} columns). "
          f"Assuming legacy k={k}.")
    return k, default_features


# =============================================================================
# 3. EVALUATION AND DIAGNOSTICS
# =============================================================================

def evaluate_rf(
    model: RandomForestClassifier,
    X_val: NDArray,
    y_val: NDArray,
    *,
    verbose: bool = True
) -> dict[str, object]:
    """
    Evaluates the model on the validation set.
    """
    y_proba = model.predict_proba(X_val)[:, 1] # type: ignore[arg-type]

    # Threshold tuning
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = 0.0
    for t in thresholds:
        pred_t = (y_proba >= t).astype(int)
        f1_t = f1_score(y_val, pred_t, average="macro", zero_division=0) # type: ignore[arg-type]
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t

    model.best_threshold_ = best_thresh # type: ignore[arg-type]
    y_pred = (y_proba >= best_thresh).astype(int)

    acc       = accuracy_score(y_val, y_pred)
    f1_exon   = f1_score(y_val, y_pred, pos_label=1, zero_division=0) # type: ignore[arg-type]
    f1_intron = f1_score(y_val, y_pred, pos_label=0, zero_division=0) # type: ignore[arg-type]
    f1_macro  = f1_score(y_val, y_pred, average="macro", zero_division=0) # type: ignore[arg-type]
    cm        = confusion_matrix(y_val, y_pred)
    report    = classification_report(
        y_val, y_pred,
        target_names=["Intron (0)", "Exon (1)"],
        zero_division=0, # type: ignore
    )

    # trivial baseline: always predict the majority class
    majority = round(float(np.mean(y_val)))
    triv = np.full_like(y_val, majority)
    triv_acc = accuracy_score(y_val, triv)
    triv_f1m = f1_score(y_val, triv, average="macro", zero_division=0) # type: ignore[arg-type]

    if verbose:
        _print_rf_results(acc, f1_exon, f1_intron, f1_macro, cm, report, # type: ignore[arg-type]
                          triv_acc, triv_f1m, majority, best_thresh)

    return {
        "accuracy": acc,
        "f1_exon": f1_exon,
        "f1_intron": f1_intron,
        "f1_macro": f1_macro,
        "cm": cm,
        "report": report,
        "trivial_accuracy": triv_acc,
        "trivial_f1_macro": triv_f1m,
        "best_threshold": best_thresh,
    }


def _print_rf_results(
    acc: float,
    f1_exon: float,
    f1_intron: float,
    f1_macro: float,
    cm: NDArray,
    report: str,
    triv_acc: float,
    triv_f1m: float,
    majority: int,
    best_thresh: float = 0.5
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  RANDOM FOREST — VALIDATION RESULTS")
    print(sep)
    print(f"  {'Optimized Threshold':<25}: {best_thresh:.3f}")
    print(f"  {'Accuracy':<25}: {acc * 100:.2f}%")
    print(f"  {'F1-Score  Exon  (1)':<25}: {f1_exon * 100:.2f}%")
    print(f"  {'F1-Score  Intron (0)':<25}: {f1_intron * 100:.2f}%")
    print(f"  {'F1-Score  Macro':<25}: {f1_macro * 100:.2f}%")
    print("\n  -- mandatory reference --")
    print(f"  {f'baseline always {majority}':<25}: acc {triv_acc * 100:.2f}%  "
          f"F1_macro {triv_f1m * 100:.2f}%")
    print("\n  Confusion Matrix:")
    print("          Pred 0   Pred 1")
    print(f"  True 0  {cm[0, 0]:>6}   {cm[0, 1]:>6}")
    print(f"  True 1  {cm[1, 0]:>6}   {cm[1, 1]:>6}")
    print(f"\n  Full Report:\n{report}")
    print(sep)


def evaluate_rf_microscope(
    model: RandomForestClassifier,
    X_val: NDArray,
    y_val_center: NDArray,
    y_val_window: NDArray | None = None,
    window_size: int = 120
) -> None:
    """
    Diagnostics by window type. This report reveals when the RF performs well on
    """
    print("\n" + "=" * 60)
    print("  LOCAL MICROSCOPE — WINDOW TYPE ANALYSIS")
    print("=" * 60)

    if y_val_window is None:
        print("  [Warning] 'y_window' missing in .npz. Recreate the data to see "
              "the pure/mixed window analysis.")
        return

    y_proba = model.predict_proba(X_val)[:, 1] # type: ignore
    best_thresh = getattr(model, "best_threshold_", 0.5)
    y_pred = (y_proba >= best_thresh).astype(int)

    idx_pure_intron = np.all(y_val_window == 0, axis=1)
    idx_pure_exon   = np.all(y_val_window == 1, axis=1)
    idx_mixed       = ~(idx_pure_intron | idx_pure_exon)

    def bucket(
        name: str,
        mask: NDArray
    ) -> None:
        n = int(np.sum(mask))
        if n == 0:
            print(f"  {name:<24}: 0 samples.")
            return
        acc = accuracy_score(y_val_center[mask], y_pred[mask])
        pct = 100.0 * n / len(mask)
        print(f"  {name:<24}: {acc * 100:>6.2f}% accuracy  "
              f"({n} windows, {pct:.0f}% of total)")

    bucket("100% Intron Windows", idx_pure_intron)
    bucket("100% Exon Windows", idx_pure_exon)
    bucket("Mixed Windows", idx_mixed)
    print("\n  If 'Mixed' is MUCH lower than the pure ones, the RF probably")
    print("  didn't see mixed composition during training.")
    print("=" * 60)


# =============================================================================
# 4. RF PROBABILITY FOR THE NETWORK
# =============================================================================

def rf_proba_oob(
    rf: RandomForestClassifier
) -> NDArray:
    """
    Out-of-Bag P(Exon) for the set on which the RF was trained.
    """
    if not hasattr(rf, "oob_decision_function_"):
        raise AttributeError(
            "RF trained without oob_score=True; no OOB estimate available."
        )
    oob = np.asarray(rf.oob_decision_function_, dtype=np.float64)
    p = oob[:, 1]
    return np.nan_to_num(p, nan=0.5).astype(np.float32)


def rf_proba(
    rf: RandomForestClassifier,
    one_hot: NDArray
) -> NDArray:
    """
    P(Exon) per window, for data NOT seen by the RF (validation/test).
    """
    k, use_features = get_feature_config(rf)
    X_tab = build_feature_matrix(one_hot, k=k, use_features=use_features)

    expected = getattr(rf, "n_features_in_", X_tab.shape[1])
    if X_tab.shape[1] != expected:
        raise ValueError(
            f"Incompatible feature schema: generated {X_tab.shape[1]} columns, "
            f"model expects {expected}. Retrain the RF (save_rf saves the "
            f"schema alongside starting from this version)."
        )

    return np.asarray(rf.predict_proba(X_tab))[:, 1].astype(np.float32)


def apply_rf_dropout(
    p_exon: NDArray,
    dropout_rate: float = 0.5,
    neutral: float = 0.5,
    rng: object = None
) -> NDArray:
    """
    Randomly zeros the RF signal, bringing it to the neutral value, so the
    """
    rng = rng or np.random
    keep = rng.binomial(1, 1.0 - dropout_rate, size=p_exon.shape) # type: ignore[arg-type]
    return np.where(keep == 1, p_exon, neutral).astype(np.float32)


def inject_rf_proba(
    rf: RandomForestClassifier,
    one_hot: NDArray,
    rf_scale: float = 0.20,
    is_training_set: bool = False,
    apply_dropout: bool = False,
    dropout_rate: float = 0.5,
    oob_proba: NDArray | None = None
) -> NDArray:
    """
    Augmented tensor (B, W, 5) with P(Exon) replicated in the 5th channel.
    """
    batch_size, window_size, _ = one_hot.shape

    if is_training_set:
        if oob_proba is None:
            raise ValueError(
                "is_training_set=True requires oob_proba=rf_proba_oob(rf), "
                "aligned row by row with one_hot."
            )
        p_exon = np.asarray(oob_proba, dtype=np.float32)
        if len(p_exon) != batch_size:
            raise ValueError(
                f"oob_proba has {len(p_exon)} rows, one_hot has {batch_size}."
            )
    else:
        p_exon = rf_proba(rf, one_hot)

    if apply_dropout:
        p_exon = apply_rf_dropout(p_exon, dropout_rate)

    p_channel = np.broadcast_to(
        (p_exon * rf_scale)[:, None, None],
        (batch_size, window_size, 1),
    ).astype(np.float32)

    return np.concatenate([one_hot.astype(np.float32), p_channel], axis=2)


# =============================================================================
# 5. PERSISTENCE
# =============================================================================

def save_rf(
    rf: RandomForestClassifier,
    path: str
) -> None:
    import joblib
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(rf, path)
    k, _ = get_feature_config(rf)
    print(f"  [RF] Model saved to: {path}  (k={k})")


def load_rf(
    path: str
) -> RandomForestClassifier | None:
    import joblib
    if not os.path.exists(path):
        print(f"  [RF] No RF model at: {path} "
              f"(validation will proceed without injection)")
        return None
    rf = joblib.load(path)
    k, _ = get_feature_config(rf)
    print(f"  [RF] Model loaded from: {path}  (k={k})")
    return rf


# =============================================================================
# 6. HIGH-LEVEL PIPELINE
# =============================================================================

def run_rf_pipeline(
    mod2_train_path: str,
    mod2_val_path: str,
    *,
    k: int = 3,
    use_features: dict[str, bool] | None = None,
) -> tuple[dict[str, object], RandomForestClassifier]:
    """
    Trains the RF on ALL training windows and evaluates on all validation
    """
    print("  [RF] Loading training data...")
    train_data = np.load(mod2_train_path)
    X_train_ohe = train_data["X"].astype(np.float32)
    y_train = train_data["y"]
    if y_train.dtype == object:
        y_train = np.stack(y_train)

    if y_train.ndim >= 3:
        y_train_window = y_train
        y_train = y_train[:, y_train.shape[1] // 2]
        if y_train.ndim > 1 and y_train.shape[-1] > 1:
            y_train = np.argmax(y_train, axis=-1)
    elif y_train.ndim == 2:
        y_train_window = y_train.astype(np.int8)
        y_train = y_train[:, y_train.shape[1] // 2].astype(np.int32)
    else:
        y_train = y_train.astype(np.int32)
        y_train_window = (train_data["y_window"].astype(np.int8) if "y_window" in train_data else None)

    if y_train.ndim > 1:
        y_train = np.squeeze(y_train)
    y_train = y_train.astype(np.int32)

    print("  [RF] Loading validation data...")
    val_data = np.load(mod2_val_path)
    X_val_ohe = val_data["X"].astype(np.float32)

    y_val = val_data["y"]
    if y_val.dtype == object:
        y_val = np.stack(y_val)

    if y_val.ndim >= 3:
        y_val_window = y_val
        y_val = y_val[:, y_val.shape[1] // 2]
        if y_val.ndim > 1 and y_val.shape[-1] > 1:
            y_val = np.argmax(y_val, axis=-1)
    elif y_val.ndim == 2:
        y_val_window = y_val.astype(np.int8)
        y_val = y_val[:, y_val.shape[1] // 2].astype(np.int32)
    else:
        y_val = y_val.astype(np.int32)
        y_val_window = (val_data["y_window"].astype(np.int8) if "y_window" in val_data else None)

    if y_val.ndim > 1:
        y_val = np.squeeze(y_val)
    y_val = y_val.astype(np.int32)

    print(f"  [RF] Train : {X_train_ohe.shape} | labels (center): {y_train.shape}")
    print(f"  [RF] Val   : {X_val_ohe.shape} | labels (center): {y_val.shape}")

    if y_train_window is not None:
        pure = (np.all(y_train_window == 0, axis=1) |
                np.all(y_train_window == 1, axis=1))
        print(f"  [RF] Training composition: {100 * pure.mean():.0f}% pure, "
              f"{100 * (1 - pure.mean()):.0f}% mixed — ALL will be used.")

    print(f"  [RF] Extracting features on a single scale (k={k})...")
    X_train = build_feature_matrix(X_train_ohe, k=k, use_features=use_features)
    X_val = build_feature_matrix(X_val_ohe, k=k, use_features=use_features)
    print(f"  [RF] Feature matrix — train: {X_train.shape} | val: {X_val.shape}")

    rf = train_rf(X_train, y_train, k=k, use_features=use_features)

    metrics = evaluate_rf(rf, X_val, y_val, verbose=True)
    evaluate_rf_microscope(rf, X_val, y_val_center=y_val,
                           y_val_window=y_val_window,
                           window_size=X_val_ohe.shape[1])

    print("\n" + "=" * 60)
    print("  RF DATA DIAGNOSTICS AND OVERFITTING")
    print("=" * 60)
    print(f"  train : {X_train_ohe.shape} | % exon: {round(100 * y_train.mean(), 1)}%")
    print(f"  val   : {X_val_ohe.shape} | % exon: {round(100 * y_val.mean(), 1)}%")
    print(f"  RF OOB score: {round(rf.oob_score_, 4)}")
    print(f"  Optim. Threshold : {round(metrics.get('best_threshold', 0.5), 3)}") # type: ignore[arg-type]
    print(f"  F1_macro       : {round(metrics['f1_macro'], 4)}") # type: ignore[arg-type]
    print("=" * 60 + "\n")

    return metrics, rf


# =============================================================================
# 7. DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tests the rf_model.py module standalone."
    )
    parser.add_argument("--train", required=True, help="Path to the training .npz.")
    parser.add_argument("--val", required=True, help="Path to the validation .npz.")
    parser.add_argument("--k", type=int, default=3, choices=[2, 3, 4],
                        help="k-mer size (default: 3).")
    parser.add_argument("--no-kmer", action="store_true", help="Disables k-mer frequencies feature")
    parser.add_argument("--no-orf", action="store_true", help="Disables max ORF length feature")
    parser.add_argument("--no-fourier", action="store_true", help="Disables Fourier Period 3 feature")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RF Standalone")
    print("=" * 60)
    use_features = {
        "kmer": not args.no_kmer,
        "orf": not args.no_orf,
        "fourier": not args.no_fourier,
    }
    metrics, rf = run_rf_pipeline(args.train, args.val,
                                  k=args.k, use_features=use_features)
    print(f"\n  Accuracy Final : {metrics['accuracy'] * 100:.2f}%") # type: ignore
    print(f"  Final Macro F1 : {metrics['f1_macro'] * 100:.2f}%") # type: ignore
    print(f"  (trivial baseline: {metrics['trivial_f1_macro'] * 100:.2f}% F1 macro)") # type: ignore

    print("\n  Testing the inference path with synthetic data...")
    W = 120
    dummy = np.eye(4)[np.random.randint(0, 4, (5, W))].astype(np.float32)
    p = rf_proba(rf, dummy)
    print(f"  rf_proba          -> {p.shape}  (expected: (5,))")
    aug = inject_rf_proba(rf, dummy)
    print(f"  inject_rf_proba   -> {aug.shape}  (expected: (5, {W}, 5))")
    print(f"  rf_proba_oob      -> {rf_proba_oob(rf).shape}  "
          f"(expected: ({len(np.load(args.train)['y'])},))")

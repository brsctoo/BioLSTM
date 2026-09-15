"""
diagnose_threshold.py — diagnostico temporario (Fase 2.2)

Roda inferencia UMA VEZ por experimento (caro: carrega o modelo, passa pelo
TensorFlow) e depois varre threshold x janela-de-suavizacao inteiramente em
memoria (barato: so numpy/sklearn) — confirma ou descarta a Hipotese 1
(miscalibracao de threshold) sem precisar retreinar nada.

Uso: ajuste EXPERIMENTS com os modelos que voce realmente tem salvos em
assets/result/, depois: python diagnose_threshold.py
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from evaluation.validation import smooth_predict, validate_model
from models import rf_model

# ---------------------------------------------------------------------------
# Ajuste aqui: um item por modelo ja treinado que voce quer diagnosticar.
# Remova as linhas de modelos que voce nao tem mais salvos localmente.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {"name": "actin_fungi_w200", "rf_scale": 0.15},
    {"name": "actin_fungi_w400", "rf_scale": 0.15},
    {"name": "actin_fungi_w600", "rf_scale": 0.15},
]

THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)
SMOOTHING_GRID = [1, 3, 5, 9, 15]  # 1 = sem suavizacao; 5 = o que roda em producao hoje


def sweep(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> pd.DataFrame:
    """Varre threshold x smoothing sem tocar no modelo de novo."""
    rows = []
    for t in THRESHOLD_GRID:
        raw = (y_prob > t).astype(int)
        for w in SMOOTHING_GRID:
            pred = np.array(smooth_predict(raw, window_size=w))
            rows.append({
                "experiment": name,
                "threshold": round(float(t), 2),
                "smooth_window": w,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "accuracy": accuracy_score(y_true, pred),
                "pct_predicted_positive": float(np.mean(pred == 1)) * 100,
            })
    return pd.DataFrame(rows)


def print_histogram(y_prob: np.ndarray, bins: int = 10) -> None:
    hist, edges = np.histogram(y_prob, bins=bins, range=(0, 1))
    total = len(y_prob)
    for h, e in zip(hist, edges):
        bar = "#" * int(60 * h / total) if total else ""
        print(f"  [{e:.1f}-{e + 1 / bins:.1f}): {h:>7d} ({100 * h / total:5.1f}%)  {bar}")


def main() -> None:
    all_results = []

    for exp in EXPERIMENTS:
        name = exp["name"]
        print("\n" + "=" * 70)
        print(f"EXPERIMENTO: {name}  (rf_scale={exp['rf_scale']})")
        print("=" * 70)

        rf = rf_model.load_rf(f"assets/result/model_{name}_rf.joblib")
        y_true, y_prob, fp_dist = validate_model(
            f"assets/result/model_{name}_onehot.h5",
            f"assets/processed_data/mod1/data_{name}_test.mod1",
            rf=rf,
            rf_scale=exp["rf_scale"],
            threshold=0.6,  # so afeta os prints internos da validate_model
        )
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        print(f"\nTotal de posicoes avaliadas: {len(y_true)}")
        print(f"Proporcao real de exon: {100 * np.mean(y_true == 1):.2f}%")

        print("\n--- Histograma de probabilidades (y_prob) ---")
        print_histogram(y_prob)

        df = sweep(y_true, y_prob, name)
        all_results.append(df)

        best = df.loc[df["f1"].idxmax()]
        baseline = df[(df["threshold"] == 0.60) & (df["smooth_window"] == 5)]

        print("\n--- Melhor combinacao (por F1) ---")
        print(best.to_string())

        if not baseline.empty:
            print("\n--- Baseline atual em producao (threshold=0.60, smoothing=5) ---")
            print(baseline.to_string(index=False))

        if len(fp_dist):
            fp_arr = np.array(fp_dist)
            print(f"\nFP a <=10bp de uma borda: {np.mean(fp_arr <= 10) * 100:.1f}%")
            print(f"Distancia mediana de FP : {np.median(fp_arr):.1f}bp")

    full = pd.concat(all_results, ignore_index=True)
    full.to_csv("diagnose_threshold_results.csv", index=False)
    print("\n\nTabela completa (todas as combinacoes) salva em: diagnose_threshold_results.csv")

    print("\n" + "=" * 70)
    print("RESUMO — melhor F1 por experimento")
    print("=" * 70)
    summary = full.loc[full.groupby("experiment")["f1"].idxmax()]
    print(summary[["experiment", "threshold", "smooth_window",
                    "precision", "recall", "f1", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()

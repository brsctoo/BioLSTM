"""
rf_model.py
===========
Random Forest classifier for Intron/Exon window classification.

Atua como modelo auxiliar do Bi-LSTM: é treinado ANTES da rede neural e
gera probabilidades por janela (predict_proba ou OOB) que são injetadas como
um 5º canal no tensor One-Hot — enriquecendo a entrada da LSTM com a
assinatura estatística global da janela (viés de dinucleotídeos e %GC).

Convenção de canais One-Hot (de modeling.py / BASE_TO_VECTOR):
    índice 0 → A (Adenina)
    índice 1 → T (Timina)
    índice 2 → G (Guanina)
    índice 3 → C (Citosina)
"""

import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =============================================================================
# 1. FEATURE ENGINEERING
# =============================================================================

def compute_gc_content(one_hot: np.ndarray) -> np.ndarray:
    """
    Calcula o conteúdo percentual de G+C para cada janela do batch.
    """
    window_size = one_hot.shape[1]

    # Soma as contagens de cada nucleotídeo ao longo da dimensão da janela.
    # nucleotide_counts shape: (Batch_Size, 4)
    nucleotide_counts = one_hot.sum(axis=1)

    # Canal 2 = G, Canal 3 = C  (conforme BASE_TO_VECTOR em modeling.py)
    gc_counts = nucleotide_counts[:, 2] + nucleotide_counts[:, 3]
    gc = (gc_counts / window_size * 100).reshape(-1, 1)

    return gc  # (Batch_Size, 1)


def compute_kmer_frequencies(one_hot: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Calcula as frequências normalizadas dos 4^k k-mers para cada janela.
    Configurado por padrão para k=2 (Dinucleotídeos) para evitar matrizes
    esparsas e overfitting em janelas curtas (ex: 120 bases).
    """
    n_kmers = 4 ** k

    # Passo 1 — One-Hot → índice inteiro (A=0, T=1, G=2, C=3)
    seq_indices = np.argmax(one_hot, axis=2)

    # Passo 2 — Janelas deslizantes de tamanho k sem cópia de dados
    kmer_windows = sliding_window_view(seq_indices, window_shape=k, axis=1)
    n_windows = kmer_windows.shape[1]

    # Passo 3 — Conversão de cada sequência em um inteiro único (base 4)
    powers = 4 ** np.arange(k - 1, -1, -1)             # (k,)
    kmer_indices = (kmer_windows * powers).sum(axis=2) # (B, n_windows)

    # Passo 4 — Contagem vetorizada via comparação booleana + soma
    kmer_onehot = (
        kmer_indices[:, :, np.newaxis] == np.arange(n_kmers)
    ).astype(np.float32)

    # Passo 5 — Frequência relativa
    kmer_freq = kmer_onehot.sum(axis=1) / n_windows    # (B, n_kmers)

    return kmer_freq  # (Batch_Size, 16)


def build_feature_matrix(one_hot: np.ndarray) -> np.ndarray:
    """
    Converte o tensor One-Hot 3D em uma matriz tabular 2D (17 features)
    pronta para o Random Forest.
    """
    gc     = compute_gc_content(one_hot)            # (B, 1)
    kmers  = compute_kmer_frequencies(one_hot, k=2) # (B, 16)
    X      = np.concatenate([gc, kmers], axis=1).astype(np.float32)  # (B, 17)
    return X


def get_feature_names() -> list[str]:
    """
    Retorna os nomes das 17 features na mesma ordem que build_feature_matrix.
    Ordem das bases alinhada com BASE_TO_VECTOR: [A, T, G, C].
    """
    bases = ["A", "T", "G", "C"]
    kmer_names = [a + b for a in bases for b in bases]
    return ["%GC"] + kmer_names  # 1 + 16 = 17


# =============================================================================
# 2. TREINAMENTO
# =============================================================================

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 300,
    max_depth: int = 8,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Configura e treina o RandomForestClassifier.
    Parâmetros restritos para forçar a generalização da assinatura global.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,            # Necessário para injeção sem Target Leakage
        random_state=random_state,
        n_jobs=-1,
    )

    print(f"  [RF] Treinando RandomForest ({n_estimators} árvores, max_depth={max_depth})...")
    model.fit(X_train, y_train)
    print(f"  [RF] Treinamento concluído. OOB Score: {model.oob_score_:.4f}")

    return model


# =============================================================================
# 3. AVALIAÇÃO
# =============================================================================

def evaluate_rf(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    verbose: bool = True,
) -> dict:
    """
    Avalia o modelo no conjunto de validação e retorna as métricas principais.
    """
    y_pred = model.predict(X_val)

    acc        = accuracy_score(y_val, y_pred)
    f1_exon    = f1_score(y_val, y_pred, pos_label=1, zero_division=0) # type: ignore
    f1_intron  = f1_score(y_val, y_pred, pos_label=0, zero_division=0) # type: ignore
    f1_macro   = f1_score(y_val, y_pred, average="macro", zero_division=0) # type: ignore
    cm         = confusion_matrix(y_val, y_pred)
    report     = classification_report(
        y_val, y_pred,
        target_names=["Íntron (0)", "Éxon (1)"],
        zero_division=0, # type: ignore
    )

    if verbose:
        _print_rf_results(acc, f1_exon, f1_intron, f1_macro, cm, report)

    return {
        "accuracy" : acc,
        "f1_exon"  : f1_exon,
        "f1_intron": f1_intron,
        "f1_macro" : f1_macro,
        "cm"       : cm,
        "report"   : report,
    }


def _print_rf_results(acc, f1_exon, f1_intron, f1_macro, cm, report):
    """Formata e imprime os resultados do RF no terminal."""
    sep = "═" * 55
    print(f"\n{sep}")
    print("  RANDOM FOREST — RESULTADOS DE VALIDAÇÃO")
    print(sep)
    print(f"  {'Acurácia':<25}: {acc * 100:.2f}%")
    print(f"  {'F1-Score  Éxon  (1)':<25}: {f1_exon * 100:.2f}%")
    print(f"  {'F1-Score  Íntron (0)':<25}: {f1_intron * 100:.2f}%")
    print(f"  {'F1-Score  Macro':<25}: {f1_macro * 100:.2f}%")
    print(f"\n  Matriz de Confusão:")
    print(f"          Pred 0   Pred 1")
    print(f"  Real 0  {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"  Real 1  {cm[1,0]:>6}   {cm[1,1]:>6}")
    print(f"\n  Relatório Completo:\n{report}")
    print(sep)


# =============================================================================
# 4. INJEÇÃO DE PROBABILIDADE RF NO TENSOR DO LSTM
# =============================================================================

def inject_rf_proba(
    rf: RandomForestClassifier,
    one_hot: np.ndarray,
    rf_scale: float = 0.20,
    is_training_set: bool = False,
    apply_dropout: bool = False,
    dropout_rate: float = 0.5
) -> np.ndarray:
    """
    Gera um tensor aumentado (B, Window_Size, 5) com injeção de probabilidade.
    Resolve Target Leakage usando oob_decision_function_ no treino.
    Evita LSTM preguiçosa usando Dropout.
    """
    batch_size, window_size, _ = one_hot.shape

    # --- PROTEÇÃO CONTRA VAZAMENTO DE DADOS (TARGET LEAKAGE) ---
    if is_training_set and hasattr(rf, 'oob_decision_function_'):
        # No treino, usamos a matriz Out-of-Bag já computada pelo fit()
        p_exon = rf.oob_decision_function_[:, 1]
    else:
        # Na validação/teste, fazemos a predição tabular clássica
        X_tabular = build_feature_matrix(one_hot)
        p_exon = np.asarray(rf.predict_proba(X_tabular))[:, 1]

    # --- PROTEÇÃO CONTRA MODELO "PREGUIÇOSO" (DROPOUT) ---
    if apply_dropout:
        mask = np.random.binomial(1, 1 - dropout_rate, size=p_exon.shape)
        p_exon = p_exon * mask

    # Escala o sinal do RF para que seja um apoio suave
    p_channel = np.broadcast_to(
        (p_exon * rf_scale)[:, np.newaxis, np.newaxis],
        (batch_size, window_size, 1)
    ).astype(np.float32)

    augmented = np.concatenate(
        [one_hot.astype(np.float32), p_channel],
        axis=2
    )
    return augmented   # (Batch_Size, Window_Size, 5)


# =============================================================================
# 5. PERSISTÊNCIA DO MODELO RF (salvar / carregar)
# =============================================================================

def save_rf(rf: RandomForestClassifier, path: str) -> None:
    import joblib
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(rf, path)
    print(f"  [RF] Modelo salvo em: {path}")


def load_rf(path: str):
    import joblib
    if not os.path.exists(path):
        print(f"  [RF] Nenhum modelo RF encontrado em: {path} (Validação prosseguirá sem injeção)")
        return None

    rf = joblib.load(path)
    print(f"  [RF] Modelo carregado de: {path}")
    return rf


# =============================================================================
# 6. PIPELINE DE ALTO NÍVEL
# =============================================================================

def run_rf_pipeline(
    mod2_train_path: str,
    mod2_val_path: str,
) -> tuple[dict, RandomForestClassifier]:

    print("  [RF] Carregando dados de treino...")
    train_data  = np.load(mod2_train_path)
    X_train_ohe = train_data["X"].astype(np.float32)
    y_train     = train_data["y"].astype(np.int32)

    print("  [RF] Carregando dados de validação...")
    val_data   = np.load(mod2_val_path)
    X_val_ohe  = val_data["X"].astype(np.float32)
    y_val      = val_data["y"].astype(np.int32)

    print(f"  [RF] Treino : {X_train_ohe.shape} | Rótulos: {y_train.shape}")
    print(f"  [RF] Val    : {X_val_ohe.shape}   | Rótulos: {y_val.shape}")

    print("  [RF] Extraindo features tabulares (17 por janela)...")
    X_train = build_feature_matrix(X_train_ohe)  # (N_train, 17)
    X_val   = build_feature_matrix(X_val_ohe)    # (N_val, 17)
    print(f"  [RF] Feature matrix — treino: {X_train.shape} | val: {X_val.shape}")

    rf = train_rf(X_train, y_train)
    metrics = evaluate_rf(rf, X_val, y_val, verbose=True)

    return metrics, rf


# =============================================================================
# 7. EXECUÇÃO DIRETA (para testes standalone)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Testa o módulo rf_model.py de forma standalone."
    )
    parser.add_argument("--train", required=True, help="Caminho para o .npz de treino.")
    parser.add_argument("--val",   required=True, help="Caminho para o .npz de validação.")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  RF Standalone — Iniciando pipeline")
    print("="*55)
    metrics, rf = run_rf_pipeline(args.train, args.val)
    print(f"\n  Acurácia Final : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Éxon Final  : {metrics['f1_exon']*100:.2f}%")

    print("\n  Testando inject_rf_proba com dados sintéticos...")
    dummy = np.eye(4)[np.random.randint(0, 4, (5, 120))].astype(np.float32)
    augmented = inject_rf_proba(rf, dummy, is_training_set=False, apply_dropout=False)
    print(f"  Tensor aumentado: {augmented.shape}  (esperado: (5, 120, 5))")

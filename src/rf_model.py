"""
rf_model.py
===========
Random Forest classifier for Intron/Exon window classification.

Atua como modelo auxiliar do Bi-LSTM: é treinado ANTES da rede neural e
gera probabilidades por janela (predict_proba) que são injetadas como um
5º canal no tensor One-Hot — enriquecendo a entrada do LSTM com a
assinatura estatística global da janela.

Convenção de canais One-Hot (de modeling.py / BASE_TO_VECTOR):
    índice 0 → A (Adenina)
    índice 1 → T (Timina)
    índice 2 → G (Guanina)
    índice 3 → C (Citosina)

Public API
----------
build_feature_matrix(one_hot)                    -> np.ndarray  (B, 65)
train_rf(X_train, y_train)                        -> RandomForestClassifier
evaluate_rf(model, X_val, y_val)                  -> dict
inject_rf_proba(rf, one_hot)                      -> np.ndarray  (B, 400, 5)
run_rf_pipeline(mod2_train, mod2_val)             -> tuple[dict, RandomForestClassifier]
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

    Parâmetros
    ----------
    one_hot : np.ndarray
        Tensor One-Hot com forma (Batch_Size, Window_Size, 4).
        Convenção de canais (de modeling.py/BASE_TO_VECTOR):
            [A=0, T=1, G=2, C=3]

    Retorna
    -------
    gc : np.ndarray
        Vetor coluna (Batch_Size, 1) com o %GC de cada janela.
    """
    window_size = one_hot.shape[1]

    # Soma as contagens de cada nucleotídeo ao longo da dimensão da janela.
    # nucleotide_counts shape: (Batch_Size, 4)
    nucleotide_counts = one_hot.sum(axis=1)

    # Canal 2 = G, Canal 3 = C  (conforme BASE_TO_VECTOR em modeling.py)
    gc_counts = nucleotide_counts[:, 2] + nucleotide_counts[:, 3]
    gc = (gc_counts / window_size * 100).reshape(-1, 1)

    return gc  # (Batch_Size, 1)


def compute_kmer_frequencies(one_hot: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Calcula as frequências normalizadas dos 4^k k-mers para cada janela.

    Usa janelas deslizantes via numpy.lib.stride_tricks.sliding_window_view
    e codificação em base 4, eliminando loops Python para máxima performance.
    """
    n_kmers = 4 ** k

    # Passo 1 — One-Hot → índice inteiro (A=0, T=1, G=2, C=3)
    seq_indices = np.argmax(one_hot, axis=2)

    # Passo 2 — Janelas deslizantes de tamanho k sem cópia de dados
    kmer_windows = sliding_window_view(seq_indices, window_shape=k, axis=1)
    n_windows = kmer_windows.shape[1]

    # Passo 3 — Conversão de cada trinca em um inteiro único (base 4)
    powers = 4 ** np.arange(k - 1, -1, -1)           # (k,)
    kmer_indices = (kmer_windows * powers).sum(axis=2) # (B, n_windows)

    # Passo 4 — Contagem vetorizada via comparação booleana + soma
    kmer_onehot = (
        kmer_indices[:, :, np.newaxis] == np.arange(n_kmers)
    ).astype(np.float32)

    # Passo 5 — Frequência relativa
    kmer_freq = kmer_onehot.sum(axis=1) / n_windows   # (B, n_kmers)

    return kmer_freq  # (Batch_Size, 64)


def build_feature_matrix(one_hot: np.ndarray) -> np.ndarray:
    """
    Converte o tensor One-Hot 3D em uma matriz tabular 2D pronta para o
    Random Forest.
    """
    gc     = compute_gc_content(one_hot)          # (B, 1)
    kmers  = compute_kmer_frequencies(one_hot, k=3)  # (B, 64)
    X      = np.concatenate([gc, kmers], axis=1).astype(np.float32)  # (B, 65)
    return X


def get_feature_names() -> list[str]:
    """
    Retorna os nomes das 65 features na mesma ordem que build_feature_matrix.

    Útil para plotar feature importances no relatório.
    Ordem dos bases alinhada com BASE_TO_VECTOR: [A, T, G, C].
    """
    bases = ["A", "T", "G", "C"]  # Ordem real do projeto (modeling.py)
    kmer_names = [a + b + c for a in bases for b in bases for c in bases]
    return ["%GC"] + kmer_names  # 1 + 64 = 65


# =============================================================================
# 2. TREINAMENTO
# =============================================================================

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 500,
    max_depth: int = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Configura e treina o RandomForestClassifier com hiperparâmetros
    recomendados para dados biológicos com possível desbalanceamento
    de classes.

    Parâmetros
    ----------
    X_train : np.ndarray
        Matriz tabular de treino (N_train, 65) gerada por build_feature_matrix.
    y_train : np.ndarray
        Vetor de rótulos de treino (N_train,) — 0=Íntron, 1=Éxon.
    n_estimators : int
        Número de árvores no ensemble (padrão 500).
    max_depth : int
        Profundidade máxima de cada árvore — principal controle de overfitting.
    min_samples_split : int
        Mínimo de amostras para dividir um nó interno.
    min_samples_leaf : int
        Mínimo de amostras em uma folha.
    random_state : int
        Semente para reprodutibilidade.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",       # Diversidade entre árvores (padrão para classificação)
        class_weight="balanced",   # Corrige desbalanceamento Íntron/Éxon automaticamente
        oob_score=True,            # Estimativa gratuita de generalização (Out-of-Bag)
        random_state=random_state,
        n_jobs=-1,                 # Paralelismo total: usa todos os núcleos de CPU
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
) -> np.ndarray:
    """
    Gera um tensor aumentado (B, Window_Size, 5) onde o 5º canal em
    cada posição da janela contém P(Éxon) predita pelo Random Forest.

    O RF opera sobre a janela INTEIRA (estatística global), produzindo
    um escalar por amostra. Esse escalar é então replicado ao longo
    de todo o eixo temporal da janela — informando ao LSTM a "confiança
    global" do RF antes mesmo de processar a sequência.

    Parâmetros
    ----------
    rf : RandomForestClassifier
        Modelo RF já treinado por train_rf().
    one_hot : np.ndarray
        Tensor One-Hot de forma (Batch_Size, Window_Size, 4).

    Retorna
    -------
    augmented : np.ndarray
        Tensor de forma (Batch_Size, Window_Size, 5), dtype float32.
        Os 4 primeiros canais são o One-Hot original; o 5º é P(Éxon).
    """
    batch_size, window_size, _ = one_hot.shape

    # Extrai features tabulares e obtém probabilidades do RF
    X_tabular = build_feature_matrix(one_hot)
    p_exon: np.ndarray = np.asarray(rf.predict_proba(X_tabular))[:, 1]

    # Replica o escalar ao longo do eixo temporal e adiciona como 5º canal
    # p_exon[:, np.newaxis, np.newaxis] → (B, 1, 1)
    # np.broadcast_to → (B, Window_Size, 1)  (sem cópia)
    p_channel = np.broadcast_to(
        p_exon[:, np.newaxis, np.newaxis],
        (batch_size, window_size, 1)
    ).astype(np.float32)

    # Concatena ao longo do eixo dos canais: (B, W, 4) + (B, W, 1) → (B, W, 5)
    augmented = np.concatenate(
        [one_hot.astype(np.float32), p_channel],
        axis=2
    )
    return augmented   # (Batch_Size, Window_Size, 5)


# =============================================================================
# 5. PIPELINE DE ALTO NÍVEL (chamado pelo pipeline.py)
# =============================================================================

def run_rf_pipeline(
    mod2_train_path: str,
    mod2_val_path: str,
) -> tuple[dict, RandomForestClassifier]:
    """
    Pipeline completo do Random Forest: carrega os dados já processados
    (arquivos .npz do projeto), extrai features, treina e avalia.

    Deve ser chamado ANTES do treinamento do LSTM. O modelo RF retornado
    é usado por inject_rf_proba() para enriquecer o tensor de entrada do LSTM.

    Parâmetros
    ----------
    mod2_train_path : str
        Caminho para o .npz de treino gerado por modeling.py.
        Chaves esperadas: 'X' (One-Hot tensor float16/32) e 'y' (rótulos int8).
    mod2_val_path : str
        Caminho para o .npz de validação (mesma estrutura).

    Retorna
    -------
    metrics : dict
        Dicionário com as métricas de validação (ver evaluate_rf).
    rf_model : RandomForestClassifier
        Modelo treinado, pronto para ser passado a inject_rf_proba().
    """
    # --- Carregamento dos dados ---
    print("  [RF] Carregando dados de treino...")
    train_data  = np.load(mod2_train_path)
    X_train_ohe = train_data["X"].astype(np.float32)  # (N_train, 400, 4)
    y_train     = train_data["y"].astype(np.int32)     # (N_train,)  — chave 'y' minúsculo

    print("  [RF] Carregando dados de validação...")
    val_data   = np.load(mod2_val_path)
    X_val_ohe  = val_data["X"].astype(np.float32)     # (N_val, 400, 4)
    y_val      = val_data["y"].astype(np.int32)        # (N_val,)

    print(f"  [RF] Treino : {X_train_ohe.shape} | Rótulos: {y_train.shape}")
    print(f"  [RF] Val    : {X_val_ohe.shape}   | Rótulos: {y_val.shape}")

    # --- Feature Engineering ---
    print("  [RF] Extraindo features tabulares (65 por janela)...")
    X_train = build_feature_matrix(X_train_ohe)  # (N_train, 65)
    X_val   = build_feature_matrix(X_val_ohe)    # (N_val, 65)
    print(f"  [RF] Feature matrix — treino: {X_train.shape} | val: {X_val.shape}")

    # --- Treinamento ---
    rf = train_rf(X_train, y_train)

    # --- Avaliação standalone do RF ---
    metrics = evaluate_rf(rf, X_val, y_val, verbose=True)

    return metrics, rf


# =============================================================================
# 6. EXECUÇÃO DIRETA (para testes standalone)
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

    # Demonstração da injeção de probabilidade (com dados sintéticos)
    print("\n  Testando inject_rf_proba com dados sintéticos...")
    dummy = np.eye(4)[np.random.randint(0, 4, (5, 400))].astype(np.float32)
    augmented = inject_rf_proba(rf, dummy)
    print(f"  Tensor aumentado: {augmented.shape}  (esperado: (5, 400, 5))")

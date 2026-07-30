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


def compute_positional_asymmetry(one_hot: np.ndarray) -> np.ndarray:
    """
    Inspirado no Fickett TESTCODE Score (1982).
    Calcula a frequência das 4 bases (A, T, G, C) agrupadas por 
    posição do códon (posição 1, 2 e 3).
    Gera 12 features que medem o viés de tradução biológica.
    """
    # pos1, pos2, pos3 pegam as bases pulando de 3 em 3
    pos1 = one_hot[:, 0::3, :].sum(axis=1) # (B, 4)
    pos2 = one_hot[:, 1::3, :].sum(axis=1) # (B, 4)
    pos3 = one_hot[:, 2::3, :].sum(axis=1) # (B, 4)
    
    codons = one_hot.shape[1] / 3
    
    pos1_freq = pos1 / codons
    pos2_freq = pos2 / codons
    pos3_freq = pos3 / codons
    
    return np.concatenate([pos1_freq, pos2_freq, pos3_freq], axis=1).astype(np.float32)


def build_feature_matrix(one_hot: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Converte o tensor One-Hot 3D em uma matriz tabular 2D
    pronta para o Random Forest. Usa k=2 (17 features) ou k=3 (65 features).
    """
    gc     = compute_gc_content(one_hot)             # (B, 1)
    kmers  = compute_kmer_frequencies(one_hot, k=k) # (B, 4^k)
    fickett = compute_positional_asymmetry(one_hot)  # (B, 12)
    X      = np.concatenate([gc, kmers, fickett], axis=1).astype(np.float32)
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
    max_depth: int = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Configura e treina o RandomForestClassifier.
    Árvores profundas (max_depth=None) para extrair regras complexas do Codon Usage.
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
# 3. AVALIAÇÃO E DIAGNÓSTICO
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


def evaluate_rf_microscope(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val_center: np.ndarray,
    y_val_window: np.ndarray = None,
    window_size: int = 120
):
    """
    Diagnóstico detalhado do Random Forest separando o desempenho por tipo de janela.
    """
    print("\n" + "═" * 55)
    print("  🔬 MICROSCÓPIO LOCAL — ANÁLISE POR TIPO DE JANELA")
    print("═" * 55)

    y_pred = model.predict(X_val)

    if y_val_window is None:
        print("  [Aviso] 'y_val_window' não fornecido no arquivo gerado. Recrie os dados (pipeline.py) para ver a análise de janelas puras.")
        return

    # 1. & 2. Criamos as máscaras booleanas REAIS garantindo que não haja bases -1 (desconhecidas) ou classes misturadas
    # Janela 100% Íntron (absolutamente todos os 120 rótulos são 0)
    idx_pure_intron = np.all(y_val_window == 0, axis=1)

    # Janela 100% Éxon (absolutamente todos os 120 rótulos são 1)
    idx_pure_exon   = np.all(y_val_window == 1, axis=1)

    # Mistas (qualquer janela que não seja perfeitamente pura, inclui bordas de splicing e -1)
    idx_mixed       = ~(idx_pure_intron | idx_pure_exon)

    # 3. Função auxiliar para calcular e printar a acurácia de cada balde
    def print_bucket_stats(name, mask):
        total_in_bucket = np.sum(mask)
        if total_in_bucket == 0:
            print(f"  {name:<22}: 0 amostras neste conjunto.")
            return

        # Filtra as predições e os gabaritos reais usando a máscara
        y_true_bucket = y_val_center[mask]
        y_pred_bucket = y_pred[mask]

        acc = accuracy_score(y_true_bucket, y_pred_bucket)
        print(f"  {name:<22}: {acc * 100:>6.2f}% de Acurácia  (Total: {total_in_bucket} janelas)")

    # 4. Resultados
    print_bucket_stats("Janelas 100% Íntron", idx_pure_intron)
    print_bucket_stats("Janelas 100% Éxon", idx_pure_exon)
    print_bucket_stats("Janelas Mistas", idx_mixed)
    print("═" * 55)


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

    # --- PROTEÇÃO CONTRA VAZAMENTO (REMOVIDA PARA LATE FUSION) ---
    # No Late Fusion, a LSTM não enxerga a P(Éxon), apenas a camada Dense final.
    # O Dropout aleatório abaixo já garante que a rede não fique viciada no RF.
    # Portanto, podemos usar predict_proba em todo o dataset (mesmo nas mistas que o RF nunca viu no treino).

    expected_features = getattr(rf, "n_features_in_", 17)
    
    # Retrocompatibilidade
    if expected_features == 77:
        k = 3 # Fickett incluído
    elif expected_features == 65:
        k = 3 # Sem Fickett (Antigo)
    else:
        k = 2 # Antigo
        
    X_tabular = build_feature_matrix(one_hot, k=3)
    # Se carregou um modelo antigo que não tem Fickett (77 features), fatia o array para não quebrar
    if expected_features < X_tabular.shape[1]:
        X_tabular = X_tabular[:, :expected_features]

    p_exon = np.asarray(rf.predict_proba(X_tabular))[:, 1] # (B,)

    # --- PROTEÇÃO CONTRA MODELO "PREGUIÇOSO" (DROPOUT) ---
    if apply_dropout:
        mask = np.random.binomial(1, 1 - dropout_rate, size=p_exon.shape)
        # O Dropout para Late Fusion precisa jogar a predição para o valor neutro (0.50),
        p_exon = np.where(mask == 1, p_exon, 0.50)

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
    y_train = train_data["y"].astype(np.int32)
    y_train_window = train_data["y_window"].astype(np.int8) if "y_window" in train_data else None

    print("  [RF] Carregando dados de validação...")
    val_data = np.load(mod2_val_path)
    X_val_ohe = val_data["X"].astype(np.float32)
    y_val = val_data["y"].astype(np.int32)
    y_val_window = val_data["y_window"].astype(np.int8) if "y_window" in val_data else None

    # --- FILTRO DE ESPECIALISTA: Treinar apenas com Janelas Puras ---
    if y_train_window is not None:
        idx_pure_intron = np.all(y_train_window == 0, axis=1)
        idx_pure_exon   = np.all(y_train_window == 1, axis=1)
        idx_pure_all    = idx_pure_intron | idx_pure_exon

        X_train_ohe_pure = X_train_ohe[idx_pure_all]
        y_train_pure     = y_train[idx_pure_all]
        print(f"  [RF] Especialização: Treinando apenas com janelas 100% puras ({len(y_train_pure)} amostras de {len(y_train)}).")
    else:
        X_train_ohe_pure = X_train_ohe
        y_train_pure     = y_train
        print("  [RF] Aviso: y_window ausente no treino, treinando com dados mistos.")

    print(f"  [RF] Treino (Puras) : {X_train_ohe_pure.shape} | Rótulos: {y_train_pure.shape}")
    print(f"  [RF] Val (Total)    : {X_val_ohe.shape}   | Rótulos: {y_val.shape}")

    print("  [RF] Extraindo features tabulares...")
    X_train = build_feature_matrix(X_train_ohe_pure)  # (N_train_pure, K)
    X_val   = build_feature_matrix(X_val_ohe)         # (N_val, K)
    print(f"  [RF] Feature matrix — treino: {X_train.shape} | val: {X_val.shape}")

    rf = train_rf(X_train, y_train_pure)

    # Avaliação Global Original
    metrics = evaluate_rf(rf, X_val, y_val, verbose=True)

    # Nova Avaliação pelo Microscópio (Usando a janela VERDADEIRA)
    evaluate_rf_microscope(rf, X_val, y_val_center=y_val, y_val_window=y_val_window, window_size=X_val_ohe.shape[1])

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

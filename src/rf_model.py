"""
rf_model.py
===========
Random Forest classifier for Intron/Exon window classification.

Atua como modelo auxiliar do Bi-LSTM: e treinado ANTES da rede neural e
gera probabilidades por janela que sao injetadas como um 5o canal / entrada
tabular — enriquecendo a entrada da LSTM com a assinatura estatistica
composicional da janela.

Convencao de canais One-Hot (de modeling.py / BASE_TO_VECTOR):
    indice 0 -> A (Adenina)
    indice 1 -> T (Timina)
    indice 2 -> G (Guanina)
    indice 3 -> C (Citosina)
"""

import os
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Versao do esquema de features. Gravada junto com o modelo.
FEATURE_SCHEMA_VERSION = 2


# =============================================================================
# 1. FEATURE ENGINEERING
# =============================================================================

def _valid_mask(one_hot: np.ndarray) -> np.ndarray:
    """
    Posicoes reais vs padding. `extract_windows_numpy` preenche as bordas da
    sequencia com [0,0,0,0]; um argmax nesse vetor devolve 0 (= 'A'), o que
    inventaria adeninas inexistentes. Esta mascara permite descontar isso.
    """
    return one_hot.sum(axis=2) > 1e-6            # (B, W)


def compute_gc_content(one_hot: np.ndarray) -> np.ndarray:
    """
    Conteudo percentual de G+C por janela, normalizado pelo numero de
    posicoes REAIS (nao pelo tamanho da janela), para nao subestimar o GC
    em janelas que caem na borda da sequencia.
    """
    counts = one_hot.sum(axis=1)                  # (B, 4)
    gc = counts[:, 2] + counts[:, 3]
    at = counts[:, 0] + counts[:, 1]
    total = gc + at
    out = np.divide(gc, total, out=np.zeros_like(gc, dtype=np.float64),
                    where=total > 0) * 100.0
    return out.reshape(-1, 1).astype(np.float32)  # (B, 1)


def compute_kmer_frequencies(one_hot: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Frequencias normalizadas dos 4^k k-mers por janela.
    k=2 -> 16 colunas, k=3 -> 64 colunas.
    """
    n_kmers = 4 ** k
    W = one_hot.shape[1]
    if W < k:
        return np.zeros((one_hot.shape[0], n_kmers), dtype=np.float32)

    seq_indices = np.argmax(one_hot, axis=2)               # (B, W)
    valid = _valid_mask(one_hot)                           # (B, W)

    kmer_windows = sliding_window_view(seq_indices, window_shape=k, axis=1)
    valid_windows = sliding_window_view(valid, window_shape=k, axis=1)
    kmer_ok = valid_windows.all(axis=2)                    # (B, n_pos) sem padding

    powers = 4 ** np.arange(k - 1, -1, -1)
    kmer_indices = (kmer_windows * powers).sum(axis=2)     # (B, n_pos)

    onehot_k = (kmer_indices[:, :, None] == np.arange(n_kmers))
    onehot_k = onehot_k & kmer_ok[:, :, None]              # descarta padding
    counts = onehot_k.sum(axis=1).astype(np.float32)       # (B, n_kmers)

    total = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, total, out=np.zeros_like(counts),
                     where=total > 0)


def compute_positional_asymmetry(one_hot: np.ndarray) -> np.ndarray:
    """
    Inspirado no Fickett TESTCODE (1982): frequencia das 4 bases agrupadas
    por posicao no codon (1a, 2a, 3a). 12 features.

    Cada fase e normalizada pelo proprio numero de posicoes, para funcionar
    quando o tamanho da janela nao e multiplo de 3 (o recorte central pode
    nao ser).
    """
    blocks = []
    for phase in range(3):
        sub = one_hot[:, phase::3, :]
        counts = sub.sum(axis=1)                            # (B, 4)
        total = counts.sum(axis=1, keepdims=True)
        blocks.append(np.divide(counts, total,
                                out=np.zeros_like(counts, dtype=np.float64),
                                where=total > 0))
    return np.concatenate(blocks, axis=1).astype(np.float32)  # (B, 12)


def compute_max_orf_length(one_hot: np.ndarray) -> np.ndarray:
    """
    Maior trecho continuo sem stop codon (TAA, TAG, TGA), avaliando os 3
    frames de leitura. Posicoes de padding interrompem a ORF, para nao
    reportar ORFs gigantes feitas de adeninas fantasma.
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

    # padding tambem interrompe a ORF
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


def compute_fourier_period_3(one_hot: np.ndarray) -> np.ndarray:
    """
    Metodo de Voss (1992): FFT sobre os 4 canais binarios para detectar a
    periodicidade-3 (f = 1/3) tipica de regiao codante.

    A normalizacao EXCLUI o bin 0 (componente DC). O DC carrega a soma do
    canal — energia ordens de magnitude maior que as outras frequencias — e
    incluindo-o na media o score fica achatado e pouco discriminativo.
    """
    W = one_hot.shape[1]
    if W < 6:
        return np.zeros((one_hot.shape[0], 1), dtype=np.float32)

    X = np.fft.fft(one_hot, axis=1)
    S = np.sum(np.abs(X) ** 2, axis=2)            # (B, W)

    idx = int(round(W / 3.0))
    idx = min(max(idx, 1), W - 1)

    s_peak = S[:, idx]
    s_mean = np.mean(S[:, 1:], axis=1)            # <- sem o DC
    score = s_peak / (s_mean + 1e-8)
    return score.astype(np.float32).reshape(-1, 1)



def _single_scale(one_hot: np.ndarray, k: int) -> np.ndarray:
    """Bloco de features de UMA janela."""
    return np.concatenate([
        compute_gc_content(one_hot),                 # 1
        compute_kmer_frequencies(one_hot, k=k),      # 4^k
        compute_positional_asymmetry(one_hot),       # 12
        compute_max_orf_length(one_hot),             # 1
        compute_fourier_period_3(one_hot),           # 1
    ], axis=1).astype(np.float32)


def build_feature_matrix(one_hot: np.ndarray, k: int = 3,
                         two_scale: bool = True) -> np.ndarray:
    """
    Converte o tensor One-Hot 3D numa matriz tabular 2D para o Random Forest.

    two_scale=True concatena dois blocos:
      - escala EXTERNA: a janela inteira (contexto composicional amplo)
      - escala INTERNA: o quarto central da janela, i.e. W/2 bases
        centradas (assinatura local, onde o rotulo de fato mora)

    Medido: +1.9 pp de acuracia sobre escala unica. Sem isso o RF ve uma
    unica sacola de k-mers e nao consegue separar centro de periferia.

    Numero de colunas: (1 + 4^k + 12 + 1 + 1) por escala.
      k=3, two_scale=True  -> 79 * 2 = 158
      k=3, two_scale=False -> 79
      k=2, two_scale=True  -> 31 * 2 = 62
    """
    outer = _single_scale(one_hot, k)
    if not two_scale:
        return outer

    W = one_hot.shape[1]
    q = max(W // 4, 3)
    center = one_hot[:, W // 2 - q: W // 2 + q, :]
    inner = _single_scale(center, k)
    return np.concatenate([outer, inner], axis=1).astype(np.float32)


def get_feature_names(k: int = 3, two_scale: bool = True) -> list[str]:
    import itertools
    """Nomes das features, na mesma ordem de build_feature_matrix."""
    bases = ["A", "T", "G", "C"]
    kmers = ["".join(p) for p in itertools.product(bases, repeat=k)]

    def block(prefix):
        n = [f"{prefix}%GC"]
        n += [f"{prefix}{m}" for m in kmers]
        n += [f"{prefix}Fickett_Pos{p}_{b}" for p in (1, 2, 3) for b in bases]
        n += [f"{prefix}Max_ORF_Length", f"{prefix}Fourier_Period3"]
        return n

    names = block("out_")
    if two_scale:
        names += block("in_")
    return names


# =============================================================================
# 2. TREINAMENTO
# =============================================================================

def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    k: int = 3,
    two_scale: bool = True,
) -> RandomForestClassifier:
    """
    Treina o RandomForestClassifier.

    max_depth=None de proposito: medi que limitar a 15 custa 2.3-3.2 pp em
    TODOS os volumes de dados testados, entao nao e artefato de dataset
    pequeno.

    O esquema de features (k, two_scale) e gravado como atributo do modelo,
    para que a inferencia use exatamente o mesmo pipeline sem ter que
    adivinhar pelo numero de colunas.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,          # necessario para injecao sem vazamento
        random_state=random_state,
        n_jobs=-1,
    )

    print(f"  [RF] Treinando RandomForest ({n_estimators} arvores, "
          f"max_depth={max_depth}, features={X_train.shape[1]})...")
    model.fit(X_train, y_train)
    print(f"  [RF] Treinamento concluido. OOB Score: {model.oob_score_:.4f}")

    # --- metadados do esquema de features ---
    model.feature_k = k                                    # type: ignore[attr-defined]
    model.feature_two_scale = two_scale                    # type: ignore[attr-defined]
    model.feature_schema_version = FEATURE_SCHEMA_VERSION   # type: ignore[attr-defined]

    return model


def get_feature_config(rf) -> tuple[int, bool]:
    """
    Recupera (k, two_scale) do modelo. Modelos antigos, salvos antes dos
    metadados existirem, caem no esquema legado de escala unica — e o k e
    deduzido do numero de colunas apenas nesse caso.
    """
    if getattr(rf, "feature_schema_version", None) == FEATURE_SCHEMA_VERSION:
        return rf.feature_k, rf.feature_two_scale

    n = getattr(rf, "n_features_in_", 17)
    k = 3 if n >= 65 else 2
    print(f"  [RF] Aviso: modelo sem metadados de esquema ({n} colunas). "
          f"Assumindo legado k={k}, escala unica.")
    return k, False


# =============================================================================
# 3. AVALIACAO E DIAGNOSTICO
# =============================================================================

def evaluate_rf(model, X_val: np.ndarray, y_val: np.ndarray,
                *, verbose: bool = True) -> dict:
    """Avalia o modelo no conjunto de validacao."""
    y_pred = model.predict(X_val)

    acc       = accuracy_score(y_val, y_pred)
    f1_exon   = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
    f1_intron = f1_score(y_val, y_pred, pos_label=0, zero_division=0)
    f1_macro  = f1_score(y_val, y_pred, average="macro", zero_division=0)
    cm        = confusion_matrix(y_val, y_pred)
    report    = classification_report(
        y_val, y_pred,
        target_names=["Intron (0)", "Exon (1)"],
        zero_division=0,
    )

    # baseline trivial: prever sempre a classe majoritaria
    majority = int(round(float(np.mean(y_val))))
    triv = np.full_like(y_val, majority)
    triv_acc = accuracy_score(y_val, triv)
    triv_f1m = f1_score(y_val, triv, average="macro", zero_division=0)

    if verbose:
        _print_rf_results(acc, f1_exon, f1_intron, f1_macro, cm, report,
                          triv_acc, triv_f1m, majority)

    return {
        "accuracy": acc,
        "f1_exon": f1_exon,
        "f1_intron": f1_intron,
        "f1_macro": f1_macro,
        "cm": cm,
        "report": report,
        "trivial_accuracy": triv_acc,
        "trivial_f1_macro": triv_f1m,
    }


def _print_rf_results(acc, f1_exon, f1_intron, f1_macro, cm, report,
                      triv_acc, triv_f1m, majority):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  RANDOM FOREST — RESULTADOS DE VALIDACAO")
    print(sep)
    print(f"  {'Acuracia':<25}: {acc * 100:.2f}%")
    print(f"  {'F1-Score  Exon  (1)':<25}: {f1_exon * 100:.2f}%")
    print(f"  {'F1-Score  Intron (0)':<25}: {f1_intron * 100:.2f}%")
    print(f"  {'F1-Score  Macro':<25}: {f1_macro * 100:.2f}%")
    print(f"\n  -- referencia obrigatoria --")
    print(f"  {f'baseline sempre {majority}':<25}: acc {triv_acc * 100:.2f}%  "
          f"F1_macro {triv_f1m * 100:.2f}%")
    print(f"\n  Matriz de Confusao:")
    print(f"          Pred 0   Pred 1")
    print(f"  Real 0  {cm[0, 0]:>6}   {cm[0, 1]:>6}")
    print(f"  Real 1  {cm[1, 0]:>6}   {cm[1, 1]:>6}")
    print(f"\n  Relatorio Completo:\n{report}")
    print(sep)


def evaluate_rf_microscope(model, X_val: np.ndarray, y_val_center: np.ndarray,
                           y_val_window: Optional[np.ndarray] = None,
                           window_size: int = 120):
    """
    Diagnostico por tipo de janela. E este relatorio que revela quando o RF
    vai bem nas puras e mal nas mistas — o sintoma do descasamento de
    distribuicao que o filtro de especialista causava.
    """
    print("\n" + "=" * 60)
    print("  MICROSCOPIO LOCAL — ANALISE POR TIPO DE JANELA")
    print("=" * 60)

    if y_val_window is None:
        print("  [Aviso] 'y_window' ausente no .npz. Recrie os dados para ver "
              "a analise por janela pura/mista.")
        return

    y_pred = model.predict(X_val)

    idx_pure_intron = np.all(y_val_window == 0, axis=1)
    idx_pure_exon   = np.all(y_val_window == 1, axis=1)
    idx_mixed       = ~(idx_pure_intron | idx_pure_exon)

    def bucket(name, mask):
        n = int(np.sum(mask))
        if n == 0:
            print(f"  {name:<24}: 0 amostras.")
            return
        acc = accuracy_score(y_val_center[mask], y_pred[mask])
        pct = 100.0 * n / len(mask)
        print(f"  {name:<24}: {acc * 100:>6.2f}% de acuracia  "
              f"({n} janelas, {pct:.0f}% do total)")

    bucket("Janelas 100% Intron", idx_pure_intron)
    bucket("Janelas 100% Exon", idx_pure_exon)
    bucket("Janelas Mistas", idx_mixed)
    print("\n  Se 'Mistas' estiver MUITO abaixo das puras, o RF provavelmente")
    print("  nao viu composicao misturada no treino.")
    print("=" * 60)


# =============================================================================
# 4. PROBABILIDADE DO RF PARA A REDE
# =============================================================================

def rf_proba_oob(rf) -> np.ndarray:
    """
    P(Exon) Out-of-Bag para o conjunto em que o RF foi treinado.

    Use ISTO para gerar o canal do conjunto de treino da rede neural.
    `predict_proba` nesses mesmos dados e otimista (o RF praticamente
    memoriza o treino), e a rede aprenderia a confiar num sinal mais
    preciso do que ele sera na inferencia — vazamento, independente de a
    fusao ser early ou late.

    Linhas que por azar ficaram sem estimativa OOB voltam como 0.5 (neutro).
    """
    if not hasattr(rf, "oob_decision_function_"):
        raise AttributeError(
            "RF treinado sem oob_score=True; nao ha estimativa OOB disponivel."
        )
    oob = np.asarray(rf.oob_decision_function_, dtype=np.float64)
    p = oob[:, 1]
    return np.nan_to_num(p, nan=0.5).astype(np.float32)


def rf_proba(rf, one_hot: np.ndarray) -> np.ndarray:
    """
    P(Exon) por janela, para dados NAO vistos pelo RF (validacao/teste).
    Retorna (B,) — sem construir o tensor de 5 canais.
    """
    k, two_scale = get_feature_config(rf)
    X_tab = build_feature_matrix(one_hot, k=k, two_scale=two_scale)

    expected = getattr(rf, "n_features_in_", X_tab.shape[1])
    if X_tab.shape[1] != expected:
        raise ValueError(
            f"Esquema de features incompativel: gerei {X_tab.shape[1]} colunas, "
            f"o modelo espera {expected}. Retreine o RF (save_rf grava o "
            f"esquema junto a partir desta versao)."
        )

    return np.asarray(rf.predict_proba(X_tab))[:, 1].astype(np.float32)


def apply_rf_dropout(p_exon: np.ndarray, dropout_rate: float = 0.5,
                     neutral: float = 0.5, rng=None) -> np.ndarray:
    """
    Zera aleatoriamente o sinal do RF, levando-o ao valor neutro, para que a
    rede nao dependa exclusivamente dele. Use apenas no TREINO da rede.
    """
    rng = rng or np.random
    keep = rng.binomial(1, 1.0 - dropout_rate, size=p_exon.shape)
    return np.where(keep == 1, p_exon, neutral).astype(np.float32)


def inject_rf_proba(
    rf,
    one_hot: np.ndarray,
    rf_scale: float = 0.20,
    is_training_set: bool = False,
    apply_dropout: bool = False,
    dropout_rate: float = 0.5,
    oob_proba: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Tensor aumentado (B, W, 5) com P(Exon) replicada no 5o canal.

    is_training_set=True exige `oob_proba` (de rf_proba_oob), porque a
    ordem das linhas do OOB precisa casar com a ordem de `one_hot` — quem
    tem essa informacao e o chamador, nao esta funcao.

    Mantido para compatibilidade com early fusion e com o validation.py.
    Para late fusion, prefira `rf_proba` / `rf_proba_oob` direto: evita
    construir e depois fatiar um tensor B x W x 5 inteiro.
    """
    batch_size, window_size, _ = one_hot.shape

    if is_training_set:
        if oob_proba is None:
            raise ValueError(
                "is_training_set=True requer oob_proba=rf_proba_oob(rf), "
                "alinhado linha a linha com one_hot."
            )
        p_exon = np.asarray(oob_proba, dtype=np.float32)
        if len(p_exon) != batch_size:
            raise ValueError(
                f"oob_proba tem {len(p_exon)} linhas, one_hot tem {batch_size}."
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
# 5. PERSISTENCIA
# =============================================================================

def save_rf(rf, path: str) -> None:
    import joblib
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(rf, path)
    k, two_scale = get_feature_config(rf)
    print(f"  [RF] Modelo salvo em: {path}  (k={k}, two_scale={two_scale})")


def load_rf(path: str):
    import joblib
    if not os.path.exists(path):
        print(f"  [RF] Nenhum modelo RF em: {path} "
              f"(validacao prosseguira sem injecao)")
        return None
    rf = joblib.load(path)
    k, two_scale = get_feature_config(rf)
    print(f"  [RF] Modelo carregado de: {path}  (k={k}, two_scale={two_scale})")
    return rf


# =============================================================================
# 6. PIPELINE DE ALTO NIVEL
# =============================================================================

def run_rf_pipeline(
    mod2_train_path: str,
    mod2_val_path: str,
    *,
    k: int = 3,
    two_scale: bool = True,
) -> tuple[dict, RandomForestClassifier]:
    """
    Treina o RF em TODAS as janelas de treino e avalia em todas as de
    validacao. Nao ha mais filtro de janelas puras — ver nota 1 no topo.
    """
    print("  [RF] Carregando dados de treino...")
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

    print("  [RF] Carregando dados de validacao...")
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

    print(f"  [RF] Treino : {X_train_ohe.shape} | rotulos (centro): {y_train.shape}")
    print(f"  [RF] Val    : {X_val_ohe.shape} | rotulos (centro): {y_val.shape}")

    if y_train_window is not None:
        pure = (np.all(y_train_window == 0, axis=1) |
                np.all(y_train_window == 1, axis=1))
        print(f"  [RF] Composicao do treino: {100 * pure.mean():.0f}% puras, "
              f"{100 * (1 - pure.mean()):.0f}% mistas — TODAS serao usadas.")

    print(f"  [RF] Extraindo features (k={k}, two_scale={two_scale})...")
    X_train = build_feature_matrix(X_train_ohe, k=k, two_scale=two_scale)
    X_val = build_feature_matrix(X_val_ohe, k=k, two_scale=two_scale)
    print(f"  [RF] Feature matrix — treino: {X_train.shape} | val: {X_val.shape}")

    rf = train_rf(X_train, y_train, k=k, two_scale=two_scale)

    metrics = evaluate_rf(rf, X_val, y_val, verbose=True)
    evaluate_rf_microscope(rf, X_val, y_val_center=y_val,
                           y_val_window=y_val_window,
                           window_size=X_val_ohe.shape[1])

    print("\n" + "=" * 60)
    print("  RF DIAGNOSTICO DE DADOS E OVERFITTING")
    print("=" * 60)
    print(f"  treino: {X_train_ohe.shape} | % exon: {round(100 * y_train.mean(), 1)}%")
    print(f"  val   : {X_val_ohe.shape} | % exon: {round(100 * y_val.mean(), 1)}%")
    print(f"  OOB score do RF: {round(rf.oob_score_, 4)}")
    print(f"  F1_macro       : {round(metrics['f1_macro'], 4)}")
    print("=" * 60 + "\n")

    return metrics, rf


# =============================================================================
# 7. EXECUCAO DIRETA
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Testa o modulo rf_model.py de forma standalone."
    )
    parser.add_argument("--train", required=True, help="Caminho do .npz de treino.")
    parser.add_argument("--val", required=True, help="Caminho do .npz de validacao.")
    parser.add_argument("--k", type=int, default=3, choices=[2, 3, 4],
                        help="Tamanho do k-mer (default: 3).")
    parser.add_argument("--single-scale", action="store_true",
                        help="Desliga as features de duas escalas (nao recomendado).")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RF Standalone")
    print("=" * 60)
    metrics, rf = run_rf_pipeline(args.train, args.val,
                                  k=args.k, two_scale=not args.single_scale)
    print(f"\n  Acuracia Final : {metrics['accuracy'] * 100:.2f}%")
    print(f"  F1 Macro Final : {metrics['f1_macro'] * 100:.2f}%")
    print(f"  (baseline trivial: {metrics['trivial_f1_macro'] * 100:.2f}% F1 macro)")

    print("\n  Testando o caminho de inferencia com dados sinteticos...")
    W = 120
    dummy = np.eye(4)[np.random.randint(0, 4, (5, W))].astype(np.float32)
    p = rf_proba(rf, dummy)
    print(f"  rf_proba          -> {p.shape}  (esperado: (5,))")
    aug = inject_rf_proba(rf, dummy)
    print(f"  inject_rf_proba   -> {aug.shape}  (esperado: (5, {W}, 5))")
    print(f"  rf_proba_oob      -> {rf_proba_oob(rf).shape}  "
          f"(esperado: ({len(np.load(args.train)['y'])},))")

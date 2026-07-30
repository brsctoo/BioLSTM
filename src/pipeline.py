"""
Bi-LSTM Pipeline for Intron/Exon Identification.

Orchestrates the full workflow: GenBank search, GenBank pre-processing
(cropping, strand correction, deduplication, degenerate-base injection),
feature extraction (sliding window + one-hot encoding), model training,
and validation.

============================================================
USAGE
============================================================

    python pipeline.py [mode] [--injection-rate R] [--injection-mode M]
                        [--alpha A] [--illumina-mode IM] [--name N] [--seed S]

    If [mode] is omitted, an interactive menu is shown.

============================================================
POSITIONAL ARGUMENTS
============================================================

    mode    Operation mode (optional — interactive menu if omitted):
              search_data                Query NCBI and save a .gb file.
              train                      Pre-process the GenBank file and train the model.
              test                       Validate an already-trained model on the test split.
              full                       Run train followed by test in a single call.
              create_train_test_files    Only run pre-processing + featurization (no training).
              validate_specific_dataset  Validate the named model against an arbitrary .mod1 file.

============================================================
OPTIONAL FLAGS
============================================================

    --injection-rate  R   (float, default: 0.0)
        Scaling factor applied on top of the chosen injection strategy.
        0.0 = no degenerate bases injected.
        1.0 = full rate as defined by the selected mode.
        Note: 100% injection does not mean 100% of bases are replaced —
        see EFFECTIVE_SCALE in noise_injector.py.

    --injection-mode  M   (str, default: conditioned)
        Selects the injection strategy used during pre-processing:
          conditioned   Annotation-based injection. Introns get the highest
                        substitution probability, exon 3rd codon position gets
                        medium probability, and 1st/2nd positions get low
                        probability. Splice sites and start codons are protected
                        (rate = 0). This is the main experimental arm — it
                        deliberately correlates with the intron/exon label.
          uniform       Control arm. Every canonical base has the same
                        constant substitution probability, regardless of its
                        position in the gene structure. Degenerate bases carry
                        no information about intron/exon class. This is the
                        arm that isolates label leakage in 'conditioned'.
          illumina      Illumina error-profile arm. Substitution probability is
                        driven by local GC content and homopolymer run length,
                        mimicking systematic sequencing errors. Not rate-matched
                        against the other arms — use it as a robustness check,
                        not as the causal control.
          mixed         Weighted combination of 'conditioned' and 'illumina',
                        controlled by --alpha. Diluting the leakage does not
                        remove it.

    --alpha  A            (float, default: 0.3)
        Weight of the conditioned channel in 'mixed' mode, in [0, 1].
        0.0 = pure Illumina (no label leakage), 1.0 = pure conditioned
        (full leakage). Ignored for all other modes.

    --illumina-mode  IM    (str, default: realistic)
        Illumina calibration sub-mode, used by 'illumina' and 'mixed':
          realistic   Literature-calibrated rates (peak ~2%).
          stress      Same curve shape, scaled 0-100%, for robustness/
                      leakage diagnostics.

    --name  N             (str, default: actin_fungi)
        Experiment label. Used both to locate the input .gb file
        (../assets/genbank_data/{N}.gb) and to name all output files.
        Use this to run multiple experiments without overwriting results.

    --seed  S             (int, default: 123865)
        Global random seed (Python random, NumPy, TensorFlow). Makes
        injection noise, dataset balancing, and weight initialization
        reproducible. NOTE: the train/test split itself always uses a
        fixed seed (123865) inside genbank_reader.separate_train_test,
        independent of --seed, so different experiments stay comparable
        on the same held-out set.
"""

import os
import gc
import argparse
from datetime import datetime

import numpy as np
import genbank_searcher
import genbank_reader
import modeling
import rf_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- Genbank Search Configuration -----------
MAX_RECORDS = 1500
BATCH_SIZE = 50
MAX_PER_SPECIES = 50  # Força alta diversidade (mínimo de 50 espécies diferentes)
MAX_GENERAL = 500000    # Tamanho do "pool" aleatório que vamos baixar os IDs
MAX_HOUSEKEEPING = 0   # Ignorado na nova abordagem

QUERY_GENERAL = '"exon"[Feature key] AND "intron"[Feature key] AND biomol_genomic[PROP]'
QUERY_HOUSEKEEPING = '' # Não será mais utilizado
# -------------------------------------------------

DEFAULT_INJECTION_RATE = 0.0
DEFAULT_INJECTION_MODE = "conditioned"  # Options: conditioned | uniform | illumina | mixed
DEFAULT_NAME = "actin_fungi"
DEFAULT_SEED = 123865
DEFAULT_WINDOW_SIZE = 400
DEFAULT_EPOCHS = 100
DEFAULT_RF_SCALE = 0.15   # influência do RF na entrada do LSTM [0.0 = desligado, 1.0 = total]

GB_FILE_NAME = "all_proteins"
OUTPUT_FILE = os.path.join(BASE_DIR, f"../assets/genbank_data/{GB_FILE_NAME}.gb")


def get_output_paths(name):
    genbank_input = os.path.join(BASE_DIR, f"../assets/genbank_data/{name}")
    mod1 = os.path.join(BASE_DIR, f"../assets/processed_data/mod1/data_{name}")
    mod2 = os.path.join(BASE_DIR, f"../assets/processed_data/mod2/data_XY_{name}.npz")
    result = os.path.join(BASE_DIR, f"../assets/result/model_{name}_onehot.h5")
    rf_result = os.path.join(BASE_DIR, f"../assets/result/model_{name}_rf.joblib")
    return genbank_input, mod1, mod2, result, rf_result


def log_stage(msg):
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    print(f"{'='*60}\n")


def set_global_seed(seed):
    """
    Seed Python's random, NumPy, and TensorFlow so a run is reproducible
    end-to-end: injection noise, dataset balancing, and weight initialization
    all draw from the same seed.

    The train/test split (genbank_reader.separate_train_test) is deliberately
    NOT covered here — it hardcodes its own fixed seed so different experiments
    in a sweep stay comparable on the same held-out set regardless of --seed.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_injector_kwargs(injection_mode, alpha, illumina_mode):
    """Strategy-specific extra parameters, forwarded to noise_injector.py."""
    injector_kwargs = {}
    if injection_mode == "illumina":
        injector_kwargs["mode"] = illumina_mode
    elif injection_mode == "mixed":
        injector_kwargs["alpha"] = alpha
        injector_kwargs["illumina_mode"] = illumina_mode
    return injector_kwargs


def save_run_metadata(name, injection_rate, injection_mode, ratio_degenerate, injector_kwargs):
    """Persist the effective degeneration rate and run hyperparameters."""
    lines = [
        f"RATIO_DEGENERATE_NUCLEOTIDES = {ratio_degenerate}",
        f"injection_mode = {injection_mode}",
        f"injection_rate = {injection_rate}",
    ]
    for key, value in injector_kwargs.items():
        lines.append(f"{key} = {value}")

    txt_filepath = os.path.join(BASE_DIR, f"../assets/RATIO_{name}.txt")
    with open(txt_filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Run metadata saved to: {txt_filepath}")

def search_data_pipeline():
    query_to_print = QUERY_GENERAL.replace(' AND ', '\nAND ')
    query_housekeeping_to_print = QUERY_HOUSEKEEPING.replace(' AND ', '\nAND ')
    log_stage("SEARCH — Querying GenBank")
    print(f"GENERAL QUERY: \n {query_to_print}")
    print(f"HOUSEKEEPING QUERY: \n {query_housekeeping_to_print}")
    genbank_searcher.main(QUERY_GENERAL, QUERY_HOUSEKEEPING, MAX_RECORDS, BATCH_SIZE,
                           MAX_PER_SPECIES, MAX_GENERAL, MAX_HOUSEKEEPING, OUTPUT_FILE)
    log_stage("SEARCH — DONE.")

def create_train_test_files(injection_rate, injection_mode, name, window_size=DEFAULT_WINDOW_SIZE, **injector_kwargs):
    genbank_input, mod1, mod2, _, _ = get_output_paths(name)

    # Derive separate paths for the two gene-level splits
    mod2_train = mod2.replace(".npz", "_train.npz")
    mod2_val = mod2.replace(".npz", "_val.npz")

    log_stage(f"PRE-PROCESSING  |  rate={injection_rate}  mode={injection_mode}  name={name}")
    print("Input .gb file:", genbank_input + ".gb")
    print("Output mod1   :", mod1)
    genbank_reader.save_preprocessed_genbank_file(
        genbank_input, mod1, injection_rate, injection_mode, **injector_kwargs
    )
    gc.collect()
    log_stage("PRE-PROCESSING — DONE. Memory freed.")

    log_stage(f"FEATURIZATION with GENE-SPLIT (window_size={window_size}, no leakage, no undersampling)")
    print("Input mod1  :", mod1 + "_train.mod1")
    print("Output train:", mod2_train)
    print("Output val  :", mod2_val)

    # Split is performed at the GENE level before window generation.
    # No degenerate ratio is returned here; stats are printed inside the function.
    modeling.modeling_train_data_gene_split(
        mod1 + "_train.mod1",
        mod2_train,
        mod2_val,
        val_gene_fraction=0.2
    )
    gc.collect()
    log_stage("FEATURIZATION — DONE. Memory freed.")

    return mod2_train, mod2_val


def train_pipeline(injection_rate, injection_mode, name, epochs=DEFAULT_EPOCHS, window_size=DEFAULT_WINDOW_SIZE, rf_scale=DEFAULT_RF_SCALE, recreate_data=True, **injector_kwargs):
    _, _, mod2, result, rf_result = get_output_paths(name)
    mod2_train = mod2.replace(".npz", "_train.npz")
    mod2_val   = mod2.replace(".npz", "_val.npz")

    if recreate_data:
        create_train_test_files(injection_rate, injection_mode, name, window_size=window_size, **injector_kwargs)
    else:
        log_stage("PRE-PROCESSING/FEATURIZATION SKIPPED (recreate_data=False)")

    # ETAPA 1: Random Forest
    log_stage("RANDOM FOREST — Extração de features + Treinamento (pré-LSTM)")
    rf_metrics, trained_rf = rf_model.run_rf_pipeline(mod2_train, mod2_val)
    log_stage(
        f"RANDOM FOREST — DONE. "
        f"Acurácia: {rf_metrics['accuracy']*100:.2f}% | "
        f"F1 Éxon: {rf_metrics['f1_exon']*100:.2f}%"
    )
    gc.collect()

    # ETAPA 2: Injeção de Probabilidade RF
    log_stage(f"AUGMENTAÇÃO — Injetando P(Éxon) do RF como 5º canal (One-Hot → 5D, W={window_size})")

    mod2_train_aug = mod2.replace(".npz", "_aug_train.npz")
    mod2_val_aug   = mod2.replace(".npz", "_aug_val.npz")

    for src_path, dst_path, label in [
        (mod2_train, mod2_train_aug, "treino"),
        (mod2_val,   mod2_val_aug,   "validação"),
    ]:
        data = np.load(src_path)
        X_ohe = data["X"].astype(np.float32)   # (N, W, 4)
        y = data["y"]

        # --- LÓGICA DE INJEÇÃO CORRIGIDA ---
        if label == "treino":
            # Treino: Ativa matriz OOB (evita vazamento) e Dropout
            X_aug = rf_model.inject_rf_proba(
                trained_rf, X_ohe,
                rf_scale=rf_scale,
                is_training_set=True,
                apply_dropout=True,
                oob_proba=rf_model.rf_proba_oob(trained_rf)
            )
        else:
            # Validação: Usa probabilidade limpa e constante (sem Dropout e sem OOB)
            X_aug = rf_model.inject_rf_proba(
                trained_rf, X_ohe,
                rf_scale=rf_scale,
                is_training_set=False,
                apply_dropout=False
            )

        np.savez_compressed(dst_path, X=X_aug, y=y)
        print(f"  [AUG] {label}: {X_ohe.shape} → {X_aug.shape}  → salvo em {dst_path}")
    gc.collect()
    log_stage("AUGMENTAÇÃO — DONE. Tensores (W, 5) salvos.")

    # ETAPA 3: Bi-LSTM treinado sobre os tensores aumentados (W, 5)
    log_stage(f"TRAINING  (Bi-LSTM com entrada aumentada {window_size}×5, epochs={epochs})")
    print("Input train (aug):", mod2_train_aug)
    print("Input val   (aug):", mod2_val_aug)
    log_stage("TRAINING — Iniciando treinamento no Keras (Bi-LSTM)")
    import train_model
    train_model.train_model_gene_split(mod2_train_aug, mod2_val_aug, result, epochs=epochs)
    gc.collect()
    log_stage("TRAINING — DONE. Model saved. Memory freed.")

    # Salva o RF no disco para ser usado na fase de validação
    rf_model.save_rf(trained_rf, rf_result)


def validate_pipeline(name, threshold=0.50):
    import validation
    _, mod1, _, result, rf_result = get_output_paths(name)

    log_stage("VALIDATION — Loading model and test data")
    print("Model    :", result)
    print("RF model :", rf_result)
    print("Test data:", mod1 + "_test.mod1")
    trained_rf = rf_model.load_rf(rf_result)
    validation.validate_model(result, mod1 + "_test.mod1", trained_rf, threshold=threshold)
    log_stage("VALIDATION — DONE.")

def validate_specific_dataset(name):
    import validation
    _, _, _, result, rf_result = get_output_paths(name)
    specific_dataset = input("Enter the name of the dataset: ")
    specific_dataset = os.path.join(BASE_DIR, f"../assets/processed_data/mod1/{specific_dataset}")

    log_stage("VALIDATION (custom dataset) — Loading model and test data")
    print("Model    :", result)
    print("Test data:", specific_dataset + ".mod1")
    trained_rf = rf_model.load_rf(rf_result)
    validation.validate_model(result, specific_dataset + ".mod1", trained_rf)
    log_stage("VALIDATION — DONE.")

def main():
    parser = argparse.ArgumentParser(description="Bi-LSTM Pipeline for Intron/Exon Identification")

    # Positional argument: operation mode. Choices cover every action reachable
    # from the interactive menu, so a full sweep can be scripted without prompts.
    parser.add_argument("mode", nargs='?',
        choices=["search_data", "train", "test", "full",
                 "validate_specific_dataset", "create_train_test_files"],
        help="Operation mode (optional — interactive menu if omitted).")

    parser.add_argument("--injection-rate", type=float, default=DEFAULT_INJECTION_RATE,
        help="Degenerate nucleotide injection rate (e.g. 0.5 for 50%%). Default: 0.0")

    parser.add_argument("--injection-mode", type=str, default=DEFAULT_INJECTION_MODE,
        choices=["conditioned", "uniform", "illumina", "mixed"],
        help="Injection strategy: 'conditioned' (annotation-based, label-correlated), "
             "'uniform' (control arm, no label information), "
             "'illumina' (sequencing error-profile), "
             "'mixed' (weighted blend of conditioned + illumina). "
             f"Default: {DEFAULT_INJECTION_MODE}")

    parser.add_argument("--alpha", type=float, default=0.3,
        help="Weight of the conditioned channel in 'mixed' mode, in [0, 1]. "
             "0.0 = pure Illumina (no leakage), 1.0 = pure conditioned "
             "(full leakage). Ignored for other modes. Default: 0.3")

    parser.add_argument("--illumina-mode", type=str, default="realistic",
        choices=["realistic", "stress"],
        help="Illumina calibration sub-mode: 'realistic' (literature-calibrated "
             "rates, peak ~2%%) or 'stress' (same curve shape, scaled 0-100%%, "
             "for robustness/leakage diagnostics). Used by 'illumina' and "
             "'mixed'. Default: realistic")

    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
        help="Experiment name — used to find the input .gb file and to name "
             f"all output files (e.g. actin_fungi_rate50). Default: {DEFAULT_NAME}")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
        help="Global random seed for reproducibility (Python random, NumPy, "
             f"TensorFlow). Default: {DEFAULT_SEED}")

    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
        help="Tamanho da janela deslizante em nucleotídeos. Propagado para modeling.py, "
             f"lstm_model.py e rf_model.py. Default: {DEFAULT_WINDOW_SIZE}")

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
        help="Número máximo de épocas de treinamento do Bi-LSTM. "
             f"Early Stopping pode parar antes. Default: {DEFAULT_EPOCHS}")

    parser.add_argument("--rf-scale", type=float, default=DEFAULT_RF_SCALE,
        help="Fator de escala do canal P(Éxon) do RF injetado na entrada do LSTM. "
             "0.0 = RF desligado, 1.0 = influência total. "
             f"Default: {DEFAULT_RF_SCALE}")

    parser.add_argument("--threshold", type=float, default=0.50,
        help="Decision threshold for probability -> Intron(0)/Exon(1) class assignment. "
             "Default: 0.50")

    parser.add_argument("--skip-data-generation", action="store_true",
        help="Se ativada, o pipeline NÃO vai recriar os arquivos mod1/mod2 se usar 'train' ou 'full', "
             "aproveitando os arquivos que já existem (Default: Recria os arquivos sempre).")

    args = parser.parse_args()

    injection_rate = args.injection_rate
    injection_mode = args.injection_mode
    name = args.name
    seed = args.seed
    window_size = args.window_size
    epochs      = args.epochs
    rf_scale    = args.rf_scale
    threshold   = args.threshold
    recreate_data = not args.skip_data_generation

    injector_kwargs = build_injector_kwargs(injection_mode, args.alpha, args.illumina_mode)

    set_global_seed(seed)

    # Propaga window_size para todos os módulos que usam janelas
    modeling.set_window_size(window_size)
    if args.mode in ["train", "full"]:
        import lstm_model
        lstm_model.set_window_size(window_size)

    # If no mode was provided, show interactive menu
    if args.mode is None:
        print("\n" + "="*50)
        print("Bi-LSTM Pipeline for Intron/Exon Identification")
        print("="*50)
        print("\nSelect an option:")
        print("0 - Search data on GenBank")
        print("1 - Train (Train the model)")
        print("2 - Test (Validate the model)")
        print("3 - Full (Train and validate)")
        print("4 - Validate specific dataset")
        print("5 - Create train and test files")
        print("="*50)
        print(f"Injection rate: {injection_rate} (use --injection-rate to change)")
        print(f"Injection mode: {injection_mode} (use --injection-mode to change)")
        print(f"Experiment name: {name} (use --name to change)")
        print(f"Window size: {window_size} (use --window-size to change)")
        print(f"Epochs: {epochs} (use --epochs to change)")
        print(f"RF scale: {rf_scale} (use --rf-scale to change)")
        print(f"Threshold: {threshold} (use --threshold to change)")
        print(f"Seed: {seed} (use --seed to change)")

        choice = input("\nEnter the option number (0/1/2/3/4/5): ").strip()

        modes_map = {
            "0": "search_data",
            "1": "train",
            "2": "test",
            "3": "full",
            "4": "validate_specific_dataset",
            "5": "create_train_test_files",
        }
        args.mode = modes_map.get(choice)

        if args.mode is None:
            print("\n❌ Invalid option! Please use 0, 1, 2, 3, 4, or 5.")
            return

    if args.mode == "search_data":
        search_data_pipeline()
    elif args.mode == "train":
        train_pipeline(injection_rate, injection_mode, name, epochs=epochs, window_size=window_size, rf_scale=rf_scale, recreate_data=recreate_data, **injector_kwargs)
    elif args.mode == "test":
        validate_pipeline(name, threshold=threshold)
    elif args.mode == "full":
        train_pipeline(injection_rate, injection_mode, name, epochs=epochs, window_size=window_size, rf_scale=rf_scale, recreate_data=recreate_data, **injector_kwargs)
        gc.collect()
        log_stage("TRANSITION — Training complete. Freeing memory before validation.")
        validate_pipeline(name, threshold=threshold)
    elif args.mode == "validate_specific_dataset":
        validate_specific_dataset(name)
    elif args.mode == "create_train_test_files":
        create_train_test_files(injection_rate, injection_mode, name, window_size=window_size, **injector_kwargs)


if __name__ == "__main__":
    main()

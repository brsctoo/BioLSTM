"""
Get the data_XY from modeling.py, train the model, and save it for later use in validation.py.

The data_XY is a list of tuples: [(X, Y), ...], where X is the input sequence - window - (list of integers) and Y is the corresponding label (0 or 1).
"""

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from lstm_model import create_model, BATCH_SIZE


def train_model_gene_split(XY_train_filepath, XY_val_filepath, result_filepath_output, epochs=100):
    """
    CORRECTED VERSION: trains using two separate .npz files produced by
    modeling.modeling_train_data_gene_split (gene-level split).

    Key differences from train_model:
      - Accepts pre-split train and validation arrays instead of a single
        shuffled pool, so genes in X_val NEVER appear in X_train.
      - Uses validation_data=(X_val, y_val) — NOT validation_split —
        which would re-contaminate from the same pool.
      - Computes class_weight from the real (unbalanced) distribution so
        BinaryFocalCrossentropy can focus on the minority class without
        discarding majority-class samples.

    SOBRE OS PESOS (importante para o relatorio)
    --------------------------------------------
    Peso de classe e ferramenta de TREINO: serve para moldar o gradiente e
    impedir que a classe majoritaria domine o aprendizado.

    Na VALIDACAO o peso de classe NAO e aplicado. Se fosse, o Keras
    calcularia acuracia/AUC/precisao/recall sobre uma distribuicao
    reponderada para 50/50 — ou seja, reportaria acuracia BALANCEADA sob o
    nome de `val_accuracy`. Esses numeros nao seriam comparaveis com o
    Random Forest (medido sem peso) nem com o baseline trivial.

    Na validacao o peso e apenas a MASCARA: 1.0 para posicao anotada, 0.0
    para posicao com rotulo -1 (padding de borda da janela / regiao sem
    anotacao). Sem essa mascara, as posicoes -1 viram 0 (intron) pelo
    y_val_clean e entram nas metricas como introns reais — cerca de 6.9%
    dos rotulos nas janelas seq2seq.

    Args:
        XY_train_filepath    : path to the training .npz file.
        XY_val_filepath      : path to the validation .npz file.
        result_filepath_output : path where the trained model (.h5) will be saved.

    Returns:
        history : Keras History object from model.fit.
    """
    lstm_model = create_model()

    # Load pre-split datasets
    print("Loading training data...")
    train_data = np.load(XY_train_filepath, allow_pickle=True)
    X_train = np.array(train_data['X'], dtype=np.float32)
    y_train = np.array(train_data['y'], dtype=np.int32)

    print("Loading validation data...")
    val_data = np.load(XY_val_filepath, allow_pickle=True)
    X_val = np.array(val_data['X'], dtype=np.float32)
    y_val = np.array(val_data['y'], dtype=np.int32)

    print(f"\n--- REAL dataset distribution (Seq2Seq) ---")
    count_0 = np.sum(y_train == 0)  # introns
    count_1 = np.sum(y_train == 1)  # exons
    total_valid = count_0 + count_1

    print(f"Train -> Exons: {count_1:,} | Introns: {count_0:,} | "
          f"Exon proportion: {count_1/total_valid*100:.1f}%")

    weight_0 = total_valid / (2.0 * max(1, count_0))
    weight_1 = total_valid / (2.0 * max(1, count_1))
    print(f"Train class weights -> intron: {weight_0:.4f} | exon: {weight_1:.4f}")

    v_count_0 = np.sum(y_val == 0)
    v_count_1 = np.sum(y_val == 1)
    v_masked = np.sum(y_val == -1)
    v_total = v_count_0 + v_count_1
    print(f"Val   -> Exons: {v_count_1:,} | Introns: {v_count_0:,} | "
          f"Exon proportion: {v_count_1/max(1, v_total)*100:.1f}%")
    print(f"Val   -> posicoes mascaradas (rotulo -1): {v_masked:,} "
          f"({100*v_masked/max(1, y_val.size):.1f}% dos rotulos)")
    print("  Metricas de validacao NAO usam peso de classe -> sao brutas,")
    print("  comparaveis com o Random Forest e com o baseline trivial.")
    print("----------------------------------------------------------\n")

    # Separa o canal do RF (5º canal) do canal de DNA (4 canais)
    # NOTA: pega o centro da janela, nao a posicao 0. Hoje o inject_rf_proba
    # replica o mesmo valor nas W posicoes, entao tanto faz — mas se o canal
    # virar um PERFIL posicional, a posicao 0 seria a borda da janela.
    mid = X_train.shape[1] // 2

    if X_train.shape[-1] == 5:
        rf_train = X_train[:, mid, 4:5]
        X_train_dna = X_train[:, :, :4]
    else:
        rf_train = np.zeros((X_train.shape[0], 1), dtype=np.float32)
        X_train_dna = X_train[:, :, :4]

    if X_val.shape[-1] == 5:
        rf_val = X_val[:, mid, 4:5]
        X_val_dna = X_val[:, :, :4]
    else:
        rf_val = np.zeros((X_val.shape[0], 1), dtype=np.float32)
        X_val_dna = X_val[:, :, :4]

    train_inputs = {'dna_input': X_train_dna, 'rf_input': rf_train}
    val_inputs = {'dna_input': X_val_dna, 'rf_input': rf_val}

    # --- TREINO: mascara + peso de classe (molda o gradiente) ---
    sample_weights_arr = np.zeros_like(y_train, dtype=np.float32)
    sample_weights_arr[y_train == 0] = weight_0
    sample_weights_arr[y_train == 1] = weight_1
    # y_train == -1 fica com peso 0 (ignorado)

    # --- VALIDACAO: SOMENTE mascara, sem peso de classe ---
    # 1.0 onde ha rotulo real, 0.0 nas posicoes -1. Assim val_accuracy,
    # val_auc, val_precision e val_recall saem na distribuicao verdadeira.
    val_sample_weights_arr = (y_val >= 0).astype(np.float32)

    # Evitar que a loss quebre com valores -1
    # (essas posicoes ja estao com peso 0, entao o valor colocado aqui e
    #  irrelevante — mas precisa ser um rotulo binario valido)
    y_train_clean = np.where(y_train == -1, 0, y_train)
    y_val_clean = np.where(y_val == -1, 0, y_val)

    # Adicionar dimensão final (N, WINDOW_SIZE, 1)
    y_train_clean = np.expand_dims(y_train_clean, -1)
    y_val_clean = np.expand_dims(y_val_clean, -1)

    train_sample_weights = {'final_out': sample_weights_arr, 'aux_lstm_out': sample_weights_arr}
    val_sample_weights = {'final_out': val_sample_weights_arr, 'aux_lstm_out': val_sample_weights_arr}

    train_targets = {'final_out': y_train_clean, 'aux_lstm_out': y_train_clean}
    val_targets = {'final_out': y_val_clean, 'aux_lstm_out': y_val_clean}

    # Configuração de Callbacks: ReduceLROnPlateau + EarlyStopping (monitorando val_auc)
    # Obs: como a métrica é um dicionário no compile com o nome final_out, o Keras
    # vai registrá-la como 'val_final_out_auc' durante o fit, então usaremos esse nome.
    #
    # Ancorados na AUC e nao na val_loss de proposito: a entropia cruzada pune
    # CONFIANCA, nao erro de decisao. Conforme o modelo treina, as probabilidades
    # migram para os extremos e a val_loss sobe mesmo quando AUC, acuracia e F1
    # continuam melhorando. Parar pela val_loss abandonaria o treino cedo demais.
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_final_out_auc',
        mode='max',
        factor=0.5,
        patience=6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_final_out_auc',
        mode='max',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # Train using validation_data with fully separated gene set
    history = lstm_model.fit(
        train_inputs,
        train_targets,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(val_inputs, val_targets, val_sample_weights), # type: ignore
        sample_weight=train_sample_weights,
        callbacks=[lr_scheduler, early_stop],
        verbose=2  # type: ignore
    )

    print("\nModel training completed.\n")

    # Avaliacao final COM a mascara. Sem sample_weight, as posicoes -1
    # (convertidas para 0 pelo y_val_clean) entrariam como introns reais e
    # contaminariam o numero — ~6.9% dos rotulos nas janelas seq2seq.
    test_results = lstm_model.evaluate(
        val_inputs,
        val_targets,
        sample_weight=val_sample_weights,  # type: ignore
        verbose=2,
        return_dict=True
    )

    if isinstance(test_results, dict):
        loss_val = test_results.get('loss', 0.0)
        acc_val = test_results.get('final_out_accuracy', test_results.get('accuracy', 0.0))
        auc_val = test_results.get('final_out_auc', 0.0)
        prec_val = test_results.get('final_out_precision', 0.0)
        rec_val = test_results.get('final_out_recall', 0.0)
    else:
        loss_val = test_results[0]
        acc_val = test_results[3]
        auc_val = prec_val = rec_val = 0.0

    f1_val = (2 * prec_val * rec_val / (prec_val + rec_val)) if (prec_val + rec_val) else 0.0

    print(f'\nValidation results (mascarado, sem peso de classe)')
    print(f'  Total Loss : {loss_val:.4f}')
    print(f'  Accuracy   : {100*acc_val:.2f}%')
    print(f'  AUC        : {auc_val:.4f}')
    print(f'  Precision  : {100*prec_val:.2f}%')
    print(f'  Recall     : {100*rec_val:.2f}%')
    print(f'  F1 (exon)  : {100*f1_val:.2f}%')

    # Baseline trivial no MESMO conjunto, para referencia obrigatoria.
    y_flat = y_val[y_val >= 0]
    if y_flat.size:
        majority = int(round(float(y_flat.mean())))
        triv_acc = float((y_flat == majority).mean())
        tp = float(np.sum((y_flat == 1) & (majority == 1)))
        fp = float(np.sum((y_flat == 0) & (majority == 1)))
        fn = float(np.sum((y_flat == 1) & (majority == 0)))
        triv_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        print(f'  -- baseline "sempre {majority}": acc {100*triv_acc:.2f}% | '
              f'F1 exon {100*triv_f1:.2f}%')
    print("")

    lstm_model.save(result_filepath_output)
    print(f"Model saved to: {result_filepath_output}")
    print(" ")
    print(" ")
    print("History:", history)
    return history

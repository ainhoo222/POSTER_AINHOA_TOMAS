
import sys
import json
import random

import numpy as np
import pandas as pd
import torch

from src_individual.config_individual import (
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
    MODELS_DIR,
    MODELS_INDIVIDUAL_DIR,
    RESULTS_INDIVIDUAL_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
    SEED,
    EXPECTED_NUM_CLASSES
)

from src.dataset import crear_dataloader
from src.preparacion import cargar_vocabulario
from src.evaluacion import evaluar_modelo

from src_individual.modelo_cnn import CNNClasificador


def fijar_semilla(seed):
    """Fija las semillas para facilitar la reproducibilidad."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def guardar_resultados(
    seed,
    metricas,
    resultados_clases,
    mejor_epoca,
    historial,
    id_to_label
):
    """Guarda las métricas y el historial del entrenamiento individual."""

    RESULTS_INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

    nombre = f"word2vec_cnn_seed{seed}"

    # 1. Guardar métricas generales.
    datos = {
        "experimento": "Word2Vec + CNN",
        "seed": seed,
        "mejor_epoca": mejor_epoca,
        **{
            clave: float(valor)
            for clave, valor in metricas.items()
        },
        "historial": historial
    }

    ruta_json = RESULTS_INDIVIDUAL_DIR / f"{nombre}.json"

    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(
            datos,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    # 2. Guardar resultados individuales por clase.
    numero_clases = len(resultados_clases["f1"])

    tabla = pd.DataFrame({
        "class_id": np.arange(numero_clases),
        "label": [
            id_to_label[i]
            for i in range(numero_clases)
        ],
        "precision": resultados_clases["precision"],
        "recall": resultados_clases["recall"],
        "f1": resultados_clases["f1"],
        "support": resultados_clases["support"],
        "train_count": resultados_clases["train_count"]
    })

    tabla.to_csv(
        RESULTS_INDIVIDUAL_DIR / f"{nombre}_clases.csv",
        index=False
    )

    # 3. Guardar historial de entrenamiento.
    pd.DataFrame(historial).to_csv(
        RESULTS_INDIVIDUAL_DIR / f"{nombre}_historial.csv",
        index=False
    )

    print(
        f"\nResultados guardados en: "
        f"{RESULTS_INDIVIDUAL_DIR}"
    )


def entrenar(seed=SEED):

    print("=" * 55)
    print("EXPERIMENTO: WORD2VEC + CNN")
    print(f"SEMILLA: {seed}")
    print("=" * 55)

    fijar_semilla(seed)

    # 1. Cargar vocabulario y frecuencias de las clases.
    print("\nCargando vocabulario...")

    _, label_to_id = cargar_vocabulario()

    num_classes = len(label_to_id)

    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Se esperaban {EXPECTED_NUM_CLASSES} clases, "
            f"pero se han encontrado {num_classes}."
        )

    id_to_label = {
        indice: etiqueta
        for etiqueta, indice in label_to_id.items()
    }

    conteos_clases = np.load(
        MODELS_DIR / "class_counts.npy"
    )

    if len(conteos_clases) != num_classes:
        raise ValueError(
            "El número de frecuencias no coincide "
            "con el número de clases."
        )

    print(f"Número de clases: {num_classes}")
    print(f"Dispositivo: {DEVICE}")

    # 2. Crear los DataLoaders.
    print("\nPreparando datos...")

    train_loader = crear_dataloader(
        ruta=TRAIN_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed
    )

    valid_loader = crear_dataloader(
        ruta=VALID_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = crear_dataloader(
        ruta=TEST_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("DataLoaders preparados correctamente.")

    # 3. Crear modelo y optimizador.
    print("\nCreando modelo CNN...")

    modelo = CNNClasificador(
        num_classes=num_classes
    ).to(DEVICE)

    criterio = torch.nn.CrossEntropyLoss()

    parametros_entrenables = [
        parametro
        for parametro in modelo.parameters()
        if parametro.requires_grad
    ]

    optimizador = torch.optim.AdamW(
        parametros_entrenables,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print("Modelo y optimizador preparados.")

    # 4. Preparar el guardado del mejor modelo.
    MODELS_INDIVIDUAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    nombre = f"word2vec_cnn_seed{seed}"

    ruta_modelo = (
        MODELS_INDIVIDUAL_DIR /
        f"{nombre}_best.pt"
    )

    mejor_macro_f1 = -1.0
    mejor_epoca = 0
    paciencia = 0

    historial = []

    # 5. Entrenamiento.
    for epoca in range(MAX_EPOCHS):

        print(f"\n{'-' * 45}")
        print(f"ÉPOCA {epoca + 1}/{MAX_EPOCHS}")
        print("-" * 45)

        modelo.train()

        # Cambiar la semilla de la aleatorización de los datos.
        train_loader.dataset.set_epoch(epoca)

        perdida_total = 0.0
        ejemplos = 0

        for numero_lote, lote in enumerate(
            train_loader,
            start=1
        ):

            input_ids, lengths, target_positions, labels = lote

            input_ids = input_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            target_positions = target_positions.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizador.zero_grad(set_to_none=True)

            logits = modelo(
                input_ids,
                lengths,
                target_positions
            )

            loss = criterio(logits, labels)

            loss.backward()
            optimizador.step()

            perdida_total += loss.item() * len(labels)
            ejemplos += len(labels)

            if numero_lote % 100 == 0:
                print(
                    f"Lote {numero_lote} | "
                    f"Pérdida: {loss.item():.4f}",
                    flush=True
                )

        if ejemplos == 0:
            raise ValueError(
                "No se han procesado ejemplos de entrenamiento."
            )

        perdida_media = perdida_total / ejemplos

        print(
            f"\nPérdida media de entrenamiento: "
            f"{perdida_media:.4f}"
        )

        # 6. Evaluación sobre validación.
        print("\nEvaluando validación...")

        metricas_valid, _ = evaluar_modelo(
            modelo=modelo,
            loader=valid_loader,
            num_classes=num_classes,
            conteos_clases=conteos_clases,
            device=DEVICE
        )

        for nombre_metrica, valor in metricas_valid.items():
            print(f"{nombre_metrica}: {valor:.4f}")

        historial.append({
            "epoca": epoca + 1,
            "perdida_train": perdida_media,
            "accuracy_valid": float(metricas_valid["accuracy"]),
            "macro_f1_valid": float(metricas_valid["macro_f1"]),
            "weighted_f1_valid": float(metricas_valid["weighted_f1"]),
            "kappa_valid": float(metricas_valid["kappa"]),
            "minority_f1_valid": float(metricas_valid["minority_f1"])
        })

        # 7. Guardar el mejor modelo según Macro-F1.
        macro_f1_actual = metricas_valid["macro_f1"]

        if macro_f1_actual > mejor_macro_f1:

            mejor_macro_f1 = macro_f1_actual
            mejor_epoca = epoca + 1
            paciencia = 0

            torch.save({
                "model_state_dict": modelo.state_dict(),
                "experimento": "Word2Vec + CNN",
                "seed": seed,
                "epoca": mejor_epoca,
                "macro_f1_valid": float(mejor_macro_f1)
            }, ruta_modelo)

            print(
                f"\n¡Nuevo mejor modelo! "
                f"Macro-F1: {mejor_macro_f1:.4f}"
            )

        else:
            paciencia += 1
            print(f"Sin mejora. Paciencia: {paciencia}")

        # 8. Early stopping.
        if paciencia >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping activado.")
            break

    if mejor_epoca == 0:
        raise RuntimeError(
            "No se ha podido guardar ningún modelo válido."
        )

    # 9. Recuperar el mejor modelo.
    print("\nCargando el mejor modelo...")

    checkpoint = torch.load(
        ruta_modelo,
        map_location=DEVICE,
        weights_only=False
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # 10. Evaluación final sobre test.
    print("\nEvaluando conjunto de test...")

    metricas_test, resultados_clases = evaluar_modelo(
        modelo=modelo,
        loader=test_loader,
        num_classes=num_classes,
        conteos_clases=conteos_clases,
        device=DEVICE
    )

    print("\n" + "=" * 55)
    print("RESULTADOS FINALES: WORD2VEC + CNN")
    print("=" * 55)

    for nombre_metrica, valor in metricas_test.items():
        print(f"{nombre_metrica}: {valor:.4f}")

    print(f"Mejor época: {mejor_epoca}")

    # 11. Guardar los resultados.
    guardar_resultados(
        seed=seed,
        metricas=metricas_test,
        resultados_clases=resultados_clases,
        mejor_epoca=mejor_epoca,
        historial=historial,
        id_to_label=id_to_label
    )

    print("\nEntrenamiento finalizado correctamente.")

    return metricas_test


if __name__ == "__main__":

    seed = int(sys.argv[1]) if len(sys.argv) >= 2 else SEED

    entrenar(seed)

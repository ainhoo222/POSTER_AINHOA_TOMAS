
import sys
import json
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import (
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    DEVICE,
    RESULTS_DIR,
    MODELS_DIR
)

from src.dataset import crear_dataloader
from src.modelo import BiLSTMClasificador
from src.preparacion import cargar_vocabulario
from src.evaluacion import evaluar_modelo


def fijar_semilla(seed):
    """Fija las semillas para facilitar la reproducibilidad."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def crear_modelo(experimento, num_classes):
    """Construye el modelo correspondiente al experimento."""

    fine_tune = experimento == "finetuning"
    attention = experimento == "attention"

    modelo = BiLSTMClasificador(
        num_classes=num_classes,
        fine_tune_embeddings=fine_tune,
        attention=attention
    )

    return modelo.to(DEVICE)


def obtener_criterio(experimento):
    """Selecciona la función de pérdida."""

    if experimento == "weighted_loss":

        pesos = np.load(
            MODELS_DIR / "class_weights.npy"
        )

        pesos = torch.tensor(
            pesos,
            dtype=torch.float32,
            device=DEVICE
        )

        return torch.nn.CrossEntropyLoss(weight=pesos)

    return torch.nn.CrossEntropyLoss()


def guardar_resultados(
    experimento,
    seed,
    metricas,
    resultados_clases,
    mejor_epoca,
    historial
):
    """Guarda las métricas generales y los resultados por clase."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nombre = f"{experimento}_seed{seed}"

    # Métricas generales
    datos = {
        "experimento": experimento,
        "seed": seed,
        "mejor_epoca": mejor_epoca,
        **{clave: float(valor) for clave, valor in metricas.items()},
        "historial": historial
    }

    ruta_json = RESULTS_DIR / f"{nombre}.json"

    with open(ruta_json, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, indent=4, ensure_ascii=False)

    # Métricas individuales por clase
    with open(
        MODELS_DIR / "vocabulario_subconjunto.pkl",
        "rb"
    ) as archivo:
        vocabulario = pickle.load(archivo)

    id_to_label = {
        indice: etiqueta
        for etiqueta, indice in vocabulario["label_to_id"].items()
    }

    tabla = pd.DataFrame({
        "class_id": np.arange(len(resultados_clases["f1"])),
        "label": [
            id_to_label[i]
            for i in range(len(resultados_clases["f1"]))
        ],
        "precision": resultados_clases["precision"],
        "recall": resultados_clases["recall"],
        "f1": resultados_clases["f1"],
        "support": resultados_clases["support"],
        "train_count": resultados_clases["train_count"]
    })

    tabla.to_csv(
        RESULTS_DIR / f"{nombre}_clases.csv",
        index=False
    )

    # Historial de entrenamiento
    pd.DataFrame(historial).to_csv(
        RESULTS_DIR / f"{nombre}_historial.csv",
        index=False
    )

    print(f"\nResultados guardados en: {RESULTS_DIR}")


def entrenar(experimento, seed=42):

    experimentos_validos = [
        "baseline",
        "finetuning",
        "attention",
        "weighted_loss"
    ]

    if experimento not in experimentos_validos:
        raise ValueError(
            f"Experimento no válido: {experimento}. "
            f"Opciones: {experimentos_validos}"
        )

    print("=" * 55)
    print(f"EXPERIMENTO: {experimento.upper()}")
    print(f"SEMILLA: {seed}")
    print("=" * 55)

    fijar_semilla(seed)

    # 1. Cargar vocabulario y frecuencias
    _, label_to_id = cargar_vocabulario()

    num_classes = len(label_to_id)

    conteos_clases = np.load(
        MODELS_DIR / "class_counts.npy"
    )

    if len(conteos_clases) != num_classes:
        raise ValueError(
            "El número de pesos de clase no coincide "
            "con el número de etiquetas."
        )

    print(f"Número de clases: {num_classes}")
    print(f"Dispositivo: {DEVICE}")

    # 2. Crear DataLoaders
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

    # 3. Crear modelo y optimizador
    modelo = crear_modelo(experimento, num_classes)

    criterio = obtener_criterio(experimento)

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

    # 4. Preparar guardado del mejor modelo
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    nombre = f"{experimento}_seed{seed}"

    ruta_modelo = RESULTS_DIR / f"{nombre}_best.pt"

    mejor_macro_f1 = -1.0
    mejor_epoca = 0
    paciencia = 0

    historial = []

    # 5. Entrenamiento
    for epoca in range(MAX_EPOCHS):

        print(f"\n{'-' * 45}")
        print(f"ÉPOCA {epoca + 1}/{MAX_EPOCHS}")
        print("-" * 45)

        modelo.train()

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

        perdida_media = perdida_total / ejemplos

        print(f"\nPérdida media de entrenamiento: {perdida_media:.4f}")

        # 6. Evaluación sobre validación
        print("\nEvaluando validación...")

        metricas_valid, _ = evaluar_modelo(
            modelo=modelo,
            loader=valid_loader,
            num_classes=num_classes,
            conteos_clases=conteos_clases,
            device=DEVICE
        )

        print(f"Accuracy: {metricas_valid['accuracy']:.4f}")
        print(f"Macro-F1: {metricas_valid['macro_f1']:.4f}")
        print(f"Weighted-F1: {metricas_valid['weighted_f1']:.4f}")
        print(f"Kappa: {metricas_valid['kappa']:.4f}")
        print(f"F1 minoritarias: {metricas_valid['minority_f1']:.4f}")

        historial.append({
            "epoca": epoca + 1,
            "perdida_train": perdida_media,
            "accuracy_valid": float(metricas_valid["accuracy"]),
            "macro_f1_valid": float(metricas_valid["macro_f1"]),
            "weighted_f1_valid": float(metricas_valid["weighted_f1"]),
            "kappa_valid": float(metricas_valid["kappa"]),
            "minority_f1_valid": float(metricas_valid["minority_f1"])
        })

        # 7. Guardar el mejor modelo
        macro_f1_actual = metricas_valid["macro_f1"]

        if macro_f1_actual > mejor_macro_f1:

            mejor_macro_f1 = macro_f1_actual
            mejor_epoca = epoca + 1
            paciencia = 0

            torch.save({
                "model_state_dict": modelo.state_dict(),
                "experimento": experimento,
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

        if paciencia >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping activado.")
            break

    # 8. Recuperar el mejor modelo
    print("\nCargando el mejor modelo...")

    checkpoint = torch.load(
        ruta_modelo,
        map_location=DEVICE,
        weights_only=False
    )

    modelo.load_state_dict(checkpoint["model_state_dict"])

    # 9. Evaluación final sobre test
    print("\nEvaluando conjunto de test...")

    metricas_test, resultados_clases = evaluar_modelo(
        modelo=modelo,
        loader=test_loader,
        num_classes=num_classes,
        conteos_clases=conteos_clases,
        device=DEVICE
    )

    print("\n" + "=" * 55)
    print("RESULTADOS FINALES")
    print("=" * 55)

    for nombre_metrica, valor in metricas_test.items():
        print(f"{nombre_metrica}: {valor:.4f}")

    print(f"Mejor época: {mejor_epoca}")

    # 10. Guardar resultados
    guardar_resultados(
        experimento=experimento,
        seed=seed,
        metricas=metricas_test,
        resultados_clases=resultados_clases,
        mejor_epoca=mejor_epoca,
        historial=historial
    )

    print("\nEntrenamiento finalizado correctamente.")

    return metricas_test


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Uso: python -m src.entrenamiento <experimento> [semilla]")
        print("Experimentos: baseline, finetuning, attention, weighted_loss")
        sys.exit(1)

    experimento = sys.argv[1]

    seed = int(sys.argv[2]) if len(sys.argv) >= 3 else 42

    entrenar(experimento, seed)

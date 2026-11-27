
import json
import torch

from src.config import (
    TEST_PATH,
    BATCH_SIZE,
    DEVICE,
    RESULTS_DIR,
    MODELS_DIR
)

from src.dataset import crear_dataloader
from src.modelo import BiLSTMClasificador
from src.preparacion import cargar_vocabulario
from src.evaluacion import evaluar_modelo
from src.entrenamiento import guardar_resultados


def main():

    experimento = "baseline"
    seed = 42

    _, label_to_id = cargar_vocabulario()
    num_classes = len(label_to_id)

    conteos_clases = __import__(
        "numpy"
    ).load(MODELS_DIR / "class_counts.npy")

    print("\nCOMPROBACIÓN DE CLASES MINORITARIAS")
    print("Número de clases:", len(conteos_clases))
    print("Frecuencia mínima:", conteos_clases.min())
    print("Frecuencia máxima:", conteos_clases.max())
    print("Clases con 20 ejemplos o menos:", (conteos_clases <= 20).sum())
    print("Clases con 10 ejemplos o menos:", (conteos_clases <= 10).sum())

    modelo = BiLSTMClasificador(
        num_classes=num_classes,
        fine_tune_embeddings=False,
        attention=False
    ).to(DEVICE)

    ruta_modelo = RESULTS_DIR / "baseline_seed42_best.pt"

    checkpoint = torch.load(
        ruta_modelo,
        map_location=DEVICE,
        weights_only=False
    )

    modelo.load_state_dict(checkpoint["model_state_dict"])

    loader = crear_dataloader(
        ruta=TEST_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    metricas, resultados_clases = evaluar_modelo(
        modelo=modelo,
        loader=loader,
        num_classes=num_classes,
        conteos_clases=conteos_clases,
        device=DEVICE
    )

    ruta_json = RESULTS_DIR / "baseline_seed42.json"

    with open(ruta_json, "r", encoding="utf-8") as archivo:
        anterior = json.load(archivo)

    guardar_resultados(
        experimento=experimento,
        seed=seed,
        metricas=metricas,
        resultados_clases=resultados_clases,
        mejor_epoca=checkpoint["epoca"],
        historial=anterior["historial"]
    )

    print("\nMétricas corregidas:")
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.4f}")


if __name__ == "__main__":
    main()

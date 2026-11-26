
import numpy as np

from tqdm import tqdm

from src.config import (
    TRAIN_PATH,
    LABEL_COLUMN,
    MODELS_DIR
)

from src.datos import leer_lotes
from src.preparacion import cargar_vocabulario


def calcular_pesos_clases():

    # Cargar el diccionario de etiquetas.
    _, label_to_id = cargar_vocabulario()

    num_classes = len(label_to_id)

    # Contador de ejemplos por clase.
    conteos = np.zeros(num_classes, dtype=np.int64)

    registros = 0

    print("Calculando frecuencias de las clases...")

    # Leer únicamente la columna LABEL del entrenamiento.
    for lote in tqdm(
        leer_lotes(
            TRAIN_PATH,
            batch_size=10000,
            columnas=[LABEL_COLUMN]
        ),
        desc="Procesando etiquetas"
    ):

        etiquetas = lote[LABEL_COLUMN]

        # Convertir etiquetas a identificadores.
        ids = np.fromiter(
            (label_to_id[etiqueta] for etiqueta in etiquetas),
            dtype=np.int64,
            count=len(etiquetas)
        )

        # Contabilizar las frecuencias.
        conteos += np.bincount(
            ids,
            minlength=num_classes
        )

        registros += len(etiquetas)

    # Comprobar que todas las clases tienen ejemplos.
    if np.any(conteos == 0):
        raise ValueError(
            "Hay clases sin ejemplos en entrenamiento."
        )

    # Calcular pesos inversamente proporcionales
    # a la raíz cuadrada de la frecuencia.
    pesos = 1.0 / np.sqrt(conteos.astype(np.float64))

    # Normalizar para que la media sea 1.
    pesos /= pesos.mean()

    # Guardar resultados.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(MODELS_DIR / "class_counts.npy", conteos)
    np.save(MODELS_DIR / "class_weights.npy", pesos)

    print("\nCálculo terminado.")
    print("Registros procesados:", f"{registros:,}")
    print("Número de clases:", num_classes)
    print("Frecuencia mínima:", conteos.min())
    print("Frecuencia máxima:", conteos.max())
    print("Frecuencia media:", round(conteos.mean(), 2))

    print(
        "Clases con 10 ejemplos o menos:",
        np.sum(conteos <= 10)
    )

    print("\nPeso mínimo:", round(pesos.min(), 4))
    print("Peso máximo:", round(pesos.max(), 4))
    print("Media de los pesos:", round(pesos.mean(), 4))

    print("\nArchivos guardados:")
    print("models/class_counts.npy")
    print("models/class_weights.npy")


if __name__ == "__main__":
    calcular_pesos_clases()


import pickle
import numpy as np
import pyarrow.parquet as pq

from src.config import DATA_DIR, MODELS_DIR, LABEL_COLUMN


def analizar(nombre, ruta, label_to_id):

    tabla = pq.read_table(
        ruta,
        columns=[LABEL_COLUMN]
    )

    etiquetas = tabla[LABEL_COLUMN].to_pylist()

    clases, frecuencias = np.unique(
        etiquetas,
        return_counts=True
    )

    print(f"\n--- {nombre} ---")
    print(f"Registros: {len(etiquetas):,}")
    print(f"Clases presentes: {len(clases):,}")
    print(f"Frecuencia mínima: {frecuencias.min()}")
    print(f"Frecuencia máxima: {frecuencias.max()}")
    print(f"Media por clase: {frecuencias.mean():.2f}")

    desconocidas = set(clases) - set(label_to_id)

    print(f"Etiquetas sin ID: {len(desconocidas)}")

    return set(clases)


def main():

    with open(
        MODELS_DIR / "vocabulario_subconjunto.pkl",
        "rb"
    ) as archivo:
        vocabulario = pickle.load(archivo)

    label_to_id = vocabulario["label_to_id"]

    print(f"Clases en vocabulario: {len(label_to_id)}")

    train = analizar(
        "TRAIN",
        DATA_DIR / "train_clases.parquet",
        label_to_id
    )

    valid = analizar(
        "VALIDACIÓN",
        DATA_DIR / "valid_clases.parquet",
        label_to_id
    )

    test = analizar(
        "TEST",
        DATA_DIR / "test_clases.parquet",
        label_to_id
    )

    print("\n--- COMPROBACIÓN FINAL ---")
    print(f"Clases comunes en los tres conjuntos: {len(train & valid & test)}")
    print(f"Clases ausentes en validación: {len(train - valid)}")
    print(f"Clases ausentes en test: {len(train - test)}")


if __name__ == "__main__":
    main()

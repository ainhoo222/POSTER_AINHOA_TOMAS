
import pyarrow.parquet as pq
import numpy as np

from src.config import LABEL_COLUMN, DATA_DIR


def analizar_conjunto(nombre, ruta):
    tabla = pq.read_table(ruta, columns=[LABEL_COLUMN])
    etiquetas = np.asarray(tabla[LABEL_COLUMN].to_pylist())

    clases, frecuencias = np.unique(
        etiquetas,
        return_counts=True
    )

    print(f"\n--- {nombre} ---")
    print(f"Registros: {len(etiquetas):,}")
    print(f"Clases presentes: {len(clases):,}")
    print(f"Ejemplos por clase (media): {frecuencias.mean():.2f}")
    print(f"Frecuencia mínima: {frecuencias.min()}")
    print(f"Frecuencia máxima: {frecuencias.max()}")

    print(f"Clases con 1 ejemplo: {np.sum(frecuencias == 1):,}")
    print(f"Clases con 2-5 ejemplos: {np.sum((frecuencias >= 2) & (frecuencias <= 5)):,}")
    print(f"Clases con 6-10 ejemplos: {np.sum((frecuencias >= 6) & (frecuencias <= 10)):,}")
    print(f"Clases con más de 10 ejemplos: {np.sum(frecuencias > 10):,}")

    return set(clases)


def main():
    train = analizar_conjunto(
        "ENTRENAMIENTO",
        DATA_DIR / "train_subconjunto.parquet"
    )

    valid = analizar_conjunto(
        "VALIDACIÓN",
        DATA_DIR / "valid_subconjunto.parquet"
    )

    test = analizar_conjunto(
        "TEST",
        DATA_DIR / "test_subconjunto.parquet"
    )

    print("\n--- COMPARACIÓN DE CLASES ---")
    print(f"Clases comunes en train y valid: {len(train & valid):,}")
    print(f"Clases comunes en train y test: {len(train & test):,}")
    print(f"Clases comunes en los tres conjuntos: {len(train & valid & test):,}")

    print(f"Clases de validación no presentes en train: {len(valid - train):,}")
    print(f"Clases de test no presentes en train: {len(test - train):,}")


if __name__ == "__main__":
    main()

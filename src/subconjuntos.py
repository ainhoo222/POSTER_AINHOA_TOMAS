
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
    DATA_DIR,
    LABEL_COLUMN
)


# Tamaños de los subconjuntos
TRAIN_SIZE = 50_000
VALID_SIZE = 10_000
TEST_SIZE = 10_000

SEED = 42


def obtener_etiquetas(ruta):
    """
    Lee únicamente las etiquetas del Parquet.
    No carga todos los textos en memoria.
    """
    archivo = pq.ParquetFile(ruta)
    etiquetas = []

    for lote in archivo.iter_batches(
        batch_size=10_000,
        columns=[LABEL_COLUMN]
    ):
        etiquetas.extend(lote.column(0).to_pylist())

    return np.asarray(etiquetas)


def seleccionar_indices_train(ruta, numero_registros):
    """
    Selecciona registros de entrenamiento de forma estratificada.
    """
    print("\nLeyendo etiquetas de entrenamiento...")

    etiquetas = obtener_etiquetas(ruta)
    indices = np.arange(len(etiquetas))

    if numero_registros > len(indices):
        raise ValueError("El tamaño solicitado supera el dataset.")

    print(f"Registros originales: {len(indices):,}")
    print(f"Clases originales: {len(np.unique(etiquetas)):,}")

    indices_seleccionados, _ = train_test_split(
        indices,
        train_size=numero_registros,
        stratify=etiquetas,
        random_state=SEED
    )

    return np.sort(indices_seleccionados)


def seleccionar_indices_aleatorios(ruta, numero_registros, seed):
    """
    Selecciona registros aleatorios de validación o test.
    """
    archivo = pq.ParquetFile(ruta)
    total = archivo.metadata.num_rows

    if numero_registros > total:
        raise ValueError("El tamaño solicitado supera el dataset.")

    generador = np.random.default_rng(seed)

    indices = generador.choice(
        total,
        size=numero_registros,
        replace=False
    )

    return np.sort(indices)


def guardar_subconjunto(ruta_origen, indices, ruta_destino):
    """
    Extrae los registros seleccionados leyendo el Parquet
    por lotes, sin cargar el dataset completo en memoria.
    """
    archivo = pq.ParquetFile(ruta_origen)

    ruta_destino = Path(ruta_destino)
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)

    escritor = pq.ParquetWriter(
        ruta_destino,
        archivo.schema_arrow
    )

    inicio = 0

    try:
        for lote in archivo.iter_batches(batch_size=10_000):
            fin = inicio + lote.num_rows

            posicion_inicio = np.searchsorted(indices, inicio)
            posicion_fin = np.searchsorted(indices, fin)

            indices_lote = indices[posicion_inicio:posicion_fin]

            if len(indices_lote) > 0:
                indices_locales = indices_lote - inicio

                lote_seleccionado = lote.take(
                    pa.array(indices_locales)
                )

                escritor.write_batch(lote_seleccionado)

            inicio = fin

    finally:
        escritor.close()

    print(f"Guardado: {ruta_destino}")
    print(f"Registros seleccionados: {len(indices):,}")


def generar_subconjuntos():
    print("=" * 50)
    print("GENERACIÓN DE SUBCONJUNTOS")
    print("=" * 50)

    # 1. Entrenamiento estratificado
    indices_train = seleccionar_indices_train(
        TRAIN_PATH,
        TRAIN_SIZE
    )

    # 2. Validación aleatoria
    indices_valid = seleccionar_indices_aleatorios(
        VALID_PATH,
        VALID_SIZE,
        SEED + 1
    )

    # 3. Test aleatorio
    indices_test = seleccionar_indices_aleatorios(
        TEST_PATH,
        TEST_SIZE,
        SEED + 2
    )

    # 4. Guardar los tres subconjuntos
    guardar_subconjunto(
        TRAIN_PATH,
        indices_train,
        DATA_DIR / "train_subconjunto.parquet"
    )

    guardar_subconjunto(
        VALID_PATH,
        indices_valid,
        DATA_DIR / "valid_subconjunto.parquet"
    )

    guardar_subconjunto(
        TEST_PATH,
        indices_test,
        DATA_DIR / "test_subconjunto.parquet"
    )

    print("\nTodos los subconjuntos se han generado correctamente.")


if __name__ == "__main__":
    generar_subconjuntos()

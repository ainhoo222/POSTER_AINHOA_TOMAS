
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
    DATA_DIR,
    MODELS_DIR,
    LABEL_COLUMN,
    VOCAB_PATH
)


# Configuración
CLASES_POR_INTERVALO = 125
SEED = 42

RUTAS_ORIGINALES = {
    "train": TRAIN_PATH,
    "valid": VALID_PATH,
    "test": TEST_PATH
}

RUTAS_DESTINO = {
    "train": DATA_DIR / "train_clases.parquet",
    "valid": DATA_DIR / "valid_clases.parquet",
    "test": DATA_DIR / "test_clases.parquet"
}

RUTA_VOCABULARIO = MODELS_DIR / "vocabulario_subconjunto.pkl"


def contar_clases(ruta):
    """Cuenta los ejemplos de cada clase sin cargar los textos."""

    conteos = Counter()

    archivo = pq.ParquetFile(ruta)

    for lote in archivo.iter_batches(
        batch_size=10000,
        columns=[LABEL_COLUMN]
    ):
        conteos.update(lote.column(0).to_pylist())

    return conteos


def seleccionar_clases(conteos):
    """Selecciona 125 clases de cada intervalo de frecuencia."""

    intervalos = [
        ("10-20", 10, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("Más de 100", 101, float("inf"))
    ]

    generador = np.random.default_rng(SEED)
    seleccionadas = []

    for nombre, minimo, maximo in intervalos:

        disponibles = [
            clase
            for clase, frecuencia in conteos.items()
            if minimo <= frecuencia <= maximo
        ]

        if len(disponibles) < CLASES_POR_INTERVALO:
            raise ValueError(
                f"No hay suficientes clases en el intervalo {nombre}."
            )

        elegidas = generador.choice(
            disponibles,
            size=CLASES_POR_INTERVALO,
            replace=False
        )

        seleccionadas.extend(elegidas.tolist())

        print(
            f"{nombre}: {len(elegidas)} clases seleccionadas "
            f"de {len(disponibles)} disponibles."
        )

    return set(seleccionadas)


def guardar_filtrado(ruta_origen, ruta_destino, clases):
    """Guarda todos los registros de las clases seleccionadas."""

    archivo = pq.ParquetFile(ruta_origen)

    escritor = pq.ParquetWriter(
        ruta_destino,
        archivo.schema_arrow
    )

    total = 0

    try:
        for lote in archivo.iter_batches(batch_size=10000):

            etiquetas = lote.column(
                lote.schema.get_field_index(LABEL_COLUMN)
            ).to_pylist()

            mascara = pa.array([
                etiqueta in clases
                for etiqueta in etiquetas
            ])

            lote_filtrado = lote.filter(mascara)

            if lote_filtrado.num_rows > 0:
                escritor.write_batch(lote_filtrado)
                total += lote_filtrado.num_rows

    finally:
        escritor.close()

    print(
        f"{Path(ruta_destino).name}: "
        f"{total:,} registros guardados."
    )

    return total


def crear_vocabulario_reducido(clases):
    """Mantiene el vocabulario de palabras y reduce las etiquetas."""

    print("\nCreando vocabulario reducido...")

    with open(VOCAB_PATH, "rb") as archivo:
        vocabulario = pickle.load(archivo)

    word_to_id = vocabulario["word_to_id"]

    # Orden determinista para que los IDs sean reproducibles.
    clases_ordenadas = sorted(clases)

    label_to_id = {
        etiqueta: indice
        for indice, etiqueta in enumerate(clases_ordenadas)
    }

    vocabulario_reducido = {
        "word_to_id": word_to_id,
        "label_to_id": label_to_id
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RUTA_VOCABULARIO, "wb") as archivo:
        pickle.dump(vocabulario_reducido, archivo)

    print(f"Palabras conservadas: {len(word_to_id):,}")
    print(f"Clases conservadas: {len(label_to_id):,}")
    print(f"Vocabulario guardado: {RUTA_VOCABULARIO}")


def main():

    print("=" * 55)
    print("SELECCIÓN DE 500 CLASES")
    print("=" * 55)

    # 1. Contar frecuencias de las tres particiones.
    conteos_train = contar_clases(RUTAS_ORIGINALES["train"])
    conteos_valid = contar_clases(RUTAS_ORIGINALES["valid"])
    conteos_test = contar_clases(RUTAS_ORIGINALES["test"])

    # 2. Comprobar que las clases cumplen los requisitos.
    clases_validas = {
        clase
        for clase in conteos_train
        if conteos_train[clase] >= 10
        and conteos_valid.get(clase, 0) >= 5
        and conteos_test.get(clase, 0) >= 5
    }

    conteos_elegibles = {
        clase: conteos_train[clase]
        for clase in clases_validas
    }

    # 3. Seleccionar las 500 clases.
    clases = seleccionar_clases(conteos_elegibles)

    print(f"\nTotal de clases seleccionadas: {len(clases)}")

    # 4. Filtrar las tres particiones originales.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for nombre in ["train", "valid", "test"]:

        print(f"\nProcesando {nombre.upper()}...")

        guardar_filtrado(
            RUTAS_ORIGINALES[nombre],
            RUTAS_DESTINO[nombre],
            clases
        )

    # 5. Crear el nuevo vocabulario de etiquetas.
    crear_vocabulario_reducido(clases)

    print("\n" + "=" * 55)
    print("SUBCONJUNTOS GENERADOS CORRECTAMENTE")
    print("=" * 55)


if __name__ == "__main__":
    main()

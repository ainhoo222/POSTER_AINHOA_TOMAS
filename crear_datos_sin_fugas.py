
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    ID_COLUMN,
    LABEL_COLUMN,
    VOCAB_PATH
)


# =====================================================
# CONFIGURACIÓN
# =====================================================

SEED = 42
BATCH_SIZE = 10_000
CLASES_POR_INTERVALO = 125

RUTAS_ORIGINALES = [
    DATA_DIR / "train_preparado.parquet",
    DATA_DIR / "valid_preparado.parquet",
    DATA_DIR / "test_preparado.parquet"
]

RUTAS_DESTINO = {
    0: DATA_DIR / "train_grouped500.parquet",
    1: DATA_DIR / "valid_grouped500.parquet",
    2: DATA_DIR / "test_grouped500.parquet"
}

RUTA_VOCABULARIO = MODELS_DIR / "vocabulario_grouped500.pkl"


# =====================================================
# 1. CARGAR LOS ABSTRACT_ID
# =====================================================

def cargar_ids():

    print("\nLeyendo ABSTRACT_ID de los archivos originales...")

    arrays = []

    for ruta in RUTAS_ORIGINALES:

        print(f"Leyendo {ruta.name}...")

        archivo = pq.ParquetFile(ruta)
        partes = []

        for lote in archivo.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[ID_COLUMN]
        ):
            partes.append(
                lote.column(0).to_numpy(zero_copy_only=False)
            )

        ids_archivo = np.concatenate(partes)
        arrays.append(ids_archivo)

        print(f"Registros leídos: {len(ids_archivo):,}")

    ids = np.concatenate(arrays)

    print(f"\nTotal de registros: {len(ids):,}")
    print(f"Abstracts únicos: {len(np.unique(ids)):,}")

    return ids


# =====================================================
# 2. DIVIDIR POR ABSTRACT_ID
# =====================================================

def dividir_abstracts(ids):

    print("\nDividiendo los abstracts por grupos...")

    # Primera división: 60% train y 40% temporal.
    divisor_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.40,
        random_state=SEED
    )

    train_idx, temporal_idx = next(
        divisor_1.split(
            np.zeros(len(ids), dtype=np.uint8),
            groups=ids
        )
    )

    # Segunda división: 20% valid y 20% test.
    divisor_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED
    )

    valid_local, test_local = next(
        divisor_2.split(
            np.zeros(len(temporal_idx), dtype=np.uint8),
            groups=ids[temporal_idx]
        )
    )

    valid_idx = temporal_idx[valid_local]
    test_idx = temporal_idx[test_local]

    # 0 = train, 1 = valid, 2 = test.
    particiones = np.empty(len(ids), dtype=np.uint8)

    particiones[train_idx] = 0
    particiones[valid_idx] = 1
    particiones[test_idx] = 2

    print("\nDistribución inicial por registros:")
    for numero, nombre in enumerate(["Train", "Valid", "Test"]):
        cantidad = int(np.sum(particiones == numero))
        porcentaje = cantidad / len(ids) * 100

        print(
            f"{nombre}: {cantidad:,} "
            f"({porcentaje:.2f}%)"
        )

    return particiones


# =====================================================
# 3. CONTAR CLASES EN CADA PARTICIÓN
# =====================================================

def contar_clases(particiones):

    print("\nContando las frecuencias de las clases...")

    conteos = {
        0: Counter(),
        1: Counter(),
        2: Counter()
    }

    posicion_global = 0

    for ruta in RUTAS_ORIGINALES:

        archivo = pq.ParquetFile(ruta)

        for lote in archivo.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[LABEL_COLUMN]
        ):

            etiquetas = lote.column(0).to_pylist()

            fin = posicion_global + len(etiquetas)
            grupos_lote = particiones[posicion_global:fin]

            for numero in range(3):

                mascara = grupos_lote == numero

                if np.any(mascara):
                    conteos[numero].update(
                        etiqueta
                        for etiqueta, pertenece in zip(
                            etiquetas, mascara
                        )
                        if pertenece
                    )

            posicion_global = fin

        print(f"Contabilizado: {ruta.name}")

    return conteos


# =====================================================
# 4. SELECCIONAR 500 CLASES
# =====================================================

def seleccionar_clases(conteos):

    print("\nBuscando clases válidas...")

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
            for clase, frecuencia in conteos[0].items()
            if minimo <= frecuencia <= maximo
            and conteos[1].get(clase, 0) >= 5
            and conteos[2].get(clase, 0) >= 5
        ]

        print(
            f"Intervalo {nombre}: "
            f"{len(disponibles)} clases disponibles."
        )

        if len(disponibles) < CLASES_POR_INTERVALO:
            raise ValueError(
                f"No hay suficientes clases en {nombre}. "
                f"Se necesitan {CLASES_POR_INTERVALO}, "
                f"pero solo hay {len(disponibles)}. "
                "No se han generado los nuevos Parquet."
            )

        elegidas = generador.choice(
            disponibles,
            size=CLASES_POR_INTERVALO,
            replace=False
        )

        seleccionadas.extend(elegidas.tolist())

    print(f"\nTotal de clases seleccionadas: {len(seleccionadas)}")

    return set(seleccionadas)


# =====================================================
# 5. GUARDAR LOS NUEVOS PARQUET
# =====================================================

def guardar_particiones(particiones, clases):

    print("\nGenerando los nuevos archivos Parquet...")

    escritores = {}
    posicion_global = 0
    totales = {0: 0, 1: 0, 2: 0}

    try:

        for ruta in RUTAS_ORIGINALES:

            archivo = pq.ParquetFile(ruta)

            for lote in archivo.iter_batches(
                batch_size=BATCH_SIZE
            ):

                etiquetas = lote.column(
                    lote.schema.get_field_index(LABEL_COLUMN)
                ).to_pylist()

                fin = posicion_global + len(etiquetas)
                grupos_lote = particiones[posicion_global:fin]

                etiquetas_array = np.asarray(
                    etiquetas,
                    dtype=object
                )

                mascara_clases = np.isin(
                    etiquetas_array,
                    list(clases)
                )

                for numero in range(3):

                    mascara = (
                        (grupos_lote == numero)
                        & mascara_clases
                    )

                    if not np.any(mascara):
                        continue

                    lote_filtrado = lote.filter(
                        pa.array(mascara)
                    )

                    if numero not in escritores:
                        escritores[numero] = pq.ParquetWriter(
                            RUTAS_DESTINO[numero],
                            archivo.schema_arrow
                        )

                    escritores[numero].write_batch(
                        lote_filtrado
                    )

                    totales[numero] += lote_filtrado.num_rows

                posicion_global = fin

            print(f"Procesado: {ruta.name}")

    finally:

        for escritor in escritores.values():
            escritor.close()

    print("\nNuevos conjuntos generados:")

    for numero, nombre in enumerate(["Train", "Valid", "Test"]):
        print(
            f"{nombre}: {totales[numero]:,} registros "
            f"-> {RUTAS_DESTINO[numero].name}"
        )

    return totales


# =====================================================
# 6. CREAR EL NUEVO VOCABULARIO
# =====================================================

def crear_vocabulario(clases, conteos):

    print("\nCreando vocabulario de etiquetas...")

    with open(VOCAB_PATH, "rb") as archivo:
        vocabulario_anterior = pickle.load(archivo)

    word_to_id = vocabulario_anterior["word_to_id"]

    clases_ordenadas = sorted(clases)

    label_to_id = {
        etiqueta: indice
        for indice, etiqueta in enumerate(clases_ordenadas)
    }

    vocabulario = {
        "word_to_id": word_to_id,
        "label_to_id": label_to_id
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RUTA_VOCABULARIO, "wb") as archivo:
        pickle.dump(vocabulario, archivo)

    # Recuentos de train en el mismo orden que label_to_id.
    class_counts = np.array(
        [
            conteos[0][etiqueta]
            for etiqueta in clases_ordenadas
        ],
        dtype=np.int64
    )

    np.save(
        MODELS_DIR / "class_counts_grouped500.npy",
        class_counts
    )

    print("Palabras:", len(word_to_id))
    print("Clases:", len(label_to_id))
    print("Vocabulario:", RUTA_VOCABULARIO)
    print("Recuentos:", MODELS_DIR / "class_counts_grouped500.npy")


# =====================================================
# 7. PROGRAMA PRINCIPAL
# =====================================================

def main():

    print("=" * 60)
    print("CREACIÓN DE DATOS SIN FUGA POR ABSTRACT_ID")
    print("=" * 60)

    for ruta in RUTAS_ORIGINALES:
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encuentra el archivo: {ruta}"
            )

    if RUTA_VOCABULARIO.exists():
        raise FileExistsError(
            f"El archivo {RUTA_VOCABULARIO} ya existe. "
            "No se sobrescribirá automáticamente."
        )

    # 1. Leer identificadores.
    ids = cargar_ids()

    # 2. Dividir abstracts completos.
    particiones = dividir_abstracts(ids)

    # 3. Contar clases.
    conteos = contar_clases(particiones)

    # 4. Seleccionar 500 clases.
    clases = seleccionar_clases(conteos)

    # 5. Guardar los nuevos conjuntos.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    guardar_particiones(particiones, clases)

    # 6. Crear vocabulario y recuentos.
    crear_vocabulario(clases, conteos)

    print("\n" + "=" * 60)
    print("PROCESO FINALIZADO")
    print("=" * 60)


if __name__ == "__main__":
    main()

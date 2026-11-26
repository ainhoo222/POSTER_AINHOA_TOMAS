
import pyarrow.parquet as pq

from pathlib import Path

from src.config import (
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
    TEXT_COLUMN,
    LOCATION_COLUMN,
    LABEL_COLUMN,
    ID_COLUMN
)


# =====================================================
# 1. INFORMACIÓN DE LOS ARCHIVOS
# =====================================================

def obtener_informacion(ruta):
    """
    Devuelve información básica de un archivo Parquet
    sin cargar todos sus registros en memoria.
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo: {ruta}"
        )

    archivo = pq.ParquetFile(ruta)

    return {
        "ruta": str(ruta),
        "registros": archivo.metadata.num_rows,
        "columnas": archivo.schema.names
    }


# =====================================================
# 2. LECTURA POR LOTES
# =====================================================

def leer_lotes(
    ruta,
    batch_size=10000,
    columnas=None
):
    """
    Generador que lee un Parquet por lotes.

    No carga el archivo completo en memoria.
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo: {ruta}"
        )

    if columnas is None:
        columnas = [
            ID_COLUMN,
            TEXT_COLUMN,
            LOCATION_COLUMN,
            LABEL_COLUMN
        ]

    archivo = pq.ParquetFile(ruta)

    columnas_disponibles = archivo.schema.names

    for columna in columnas:
        if columna not in columnas_disponibles:
            raise ValueError(
                f"No existe la columna {columna} en {ruta.name}"
            )

    for lote in archivo.iter_batches(
        batch_size=batch_size,
        columns=columnas
    ):
        yield lote.to_pydict()


# =====================================================
# 3. VALIDACIÓN DE REGISTROS
# =====================================================

def validar_registro(registro):
    """
    Comprueba que un registro contiene los campos
    necesarios y que la posición de la abreviatura
    es válida.
    """

    texto = registro[TEXT_COLUMN]
    posicion = registro[LOCATION_COLUMN]
    etiqueta = registro[LABEL_COLUMN]

    if not isinstance(texto, str) or not texto.strip():
        return False

    if not isinstance(etiqueta, str) or not etiqueta.strip():
        return False

    if posicion is None:
        return False

    palabras = texto.split()

    if not isinstance(posicion, int):
        return False

    if posicion < 0 or posicion >= len(palabras):
        return False

    return True


# =====================================================
# 4. COMPROBACIÓN DE UNA MUESTRA
# =====================================================

def comprobar_muestra(ruta, numero_registros=100):
    """
    Comprueba los primeros registros del dataset
    sin recorrer el archivo completo.
    """

    total = 0
    validos = 0
    invalidos = 0

    for lote in leer_lotes(
        ruta,
        batch_size=numero_registros
    ):
        for i in range(len(lote[TEXT_COLUMN])):

            registro = {
                ID_COLUMN: lote[ID_COLUMN][i],
                TEXT_COLUMN: lote[TEXT_COLUMN][i],
                LOCATION_COLUMN: lote[LOCATION_COLUMN][i],
                LABEL_COLUMN: lote[LABEL_COLUMN][i]
            }

            total += 1

            if validar_registro(registro):
                validos += 1
            else:
                invalidos += 1

            if total >= numero_registros:
                break

        break

    return {
        "total": total,
        "validos": validos,
        "invalidos": invalidos
    }


# =====================================================
# 5. RESUMEN DE LOS TRES CONJUNTOS
# =====================================================

def mostrar_resumen_datasets():

    datasets = {
        "TRAIN": TRAIN_PATH,
        "VALIDACIÓN": VALID_PATH,
        "TEST": TEST_PATH
    }

    for nombre, ruta in datasets.items():

        info = obtener_informacion(ruta)

        print(f"\n--- {nombre} ---")
        print(f"Registros: {info['registros']:,}")
        print(f"Columnas: {info['columnas']}")

        muestra = comprobar_muestra(ruta)

        print(
            f"Muestra: {muestra['validos']} válidos, "
            f"{muestra['invalidos']} inválidos"
        )


if __name__ == "__main__":
    mostrar_resumen_datasets()

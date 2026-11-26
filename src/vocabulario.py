
import pickle
from collections import Counter

from tqdm import tqdm

from src.config import (
    TRAIN_PATH,
    MAX_VOCAB_SIZE,
    PAD_TOKEN,
    UNK_TOKEN,
    VOCAB_PATH,
    MODELS_DIR,
    TEXT_COLUMN,
    LOCATION_COLUMN,
    LABEL_COLUMN
)

from src.datos import leer_lotes


def construir_vocabulario(
    batch_size=10000,
    max_records=None
):
    """
    Construye el vocabulario utilizando exclusivamente
    el conjunto de entrenamiento.

    max_records permite realizar pruebas con una muestra.
    Si es None, procesa los 3 millones de registros.
    """

    contador_palabras = Counter()
    abreviaturas = set()
    clases = set()

    print("\nConstruyendo vocabulario...")
    print("Procesando el conjunto de entrenamiento.")

    registros_procesados = 0

    for lote in tqdm(
        leer_lotes(
            TRAIN_PATH,
            batch_size=batch_size,
            columnas=[
                TEXT_COLUMN,
                LOCATION_COLUMN,
                LABEL_COLUMN
            ]
        ),
        desc="Leyendo train"
    ):

        textos = lote[TEXT_COLUMN]
        posiciones = lote[LOCATION_COLUMN]
        etiquetas = lote[LABEL_COLUMN]

        for texto, posicion, etiqueta in zip(
            textos,
            posiciones,
            etiquetas
        ):

            if not isinstance(texto, str):
                continue

            palabras = texto.split()

            # Contar las palabras del texto
            contador_palabras.update(palabras)

            # Guardar la abreviatura situada en LOCATION
            if (
                isinstance(posicion, int)
                and 0 <= posicion < len(palabras)
            ):
                abreviaturas.add(palabras[posicion])

            # Guardar los significados
            if isinstance(etiqueta, str):
                clases.add(etiqueta)

            registros_procesados += 1

            if (
                max_records is not None
                and registros_procesados >= max_records
            ):
                break

        if (
            max_records is not None
            and registros_procesados >= max_records
        ):
            break

    print("\nProcesamiento terminado.")
    print(f"Registros procesados: {registros_procesados:,}")
    print(f"Palabras diferentes: {len(contador_palabras):,}")
    print(f"Abreviaturas diferentes: {len(abreviaturas):,}")
    print(f"Clases diferentes: {len(clases):,}")

    # Seleccionar las palabras más frecuentes
    palabras_frecuentes = [
        palabra
        for palabra, frecuencia
        in contador_palabras.most_common(MAX_VOCAB_SIZE)
    ]

    # Añadir abreviaturas que no estén ya en el vocabulario
    vocabulario_palabras = set(palabras_frecuentes)

    abreviaturas_adicionales = sorted(
        abreviaturas - vocabulario_palabras
    )

    palabras_vocabulario = (
        palabras_frecuentes + abreviaturas_adicionales
    )

    # Crear identificadores especiales
    word_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    # Asignar un identificador a cada palabra
    for palabra in palabras_vocabulario:
        if palabra not in word_to_id:
            word_to_id[palabra] = len(word_to_id)

    # Crear identificadores para las clases
    # El orden alfabético garantiza reproducibilidad
    clases_ordenadas = sorted(clases)

    label_to_id = {
        etiqueta: i
        for i, etiqueta in enumerate(clases_ordenadas)
    }

    # Diccionario completo
    vocabulario = {
        "word_to_id": word_to_id,
        "label_to_id": label_to_id,
        "num_words": len(word_to_id),
        "num_classes": len(label_to_id),
        "num_abbreviations": len(abreviaturas),
        "num_extra_abbreviations": len(abreviaturas_adicionales)
    }

    # Crear carpeta si no existe
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar vocabulario
    with open(VOCAB_PATH, "wb") as archivo:
        pickle.dump(
            vocabulario,
            archivo,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print("\nResumen del vocabulario:")
    print(f"Palabras totales: {len(word_to_id):,}")
    print(f"Clases totales: {len(label_to_id):,}")
    print(
        "Abreviaturas adicionales: "
        f"{len(abreviaturas_adicionales):,}"
    )

    print(f"\nVocabulario guardado en: {VOCAB_PATH}")

    return vocabulario


def cargar_vocabulario():

    with open(VOCAB_PATH, "rb") as archivo:
        return pickle.load(archivo)


if __name__ == "__main__":

    construir_vocabulario()

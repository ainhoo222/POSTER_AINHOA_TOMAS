
import gc
import pickle

import fasttext
import numpy as np
from tqdm import tqdm

from src.config import (
    FASTTEXT_PATH,
    VOCAB_PATH,
    EMBEDDINGS_PATH,
    EMBEDDINGS_TEST_PATH,
    EMBEDDING_DIM,
    EMBEDDINGS_DTYPE,
    PAD_TOKEN,
    UNK_TOKEN,
    MODELS_DIR
)


def generar_embeddings(
    max_words=None,
    output_path=None
):
    """
    Genera una matriz de embeddings fastText para
    todas las palabras del vocabulario.

    max_words permite realizar una prueba parcial.
    Si es None, genera la matriz completa.

    output_path permite guardar la prueba en otro archivo.
    """

    # -------------------------------------------------
    # 1. Comprobar los archivos necesarios
    # -------------------------------------------------

    if not FASTTEXT_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo fastText: {FASTTEXT_PATH}"
        )

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el vocabulario: {VOCAB_PATH}"
        )

    # -------------------------------------------------
    # 2. Cargar el vocabulario
    # -------------------------------------------------

    print("Cargando vocabulario...")

    with open(VOCAB_PATH, "rb") as archivo:
        vocabulario = pickle.load(archivo)

    word_to_id = vocabulario["word_to_id"]

    palabras = [
        palabra
        for palabra, indice in sorted(
            word_to_id.items(),
            key=lambda elemento: elemento[1]
        )
    ]

    numero_palabras = len(palabras)

    if max_words is not None:
        numero_palabras = min(max_words, numero_palabras)
        palabras = palabras[:numero_palabras]

    print(f"Palabras que se procesarán: {numero_palabras:,}")

    # -------------------------------------------------
    # 3. Cargar fastText
    # -------------------------------------------------

    print("\nCargando modelo fastText...")
    modelo = fasttext.load_model(str(FASTTEXT_PATH))

    print("Modelo fastText cargado.")

    # -------------------------------------------------
    # 4. Crear la matriz en disco
    # -------------------------------------------------

    if output_path is None:
        output_path = EMBEDDINGS_PATH

    output_path = output_path

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    matriz = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=EMBEDDINGS_DTYPE,
        shape=(numero_palabras, EMBEDDING_DIM)
    )

    # Inicializar todos los vectores a cero
    matriz[:] = 0.0

    # -------------------------------------------------
    # 5. Obtener los vectores de cada palabra
    # -------------------------------------------------

    encontradas = 0
    desconocidas = 0

    print("\nGenerando embeddings...")

    for indice, palabra in enumerate(
        tqdm(palabras, desc="Procesando palabras")
    ):

        # PAD y UNK mantienen vector cero
        if palabra in (PAD_TOKEN, UNK_TOKEN):
            continue

        # Comprobar si la palabra existe directamente
        # en el vocabulario de fastText
        id_fasttext = modelo.get_word_id(palabra)

        if id_fasttext >= 0:
            encontradas += 1
        else:
            desconocidas += 1

        # fastText también genera vectores para palabras
        # desconocidas mediante sus subwords
        vector = modelo.get_word_vector(palabra)

        if vector.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Dimensión incorrecta para la palabra: {palabra}"
            )

        matriz[indice] = vector

    # -------------------------------------------------
    # 6. Guardar la matriz
    # -------------------------------------------------

    matriz.flush()

    del matriz
    del modelo
    gc.collect()

    print("\nGeneración terminada.")
    print(f"Palabras procesadas: {numero_palabras:,}")
    print(f"Encontradas en fastText: {encontradas:,}")
    print(f"Generadas mediante subwords: {desconocidas:,}")
    print(f"Matriz guardada en: {output_path}")

    return output_path


def cargar_embeddings(ruta=None):

    if ruta is None:
        ruta = EMBEDDINGS_PATH

    return np.load(
        ruta,
        mmap_mode="r"
    )


if __name__ == "__main__":

    generar_embeddings()

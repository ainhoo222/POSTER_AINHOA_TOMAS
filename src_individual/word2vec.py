
import pickle
import numpy as np

from pathlib import Path
from gensim.models import Word2Vec

from src.datos import leer_lotes

from src_individual.config_individual import (
    TRAIN_PATH,
    VOCAB_PATH,
    WORD2VEC_PATH,
    EMBEDDINGS_PATH,
    TEXT_COLUMN,
    EMBEDDING_DIM,
    WORD2VEC_WINDOW,
    WORD2VEC_MIN_COUNT,
    WORD2VEC_SG,
    WORD2VEC_NEGATIVE,
    WORD2VEC_EPOCHS,
    WORD2VEC_WORKERS,
    SEED
)


# =====================================================
# 1. CARGAR VOCABULARIO
# =====================================================

def cargar_vocabulario():

    with open(VOCAB_PATH, "rb") as archivo:
        vocabulario = pickle.load(archivo)

    return vocabulario["word_to_id"]


# =====================================================
# 2. CORPUS DE ENTRENAMIENTO
# =====================================================

class CorpusWord2Vec:

    def __init__(self, ruta, word_to_id):
        self.ruta = ruta
        self.word_to_id = word_to_id

    def __iter__(self):

        for lote in leer_lotes(
            ruta=self.ruta,
            batch_size=1000,
            columnas=[TEXT_COLUMN]
        ):

            for texto in lote[TEXT_COLUMN]:

                if not isinstance(texto, str):
                    continue

                palabras = texto.split()

                # Solo utilizamos palabras del vocabulario existente.
                palabras = [
                    palabra for palabra in palabras
                    if palabra in self.word_to_id
                ]

                if palabras:
                    yield palabras


# =====================================================
# 3. ENTRENAMIENTO DE WORD2VEC
# =====================================================

def entrenar_word2vec(corpus):

    print("\n" + "=" * 55)
    print("ENTRENAMIENTO WORD2VEC")
    print("=" * 55)

    modelo = Word2Vec(
        vector_size=EMBEDDING_DIM,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=WORD2VEC_WORKERS,
        sg=WORD2VEC_SG,
        negative=WORD2VEC_NEGATIVE,
        seed=SEED,
        epochs=WORD2VEC_EPOCHS
    )

    print("\nConstruyendo vocabulario de Word2Vec...")

    modelo.build_vocab(corpus)

    print(
        f"Palabras aprendidas: "
        f"{len(modelo.wv):,}"
    )

    print(
        f"Documentos procesados: "
        f"{modelo.corpus_count:,}"
    )

    print("\nEntrenando embeddings...")

    modelo.train(
        corpus,
        total_examples=modelo.corpus_count,
        epochs=WORD2VEC_EPOCHS
    )

    WORD2VEC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    modelo.save(str(WORD2VEC_PATH))

    print("\nModelo Word2Vec guardado:")
    print(WORD2VEC_PATH)

    return modelo


# =====================================================
# 4. CONSTRUIR MATRIZ DE EMBEDDINGS
# =====================================================


def construir_matriz_embeddings(modelo, word_to_id):

    print("\nConstruyendo matriz de embeddings...")

    vocab_size = len(word_to_id)

    # Vector medio de los embeddings aprendidos.
    vector_desconocido = np.mean(
        modelo.wv.vectors,
        axis=0
    )

    # Crear matriz de embeddings.
    matriz = np.zeros(
        (vocab_size, EMBEDDING_DIM),
        dtype=np.float32
    )

    palabras_encontradas = 0
    palabras_no_encontradas = 0

    for palabra, indice in word_to_id.items():

        if palabra == "<PAD>":
            matriz[indice] = np.zeros(EMBEDDING_DIM)

        elif palabra == "<UNK>":
            matriz[indice] = vector_desconocido

        elif palabra in modelo.wv:
            matriz[indice] = modelo.wv[palabra]
            palabras_encontradas += 1

        else:
            matriz[indice] = vector_desconocido
            palabras_no_encontradas += 1

    # Guardar matriz.
    EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(EMBEDDINGS_PATH, matriz)

    print("\nMatriz de embeddings creada.")
    print("Dimensiones:", matriz.shape)
    print("Palabras encontradas:", palabras_encontradas)
    print("Palabras no encontradas:", palabras_no_encontradas)
    print("Dimensión de cada vector:", EMBEDDING_DIM)
    print("PAD:", matriz[word_to_id["<PAD>"]].sum())
    print("UNK:", matriz[word_to_id["<UNK>"]].shape)

    print("\nMatriz guardada en:")
    print(EMBEDDINGS_PATH)

    return matriz


# =====================================================
# 5. PROCESO COMPLETO
# =====================================================


def main():

    print("Cargando vocabulario...")

    word_to_id = cargar_vocabulario()

    print("Palabras del vocabulario:", len(word_to_id))

    if WORD2VEC_PATH.exists():

        print("\nCargando modelo Word2Vec existente...")

        modelo = Word2Vec.load(str(WORD2VEC_PATH))

    else:

        print("\nEntrenando nuevo modelo Word2Vec...")

        corpus = CorpusWord2Vec(
            ruta=TRAIN_PATH,
            word_to_id=word_to_id
        )

        modelo = entrenar_word2vec(corpus)

    matriz = construir_matriz_embeddings(
        modelo,
        word_to_id
    )

    print("\n" + "=" * 55)
    print("PROCESO WORD2VEC FINALIZADO")
    print("=" * 55)

    print("Forma final de la matriz:", matriz.shape)


if __name__ == "__main__":
    main()

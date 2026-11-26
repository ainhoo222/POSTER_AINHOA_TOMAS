
import pickle
import numpy as np

from src.config import (
    VOCAB_PATH,
    MAX_LENGTH,
    CONTEXT_RADIUS,
    PAD_TOKEN,
    UNK_TOKEN,
    LABEL_COLUMN,
    TEXT_COLUMN,
    LOCATION_COLUMN
)


def cargar_vocabulario():
    """Carga los diccionarios de palabras y etiquetas."""
    with open(VOCAB_PATH, "rb") as archivo:
        vocabulario = pickle.load(archivo)

    return (
        vocabulario["word_to_id"],
        vocabulario["label_to_id"]
    )


def codificar_registro(
    texto,
    posicion,
    etiqueta,
    word_to_id,
    label_to_id
):
    """
    Convierte un texto médico en una secuencia de IDs.

    Conserva una ventana de hasta 257 palabras centrada
    alrededor de la abreviatura.
    """

    palabras = texto.split()

    if posicion < 0 or posicion >= len(palabras):
        raise ValueError(
            f"Posición {posicion} fuera del texto."
        )

    if etiqueta not in label_to_id:
        raise ValueError(
            f"Etiqueta desconocida: {etiqueta}"
        )

    # Calcular los límites de la ventana.
    inicio = max(0, posicion - CONTEXT_RADIUS)
    fin = min(len(palabras), posicion + CONTEXT_RADIUS + 1)

    # Extraer el contexto.
    palabras_ventana = palabras[inicio:fin]

    # Calcular la posición de la abreviatura dentro de la ventana.
    posicion_objetivo = posicion - inicio

    # Convertir las palabras en identificadores.
    secuencia = [
        word_to_id.get(palabra, word_to_id[UNK_TOKEN])
        for palabra in palabras_ventana
    ]

    longitud = len(secuencia)

    # Añadir PAD al final hasta alcanzar MAX_LENGTH.
    secuencia += [word_to_id[PAD_TOKEN]] * (
        MAX_LENGTH - longitud
    )

    return {
        "input_ids": np.array(secuencia, dtype=np.int32),
        "length": longitud,
        "target_position": posicion_objetivo,
        "label_id": label_to_id[etiqueta]
    }


def probar_codificacion(numero_registros=5):
    """Prueba la conversión con unos pocos registros."""

    from src.datos import leer_lotes

    word_to_id, label_to_id = cargar_vocabulario()

    lote = next(
        leer_lotes(
            ruta=__import__(
                "src.config", fromlist=["TRAIN_PATH"]
            ).TRAIN_PATH,
            batch_size=numero_registros
        )
    )

    print("\nComprobando codificación...\n")

    for i in range(numero_registros):

        resultado = codificar_registro(
            texto=lote[TEXT_COLUMN][i],
            posicion=lote[LOCATION_COLUMN][i],
            etiqueta=lote[LABEL_COLUMN][i],
            word_to_id=word_to_id,
            label_to_id=label_to_id
        )

        palabras = lote[TEXT_COLUMN][i].split()
        posicion = lote[LOCATION_COLUMN][i]

        print(f"Registro {i + 1}")
        print("Abreviatura:", palabras[posicion])
        print("Longitud de ventana:", resultado["length"])
        print("Posición objetivo:", resultado["target_position"])
        print("ID de etiqueta:", resultado["label_id"])
        print("Dimensiones:", resultado["input_ids"].shape)
        print("ID de la abreviatura:",
              resultado["input_ids"][resultado["target_position"]])
        print("-" * 45)


if __name__ == "__main__":
    probar_codificacion()


import numpy as np

from src.config import (
    VALID_PATH,
    TEST_PATH,
    LABEL_COLUMN
)

from src.datos import leer_lotes
from src.preparacion import cargar_vocabulario


def comprobar_clases(ruta, nombre, num_classes, label_to_id):

    presentes = np.zeros(num_classes, dtype=bool)
    total = 0

    print(f"\nComprobando {nombre}...")

    for lote in leer_lotes(
        ruta,
        batch_size=10000,
        columnas=[LABEL_COLUMN]
    ):

        etiquetas = lote[LABEL_COLUMN]

        ids = np.fromiter(
            (label_to_id[e] for e in etiquetas),
            dtype=np.int64,
            count=len(etiquetas)
        )

        presentes[ids] = True
        total += len(etiquetas)

    print("Registros:", f"{total:,}")
    print("Clases presentes:", int(presentes.sum()))
    print("Clases ausentes:", int((~presentes).sum()))

    if not presentes.all():
        ausentes = np.where(~presentes)[0]
        print("Primeras clases ausentes:", ausentes[:20])


if __name__ == "__main__":

    _, label_to_id = cargar_vocabulario()
    num_classes = len(label_to_id)

    comprobar_clases(
        VALID_PATH,
        "VALIDACIÓN",
        num_classes,
        label_to_id
    )

    comprobar_clases(
        TEST_PATH,
        "TEST",
        num_classes,
        label_to_id
    )

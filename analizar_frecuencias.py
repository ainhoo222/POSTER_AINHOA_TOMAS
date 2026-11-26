
from collections import Counter

from src.config import (
    TRAIN_PATH,
    VALID_PATH,
    TEST_PATH,
    LABEL_COLUMN
)

from src.datos import leer_lotes


def contar_clases(ruta, nombre):
    conteos = Counter()

    print(f"\nAnalizando {nombre}...")

    for lote in leer_lotes(
        ruta,
        batch_size=10000,
        columnas=[LABEL_COLUMN]
    ):
        conteos.update(lote[LABEL_COLUMN])

    print(f"Clases encontradas: {len(conteos):,}")
    return conteos


def main():

    train = contar_clases(TRAIN_PATH, "TRAIN")
    valid = contar_clases(VALID_PATH, "VALIDACIÓN")
    test = contar_clases(TEST_PATH, "TEST")

    clases_comunes = set(train) & set(valid) & set(test)

    intervalos = [
        ("10-20", 10, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("Más de 100", 101, float("inf"))
    ]

    print("\n" + "=" * 50)
    print("DISTRIBUCIÓN DE CLASES DISPONIBLES")
    print("=" * 50)

    for nombre, minimo, maximo in intervalos:

        elegibles = [
            clase for clase in clases_comunes
            if minimo <= train[clase] <= maximo
            and valid[clase] >= 5
            and test[clase] >= 5
        ]

        print(
            f"{nombre} ejemplos en train: "
            f"{len(elegibles):,} clases disponibles"
        )


if __name__ == "__main__":
    main()


import pandas as pd

from src.config import DATA_DIR

rutas = {
    "Train": DATA_DIR / "train_grouped500.parquet",
    "Valid": DATA_DIR / "valid_grouped500.parquet",
    "Test": DATA_DIR / "test_grouped500.parquet"
}

datos = {}

for nombre, ruta in rutas.items():
    datos[nombre] = pd.read_parquet(
        ruta,
        columns=["ABSTRACT_ID", "LABEL"]
    )

    print(f"{nombre}: {len(datos[nombre]):,} registros")
    print(
        f"  Clases: {datos[nombre]['LABEL'].nunique()}"
    )
    print(
        f"  Abstracts únicos: "
        f"{datos[nombre]['ABSTRACT_ID'].nunique():,}"
    )

print("\nCOMPROBACIÓN DE ABSTRACT_ID")

particiones = [
    ("Train", "Valid"),
    ("Train", "Test"),
    ("Valid", "Test")
]

for nombre_a, nombre_b in particiones:
    ids_a = set(datos[nombre_a]["ABSTRACT_ID"])
    ids_b = set(datos[nombre_b]["ABSTRACT_ID"])

    coincidencias = ids_a.intersection(ids_b)

    print(
        f"{nombre_a} - {nombre_b}: "
        f"{len(coincidencias)} coincidencias"
    )

print("\nCOMPROBACIÓN DE CLASES")

clases_train = set(datos["Train"]["LABEL"])
clases_valid = set(datos["Valid"]["LABEL"])
clases_test = set(datos["Test"]["LABEL"])

print("Clases en train:", len(clases_train))
print("Clases en valid:", len(clases_valid))
print("Clases en test:", len(clases_test))

print(
    "Clases que faltan en valid:",
    len(clases_train - clases_valid)
)

print(
    "Clases que faltan en test:",
    len(clases_train - clases_test)
)

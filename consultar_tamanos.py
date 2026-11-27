
import pandas as pd
from pathlib import Path

# Carpeta donde están los datasets
DATA_DIR = Path("data")

# Archivos de los subconjuntos
archivos = {
    "Train": DATA_DIR / "train_clases.parquet",
    "Valid": DATA_DIR / "valid_clases.parquet",
    "Test": DATA_DIR / "test_clases.parquet"
}

total = 0

print("=" * 50)
print("TAMAÑO DE LOS SUBCONJUNTOS")
print("=" * 50)

for nombre, ruta in archivos.items():

    if not ruta.exists():
        print(f"{nombre}: archivo no encontrado ({ruta})")
        continue

    df = pd.read_parquet(ruta)

    numero_registros = len(df)
    numero_clases = df["LABEL"].nunique()

    total += numero_registros

    print(f"\n{nombre}:")
    print(f"  Registros: {numero_registros:,}")
    print(f"  Clases: {numero_clases:,}")
    print(f"  Tamaño en disco: {ruta.stat().st_size / (1024**2):.2f} MB")

print("\n" + "=" * 50)
print(f"TOTAL DE REGISTROS: {total:,}")
print("=" * 50)

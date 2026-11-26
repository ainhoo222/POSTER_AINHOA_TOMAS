
from pathlib import Path
import pyarrow.parquet as pq

# Directorio principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorio donde están los datos
DATA_DIR = BASE_DIR / "data"

# Archivos que vamos a comprobar
archivos = [
    "train_preparado.parquet",
    "valid_preparado.parquet",
    "test_preparado.parquet"
]

for nombre in archivos:

    ruta = DATA_DIR / nombre

    print("\n" + "=" * 60)
    print(f"ARCHIVO: {nombre}")
    print("=" * 60)

    # Abrimos los metadatos sin cargar todo el archivo
    parquet = pq.ParquetFile(ruta)

    print(f"Número de registros: {parquet.metadata.num_rows}")
    print(f"Número de columnas: {parquet.metadata.num_columns}")
    print(f"Columnas: {parquet.schema.names}")

    # Leemos solamente cinco registros
    lote = next(parquet.iter_batches(batch_size=5))
    datos = lote.to_pandas()

    print("\nColumnas individuales:")
    for columna in datos.columns:
        print(f"- {columna}")

    print("\nDimensiones del lote:", datos.shape)

    print("\nPrimeros registros:")
    print(datos[["TEXT", "LOCATION", "LABEL"]].to_string(index=False))

    print("\nTipos de datos:")
    print(datos.dtypes)

print("\nComprobación finalizada.")

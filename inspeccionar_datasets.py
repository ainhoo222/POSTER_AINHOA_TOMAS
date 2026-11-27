
import pyarrow.parquet as pq

from pathlib import Path

carpeta = Path("data")

archivos = [
    "train_preparado.parquet",
    "valid_preparado.parquet",
    "test_preparado.parquet",
    "train_subconjunto.parquet",
    "valid_subconjunto.parquet",
    "test_subconjunto.parquet",
    "train_clases.parquet",
    "valid_clases.parquet",
    "test_clases.parquet"
]

for nombre in archivos:
    ruta = carpeta / nombre

    archivo = pq.ParquetFile(ruta)

    print("\n" + "=" * 60)
    print(nombre)
    print("=" * 60)
    print("Número de registros:", f"{archivo.metadata.num_rows:,}")
    print("Columnas:", archivo.schema_arrow.names)
    print("Esquema:")
    print(archivo.schema_arrow)

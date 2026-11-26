
print("1. Iniciando prueba...", flush=True)

import torch
print("2. PyTorch importado.", flush=True)

from src.dataset import crear_dataloader
print("3. Dataset importado.", flush=True)

from src.config import TRAIN_PATH
print("4. Configuración importada.", flush=True)

print("5. Creando DataLoader...", flush=True)

loader = crear_dataloader(
    ruta=TRAIN_PATH,
    batch_size=5,
    shuffle=True,
    max_records=20
)

print("6. DataLoader creado.", flush=True)
print("7. Intentando obtener el primer lote...", flush=True)

for input_ids, lengths, target_positions, labels in loader:

    print("8. ¡Primer lote recibido!", flush=True)

    print("Entrada:", input_ids.shape)
    print("Longitudes:", lengths.shape)
    print("Posiciones:", target_positions.shape)
    print("Etiquetas:", labels.shape)

    break

print("9. Prueba finalizada correctamente.", flush=True)

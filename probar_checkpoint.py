
import torch

from src.modelo import BiLSTMClasificador
from src.preparacion import cargar_vocabulario
from src.checkpoints import (
    guardar_checkpoint,
    cargar_checkpoint
)

from src.config import MODELS_DIR


# Cargar vocabulario.
word_to_id, label_to_id = cargar_vocabulario()

# Crear modelo.
modelo = BiLSTMClasificador(
    num_classes=len(label_to_id),
    fine_tune_embeddings=False,
    attention=False
)

# Crear optimizador.
optimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=0.001
)

# Guardar checkpoint.
ruta = MODELS_DIR / "checkpoint_prueba.pt"

guardar_checkpoint(
    ruta=ruta,
    modelo=modelo,
    optimizador=optimizador,
    epoch=1,
    batch=100,
    mejor_macro_f1=0.75,
    paciencia=0
)

# Recuperar checkpoint.
progreso = cargar_checkpoint(
    ruta=ruta,
    modelo=modelo,
    optimizador=optimizador
)

print("\nProgreso recuperado:")
print(progreso)

print("\nPrueba de checkpoint completada.")


import torch
import numpy as np

from src.modelo import BiLSTMClasificador
from src.preparacion import cargar_vocabulario, codificar_registro
from src.datos import leer_lotes
from src.config import TRAIN_PATH, TEXT_COLUMN, LOCATION_COLUMN, LABEL_COLUMN


# Cargar vocabulario.
word_to_id, label_to_id = cargar_vocabulario()

# Obtener cinco registros.
lote = next(leer_lotes(TRAIN_PATH, batch_size=5))

# Codificar registros.
resultados = []

for i in range(5):
    resultados.append(
        codificar_registro(
            lote[TEXT_COLUMN][i],
            lote[LOCATION_COLUMN][i],
            lote[LABEL_COLUMN][i],
            word_to_id,
            label_to_id
        )
    )

# Crear tensores.
input_ids = torch.tensor(
    np.array([r["input_ids"] for r in resultados]),
    dtype=torch.long
)

lengths = torch.tensor(
    [r["length"] for r in resultados],
    dtype=torch.long
)

target_positions = torch.tensor(
    [r["target_position"] for r in resultados],
    dtype=torch.long
)


# Probar las tres variantes.
experimentos = {
    "Baseline": {
        "fine_tune_embeddings": False,
        "attention": False
    },
    "Fine-tuning": {
        "fine_tune_embeddings": True,
        "attention": False
    },
    "Atención": {
        "fine_tune_embeddings": False,
        "attention": True
    }
}

for nombre, configuracion in experimentos.items():

    print(f"\nProbando: {nombre}")

    modelo = BiLSTMClasificador(
        num_classes=len(label_to_id),
        **configuracion
    )

    modelo.eval()

    with torch.no_grad():
        salida = modelo(
            input_ids,
            lengths,
            target_positions
        )

    print("Dimensiones:", salida.shape)
    print(
        "Embeddings entrenables:",
        modelo.embedding.weight.requires_grad
    )

    print("Prueba correcta.")
    
    del modelo


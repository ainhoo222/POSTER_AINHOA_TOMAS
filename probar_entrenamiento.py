
import torch

from src.dataset import crear_dataloader
from src.modelo import BiLSTMClasificador
from src.config import TRAIN_PATH
from src.preparacion import cargar_vocabulario


# Cargar vocabulario.
word_to_id, label_to_id = cargar_vocabulario()

# Crear DataLoader de prueba.
loader = crear_dataloader(
    ruta=TRAIN_PATH,
    batch_size=5,
    shuffle=True,
    max_records=20
)

# Crear modelo.
modelo = BiLSTMClasificador(
    num_classes=len(label_to_id),
    fine_tune_embeddings=False,
    attention=False
)

# Función de pérdida y optimizador.
criterio = torch.nn.CrossEntropyLoss()
optimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=0.001
)

modelo.train()

print("Comenzando prueba de entrenamiento...\n")

for input_ids, lengths, target_positions, labels in loader:

    # Reiniciar gradientes.
    optimizador.zero_grad()

    # Obtener predicciones.
    logits = modelo(
        input_ids,
        lengths,
        target_positions
    )

    # Calcular pérdida.
    loss = criterio(logits, labels)

    # Calcular gradientes.
    loss.backward()

    # Actualizar parámetros.
    optimizador.step()

    print("Pérdida:", loss.item())
    print("Dimensiones de salida:", logits.shape)

    break

print("\nEntrenamiento de prueba completado.")

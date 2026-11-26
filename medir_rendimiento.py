
import time
import torch

from src.config import (
    TRAIN_PATH,
    BATCH_SIZE,
    LEARNING_RATE
)

from src.dataset import crear_dataloader
from src.modelo import BiLSTMClasificador
from src.preparacion import cargar_vocabulario

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def main():

    print("Preparando prueba de rendimiento...\n")

    _, label_to_id = cargar_vocabulario()

    # Utilizar 10 lotes reales.
    loader = crear_dataloader(
        ruta=TRAIN_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        max_records=BATCH_SIZE * 10
    )

    modelo = BiLSTMClasificador(
        num_classes=len(label_to_id),
        fine_tune_embeddings=False,
        attention=False
    )

    criterio = torch.nn.CrossEntropyLoss()

    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=LEARNING_RATE
    )

    modelo.train()

    tiempo_inicio = time.perf_counter()
    tiempo_medicion = None
    ejemplos_medidos = 0

    print("Ejecutando 10 lotes...\n")

    for numero_lote, lote in enumerate(loader, start=1):

        input_ids, lengths, target_positions, labels = lote

        optimizador.zero_grad()

        logits = modelo(
            input_ids,
            lengths,
            target_positions
        )

        loss = criterio(logits, labels)

        loss.backward()
        optimizador.step()

        # Los dos primeros lotes sirven como calentamiento.
        if numero_lote == 2:
            tiempo_medicion = time.perf_counter()

        elif numero_lote > 2:
            ejemplos_medidos += len(labels)

        print(
            f"Lote {numero_lote}/10 completado "
            f"| Pérdida: {loss.item():.4f}",
            flush=True
        )

    tiempo_total = time.perf_counter() - tiempo_inicio
    tiempo_calculado = time.perf_counter() - tiempo_medicion

    velocidad = ejemplos_medidos / tiempo_calculado

    # Estimación para los 3 millones de ejemplos.
    ejemplos_train = 3_000_000

    segundos_epoca = ejemplos_train / velocidad

    print("\n" + "=" * 45)
    print("RESULTADOS DE RENDIMIENTO")
    print("=" * 45)

    print(f"Tiempo total de prueba: {tiempo_total:.2f} segundos")
    print(f"Ejemplos medidos: {ejemplos_medidos}")
    print(f"Velocidad: {velocidad:.2f} ejemplos/segundo")

    print(
        f"Tiempo estimado por época: "
        f"{segundos_epoca / 3600:.2f} horas"
    )

    print(
        f"Tiempo estimado para 10 épocas: "
        f"{segundos_epoca * 10 / 3600:.2f} horas"
    )

    print("\nEstimación orientativa: no incluye validación ni")
    print("el coste adicional de guardar checkpoints.")


if __name__ == "__main__":
    main()

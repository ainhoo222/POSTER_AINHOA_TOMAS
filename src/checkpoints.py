
import os
import random
import numpy as np
import torch

from pathlib import Path


def guardar_checkpoint(
    ruta,
    modelo,
    optimizador,
    epoch,
    batch,
    mejor_macro_f1,
    paciencia
):
    """
    Guarda el estado necesario para reanudar el entrenamiento.
    """

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": modelo.state_dict(),
        "optimizer_state_dict": optimizador.state_dict(),

        "epoch": epoch,
        "batch": batch,
        "mejor_macro_f1": mejor_macro_f1,
        "paciencia": paciencia,

        # Estados aleatorios para reproducibilidad.
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state()
    }

    # Guardado temporal para evitar dejar un checkpoint incompleto.
    ruta_temporal = ruta.with_suffix(ruta.suffix + ".tmp")

    torch.save(checkpoint, ruta_temporal)
    os.replace(ruta_temporal, ruta)

    print(
        f"Checkpoint guardado: época {epoch}, lote {batch}",
        flush=True
    )


def cargar_checkpoint(
    ruta,
    modelo,
    optimizador,
    device="cpu"
):
    """
    Recupera el modelo, optimizador y progreso anterior.
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: {ruta}"
        )

    checkpoint = torch.load(
        ruta,
        map_location=device,
        weights_only=False
    )

    modelo.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizador.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["numpy_state"])
    torch.set_rng_state(checkpoint["torch_state"])

    print("Checkpoint recuperado correctamente.")

    return {
        "epoch": checkpoint["epoch"],
        "batch": checkpoint["batch"],
        "mejor_macro_f1": checkpoint["mejor_macro_f1"],
        "paciencia": checkpoint["paciencia"]
    }

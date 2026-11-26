
import random
import torch

from torch.utils.data import IterableDataset, DataLoader

from src.config import (
    TRAIN_PATH,
    BATCH_SIZE,
    TEXT_COLUMN,
    LOCATION_COLUMN,
    LABEL_COLUMN
)

from src.datos import leer_lotes
from src.preparacion import cargar_vocabulario, codificar_registro


class MeDALIterableDataset(IterableDataset):

    def __init__(
        self,
        ruta,
        shuffle=False,
        buffer_size=10000,
        seed=42,
        max_records=None
    ):
        super().__init__()

        self.ruta = ruta
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed
        self.max_records = max_records
        self.epoch = 0

        self.word_to_id, self.label_to_id = cargar_vocabulario()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):

        generador = random.Random(self.seed + self.epoch)
        buffer = []
        procesados = 0

        for lote in leer_lotes(self.ruta, batch_size=1000):

            numero_registros = len(lote[TEXT_COLUMN])

            for i in range(numero_registros):

                resultado = codificar_registro(
                    texto=lote[TEXT_COLUMN][i],
                    posicion=lote[LOCATION_COLUMN][i],
                    etiqueta=lote[LABEL_COLUMN][i],
                    word_to_id=self.word_to_id,
                    label_to_id=self.label_to_id
                )

                ejemplo = (
                    torch.from_numpy(resultado["input_ids"].astype("int64")),
                    resultado["length"],
                    resultado["target_position"],
                    resultado["label_id"]
                )

                procesados += 1

                if self.shuffle:

                    buffer.append(ejemplo)

                    if len(buffer) >= self.buffer_size:
                        indice = generador.randrange(len(buffer))
                        yield buffer.pop(indice)

                else:
                    yield ejemplo

                if (
                    self.max_records is not None
                    and procesados >= self.max_records
                ):
                    break

            if (
                self.max_records is not None
                and procesados >= self.max_records
            ):
                break

        # Vaciar el búfer al terminar.
        if self.shuffle:
            generador.shuffle(buffer)

            for ejemplo in buffer:
                yield ejemplo


def crear_dataloader(
    ruta,
    batch_size=BATCH_SIZE,
    shuffle=False,
    max_records=None,
    seed=42
):

    dataset = MeDALIterableDataset(
        ruta=ruta,
        shuffle=shuffle,
        seed=seed,
        max_records=max_records
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False
    )

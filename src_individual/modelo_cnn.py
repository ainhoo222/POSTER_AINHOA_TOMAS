
import numpy as np
import torch
import torch.nn as nn

from src_individual.config_individual import (
    EMBEDDINGS_PATH,
    EMBEDDING_DIM,
    CNN_FILTERS,
    CNN_KERNEL_SIZES,
    DROPOUT,
    EXPECTED_NUM_CLASSES,
    DEVICE
)


class CNNClasificador(nn.Module):

    def __init__(
        self,
        num_classes,
        embeddings_path=EMBEDDINGS_PATH,
        freeze_embeddings=True
    ):
        super().__init__()

        # =============================================
        # 1. CARGAR MATRIZ WORD2VEC
        # =============================================

        matriz_embeddings = np.load(embeddings_path)

        if matriz_embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                "La dimensión de los embeddings no coincide."
            )

        pesos = torch.tensor(
            matriz_embeddings,
            dtype=torch.float32
        )

        self.embedding = nn.Embedding.from_pretrained(
            pesos,
            freeze=freeze_embeddings,
            padding_idx=0
        )

        # =============================================
        # 2. CAPAS CONVOLUCIONALES
        # =============================================

        self.convoluciones = nn.ModuleList([
            nn.Conv1d(
                in_channels=EMBEDDING_DIM,
                out_channels=CNN_FILTERS,
                kernel_size=k,
                padding="same"
            )
            for k in CNN_KERNEL_SIZES
        ])

        self.activacion = nn.ReLU()

        # =============================================
        # 3. DROPOUT Y CLASIFICADOR
        # =============================================

        # Salida CNN: 128 filtros por cada uno
        # de los tres tamaños de kernel.
        dimension_cnn = CNN_FILTERS * len(CNN_KERNEL_SIZES)

        # Añadimos el embedding de la abreviatura.
        dimension_final = dimension_cnn + EMBEDDING_DIM

        self.dropout = nn.Dropout(DROPOUT)

        self.clasificador = nn.Linear(
            dimension_final,
            num_classes
        )

    # =============================================
    # 4. PROPAGACIÓN HACIA DELANTE
    # =============================================

    def forward(
        self,
        input_ids,
        lengths,
        target_positions
    ):

        # Obtener embeddings.
        x = self.embedding(input_ids)

        # Forma: (batch, embedding_dim, longitud).
        x = x.transpose(1, 2)

        # Máscara para ignorar posiciones PAD.
        posiciones = torch.arange(
            input_ids.size(1),
            device=input_ids.device
        ).unsqueeze(0)

        mascara = posiciones < lengths.unsqueeze(1)

        # Aplicar las convoluciones.
        caracteristicas = []

        for conv in self.convoluciones:

            salida = self.activacion(conv(x))

            # Ignorar posiciones posteriores a la longitud real.
            mascara_expandida = mascara.unsqueeze(1)

            salida = salida.masked_fill(
                ~mascara_expandida,
                float("-inf")
            )

            # Max pooling global.
            pooled = torch.max(
                salida,
                dim=2
            ).values

            caracteristicas.append(pooled)

        # Concatenar las características de las CNN.
        caracteristicas_cnn = torch.cat(
            caracteristicas,
            dim=1
        )

        # =============================================
        # 5. EMBEDDING DE LA ABREVIATURA OBJETIVO
        # =============================================

        indices = target_positions.unsqueeze(1)

        indices = indices.unsqueeze(2).expand(
            -1,
            -1,
            EMBEDDING_DIM
        )

        embedding_objetivo = torch.gather(
            self.embedding(input_ids),
            dim=1,
            index=indices
        ).squeeze(1)

        # =============================================
        # 6. CLASIFICACIÓN
        # =============================================

        representacion = torch.cat(
            [caracteristicas_cnn, embedding_objetivo],
            dim=1
        )

        representacion = self.dropout(representacion)

        logits = self.clasificador(representacion)

        return logits


# =============================================
# 7. PRUEBA DE LA ARQUITECTURA
# =============================================

def probar_modelo():

    print("\nComprobando arquitectura CNN...")

    modelo = CNNClasificador(
        num_classes=EXPECTED_NUM_CLASSES
    ).to(DEVICE)

    batch_size = 4
    max_length = 257

    # Crear datos ficticios para probar dimensiones.
    input_ids = torch.randint(
        low=1,
        high=100,
        size=(batch_size, max_length)
    )

    lengths = torch.tensor([257, 200, 150, 100])

    target_positions = torch.tensor([128, 80, 60, 40])

    input_ids = input_ids.to(DEVICE)
    lengths = lengths.to(DEVICE)
    target_positions = target_positions.to(DEVICE)

    modelo.eval()

    with torch.no_grad():

        salida = modelo(
            input_ids,
            lengths,
            target_positions
        )

    print("Entrada:", tuple(input_ids.shape))
    print("Salida:", tuple(salida.shape))
    print("Número de clases:", salida.shape[1])

    print("\nParámetros entrenables:")

    parametros = sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )

    print(f"{parametros:,}")

    print("\nModelo CNN comprobado correctamente.")


if __name__ == "__main__":
    probar_modelo()

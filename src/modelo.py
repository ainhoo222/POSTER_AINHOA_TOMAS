
import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)

from src.config import (
    EMBEDDINGS_PATH,
    EMBEDDING_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT
)


class BiLSTMClasificador(nn.Module):

    def __init__(
        self,
        num_classes,
        fine_tune_embeddings=False,
        attention=False
    ):
        super().__init__()

        # 1. Cargar la matriz de embeddings fastText.
        matriz_embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r"
        )

        pesos = torch.tensor(
            np.asarray(matriz_embeddings),
            dtype=torch.float32
        )

        # 2. Crear la capa de embeddings.
        self.embedding = nn.Embedding.from_pretrained(
            pesos,
            freeze=not fine_tune_embeddings,
            padding_idx=0
        )

        # 3. Crear la BiLSTM.
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )

        # 4. Dropout.
        self.dropout = nn.Dropout(DROPOUT)

        # 5. Configurar la atención.
        self.attention = attention

        if attention:
            self.classifier = nn.Linear(
                HIDDEN_SIZE * 4,
                num_classes
            )
        else:
            self.classifier = nn.Linear(
                HIDDEN_SIZE * 2,
                num_classes
            )

    def forward(
        self,
        input_ids,
        lengths,
        target_positions
    ):

        # 1. Obtener los embeddings de las palabras.
        x = self.embedding(input_ids)

        # 2. Empaquetar las secuencias para ignorar el padding.
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # 3. Procesar las secuencias con la BiLSTM.
        packed_output, _ = self.lstm(packed)

        # 4. Recuperar las salidas de la LSTM.
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1)
        )

        # 5. Obtener la representación de la abreviatura.
        batch_indices = torch.arange(
            input_ids.size(0),
            device=input_ids.device
        )

        target_output = output[
            batch_indices,
            target_positions
        ]

        if self.attention:

            # 6. Calcular atención usando la abreviatura como consulta.
            scores = torch.bmm(
                output,
                target_output.unsqueeze(2)
            ).squeeze(2)

            scores = scores / np.sqrt(HIDDEN_SIZE * 2)

            # Ignorar las posiciones de padding.
            posiciones = torch.arange(
                input_ids.size(1),
                device=input_ids.device
            ).unsqueeze(0)

            mascara = posiciones >= lengths.unsqueeze(1)

            scores = scores.masked_fill(
                mascara,
                float("-inf")
            )

            # Calcular los pesos de atención.
            pesos_atencion = torch.softmax(scores, dim=1)

            # Obtener el contexto ponderado.
            contexto = torch.bmm(
                pesos_atencion.unsqueeze(1),
                output
            ).squeeze(1)

            # Combinar abreviatura y contexto.
            representacion = torch.cat(
                (target_output, contexto),
                dim=1
            )

        else:
            representacion = target_output

        # 7. Clasificación final.
        representacion = self.dropout(representacion)

        logits = self.classifier(representacion)

        return logits

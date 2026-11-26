
import numpy as np
import torch

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)


def calcular_metricas(
    etiquetas_reales,
    predicciones,
    num_classes,
    conteos_clases
):
    """
    Calcula las métricas generales y por clase.
    """

    etiquetas_reales = np.asarray(etiquetas_reales)
    predicciones = np.asarray(predicciones)

    labels = np.arange(num_classes)

    # Calcular precision, recall y F1 de cada clase.
    precision, recall, f1, soporte = (
        precision_recall_fscore_support(
            etiquetas_reales,
            predicciones,
            labels=labels,
            average=None,
            zero_division=0
        )
    )

    # Métricas generales.
    accuracy = accuracy_score(
        etiquetas_reales,
        predicciones
    )

    macro_f1 = np.mean(f1)

    weighted_f1 = np.average(
        f1,
        weights=soporte
    )

    # Calcular Cohen's Kappa sin matriz de confusión.
    frecuencia_real = np.bincount(
        etiquetas_reales,
        minlength=num_classes
    )

    frecuencia_predicha = np.bincount(
        predicciones,
        minlength=num_classes
    )

    n = len(etiquetas_reales)

    acuerdo_observado = accuracy

    acuerdo_azar = np.dot(
        frecuencia_real,
        frecuencia_predicha
    ) / (n ** 2)

    if np.isclose(acuerdo_azar, 1.0):
        kappa = 1.0 if np.isclose(accuracy, 1.0) else 0.0
    else:
        kappa = (
            acuerdo_observado - acuerdo_azar
        ) / (
            1.0 - acuerdo_azar
        )

    # Identificar las clases minoritarias.
    clases_minoritaria = conteos_clases <= 10

    f1_minoritaria = np.mean(
        f1[clases_minoritaria]
    )

    metricas = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "kappa": float(kappa),
        "minority_f1": float(f1_minoritaria)
    }

    # Resultados detallados por clase.
    resultados_clases = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": soporte,
        "train_count": conteos_clases
    }

    return metricas, resultados_clases


def evaluar_modelo(
    modelo,
    loader,
    num_classes,
    conteos_clases,
    device="cpu"
):
    """
    Evalúa un modelo recorriendo el DataLoader por lotes.
    """

    modelo.eval()

    etiquetas_reales = []
    predicciones = []

    with torch.no_grad():

        for input_ids, lengths, target_positions, labels in loader:

            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            target_positions = target_positions.to(device)

            logits = modelo(
                input_ids,
                lengths,
                target_positions
            )

            predichas = torch.argmax(
                logits,
                dim=1
            )

            etiquetas_reales.extend(
                labels.numpy().tolist()
            )

            predicciones.extend(
                predichas.cpu().numpy().tolist()
            )

    metricas, resultados_clases = calcular_metricas(
        etiquetas_reales,
        predicciones,
        num_classes,
        conteos_clases
    )

    return metricas, resultados_clases

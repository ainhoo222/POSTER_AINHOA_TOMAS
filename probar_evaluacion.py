
import numpy as np

from src.evaluacion import calcular_metricas


# Etiquetas reales de prueba.
y_real = np.array([
    0, 0, 1, 1, 2, 2, 3, 3
])

# Predicciones artificiales.
y_pred = np.array([
    0, 0, 1, 2, 2, 2, 3, 0
])

# Frecuencias de entrenamiento ficticias.
conteos = np.array([
    100, 50, 8, 6
])

metricas, resultados = calcular_metricas(
    etiquetas_reales=y_real,
    predicciones=y_pred,
    num_classes=4,
    conteos_clases=conteos
)

print("Métricas generales:\n")

for nombre, valor in metricas.items():
    print(f"{nombre}: {valor:.4f}")

print("\nF1 por clase:")
print(resultados["f1"])

print("\nSoporte por clase:")
print(resultados["support"])

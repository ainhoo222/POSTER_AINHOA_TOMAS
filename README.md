# Baseline común: BiLSTM + fastText

## 1. Descripción

La baseline común del proyecto aborda la desambiguación de abreviaturas médicas mediante un modelo de clasificación supervisada basado en una red neuronal recurrente **BiLSTM (Bidirectional Long Short-Term Memory)** y representaciones vectoriales de palabras obtenidas mediante **fastText**.

El objetivo es clasificar cada aparición de una abreviatura médica según su significado, utilizando el contexto textual en el que aparece.

El problema se plantea como una tarea de clasificación multiclase con 500 clases correspondientes a los distintos significados de las abreviaturas.

## 2. Estructura del proyecto

La implementación común se encuentra en la carpeta `src/`:

```text
src/
├── config.py
├── datos.py
├── preparacion.py
├── dataset.py
├── embeddings.py
├── modelo.py
├── entrenamiento.py
├── evaluacion.py
└── pesos_clases.py
```

Los datos, vocabularios, modelos y resultados se almacenan en sus respectivas carpetas:

```text
data/
models/
results/
```

## 3. Preparación de los datos

Se utilizan los conjuntos de entrenamiento, validación y test generados previamente:

| Conjunto | Registros |
|---|---:|
| Entrenamiento | 43.732 |
| Validación | 14.586 |
| Test | 14.583 |
| **Total** | **72.901** |

El problema contiene 500 clases.

Para cada ejemplo se construye una secuencia de contexto de hasta 257 posiciones, considerando un máximo de 128 palabras a cada lado de la abreviatura objetivo.

Las palabras se convierten en identificadores numéricos mediante un vocabulario previamente generado. Las secuencias se completan mediante padding cuando es necesario.

## 4. Representación de palabras: fastText

Se utiliza el modelo preentrenado `cc.en.300.bin` de fastText para obtener representaciones vectoriales de las palabras.

Las características principales son:

- Dimensión de los embeddings: 300.
- Modelo preentrenado de fastText.
- Representaciones densas de palabras.
- Embeddings congelados durante el entrenamiento de la baseline.

Estas representaciones permiten incorporar información semántica y características de las palabras en la clasificación.

## 5. Arquitectura BiLSTM

El clasificador utiliza una red neuronal recurrente bidireccional para procesar el contexto de la abreviatura.

La arquitectura está formada por:

1. Una capa de embeddings fastText de 300 dimensiones.
2. Una capa BiLSTM bidireccional con tamaño oculto de 256.
3. Dropout del 30 %.
4. Una capa de clasificación que produce 500 salidas, una por cada clase.

La red procesa el contexto en ambas direcciones, permitiendo utilizar información tanto anterior como posterior a la abreviatura.

El entrenamiento se realiza mediante la función de pérdida CrossEntropyLoss.

## 6. Configuración del entrenamiento

La configuración de la baseline es:

- Semilla: 42.
- Optimizador: AdamW.
- Learning rate: 0.001.
- Weight decay: 0.01.
- Batch size: 64.
- Número máximo de épocas: 10.
- Early stopping: paciencia de 2 épocas.
- Dispositivo: CPU.

El mejor modelo se selecciona según el Macro-F1 obtenido sobre el conjunto de validación. Posteriormente, se recupera ese modelo y se realiza la evaluación final sobre el conjunto de test.

## 7. Métricas de evaluación

Se utilizan las siguientes métricas para evaluar el rendimiento:

- Accuracy: proporción de predicciones correctas.
- Macro-F1: media del F1 de todas las clases, dando el mismo peso a cada una.
- Weighted-F1: media del F1 ponderada por el número de ejemplos de cada clase.
- Cohen's Kappa: mide el acuerdo entre las predicciones y las etiquetas reales, teniendo en cuenta el acuerdo esperado por azar.
- Minority-F1: F1 medio de las clases con 20 ejemplos o menos en entrenamiento, cuando existen clases que cumplen ese criterio.

También se generan resultados individuales por clase con precisión, recall, F1 y soporte.

## 8. Ejecución

Desde la carpeta principal del proyecto, con el entorno virtual activado:

**Entrenar la baseline:**

```bash
python -u -m src.entrenamiento baseline 42
```

El comando ejecuta el entrenamiento con la configuración baseline y la semilla 42.

Los resultados se guardan en la carpeta `results/`, mientras que los modelos y recursos utilizados se almacenan en `models/`.

## 9. Resultados de referencia

En la ejecución de referencia de la baseline se obtuvieron los siguientes resultados sobre el conjunto de test:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9915 |
| Macro-F1 | 0.9847 |
| Weighted-F1 | 0.9911 |
| Cohen's Kappa | 0.9915 |
| Mejor época | 7 |

La métrica Minority-F1 no estuvo disponible en esta ejecución porque ninguna clase cumplía el criterio de tener 20 ejemplos o menos en entrenamiento.

Estos resultados constituyen la referencia para comparar las mejoras y alternativas implementadas posteriormente.

## 10. Relación con la implementación individual

Esta baseline constituye el punto de partida del proyecto. La implementación individual desarrolla una arquitectura alternativa basada en Word2Vec + CNN.

Ambas aproximaciones utilizan las mismas particiones de datos y las mismas métricas generales de evaluación, permitiendo comparar sus resultados bajo condiciones experimentales equivalentes.

# Baseline individual: Word2Vec + CNN

## 1. Descripción

Este proyecto implementa una baseline individual de clasificación supervisada de abreviaturas médicas utilizando una arquitectura **CNN (Convolutional Neural Network)** y representaciones vectoriales de palabras obtenidas mediante **Word2Vec**.

El objetivo es estudiar una alternativa a la baseline común del proyecto, basada en BiLSTM y fastText, manteniendo los mismos conjuntos de datos y condiciones generales de experimentación para poder comparar los resultados.

El problema consiste en clasificar las apariciones de abreviaturas médicas en su contexto, asignándolas a una de las 500 clases correspondientes a sus posibles significados.

## 2. Estructura de archivos

Los archivos específicos de esta implementación se encuentran en `src_individual/`:

```text
src_individual/
├── __init__.py
├── config_individual.py
├── word2vec.py
├── modelo_cnn.py
├── entrenamiento_individual.py
└── README.md
```

Los modelos y resultados se almacenan de forma independiente:

```text
models_individual/
├── word2vec.model
├── embeddings_word2vec.npy
└── word2vec_cnn_seed42_best.pt

results_individual/
├── word2vec_cnn_seed42.json
├── word2vec_cnn_seed42_clases.csv
└── word2vec_cnn_seed42_historial.csv
```

La implementación reutiliza algunos módulos comunes del proyecto, especialmente los encargados de cargar y preparar los datos y calcular las métricas.

## 3. Representación de palabras: Word2Vec

Se utiliza Word2Vec para aprender representaciones vectoriales densas a partir del texto del conjunto de entrenamiento.

La configuración empleada es:

- Arquitectura: Skip-gram.
- Dimensión de los embeddings: 300.
- Ventana de contexto: 5 palabras.
- Frecuencia mínima: 1.
- Muestreo negativo: 5.
- Épocas: 10.
- Semilla: 42.

Una vez entrenado Word2Vec, se construye una matriz de embeddings alineada con el vocabulario utilizado por el proyecto. Las palabras que no aparecen en el modelo reciben el vector medio de los embeddings aprendidos, mientras que el token de padding utiliza un vector de ceros.

Los archivos generados son `word2vec.model` y `embeddings_word2vec.npy`.

## 4. Arquitectura CNN

El clasificador utiliza la matriz de embeddings de Word2Vec como capa de entrada.

La arquitectura está formada por:

1. Una capa de embeddings de 300 dimensiones, cuyos pesos permanecen congelados.
2. Tres capas convolucionales paralelas, con tamaños de kernel 3, 4 y 5.
3. 128 filtros por capa convolucional.
4. Activación ReLU.
5. Max pooling global sobre las posiciones válidas de cada secuencia.
6. Concatenación de las características convolucionales con el embedding de la posición objetivo, correspondiente a la abreviatura que se desea desambiguar.
7. Dropout del 30 %.
8. Una capa lineal de clasificación con 500 salidas.

El modelo produce una puntuación para cada clase y se entrena mediante CrossEntropyLoss.

## 5. Datos experimentales

Para mantener la comparabilidad con la baseline común, se utilizan las mismas particiones:

| Conjunto | Número de registros |
|---|---:|
| Entrenamiento | 43.732 |
| Validación | 14.586 |
| Test | 14.583 |
| Total | 72.901 |

Se consideran 500 clases.

## 6. Configuración del entrenamiento

La configuración utilizada es:

- Semilla: 42.
- Optimizador: AdamW.
- Learning rate: 0.001.
- Weight decay: 0.01.
- Batch size: 64.
- Número máximo de épocas: 10.
- Early stopping: paciencia de 2 épocas.
- Dispositivo: CPU.

La selección del mejor modelo se realiza utilizando el Macro-F1 obtenido en el conjunto de validación. Al finalizar, se recupera el mejor modelo y se evalúa sobre el conjunto de test.

## 7. Métricas de evaluación

Se calculan las siguientes métricas:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Cohen's Kappa.
- F1 de las clases minoritarias.

También se guardan las métricas individuales de cada clase, incluyendo precisión, recall, F1 y soporte.

## 8. Ejecución

Desde la carpeta principal del proyecto y con el entorno virtual activado:

**Entrenar Word2Vec y generar los embeddings** (solo si todavía no existen):

```bash
python -m src_individual.word2vec
```

**Comprobar la arquitectura CNN:**

```bash
python -m src_individual.modelo_cnn
```

**Entrenar y evaluar el clasificador:**

```bash
python -u -m src_individual.entrenamiento_individual
```

El entrenamiento utiliza la semilla 42 por defecto. También puede indicarse otra semilla como argumento:

```bash
python -u -m src_individual.entrenamiento_individual 42
```

Los modelos y resultados se guardan en `models_individual/` y `results_individual/`, respectivamente.

## 9. Comparación experimental

Esta implementación permite comparar la baseline individual Word2Vec + CNN con la baseline común BiLSTM + fastText, manteniendo los mismos conjuntos de datos y las métricas de evaluación.

Los resultados cuantitativos se incorporarán una vez finalizado el entrenamiento y la evaluación del modelo.
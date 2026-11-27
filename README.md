# Baseline común: BiLSTM + fastText

# Clasificación supervisada de abreviaturas médicas: BiLSTM + fastText

## 1. Descripción del proyecto

Este proyecto implementa un sistema de clasificación supervisada para la desambiguación de abreviaturas médicas mediante una arquitectura neuronal **BiLSTM (Bidirectional Long Short-Term Memory)** y representaciones vectoriales de palabras obtenidas mediante **fastText**.

El objetivo es clasificar cada aparición de una abreviatura médica según su significado, utilizando el contexto textual en el que aparece.

El problema se plantea como una tarea de clasificación multiclase con 500 clases.

La implementación permite estudiar cuatro experimentos:

| Experimento | Descripción |
|---|---|
| `baseline` | BiLSTM con embeddings fastText congelados. |
| `finetuning` | BiLSTM permitiendo actualizar los embeddings durante el entrenamiento. |
| `attention` | BiLSTM incorporando un mecanismo de atención. |
| `weighted_loss` | BiLSTM utilizando una función de pérdida ponderada según la frecuencia de las clases. |

## 2. Estructura del proyecto

La implementación se encuentra en la carpeta `src/`:

```text
POSTER_AINHOA_TOMAS/
│
├── data/
│   ├── train_clases.parquet
│   ├── valid_clases.parquet
│   └── test_clases.parquet
│
├── models/
│   ├── cc.en.300.bin
│   ├── vocabulario_subconjunto.pkl
│   ├── embeddings_fasttext.npy
│   ├── class_counts.npy
│   └── class_weights.npy
│
├── results/
│
├── src/
│   ├── config.py
│   ├── datos.py
│   ├── vocabulario.py
│   ├── preparacion.py
│   ├── embeddings.py
│   ├── dataset.py
│   ├── modelo.py
│   ├── entrenamiento.py
│   ├── evaluacion.py
│   └── pesos_clases.py
│
├── requirements.txt
└── README.md
```

Las carpetas `models/` y `results/` se generan durante la ejecución cuando no existen.

## 3. Requisitos previos

### 3.1. Software necesario

- Python 3.
- Git.
- Windows, Linux o macOS.
- Conexión a Internet para descargar el modelo preentrenado de fastText.

### 3.2. Clonar el repositorio

Desde una terminal, ejecutar:

```bash
git clone https://github.com/ainhoo222/POSTER_AINHOA_TOMAS.git
```

Entrar en la carpeta del proyecto:

```bash
cd POSTER_AINHOA_TOMAS
```

### 3.3. Crear y activar el entorno virtual

En Windows (PowerShell):

```powershell
py -m venv .venv
```

Activar el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

En Linux o macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Actualizar pip:

```bash
python -m pip install --upgrade pip
```

### 3.4. Instalar las dependencias

Instalar las dependencias del archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Además, instalar PyTorch y fastText, que son necesarios para la arquitectura y la generación de embeddings:

Para PyTorch con CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Para fastText:

```bash
pip install fasttext-wheel
```

Las principales bibliotecas utilizadas son:

- NumPy.
- Pandas.
- PyArrow.
- Scikit-learn.
- PyTorch.
- fastText.
- tqdm.

Comprobar que las bibliotecas principales se han instalado correctamente:

```bash
python -c "import numpy, pandas, pyarrow, sklearn, torch, fasttext; print('Entorno preparado correctamente')"
```

## 4. Preparación de los datos

Antes de generar los modelos es necesario disponer de los tres conjuntos de datos en formato Parquet.

Los archivos deben encontrarse en la carpeta `data/` y tener exactamente los siguientes nombres:

| Archivo | Utilización |
|---|---|
| `train_clases.parquet` | Entrenamiento del modelo y generación del vocabulario y las frecuencias de las clases. |
| `valid_clases.parquet` | Selección del mejor modelo durante el entrenamiento. |
| `test_clases.parquet` | Evaluación final del modelo. |

Los tres archivos deben contener las siguientes columnas:

- `ABSTRACT_ID`: identificador del abstract.
- `TEXT`: texto del contexto médico.
- `LOCATION`: posición de la abreviatura dentro del texto.
- `LABEL`: significado correspondiente a la abreviatura.

La posición `LOCATION` se interpreta como un índice basado en cero sobre las palabras obtenidas mediante `TEXT.split()`.

**Importante:** los tres conjuntos de datos deben estar preparados antes de continuar. Los scripts de entrenamiento no generan estos archivos Parquet.

## 5. Descarga del modelo preentrenado fastText

El proyecto utiliza el modelo preentrenado `cc.en.300.bin`, correspondiente a fastText en inglés, con vectores de 300 dimensiones entrenados sobre Common Crawl.

El modelo original puede descargarse desde:

https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz

Es necesario descargar el archivo comprimido, descomprimirlo y colocar el archivo `cc.en.300.bin` dentro de `models/`.

En Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force models

Invoke-WebRequest `
    -Uri "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz" `
    -OutFile "models/cc.en.300.bin.gz"
```

Descomprimir:

```powershell
tar -xzf models/cc.en.300.bin.gz -C models
```

Una vez terminado, la estructura debe contener:

```text
models/
└── cc.en.300.bin
```

El archivo binario es de gran tamaño, por lo que no es necesario incluirlo en el repositorio GitHub. Cada usuario puede descargarlo desde la fuente original.

## 6. Generación de los recursos necesarios para el entrenamiento

Una vez preparados los datos y descargado fastText, es necesario ejecutar tres módulos en un orden determinado.

Todos los comandos deben ejecutarse desde la carpeta principal del proyecto, con el entorno virtual activado.

### 6.1. Generar el vocabulario

Ejecutar:

```bash
python -m src.vocabulario
```

Este módulo procesa el conjunto de entrenamiento y construye dos diccionarios:

- `word_to_id`: asigna un identificador numérico a cada palabra.
- `label_to_id`: asigna un identificador numérico a cada clase.

Se seleccionan las palabras más frecuentes, con un máximo de 100.000 palabras, y se incorporan las abreviaturas que no estén incluidas.

También se añaden los tokens especiales `<PAD>` y `<UNK>`.

El vocabulario se genera exclusivamente utilizando el conjunto de entrenamiento para evitar introducir información de validación o test.

**Archivo generado:**

```text
models/vocabulario_subconjunto.pkl
```

### 6.2. Generar la matriz de embeddings fastText

Una vez creado el vocabulario, ejecutar:

```bash
python -m src.embeddings
```

Este módulo realiza las siguientes operaciones:

1. Carga el vocabulario generado anteriormente.
2. Carga el modelo preentrenado `cc.en.300.bin`.
3. Obtiene el vector fastText correspondiente a cada palabra.
4. Genera una matriz de embeddings alineada con los identificadores del vocabulario.
5. Guarda la matriz en formato NumPy.

Los embeddings tienen una dimensión de 300. Para palabras que no aparecen directamente en el vocabulario de fastText se utilizan las subpalabras para generar sus representaciones.

Los tokens `<PAD>` y `<UNK>` mantienen vectores de ceros.

**Archivo generado:**

```text
models/embeddings_fasttext.npy
```

Esta matriz se utilizará posteriormente como capa de embeddings de la arquitectura BiLSTM.

### 6.3. Calcular las frecuencias y los pesos de las clases

Ejecutar:

```bash
python -m src.pesos_clases
```

Este módulo recorre el conjunto de entrenamiento y calcula el número de ejemplos pertenecientes a cada clase.

Posteriormente, calcula los pesos mediante la fórmula:

\[
w_i = \frac{1}{\sqrt{n_i}}
\]

Donde \(n_i\) representa el número de ejemplos de la clase \(i\).

Los pesos se normalizan para que su media sea igual a 1.

Se generan dos archivos:

```text
models/
├── class_counts.npy
└── class_weights.npy
```

- `class_counts.npy`: contiene las frecuencias de las clases. Se utiliza durante la evaluación para identificar las clases minoritarias.
- `class_weights.npy`: contiene los pesos calculados para cada clase. Se utiliza en el experimento `weighted_loss`.

## 7. Comprobaciones antes del entrenamiento

Antes de iniciar los experimentos es recomendable comprobar que los datos, los recursos generados y la arquitectura funcionan correctamente.

### 7.1. Comprobar el entorno

```bash
python Comprobar_entorno.py
```

Este script verifica la importación de las principales bibliotecas utilizadas en el proyecto.

### 7.2. Comprobar los conjuntos de datos

```bash
python -m src.datos
```

Se comprueba que los tres archivos Parquet existen, se muestra su número de registros y se inspecciona una pequeña muestra de cada conjunto.

### 7.3. Comprobar la codificación de los registros

```bash
python -m src.preparacion
```

Se comprueba que los textos pueden convertirse correctamente en secuencias de identificadores numéricos.

Se verifica la ventana contextual, la posición de la abreviatura y el identificador de su clase.

### 7.4. Comprobar el DataLoader

```bash
python probar_dataset.py
```

Este script comprueba que el DataLoader puede leer los datos y generar lotes correctamente.

### 7.5. Comprobar el entrenamiento del modelo

```bash
python probar_entrenamiento.py
```

Se ejecuta un paso de entrenamiento de prueba con un lote pequeño para comprobar que:

- Los datos llegan correctamente al modelo.
- La arquitectura produce las predicciones.
- Se calcula la función de pérdida.
- Se pueden calcular los gradientes y actualizar los parámetros.

Esta prueba no realiza el entrenamiento completo.

## 8. Ejecución del entrenamiento

Una vez generados todos los recursos y realizadas las comprobaciones, se puede iniciar el entrenamiento.

El módulo principal es:

```text
src/entrenamiento.py
```

Su ejecución sigue el siguiente formato:

```bash
python -u -m src.entrenamiento <experimento> <semilla>
```

El primer argumento indica el experimento y el segundo la semilla aleatoria.

### 8.1. Entrenar la baseline

```bash
python -u -m src.entrenamiento baseline 42
```

### 8.2. Entrenar con fine-tuning

```bash
python -u -m src.entrenamiento finetuning 42
```

### 8.3. Entrenar con atención

```bash
python -u -m src.entrenamiento attention 42
```

### 8.4. Entrenar con pérdida ponderada

```bash
python -u -m src.entrenamiento weighted_loss 42
```

También pueden repetirse los experimentos utilizando una segunda semilla:

```bash
python -u -m src.entrenamiento baseline 43
python -u -m src.entrenamiento finetuning 43
python -u -m src.entrenamiento attention 43
python -u -m src.entrenamiento weighted_loss 43
```

El parámetro `-u` permite mostrar los mensajes de progreso durante la ejecución sin esperar al almacenamiento en búfer de la salida.

## 9. Configuración del entrenamiento

La configuración se encuentra centralizada en `src/config.py`.

| Parámetro | Valor |
|---|---|
| Arquitectura | BiLSTM |
| Dimensión de embeddings | 300 |
| Tamaño oculto | 256 |
| Número de capas LSTM | 1 |
| Dropout | 0.3 |
| Longitud máxima de secuencia | 257 |
| Radio contextual | 128 |
| Número de clases | 500 |
| Optimizador | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 64 |
| Épocas máximas | 10 |
| Early stopping | Paciencia de 2 épocas |
| Dispositivo | CPU |

La ventana contextual contiene hasta 128 palabras anteriores a la abreviatura, la propia abreviatura y hasta 128 palabras posteriores.

Durante cada época se realiza el entrenamiento sobre el conjunto de entrenamiento y se evalúa el rendimiento sobre validación.

La selección del mejor modelo se realiza utilizando el Macro-F1 de validación. Si no se obtiene una mejora durante dos épocas consecutivas, se activa el mecanismo de Early Stopping.

Una vez finalizado el entrenamiento, se recupera el mejor checkpoint y se realiza la evaluación final sobre el conjunto de test.

## 10. Resultados generados

Los resultados se almacenan automáticamente en la carpeta `results/`.

Para cada experimento y semilla se generan los siguientes archivos:

```text
results/
├── baseline_seed42_best.pt
├── baseline_seed42.json
├── baseline_seed42_clases.csv
├── baseline_seed42_historial.csv
│
├── finetuning_seed42_best.pt
├── finetuning_seed42.json
├── finetuning_seed42_clases.csv
├── finetuning_seed42_historial.csv
│
├── attention_seed42_best.pt
├── attention_seed42.json
├── attention_seed42_clases.csv
├── attention_seed42_historial.csv
│
└── weighted_loss_seed42_best.pt
    ├── weighted_loss_seed42.json
    ├── weighted_loss_seed42_clases.csv
    └── weighted_loss_seed42_historial.csv
```

Los archivos contienen:

- **Checkpoint (`.pt`)**: pesos del mejor modelo, experimento, semilla y época seleccionada.
- **Resultados generales (`.json`)**: métricas finales y configuración de la ejecución.
- **Resultados por clase (`_clases.csv`)**: precisión, recall, F1, soporte y número de ejemplos de entrenamiento de cada clase.
- **Historial (`_historial.csv`)**: evolución de la pérdida de entrenamiento y de las métricas de validación durante las épocas.

### Métricas de evaluación

Se calculan las siguientes métricas:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Cohen's Kappa.
- Minority-F1.

También se calculan precisión, recall y F1 individualmente para cada clase.

Minority-F1 corresponde al F1 medio de las clases que tienen 20 ejemplos o menos en el conjunto de entrenamiento.

## 11. Consideraciones importantes

1. Es necesario disponer de los tres archivos Parquet antes de iniciar el entrenamiento.
2. El vocabulario debe generarse antes que la matriz de embeddings y los pesos de las clases.
3. Si se modifica o regenera el vocabulario, también deben regenerarse los embeddings y los pesos de las clases para mantener la correspondencia entre identificadores.
4. Todos los experimentos utilizan las mismas particiones de entrenamiento, validación y test.
5. Los resultados se guardan automáticamente y no es necesario crearlos manualmente.
6. Los archivos grandes de datos, el modelo fastText y los recursos generados deben estar disponibles localmente, aunque no estén incluidos en el repositorio GitHub.

## 12. Resumen del orden de ejecución

Para reproducir el entrenamiento desde cero, seguir este orden:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fasttext-wheel

# 2. Comprobar los datos
python -m src.datos

# 3. Generar el vocabulario
python -m src.vocabulario

# 4. Generar los embeddings fastText
python -m src.embeddings

# 5. Calcular las frecuencias y pesos de las clases
python -m src.pesos_clases

# 6. Comprobar el entorno y los recursos
python Comprobar_entorno.py
python -m src.preparacion
python probar_dataset.py
python probar_entrenamiento.py

# 7. Ejecutar el experimentos
python -u -m src.entrenamiento baseline 42
```

Una vez completados estos pasos, el proyecto habrá generado los modelos entrenados y sus correspondientes resultados experimentales.

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
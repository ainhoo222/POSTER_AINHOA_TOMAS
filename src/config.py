
from pathlib import Path

# =====================================================
# 1. RUTAS DEL PROYECTO
# =====================================================

# Carpeta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpetas de datos, modelos y resultados
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Archivos del dataset
TRAIN_PATH = DATA_DIR / "train_preparado.parquet"
VALID_PATH = DATA_DIR / "valid_preparado.parquet"
TEST_PATH = DATA_DIR / "test_preparado.parquet"

# Modelo fastText (todavía no descargado)
FASTTEXT_PATH = MODELS_DIR / "cc.en.300.bin"


# =====================================================
# 2. COLUMNAS DEL DATASET
# =====================================================

TEXT_COLUMN = "TEXT"
LOCATION_COLUMN = "LOCATION"
LABEL_COLUMN = "LABEL"
ID_COLUMN = "ABSTRACT_ID"


# =====================================================
# 3. CONFIGURACIÓN DEL MODELO
# =====================================================

# Dimensiones de los embeddings fastText
EMBEDDING_DIM = 300

# Tamaño de la ventana contextual:
# 128 palabras antes + abreviatura + 128 después
MAX_LENGTH = 257
CONTEXT_RADIUS = 128

# Arquitectura BiLSTM
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.3


# =====================================================
# 4. CONFIGURACIÓN DEL ENTRENAMIENTO
# =====================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2

# Semillas para reproducibilidad
SEEDS = [42, 43]

# Dispositivo: entrenamiento en CPU
DEVICE = "cpu"


# =====================================================
# 5. CONFIGURACIÓN DE LOS EXPERIMENTOS
# =====================================================

# Experimentos que estudiaremos:
# E1: fastText congelado (baseline)
# E2: fastText entrenable (fine-tuning)
# E3: BiLSTM con atención
# E4: función de pérdida ponderada

EXPERIMENTS = [
    "baseline",
    "finetuning",
    "attention",
    "weighted_loss"
]


# =====================================================
# 6. TAMAÑOS ESPERADOS DEL DATASET
# =====================================================

EXPECTED_TRAIN_SIZE = 3_000_000
EXPECTED_VALID_SIZE = 999_950
EXPECTED_TEST_SIZE = 999_956

EXPECTED_NUM_CLASSES = 22_555

# =====================================================
# 7. CONFIGURACIÓN DEL VOCABULARIO
# =====================================================

MAX_VOCAB_SIZE = 100_000

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

VOCAB_PATH = MODELS_DIR / "vocabulario.pkl"

# =====================================================
# 8. MATRIZ DE EMBEDDINGS
# =====================================================

import numpy as np

EMBEDDINGS_PATH = MODELS_DIR / "embeddings_fasttext.npy"
EMBEDDINGS_TEST_PATH = MODELS_DIR / "embeddings_prueba.npy"
EMBEDDINGS_DTYPE = np.float32
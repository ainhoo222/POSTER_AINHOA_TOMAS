
from pathlib import Path

# =====================================================
# 1. RUTAS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_INDIVIDUAL_DIR = BASE_DIR / "models_individual"
RESULTS_INDIVIDUAL_DIR = BASE_DIR / "results_individual"

MODELS_INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_clases.parquet"
VALID_PATH = DATA_DIR / "valid_clases.parquet"
TEST_PATH = DATA_DIR / "test_clases.parquet"

# Reutilizamos el vocabulario de la baseline común.
VOCAB_PATH = MODELS_DIR / "vocabulario_subconjunto.pkl"

# Archivos propios de Word2Vec.
WORD2VEC_PATH = MODELS_INDIVIDUAL_DIR / "word2vec.model"
EMBEDDINGS_PATH = MODELS_INDIVIDUAL_DIR / "embeddings_word2vec.npy"


# =====================================================
# 2. COLUMNAS DEL DATASET
# =====================================================

TEXT_COLUMN = "TEXT"
LOCATION_COLUMN = "LOCATION"
LABEL_COLUMN = "LABEL"
ID_COLUMN = "ABSTRACT_ID"


# =====================================================
# 3. PREPARACIÓN DE LOS TEXTOS
# =====================================================

MAX_LENGTH = 257
CONTEXT_RADIUS = 128

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# =====================================================
# 4. WORD2VEC
# =====================================================

EMBEDDING_DIM = 300

WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 1
WORD2VEC_SG = 1
WORD2VEC_NEGATIVE = 5
WORD2VEC_EPOCHS = 10
WORD2VEC_WORKERS = 1


# =====================================================
# 5. ARQUITECTURA CNN
# =====================================================

CNN_FILTERS = 128
CNN_KERNEL_SIZES = (3, 4, 5)
DROPOUT = 0.3


# =====================================================
# 6. ENTRENAMIENTO
# =====================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01

MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2

SEED = 42
DEVICE = "cpu"


# =====================================================
# 7. TAMAÑOS ESPERADOS
# =====================================================

EXPECTED_TRAIN_SIZE = 43_732
EXPECTED_VALID_SIZE = 14_586
EXPECTED_TEST_SIZE = 14_583

EXPECTED_NUM_CLASSES = 500

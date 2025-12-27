import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROFILE = "cpu_dev"  # "cpu_dev", "cpu_full", "gpu_train"


if PROFILE == "cpu_dev":
    SEQ_LEN = 16
    BATCH_SIZE = 16
    EMBED_DIM = 32
    N_HEADS = 1
    N_LAYERS = 1
    LR = 3e-4

    EPOCHS_PER_RUN = 50
    MAX_EPOCHS = None
    MAX_BATCHES = 10
    TRAIN_FRACTION = 1
    VAL_BATCHES= 0

elif PROFILE == "cpu_full":
    SEQ_LEN = 32
    BATCH_SIZE = 32
    EMBED_DIM = 64
    N_HEADS = 2
    N_LAYERS = 2
    LR = 3e-4

    EPOCHS_PER_RUN = 50
    MAX_EPOCHS = None
    MAX_BATCHES = None
    TRAIN_FRACTION = 1.0
    VAL_BATCHES= 20

elif PROFILE == "gpu_train":
    SEQ_LEN = 3072
    BATCH_SIZE = 12
    EMBED_DIM = 1536
    N_HEADS = 24
    N_LAYERS = 32
    LR = 1.5e-4
    PIN_MEMORY = True
    EPOCHS_PER_RUN = 100
    MAX_EPOCHS = None
    MAX_BATCHES = None
    TRAIN_FRACTION = 1.0
    VAL_BATCHES= 50
    GRAD_ACCUM_STEPS= 4
else:
    raise ValueError(f"unknown PROFILE: {PROFILE}")

if DEVICE.startswith("cuda"):
    NUM_WORKERS = 8
    PIN_MEMORY = True
    USE_AMP = True
else:
    NUM_WORKERS = 0
    PIN_MEMORY = False
    USE_AMP = False

GEN_MAX_NEW_TOKENS = 200
GEN_TEMPERATURE = 0.8
GEN_TOP_K = None
GEN_TOP_P = 0.95
DROPOUT = 0.1

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(ROOT_DIR, "data", "char", "raw.txt")
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "char", "SolenaTiny.pth")
VAL_PATH = os.path.join(ROOT_DIR, "data", "char", "val.txt")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "checkpoints", "char", "tokenizer.json")

RESUME = True
SAVE_BEST_ONLY = True
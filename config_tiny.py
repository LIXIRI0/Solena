import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROFILE = "cpu_dev"  # "cpu_dev", "cpu_full", "gpu_train"

GEN_MAX_NEW_TOKENS = 200
GEN_TEMPERATURE = 0.9
GEN_TOP_K = 20
GEN_TOP_P = 0.9
DROPOUT = 0.1

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
    SEQ_LEN = 128
    BATCH_SIZE = 64
    EMBED_DIM = 128
    N_HEADS = 4
    N_LAYERS = 4
    LR = 3e-4
    PIN_MEMORY = True
    EPOCHS_PER_RUN = 10
    MAX_EPOCHS = None
    MAX_BATCHES = None
    TRAIN_FRACTION = 1.0
    VAL_BATCHES= 50
else:
    raise ValueError(f"unknown PROFILE: {PROFILE}")

if DEVICE.startswith("cuda"):
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_AMP = True
else:
    NUM_WORKERS = 0
    PIN_MEMORY = False
    USE_AMP = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(ROOT_DIR, "data", "char", "raw.txt")
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "char", "SolenaTiny.pth")
VAL_PATH = os.path.join(ROOT_DIR, "data", "char", "val.txt")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "checkpoints", "char", "tokenizer.json")

RESUME = True
SAVE_BEST_ONLY = True
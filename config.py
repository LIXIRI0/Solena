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
    TRAIN_FRACTION = 0.1

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

elif PROFILE == "gpu_train":
    SEQ_LEN = 64
    BATCH_SIZE = 64
    EMBED_DIM = 128
    N_HEADS = 4
    N_LAYERS = 4
    LR = 3e-4

    EPOCHS_PER_RUN = 5
    MAX_EPOCHS = 200
    MAX_BATCHES = None
    TRAIN_FRACTION = 1.0

else:
    raise ValueError(f"unknown PROFILE: {PROFILE}")

if DEVICE.startswith("cuda"):
    NUM_WORKERS = 2
    PIN_MEMORY = True
    USE_AMP = True
else:
    NUM_WORKERS = 0
    PIN_MEMORY = False
    USE_AMP = False

DATA_PATH = "data/raw.txt"
CHECKPOINT_PATH = "checkpoints/SolenaTiny.pth"
RESUME = True
SAVE_BEST_ONLY = True  
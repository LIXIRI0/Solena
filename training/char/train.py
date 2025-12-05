import os
import sys
import torch
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_tiny import *
import config_tiny
from utils.char.tokenizer import SimpleCharTokenizer
from utils.char.dataset import TextDataset
from models.char.solena_tiny import SolenaTiny

torch.set_num_threads(4)

text = open(config_tiny.DATA_PATH, "r", encoding="utf-8").read()

if hasattr(config_tiny, "TRAIN_FRACTION"):
    cut = int(len(text) * config_tiny.TRAIN_FRACTION)
    text = text[:cut]

tokenizer = SimpleCharTokenizer(text)
dataset = TextDataset(text, tokenizer, config_tiny.SEQ_LEN)

loader = DataLoader(
    dataset,
    batch_size=config_tiny.BATCH_SIZE,
    shuffle=True,
    num_workers=config_tiny.NUM_WORKERS,
    pin_memory=config_tiny.PIN_MEMORY,
    persistent_workers=False,
)

model = SolenaTiny(
    vocab_size=tokenizer.vocab_size,
    embed_dim=config_tiny.EMBED_DIM,
    n_heads=config_tiny.N_HEADS,
    n_layers=config_tiny.N_LAYERS,
    seq_len=config_tiny.SEQ_LEN,
).to(config_tiny.DEVICE)

optim = torch.optim.AdamW(model.parameters(), lr=config_tiny.LR)

os.makedirs(os.path.dirname(config_tiny.CHECKPOINT_PATH), exist_ok=True)

start_epoch = 0
best_loss = float("inf")
best_epoch = None

if getattr(config_tiny, "RESUME", False) and os.path.exists(config_tiny.CHECKPOINT_PATH):
    ckpt = torch.load(config_tiny.CHECKPOINT_PATH, map_location=config_tiny.DEVICE)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", None)
        if best_epoch is not None:
            print(
                f"resumed from epoch {start_epoch}, "
                f"best_loss={best_loss:.4f} (epoch {best_epoch})"
            )
        else:
            print(f"resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
    else:
        model.load_state_dict(ckpt)
        print("loaded raw state_dict checkpoint (model only)")
else:
    print("no checkpoint, starting from scratch")

end_epoch = start_epoch + config_tiny.EPOCHS_PER_RUN
if getattr(config_tiny, "MAX_EPOCHS", None) is not None:
    end_epoch = min(end_epoch, config_tiny.MAX_EPOCHS)

for epoch in range(start_epoch, end_epoch):
    epoch_loss = 0.0
    batches = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(config_tiny.DEVICE)
        y = y.to(config_tiny.DEVICE)

        optim.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, tokenizer.vocab_size),
            y.view(-1),
        )
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
        batches += 1

        if getattr(config_tiny, "MAX_BATCHES", None):
            if i + 1 >= config_tiny.MAX_BATCHES:
                break

    if batches == 0:
        print(f"epoch {epoch}: no batches, skipping")
        continue

    avg_loss = epoch_loss / batches
    print(f"epoch {epoch} avg_loss {avg_loss:.4f}")

    improved = avg_loss < best_loss
    if improved:
        best_loss = avg_loss
        best_epoch = epoch

    save_payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
    }

    if getattr(config_tiny, "SAVE_BEST_ONLY", False):
        if improved:
            torch.save(save_payload, config_tiny.CHECKPOINT_PATH)
            print(
                f"saved BEST checkpoint at epoch {epoch}, "
                f"best_loss={best_loss:.4f}"
            )
        else:
            print("no improvement, not saving checkpoint this epoch")
    else:
        torch.save(save_payload, config_tiny.CHECKPOINT_PATH)
        print(f"saved checkpoint at epoch {epoch}")

if best_epoch is not None and best_loss < float("inf"):
    print(
        f"\n✔ finished training — best checkpoint at epoch {best_epoch}, "
        f"best_loss={best_loss:.4f}"
    )
else:
    print("\n✔ finished training — no best checkpoint recorded")

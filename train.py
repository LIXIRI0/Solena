import os
import torch
from torch.utils.data import DataLoader

import config
from utils.tokenizer import SimpleCharTokenizer
from utils.dataset import TextDataset
from models.tiny_gpt import TinyGPT

torch.set_num_threads(4)

text = open(config.DATA_PATH, "r", encoding="utf-8").read()

if hasattr(config, "TRAIN_FRACTION"):
    cut = int(len(text) * config.TRAIN_FRACTION)
    text = text[:cut]

tokenizer = SimpleCharTokenizer(text)
dataset = TextDataset(text, tokenizer, config.SEQ_LEN)

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
    persistent_workers=False,
)

model = TinyGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=config.EMBED_DIM,
    n_heads=config.N_HEADS,
    n_layers=config.N_LAYERS,
    seq_len=config.SEQ_LEN,
).to(config.DEVICE)

optim = torch.optim.AdamW(model.parameters(), lr=config.LR)

os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)

start_epoch = 0
best_loss = float("inf")
best_epoch = None

if getattr(config, "RESUME", False) and os.path.exists(config.CHECKPOINT_PATH):
    ckpt = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)

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

end_epoch = start_epoch + config.EPOCHS_PER_RUN
if getattr(config, "MAX_EPOCHS", None) is not None:
    end_epoch = min(end_epoch, config.MAX_EPOCHS)

for epoch in range(start_epoch, end_epoch):
    epoch_loss = 0.0
    batches = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

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

        if getattr(config, "MAX_BATCHES", None):
            if i + 1 >= config.MAX_BATCHES:
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

    if getattr(config, "SAVE_BEST_ONLY", False):
        if improved:
            torch.save(save_payload, config.CHECKPOINT_PATH)
            print(
                f"saved BEST checkpoint at epoch {epoch}, "
                f"best_loss={best_loss:.4f}"
            )
        else:
            print("no improvement, not saving checkpoint this epoch")
    else:
        torch.save(save_payload, config.CHECKPOINT_PATH)
        print(f"saved checkpoint at epoch {epoch}")

if best_epoch is not None and best_loss < float("inf"):
    print(
        f"\n✔ finished training — best checkpoint at epoch {best_epoch}, "
        f"best_loss={best_loss:.4f}"
    )
else:
    print("\n✔ finished training — no best checkpoint recorded")
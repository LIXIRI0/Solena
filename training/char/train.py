import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import config_tiny
from utils.char.tokenizer import SimpleCharTokenizer
from utils.char.dataset import TextDataset
from models.char.solena_tiny import SolenaTiny

torch.set_num_threads(4)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

text = open(config_tiny.DATA_PATH, "r", encoding="utf-8").read()
val_text = open(config_tiny.VAL_PATH, "r", encoding="utf-8").read()

if hasattr(config_tiny, "TRAIN_FRACTION"):
    cut = int(len(text) * config_tiny.TRAIN_FRACTION)
    text = text[:cut]

tokenizer = SimpleCharTokenizer(text)
os.makedirs(os.path.dirname(config_tiny.TOKENIZER_PATH), exist_ok=True)
tokenizer.save(config_tiny.TOKENIZER_PATH)

dataset = TextDataset(text, tokenizer, config_tiny.SEQ_LEN)
val_dataset = TextDataset(val_text, tokenizer, config_tiny.SEQ_LEN)

loader = DataLoader(
    dataset,
    batch_size=config_tiny.BATCH_SIZE,
    shuffle=True,
    num_workers=config_tiny.NUM_WORKERS,
    pin_memory=config_tiny.PIN_MEMORY,
    persistent_workers=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config_tiny.BATCH_SIZE,
    shuffle=False,
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
    dropout=config_tiny.DROPOUT,
).to(config_tiny.DEVICE)

optim = torch.optim.AdamW(model.parameters(), lr=config_tiny.LR)

use_amp = (
    bool(getattr(config_tiny, "USE_AMP", False))
    and str(config_tiny.DEVICE).startswith("cuda")
    and torch.cuda.is_available()
)

amp_dtype = torch.float16
if use_amp and torch.cuda.is_available():
    try:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
    except Exception:
        amp_dtype = torch.float16

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

grad_accum = int(getattr(config_tiny, "GRAD_ACCUM_STEPS", 1))
if grad_accum < 1:
    grad_accum = 1

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
            print(f"resumed from epoch {start_epoch}, best_loss={best_loss:.4f} (epoch {best_epoch})")
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
    model.train()
    epoch_loss = 0.0
    batches = 0

    optim.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(loader):
        x = x.to(config_tiny.DEVICE, non_blocking=True)
        y = y.to(config_tiny.DEVICE, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, tokenizer.vocab_size),
                    y.view(-1),
                )
                loss = loss / grad_accum
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1),
            )
            loss = loss / grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum == 0:
            if use_amp:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * grad_accum
        batches += 1

        if config_tiny.MAX_BATCHES is not None and (i + 1) >= config_tiny.MAX_BATCHES:
            break

    if batches > 0 and (batches % grad_accum) != 0:
        if use_amp:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad(set_to_none=True)

    if batches == 0:
        print(f"epoch {epoch}: no batches, skipping")
        continue

    avg_loss = epoch_loss / batches
    print(f"epoch {epoch} avg_loss {avg_loss:.4f}")

    model.eval()
    avg_val_loss = None
    val_ppl = None

    if getattr(config_tiny, "VAL_BATCHES", None) != 0:
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for _, (x, y) in enumerate(val_loader):
                x = x.to(config_tiny.DEVICE, non_blocking=True)
                y = y.to(config_tiny.DEVICE, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.view(-1, tokenizer.vocab_size),
                            y.view(-1),
                        )
                else:
                    logits = model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, tokenizer.vocab_size),
                        y.view(-1),
                    )

                val_loss += loss.item()
                val_batches += 1

                if config_tiny.VAL_BATCHES is not None and val_batches >= config_tiny.VAL_BATCHES:
                    break

        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            val_ppl = float(torch.exp(torch.tensor(avg_val_loss)))
            print(f"val_loss {avg_val_loss:.4f} | val_ppl {val_ppl:.2f}")

    model.train()

    metric = avg_val_loss if avg_val_loss is not None else avg_loss
    improved = metric < best_loss
    if improved:
        best_loss = metric
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
            if avg_val_loss is not None:
                print(f"epoch {epoch} train_loss {avg_loss:.4f} val_loss {avg_val_loss:.4f} ppl {val_ppl:.2f} — new best, saved checkpoint")
            else:
                print(f"epoch {epoch} train_loss {avg_loss:.4f} — new best, saved checkpoint")
        else:
            print("no improvement, not saving checkpoint this epoch")
    else:
        torch.save(save_payload, config_tiny.CHECKPOINT_PATH)
        print(f"saved checkpoint at epoch {epoch}")

if best_epoch is not None and best_loss < float("inf"):
    print(f"\n✔ finished training — best checkpoint at epoch {best_epoch}, best_loss={best_loss:.4f}")
else:
    print("\n✔ finished training — no best checkpoint recorded")
import os
import torch
import config
from utils.tokenizer import SimpleCharTokenizer
from models.tiny_gpt import TinyGPT

torch.set_num_threads(4)

DEVICE = config.DEVICE
SEQ_LEN = config.SEQ_LEN
CHECKPOINT_PATH = config.CHECKPOINT_PATH
DATA_PATH = config.DATA_PATH

def load_tokenizer():
    text = open(DATA_PATH, "r", encoding="utf-8").read()
    if hasattr(config, "TRAIN_FRACTION"):
        cut = int(len(text) * config.TRAIN_FRACTION)
        text = text[:cut]
    tokenizer = SimpleCharTokenizer(text)
    return tokenizer, text

def load_model(tokenizer):
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.EMBED_DIM,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        seq_len=SEQ_LEN,
    ).to(DEVICE)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"no checkpoint at {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # optional sanity check
    emb_key = "token_emb.weight"
    if emb_key in state_dict:
        ckpt_vocab = state_dict[emb_key].shape[0]
        if ckpt_vocab != tokenizer.vocab_size:
            raise RuntimeError(
                f"vocab size mismatch: checkpoint={ckpt_vocab}, tokenizer={tokenizer.vocab_size}. "
                "make sure TRAIN_FRACTION and data are the same as during training."
            )

    model.load_state_dict(state_dict)
    model.eval()
    return model

def sample(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=None):
    encoded = tokenizer.encode(prompt)
    if len(encoded) == 0:
        encoded = [0]
    tokens = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) > SEQ_LEN:
                tokens = tokens[:, -SEQ_LEN:]

            logits = model(tokens)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < thresh,
                    torch.full_like(logits, -1e10),
                    logits,
                )

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_id], dim=1)

    ids = tokens[0].tolist()
    return tokenizer.decode(ids)

def main():
    tokenizer, _ = load_tokenizer()
    model = load_model(tokenizer)

    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:
            break

        if not prompt.strip():
            continue

        out = sample(
            model,
            tokenizer,
            prompt,
            max_new_tokens=200,
            temperature=0.9,
            top_k=20,
        )
        print("----")
        print(out)
        print("----")

if __name__ == "__main__":
    main()
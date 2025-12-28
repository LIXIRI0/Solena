import os
import torch
import config_tiny as config_tiny
from utils.char.tokenizer import SimpleCharTokenizer
from models.char.solena_tiny import SolenaTiny

torch.set_num_threads(4)

DEVICE = config_tiny.DEVICE
SEQ_LEN = config_tiny.SEQ_LEN
INFER_PATH = config_tiny.INFER_PATH
DATA_PATH = config_tiny.DATA_PATH

def load_tokenizer():
    if hasattr(config_tiny, "TOKENIZER_PATH") and os.path.exists(config_tiny.TOKENIZER_PATH):
        return SimpleCharTokenizer.load(config_tiny.TOKENIZER_PATH)
    text = open(DATA_PATH, "r", encoding="utf-8").read()
    if hasattr(config_tiny, "TRAIN_FRACTION"):
        cut = int(len(text) * config_tiny.TRAIN_FRACTION)
        text = text[:cut]
    return SimpleCharTokenizer(text)

def load_model(tokenizer):
    model = SolenaTiny(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config_tiny.EMBED_DIM,
        n_heads=config_tiny.N_HEADS,
        n_layers=config_tiny.N_LAYERS,
        seq_len=SEQ_LEN,
        dropout=config_tiny.DROPOUT,
    ).to(DEVICE)

    if not os.path.exists(INFER_PATH):
        raise FileNotFoundError(f"no checkpoint at {INFER_PATH}")

    ckpt = torch.load(INFER_PATH, map_location=DEVICE)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    emb_key = "token_emb.weight"
    if emb_key in state_dict:
        ckpt_vocab = state_dict[emb_key].shape[0]
        if ckpt_vocab != tokenizer.vocab_size:
            raise RuntimeError(
                f"vocab mismatch: checkpoint={ckpt_vocab} tokenizer={tokenizer.vocab_size}. "
                f"delete {config_tiny.TOKENIZER_PATH} + retrain, or keep tokenizer/checkpoint paired."
            )
    model.load_state_dict(state_dict)
    model.eval()
    return model

def sample(model, tokenizer, prompt):
    encoded = tokenizer.encode(prompt)
    tokens = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(config_tiny.GEN_MAX_NEW_TOKENS):
            if tokens.size(1) > SEQ_LEN:
                tokens = tokens[:, -SEQ_LEN:]

            logits = model(tokens)
            logits = logits[:, -1, :] / max(config_tiny.GEN_TEMPERATURE, 1e-6)

            if config_tiny.GEN_TOP_K is not None:
                v, _ = torch.topk(logits, config_tiny.GEN_TOP_K)
                thresh = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < thresh, torch.full_like(logits, -1e10), logits)
                
            if getattr(config_tiny, "GEN_TOP_P", None) is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                keep = cum <= config_tiny.GEN_TOP_P
                keep[..., 0] = True
                masked_sorted = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, -1e10))
                inv = torch.empty_like(sorted_idx)
                inv.scatter_(1, sorted_idx, torch.arange(sorted_idx.size(1), device=sorted_idx.device).unsqueeze(0).expand_as(sorted_idx))
                logits = torch.gather(masked_sorted, 1, inv)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_id], dim=1)

            text = tokenizer.decode(tokens[0].tolist())

            start = len(prompt)
            if start < len(text):
                tail = text[start:]
                if "\nUser:" in tail:
                    break

    return tokenizer.decode(tokens[0].tolist())

def extract_assistant(text: str):
    last = text.rfind("Assistant:")
    if last == -1:
        return text.strip()
    out = text[last + len("Assistant:"):]
    cut_user = out.find("\nUser:")
    if cut_user != -1:
        out = out[:cut_user]
    cut_ass = out.find("\nAssistant:")
    if cut_ass != -1:
        out = out[:cut_ass]
    return out.strip(" \n\t")

def main():
    tokenizer = load_tokenizer()
    model = load_model(tokenizer)

    while True:
        try:
            user = input("prompt> ").strip()
        except EOFError:
            break

        if not user:
            continue

        prompt = f"User: {user}\nAssistant:"
        out = sample(model, tokenizer, prompt)
        print("----")
        print(extract_assistant(out))
        print("----")

if __name__ == "__main__":
    main()
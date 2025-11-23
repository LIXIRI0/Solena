import torch

import config
from utils.tokenizer import SimpleCharTokenizer
from models.tiny_gpt import TinyGPT

text = open(config.DATA_PATH, "r", encoding="utf-8").read()
tokenizer = SimpleCharTokenizer(text)

model = TinyGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=config.EMBED_DIM,
    n_heads=config.N_HEADS,
    n_layers=config.N_LAYERS,
    seq_len=config.SEQ_LEN
)
model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location="cpu"))
model.eval()

prompt = "hello"
encoded = tokenizer.encode(prompt)
idx = torch.tensor([encoded], dtype=torch.long)

out = model.generate(idx, max_new_tokens=200)
print(tokenizer.decode(out[0].tolist()))
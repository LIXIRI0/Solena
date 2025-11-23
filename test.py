import torch
from models.solena_tiny import SolenaTiny

vocab_size = 50
embed_dim = 64
n_heads = 4
n_layers = 2
seq_len = 16

model = SolenaTiny(vocab_size, embed_dim, n_heads, n_layers, seq_len)

x = torch.randint(0, vocab_size, (2, seq_len))  # [B, T]
logits = model(x)

print("input shape:", x.shape)
print("logits shape:", logits.shape)
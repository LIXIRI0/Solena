import torch
from torch.utils.data import DataLoader

import config
from utils.tokenizer import SimpleCharTokenizer
from utils.dataset import TextDataset
from models.tiny_gpt import TinyGPT

text = open(config.DATA_PATH, "r", encoding="utf-8").read()

tokenizer = SimpleCharTokenizer(text)
dataset = TextDataset(text, tokenizer, config.SEQ_LEN)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

model = TinyGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=config.EMBED_DIM,
    n_heads=config.N_HEADS,
    n_layers=config.N_LAYERS,
    seq_len=config.SEQ_LEN
).to(config.DEVICE)

optim = torch.optim.AdamW(model.parameters(), lr=config.LR)

for epoch in range(config.EPOCHS):
    for x, y in loader:
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, tokenizer.vocab_size),
            y.view(-1)
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

    print("epoch", epoch, "loss", loss.item())

torch.save(model.state_dict(), config.CHECKPOINT_PATH)
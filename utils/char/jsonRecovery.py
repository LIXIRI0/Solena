import os
import config_tiny
from utils.char.tokenizer import SimpleCharTokenizer

text = open(config_tiny.DATA_PATH, "r", encoding="utf-8").read()

if hasattr(config_tiny, "TRAIN_FRACTION"):
    cut = int(len(text) * config_tiny.TRAIN_FRACTION)
    text = text[:cut]

tokenizer = SimpleCharTokenizer(text)

os.makedirs(os.path.dirname(config_tiny.TOKENIZER_PATH), exist_ok=True)
tokenizer.save(config_tiny.TOKENIZER_PATH)

print(config_tiny.TOKENIZER_PATH)
print(tokenizer.vocab_size)
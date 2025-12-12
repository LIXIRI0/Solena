
---

# 🌙 Solena — A Minimal GPT-Style Transformer

Solena is a tiny educational GPT-like language model built completely from scratch using PyTorch.
It trains on plain text, learns next-character prediction, and can generate coherent sequences.

This project is made to be:

🔥 Simple — minimal, readable architecture

🧪 Hackable — ideal for learning LLM internals

♻️ Resumable — seamless checkpoint save/load

⚡ Lightweight — runs even on weak CPUs / WSL

🎯 Consistent — tokenizer & vocab always align



---

## 📁 Project Structure
```
Solena/
│
├── models/
│   └── solena_tiny.py
│
├── utils/
│   ├── tokenizer.py
│   └── dataset.py
│
├── data/
│   └── raw.txt
│
├── checkpoints/
│   └── SolenaTiny.pth
│
├── train.py
├── generate.py
├── config.py
├── requirements.txt
└── README.md
```

---

### 🚀 Installation

Requires Python 3.9+
```
pip install -r requirements.txt
```
If you're on WSL:
```
sudo apt update
sudo apt install build-essential python3-dev
```

---

### 📦 Add Training Data

Place your text dataset at:
```
data/raw.txt
```
You can train on:

Shakespeare

Song lyrics

Books

Chat logs

Your own writing

ANY plain-text file



---

### 🧠 Training

Just run:
```
python3 train.py
```
The trainer automatically:

Loads config

Resumes from checkpoint if RESUME=True

Saves only the best model if SAVE_BEST_ONLY=True

Supports fractional dataset training (fast debug)


You can run it multiple times — it will continue training seamlessly.


---

### 📝 Text Generation

After training:
```
python3 generate.py
```
Example:
```
prompt> hello
----
helolo hera sor thi...
```
As loss decreases, output quality improves.


---

### ⚙️ Configuration (config.py)

All model & training parameters live inside config.py, including:

Sequence length (SEQ_LEN)

Batch size (BATCH_SIZE)

Learning rate (LR)

Embedding dims, layers, heads

CPU/GPU automatic detection

Checkpoint behavior

Train subset fraction (TRAIN_FRACTION)

Dev/debug modes


You can modify anything at any time.


---

### 🔧 Example Dev Mode Settings
```
SEQ_LEN        = 16
BATCH_SIZE     = 16
EMBED_DIM      = 32
N_HEADS        = 1
N_LAYERS       = 1
EPOCHS_PER_RUN = 10
TRAIN_FRACTION = 0.1
LR             = 3e-4
```
Perfect for weak hardware or WSL.


---

🧪 Example Output
```
prompt> To be or not to be
----
To be or not to beren tomas hir...
```
(Improves significantly over training.)


---

## 🛣️ Roadmap

[ ] Add dropout

[ ] Add learned/rope positional encodings

[ ] Add attention mask

[ ] Add perplexity evaluation

[ ] Add sampling options (top-k, nucleus, temp)

[ ] Add web UI for inference

[ ] Multi-GPU support for cloud GPUs (A10G / T4)



---

###  🤝 Contributing

PRs, issues, and improvements are welcome.
Solena is intentionally minimal to encourage learning and experimentation.


---

### ⚖️ License

MIT License


---

# 🧡 Solena is just the beginning.

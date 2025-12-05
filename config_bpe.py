from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
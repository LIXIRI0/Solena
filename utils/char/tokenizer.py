import json

class SimpleCharTokenizer:
    def __init__(self, text, add_special_tokens=True):
        chars = sorted(list(set(text)))

        self.special_tokens = []
        if add_special_tokens:
            self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

        self.vocab = {}
        self.id_to_token = {}

        idx = 0
        for tok in self.special_tokens:
            self.vocab[tok] = idx
            self.id_to_token[idx] = tok
            idx += 1

        for ch in chars:
            if ch not in self.vocab:
                self.vocab[ch] = idx
                self.id_to_token[idx] = ch
                idx += 1

        self.pad_id = self.vocab.get("<pad>", 0)
        self.bos_id = self.vocab.get("<bos>", 1)
        self.eos_id = self.vocab.get("<eos>", 2)
        self.unk_id = self.vocab.get("<unk>", 3)

        self.vocab_size = len(self.vocab)

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        for ch in text:
            tokens.append(self.vocab.get(ch, self.unk_id))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids, skip_special=True):
        out = []
        for tid in token_ids:
            tok = self.id_to_token.get(int(tid), "")
            if skip_special and tok in self.special_tokens:
                continue
            out.append(tok)
        return "".join(out)

    def to_dict(self):
        return {
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.vocab = {k: int(v) for k, v in d["vocab"].items()}
        obj.id_to_token = {int(v): k for k, v in obj.vocab.items()}
        obj.special_tokens = list(d.get("special_tokens", []))
        obj.pad_id = int(d.get("pad_id", obj.vocab.get("<pad>", 0)))
        obj.bos_id = int(d.get("bos_id", obj.vocab.get("<bos>", 1)))
        obj.eos_id = int(d.get("eos_id", obj.vocab.get("<eos>", 2)))
        obj.unk_id = int(d.get("unk_id", obj.vocab.get("<unk>", 3)))
        obj.vocab_size = len(obj.vocab)
        return obj

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)
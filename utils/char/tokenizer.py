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

        self.pad_id = self.vocab["<pad>"]
        self.bos_id = self.vocab["<bos>"]
        self.eos_id = self.vocab["<eos>"]
        self.unk_id = self.vocab["<unk>"]

        self.vocab_size = len(self.vocab)

    @classmethod
    def from_dict(cls, d):
        obj = cls("", add_special_tokens=False)
        obj.special_tokens = list(d["special_tokens"])
        obj.vocab = {k: int(v) for k, v in d["vocab"].items()}
        obj.id_to_token = {int(k): v for k, v in d["id_to_token"].items()}
        obj.pad_id = obj.vocab["<pad>"]
        obj.bos_id = obj.vocab["<bos>"]
        obj.eos_id = obj.vocab["<eos>"]
        obj.unk_id = obj.vocab["<unk>"]
        obj.vocab_size = len(obj.vocab)
        return obj

    def to_dict(self):
        return {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

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
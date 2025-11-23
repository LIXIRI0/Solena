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
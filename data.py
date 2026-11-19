"""Tiny Shakespeare, character-level: download, tokenize to train.bin/val.bin, sample batches."""
import json
import os
import urllib.request

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def prepare():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(raw):
        urllib.request.urlretrieve(URL, raw)
    with open(raw, encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer("".join(sorted(set(text))))
    ids = np.array(tok.encode(text), dtype=np.uint16)
    n = int(0.9 * len(ids))
    ids[:n].tofile(os.path.join(DATA_DIR, "train.bin"))
    ids[n:].tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump({"chars": tok.chars}, f)
    print(f"{len(text):,} chars, vocab {tok.vocab_size}, train {n:,} / val {len(ids) - n:,} tokens")


def load_tokenizer():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        return CharTokenizer(json.load(f)["chars"])


def load_split(split):
    return np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device):
    """Random windows: x = tokens[i:i+T], y = tokens[i+1:i+T+1]."""
    ix = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    prepare()

"""Tiny Shakespeare: download, tokenize (char or BPE) to data/<name>/{train,val}.bin, sample batches.

python data.py                                  # -> data/char
python data.py --tokenizer bpe --vocab_size 1024  # -> data/bpe1024
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np
import torch

from bpe import BPETokenizer

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


def prepare(tokenizer="char", vocab_size=1024):
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(raw):
        urllib.request.urlretrieve(URL, raw)
    with open(raw, encoding="utf-8") as f:
        text = f.read()
    n = int(0.9 * len(text))
    splits = {"train": text[:n], "val": text[n:]}

    t0 = time.time()
    if tokenizer == "char":
        tok = CharTokenizer("".join(sorted(set(text))))
        meta, out = {"type": "char", "chars": tok.chars}, os.path.join(DATA_DIR, "char")
    else:
        tok = BPETokenizer.train(splits["train"], vocab_size, verbose=True)  # train split only: no val leakage
        meta, out = {"type": "bpe", "merges": tok.merge_list}, os.path.join(DATA_DIR, f"bpe{vocab_size}")
    assert tok.vocab_size <= 2**16, "token ids are stored as uint16"
    os.makedirs(out, exist_ok=True)

    for split, s in splits.items():
        ids = np.array(tok.encode(s), dtype=np.uint16)
        assert tok.decode(ids.tolist()) == s, f"{split} does not round-trip"
        ids.tofile(os.path.join(out, f"{split}.bin"))
        meta[f"{split}_tokens"], meta[f"{split}_bytes"] = len(ids), len(s.encode("utf-8"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"{out}: vocab {tok.vocab_size}, train {meta['train_tokens']:,} / val {meta['val_tokens']:,} tokens, "
          f"{meta['train_bytes'] / meta['train_tokens']:.2f} bytes/token, {time.time() - t0:.1f}s")


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def load_tokenizer(data_dir):
    meta = load_meta(data_dir)
    return CharTokenizer(meta["chars"]) if meta["type"] == "char" else BPETokenizer(meta["merges"])


def load_split(data_dir, split):
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device):
    """Random windows: x = tokens[i:i+T], y = tokens[i+1:i+T+1]."""
    ix = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--vocab_size", type=int, default=1024, help="BPE only")
    a = p.parse_args()
    prepare(a.tokenizer, a.vocab_size)

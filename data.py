"""Download a corpus, tokenize (char or BPE) to data/<name>/{train,val}.bin, sample batches.

python data.py                                                        # Tiny Shakespeare, char   -> data/char
python data.py --tokenizer bpe --vocab_size 1024                      # Tiny Shakespeare, BPE    -> data/bpe1024
python data.py --dataset tinystories --tokenizer bpe --vocab_size 4096  # TinyStories (slice), BPE -> data/tinystories_bpe4096
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
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt"
EOT = "<|endoftext|>"


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


def download(url, path, max_bytes=None):
    if os.path.exists(path):
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {})
    print(f"downloading {url}" + (f" (first {max_bytes / 1e6:.0f} MB)" if max_bytes else ""))
    with urllib.request.urlopen(req) as r, open(path + ".tmp", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(path + ".tmp", path)


def load_text(dataset, train_mb):
    """Returns {"train": str, "val": str}."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if dataset == "shakespeare":
        raw = os.path.join(DATA_DIR, "input.txt")
        download(SHAKESPEARE_URL, raw)
        text = open(raw, encoding="utf-8").read()
        n = int(0.9 * len(text))
        return {"train": text[:n], "val": text[n:]}
    splits = {}
    for split, remote, max_bytes in [("train", "train", train_mb * 10**6), ("val", "valid", None)]:
        raw = os.path.join(DATA_DIR, f"tinystories_{split}{f'_{train_mb}mb' if max_bytes else ''}.txt")
        download(TINYSTORIES_URL.format(remote), raw, max_bytes)
        text = open(raw, encoding="utf-8", errors="ignore").read()  # a byte-range cut can split a UTF-8 char
        # Keep whole stories only: drop text before the first and after the last separator.
        splits[split] = text[text.index(EOT) + len(EOT) : text.rindex(EOT) + len(EOT)].lstrip("\n")
    return splits


def encode_to_file(tok, text, path, piece_chars=1 << 20):
    """Encode in ~1M-char pieces cut at newlines, so a 400MB split never becomes one huge Python list."""
    n_tokens = 0
    with open(path, "wb") as f:
        start = 0
        while start < len(text):
            end = text.find("\n", start + piece_chars)
            end = len(text) if end == -1 else end + 1
            ids = np.array(tok.encode(text[start:end]), dtype=np.uint16)
            ids.tofile(f)
            n_tokens += len(ids)
            start = end
    return n_tokens


def prepare(dataset="shakespeare", tokenizer="char", vocab_size=1024, train_mb=400, bpe_train_mb=50):
    splits = load_text(dataset, train_mb)
    t0 = time.time()
    prefix = "" if dataset == "shakespeare" else f"{dataset}_"
    if tokenizer == "char":
        tok = CharTokenizer("".join(sorted(set(splits["train"]) | set(splits["val"]))))
        meta, out = {"type": "char", "chars": tok.chars}, os.path.join(DATA_DIR, f"{prefix}char")
    else:
        specials = [EOT] if dataset == "tinystories" else []
        # Merges are learned from (a prefix of) the train split only: no val leakage.
        tok = BPETokenizer.train(splits["train"][: bpe_train_mb * 10**6], vocab_size, specials, verbose=True)
        print(f"learned {len(tok.merge_list)} merges in {time.time() - t0:.0f}s")
        meta = {"type": "bpe", "merges": tok.merge_list, "special_tokens": specials}
        out = os.path.join(DATA_DIR, f"{prefix}bpe{vocab_size}")
    assert tok.vocab_size <= 2**16, "token ids are stored as uint16"
    os.makedirs(out, exist_ok=True)

    for split, s in splits.items():
        path = os.path.join(out, f"{split}.bin")
        meta[f"{split}_tokens"] = encode_to_file(tok, s, path)
        meta[f"{split}_bytes"] = len(s.replace(EOT, "").encode("utf-8"))  # bits/byte counts real text only
    val_ids = np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16).tolist()
    assert tok.decode(val_ids) == splits["val"], "val split does not round-trip"
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"{out}: vocab {tok.vocab_size}, train {meta['train_tokens']:,} / val {meta['val_tokens']:,} tokens, "
          f"{meta['train_bytes'] / meta['train_tokens']:.2f} bytes/token, {time.time() - t0:.0f}s total")


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def load_tokenizer(data_dir):
    meta = load_meta(data_dir)
    if meta["type"] == "char":
        return CharTokenizer(meta["chars"])
    return BPETokenizer(meta["merges"], meta.get("special_tokens", []))


def load_split(data_dir, split):
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device):
    """Random windows: x = tokens[i:i+T], y = tokens[i+1:i+T+1]."""
    ix = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":  # pinned memory + non_blocking lets the copy overlap with GPU work
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["shakespeare", "tinystories"], default="shakespeare")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--vocab_size", type=int, default=1024, help="BPE only; includes special tokens")
    p.add_argument("--train_mb", type=int, default=400, help="TinyStories: MB of the 2.2GB train file to use")
    p.add_argument("--bpe_train_mb", type=int, default=50, help="MB of train text to learn BPE merges from")
    a = p.parse_args()
    prepare(a.dataset, a.tokenizer, a.vocab_size, a.train_mb, a.bpe_train_mb)

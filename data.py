"""Download a corpus, tokenize (char or BPE) to data/<name>/{train,val}.bin, sample batches.

python data.py                                                        # Tiny Shakespeare, char   -> data/char
python data.py --tokenizer bpe --vocab_size 1024                      # Tiny Shakespeare, BPE    -> data/bpe1024
python data.py --dataset tinystories --tokenizer bpe --vocab_size 4096  # TinyStories (slice), BPE -> data/tinystories_bpe4096
python data.py --add_v1 data/tinystories_bpe4096_2300mb   # + TinyStories V1 stories not in V2 -> <dir>_v1
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from itertools import chain
from multiprocessing import Pool

import numpy as np
import torch

from bpe import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt"
TINYSTORIES_V1_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-{}.txt"
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


def _text_pieces(text, piece_chars=1 << 20):
    """Cut an in-memory string into ~piece_chars pieces at newlines."""
    start = 0
    while start < len(text):
        end = text.find("\n", start + piece_chars)
        end = len(text) if end == -1 else end + 1
        yield text[start:end]
        start = end


def _story_pieces(path, piece_chars=1 << 20):
    """Stream whole stories from a TinyStories file in ~piece_chars pieces, without loading the file: skip text
    before the first separator and after the last (a byte-range download cuts stories at both ends). Pieces end
    right after an EOT, which the tokenizer splits out first, so the cuts never change the tokenization."""
    with open(path, encoding="utf-8", errors="ignore") as f:  # a byte-range cut can split a UTF-8 char
        buf, started = "", False
        while chunk := f.read(piece_chars):
            buf += chunk
            if not started:
                i = buf.find(EOT)
                if i == -1:
                    continue
                buf, started = buf[i + len(EOT) :].lstrip("\n"), True
            cut = buf.rfind(EOT)
            if cut != -1:
                cut += len(EOT)
                yield buf[:cut]
                buf = buf[cut:]
        # whatever is left after the last EOT is a partial story: dropped


def split_sources(dataset, train_mb):
    """{split: zero-arg callable returning an iterator of text pieces}; nothing large is held in memory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if dataset == "shakespeare":
        raw = os.path.join(DATA_DIR, "input.txt")
        download(SHAKESPEARE_URL, raw)
        text = open(raw, encoding="utf-8").read()
        n = int(0.9 * len(text))
        return {"train": lambda: _text_pieces(text[:n]), "val": lambda: _text_pieces(text[n:])}
    sources = {}
    for split, remote, max_bytes in [("train", "train", train_mb * 10**6), ("val", "valid", None)]:
        raw = os.path.join(DATA_DIR, f"tinystories_{split}{f'_{train_mb}mb' if max_bytes else ''}.txt")
        download(TINYSTORIES_URL.format(remote), raw, max_bytes)
        sources[split] = lambda raw=raw: _story_pieces(raw)
    return sources


_worker_tok = None


def _init_worker(tok):
    global _worker_tok
    _worker_tok = tok


def _encode_piece(piece):
    ids = np.array(_worker_tok.encode(piece), dtype=np.uint16)
    return ids, len(piece.replace(EOT, "").encode("utf-8"))  # bits/byte counts real text only


def encode_to_file(tok, pieces, path, in_flight=32):
    """Encode pieces on all CPU cores and append them to path in order. At most `in_flight` pieces are queued at
    a time (Pool.imap alone would read the whole input ahead), so memory stays flat for multi-GB splits.
    Returns (tokens, bytes)."""
    n_tokens = n_bytes = 0
    with open(path, "wb") as f, Pool(os.cpu_count(), initializer=_init_worker, initargs=(tok,)) as pool:
        batch = []
        for piece in chain(pieces, [None]):  # lazy; None marks the end so the last partial batch is flushed
            if piece is not None:
                batch.append(piece)
            if len(batch) == in_flight or (piece is None and batch):
                for ids, nb in pool.imap(_encode_piece, batch):
                    ids.tofile(f)
                    n_tokens, n_bytes = n_tokens + len(ids), n_bytes + nb
                batch = []
    return n_tokens, n_bytes


def prepare(dataset="shakespeare", tokenizer="char", vocab_size=1024, train_mb=400, bpe_train_mb=50):
    sources = split_sources(dataset, train_mb)
    t0 = time.time()
    prefix = "" if dataset == "shakespeare" else f"{dataset}_"
    if tokenizer == "char":
        chars = set()
        for src in sources.values():
            for piece in src():
                chars |= set(piece)
        tok = CharTokenizer("".join(sorted(chars)))
        meta, out = {"type": "char", "chars": tok.chars}, os.path.join(DATA_DIR, f"{prefix}char")
    else:
        specials = [EOT] if dataset == "tinystories" else []
        bpe_text, n = [], 0
        for piece in sources["train"]():  # merges are learned from a train prefix only: no val leakage
            bpe_text.append(piece)
            n += len(piece)
            if n >= bpe_train_mb * 10**6:
                break
        tok = BPETokenizer.train("".join(bpe_text)[: bpe_train_mb * 10**6], vocab_size, specials, verbose=True)
        del bpe_text
        print(f"learned {len(tok.merge_list)} merges in {time.time() - t0:.0f}s")
        meta = {"type": "bpe", "merges": tok.merge_list, "special_tokens": specials}
        size = f"_{train_mb}mb" if dataset == "tinystories" and train_mb != 400 else ""  # 400 MB: original dir name
        out = os.path.join(DATA_DIR, f"{prefix}bpe{vocab_size}{size}")
    assert tok.vocab_size <= 2**16, "token ids are stored as uint16"
    os.makedirs(out, exist_ok=True)

    for split, src in sources.items():
        meta[f"{split}_tokens"], meta[f"{split}_bytes"] = encode_to_file(tok, src(), os.path.join(out, f"{split}.bin"))
    val_ids = np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16).tolist()
    assert tok.decode(val_ids) == "".join(sources["val"]()), "val split does not round-trip"
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"{out}: vocab {tok.vocab_size}, train {meta['train_tokens']:,} / val {meta['val_tokens']:,} tokens, "
          f"{meta['train_bytes'] / meta['train_tokens']:.2f} bytes/token, {time.time() - t0:.0f}s total")


def _story_key(story):
    return hashlib.blake2b(" ".join(story.split()).encode(), digest_size=8).digest()  # whitespace-insensitive


def new_stories(pieces, seen, piece_chars=1 << 20):
    """Re-chunk the stories in `pieces` whose text is not in `seen` (adding each kept one, so repeats are dropped
    too) into ~piece_chars pieces that end right after an EOT. Returns a generator; counts go in new_stories.stats."""
    new_stories.stats = stats = {"kept": 0, "dropped": 0}

    def gen():
        buf, size = [], 0
        for piece in pieces:
            for story in piece.split(EOT):
                if not story.strip():
                    continue
                k = _story_key(story)
                if k in seen:
                    stats["dropped"] += 1
                    continue
                seen.add(k)
                stats["kept"] += 1
                buf.append(story.strip("\n") + "\n" + EOT + "\n")
                size += len(buf[-1])
                if size >= piece_chars:
                    yield "".join(buf)
                    buf, size = [], 0
        if buf:
            yield "".join(buf)

    return gen()


def add_v1(data_dir):
    """data_dir + TinyStories V1 train stories that are in neither V2 split, with data_dir's tokenizer and val set.
    V1 includes GPT-3.5 stories; V2 is GPT-4 only, so the extra text is more of the same world, a bit noisier."""
    out = data_dir.rstrip("/") + "_v1"
    raw = os.path.join(DATA_DIR, "tinystories_v1_train.txt")
    download(TINYSTORIES_V1_URL.format("train"), raw)
    tok, t0 = load_tokenizer(data_dir), time.time()
    meta = load_meta(data_dir)
    seen = set()
    for path in [os.path.join(DATA_DIR, "tinystories_train_2300mb.txt"), os.path.join(DATA_DIR, "tinystories_val.txt")]:
        for _ in new_stories(_story_pieces(path), seen):
            pass
    print(f"{len(seen):,} V2 stories hashed in {time.time() - t0:.0f}s")
    os.makedirs(out, exist_ok=True)
    extra = new_stories(_story_pieces(raw), seen)
    n_tok, n_bytes = encode_to_file(tok, extra, os.path.join(out, "extra.bin"))
    print(f"V1: {new_stories.stats['kept']:,} new stories, {new_stories.stats['dropped']:,} already in V2; "
          f"{n_tok:,} new tokens")
    with open(os.path.join(out, "train.bin"), "wb") as f:
        for part in [os.path.join(data_dir, "train.bin"), os.path.join(out, "extra.bin")]:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, f, 1 << 24)
    os.remove(os.path.join(out, "extra.bin"))
    shutil.copy(os.path.join(data_dir, "val.bin"), os.path.join(out, "val.bin"))
    meta["train_tokens"] += n_tok
    meta["train_bytes"] += n_bytes
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"{out}: train {meta['train_tokens']:,} tokens, {time.time() - t0:.0f}s total")


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def load_tokenizer(data_dir, extra_special_tokens=()):
    """extra_special_tokens: added after fine-tuning (e.g. chat roles); they get the next ids."""
    meta = load_meta(data_dir)
    if meta["type"] == "char":
        return CharTokenizer(meta["chars"])
    return BPETokenizer(meta["merges"], meta.get("special_tokens", []) + list(extra_special_tokens))


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
    p.add_argument("--add_v1", metavar="DATA_DIR", help="extend a TinyStories BPE dir with deduplicated V1 stories")
    a = p.parse_args()
    if a.add_v1:
        add_v1(a.add_v1)
    else:
        prepare(a.dataset, a.tokenizer, a.vocab_size, a.train_mb, a.bpe_train_mb)

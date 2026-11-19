"""Byte-level BPE (GPT-2/4 style): regex pre-split, learn merges over UTF-8 bytes, so any string encodes."""
import math
import re
from collections import Counter, defaultdict

import regex

# GPT-4's split: contractions, words (with one leading non-letter), up to 3 digits, punctuation runs, whitespace.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
_split = regex.compile(SPLIT_PATTERN)


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges, special_tokens=()):
        """merges: (a, b) id pairs in the order learned; merge k creates id 256 + k.
        special_tokens: strings like "<|endoftext|>", given the ids after the merges; never split or merged."""
        self.merge_list = [tuple(p) for p in merges]
        self.ranks = {p: 256 + k for k, p in enumerate(self.merge_list)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in self.ranks.items():
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.special_tokens = list(special_tokens)
        self.special_ids = {s: 256 + len(self.merge_list) + k for k, s in enumerate(self.special_tokens)}
        for s, i in self.special_ids.items():
            self.vocab[i] = s.encode("utf-8")
        self._special_split = re.compile("(" + "|".join(map(re.escape, self.special_tokens)) + ")") if special_tokens else None
        self.cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merge_list) + len(self.special_tokens)

    @classmethod
    def train(cls, text, vocab_size, special_tokens=(), verbose=False):
        """vocab_size includes the special tokens."""
        n_merges = vocab_size - 256 - len(special_tokens)
        assert n_merges >= 0
        for s in special_tokens:  # specials are atomic: learn merges only from the text between them
            text = text.replace(s, "\n")
        # Merges never cross chunk boundaries, so train on unique chunks weighted by frequency.
        counts = Counter(_split.findall(text))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())

        # Incremental pair counts: after a merge, only words containing the merged pair are re-counted.
        stats, where = Counter(), defaultdict(set)
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                stats[p] += freqs[i]
                where[p].add(i)
        merges = []
        for new_id in range(256, 256 + n_merges):
            if not stats:
                break  # every chunk is a single token already
            pair = max(stats, key=lambda p: (stats[p], p))  # ties broken by pair, so results don't depend on dict order
            touched = set()
            for i in where.pop(pair):
                w, f = words[i], freqs[i]
                for p in zip(w, w[1:]):
                    stats[p] -= f
                    touched.add(p)
                words[i] = w = merge(w, pair, new_id)
                for p in zip(w, w[1:]):
                    stats[p] += f
                    where[p].add(i)  # `where` may keep stale indices; re-merging those is a harmless no-op
            for p in touched:
                if stats[p] <= 0:
                    del stats[p]
            merges.append(pair)
            if verbose and len(merges) % 500 == 0:
                print(f"merge {len(merges)}/{n_merges}: {cls(merges).vocab[new_id]!r}")
        return cls(merges, special_tokens)

    def _encode_chunk(self, data):
        ids = list(data)
        while len(ids) >= 2:
            # Apply the earliest-learned merge present; this reproduces how training tokenized the chunk.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, math.inf))
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, self.ranks[pair])
        return ids

    def encode_ordinary(self, text):
        """Encode ignoring special tokens (their text is encoded as plain bytes)."""
        out = []
        for chunk in _split.findall(text):
            ids = self.cache.get(chunk)
            if ids is None:
                ids = self.cache[chunk] = self._encode_chunk(chunk.encode("utf-8"))
            out.extend(ids)
        return out

    def encode(self, text):
        if self._special_split is None:
            return self.encode_ordinary(text)
        out = []
        for part in self._special_split.split(text):
            if part in self.special_ids:
                out.append(self.special_ids[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids):
        # A token boundary can split a multi-byte character (e.g. mid-sample), so don't crash on it.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

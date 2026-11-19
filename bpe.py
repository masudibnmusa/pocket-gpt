"""Byte-level BPE (GPT-2/4 style): regex pre-split, learn merges over UTF-8 bytes, so any string encodes."""
import math
from collections import Counter

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
    def __init__(self, merges):
        """merges: list of (a, b) id pairs in the order learned; merge k creates id 256 + k."""
        self.merge_list = [tuple(p) for p in merges]
        self.ranks = {p: 256 + k for k, p in enumerate(self.merge_list)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in self.ranks.items():
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merge_list)

    @classmethod
    def train(cls, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # Merges never cross chunk boundaries, so train on unique chunks weighted by frequency.
        counts = Counter(_split.findall(text))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())
        merges = []
        # shortcut: recounts all pairs every merge (O(merges x corpus words)); fine for ~1MB / a few k vocab,
        # switch to incremental pair counts or HF `tokenizers` for 100MB+ corpora.
        for new_id in range(256, vocab_size):
            stats = Counter()
            for w, f in zip(words, freqs):
                for p in zip(w, w[1:]):
                    stats[p] += f
            if not stats:
                break  # every chunk is a single token already
            pair = max(stats, key=stats.get)
            words = [merge(w, pair, new_id) if pair[0] in w else w for w in words]
            merges.append(pair)
            if verbose and (new_id - 255) % 100 == 0:
                tok = cls(merges)
                print(f"merge {new_id - 255}/{vocab_size - 256}: {tok.vocab[new_id]!r} ({stats[pair]:,} occurrences)")
        return cls(merges)

    def _encode_chunk(self, data):
        ids = list(data)
        while len(ids) >= 2:
            # Apply the earliest-learned merge present; this reproduces how training tokenized the chunk.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, math.inf))
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, self.ranks[pair])
        return ids

    def encode(self, text):
        out = []
        for chunk in _split.findall(text):
            ids = self.cache.get(chunk)
            if ids is None:
                ids = self.cache[chunk] = self._encode_chunk(chunk.encode("utf-8"))
            out.extend(ids)
        return out

    def decode(self, ids):
        # A token boundary can split a multi-byte character (e.g. mid-sample), so don't crash on it.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

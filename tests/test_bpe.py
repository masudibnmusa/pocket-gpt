from bpe import BPETokenizer

CORPUS = "the cat sat on the mat. the cat ate the rat! " * 20


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer.train("aaabdaaabac", 257)
    assert tok.merge_list == [(97, 97)]  # "aa" occurs 4 times, more than any other pair


def test_roundtrip_including_unseen_unicode():
    tok = BPETokenizer.train(CORPUS, 300)
    for s in [CORPUS, "", "héllo wörld 🤖 日本語\n\n  tabs\tand 1234567", "the"]:
        assert tok.decode(tok.encode(s)) == s


def test_merges_compress():
    tok = BPETokenizer.train(CORPUS, 300)
    assert len(tok.encode(CORPUS)) < len(CORPUS.encode()) / 2


def test_reload_from_merges_encodes_identically():
    tok = BPETokenizer.train(CORPUS, 300)
    again = BPETokenizer(tok.merge_list)
    assert again.encode(CORPUS) == tok.encode(CORPUS)


def test_incremental_training_matches_naive_recount():
    from collections import Counter

    from bpe import _split, merge

    text = CORPUS + "héllo wörld, the theatre thereafter! 12345 " * 5
    words = [list(w.encode()) for w in _split.findall(text)]
    naive = []
    for new_id in range(256, 400):
        stats = Counter(p for w in words for p in zip(w, w[1:]))
        if not stats:
            break
        pair = max(stats, key=lambda p: (stats[p], p))
        naive.append(pair)
        words = [merge(w, pair, new_id) for w in words]
    assert BPETokenizer.train(text, 400).merge_list == naive


def test_special_tokens_are_atomic():
    tok = BPETokenizer.train(CORPUS, 300, special_tokens=["<|endoftext|>"])
    eot = tok.special_ids["<|endoftext|>"]
    assert eot == tok.vocab_size - 1
    ids = tok.encode("the cat<|endoftext|>the rat")
    assert ids.count(eot) == 1
    assert tok.decode(ids) == "the cat<|endoftext|>the rat"
    assert eot not in tok.encode_ordinary("<|endoftext|>")

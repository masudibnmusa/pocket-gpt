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

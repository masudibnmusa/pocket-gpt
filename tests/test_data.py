from data import EOT, new_stories


def test_new_stories_drops_seen_and_repeats():
    seen = set()
    list(new_stories([f"\nA cat sat.\n{EOT}\nA dog ran.\n{EOT}\n"], seen))
    out = "".join(new_stories([f"A  cat sat.\n{EOT}\nA bird sang.\n{EOT}\nA bird sang.\n{EOT}\n"], seen))
    assert out == f"A bird sang.\n{EOT}\n"
    assert new_stories.stats == {"kept": 1, "dropped": 2}


def test_new_stories_pieces_end_at_eot():
    pieces = list(new_stories([f"story {i}\n{EOT}\n" for i in range(50)], set(), piece_chars=40))
    assert len(pieces) > 1 and all(p.endswith(EOT + "\n") for p in pieces)
    assert "".join(pieces).count(EOT) == 50

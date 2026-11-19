from finetune import make_batch, parse


def test_parse_fields_before_and_after_story():
    fields, story = parse("Words: oak, quit\nStory: \nTom saw an oak.\n\"Hi!\" he said.\nSummary: Tom sees a tree.\n")
    assert fields == {"Words": "oak, quit", "Summary": "Tom sees a tree."}
    assert story == 'Tom saw an oak.\n"Hi!" he said.'


def test_loss_only_on_response_tokens():
    # ids = prompt [10, 11, 12] + response [20, 21, 99]; second example is shorter, so it gets padded
    x, y = make_batch([([10, 11, 12, 20, 21, 99], 3, {}), ([10, 11, 20, 99], 2, {})], "cpu")
    assert x[0].tolist() == [10, 11, 12, 20, 21]
    assert y[0].tolist() == [-100, -100, 20, 21, 99]  # position 2 (last prompt token) predicts the first story token
    assert y[1].tolist() == [-100, 20, 99, -100, -100]  # prompt and padding ignored

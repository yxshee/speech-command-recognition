from speech_commands.labels import MODEL_LABELS, NON_WORD_LABELS, WORD_MASK, WORDS

README_WORDS = (
    "backward, bed, bird, cat, dog, down, eight, five, follow, forward, four, "
    "go, happy, house, learn, left, marvin, nine, no, off, on, one, right, "
    "seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero"
).split(", ")


def test_model_has_41_outputs_of_which_35_are_words():
    assert len(MODEL_LABELS) == 41
    assert len(set(MODEL_LABELS)) == 41
    assert int(WORD_MASK.sum()) == 35
    assert len(NON_WORD_LABELS) == 6


def test_words_match_the_list_in_the_readme():
    assert list(WORDS) == README_WORDS


def test_mask_marks_exactly_the_non_word_labels():
    for label, is_word in zip(MODEL_LABELS, WORD_MASK):
        assert is_word == (label not in NON_WORD_LABELS)

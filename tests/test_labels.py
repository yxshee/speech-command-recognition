from speech_commands.labels import WORDS


def test_there_are_35_unique_words_in_alphabetical_order():
    assert len(WORDS) == 35
    assert len(set(WORDS)) == 35
    assert list(WORDS) == sorted(WORDS)


def test_the_ten_core_commands_are_present():
    for word in ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"):
        assert word in WORDS

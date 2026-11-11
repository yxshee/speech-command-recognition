import pytest

from speech_commands import dataset
from speech_commands.labels import WORDS


def make_fake_dataset(root, words=WORDS):
    """A tiny directory tree shaped like Speech Commands, with empty wav files."""
    for word in words:
        (root / word).mkdir(parents=True)
        for name in ("aaa_nohash_0.wav", "bbb_nohash_0.wav", "ccc_nohash_0.wav"):
            (root / word / name).write_bytes(b"")
    (root / "_background_noise_").mkdir()
    (root / "_background_noise_" / "white_noise.wav").write_bytes(b"")
    (root / "validation_list.txt").write_text("".join(f"{word}/bbb_nohash_0.wav\n" for word in words))
    (root / "testing_list.txt").write_text("".join(f"{word}/ccc_nohash_0.wav\n" for word in words))
    (root / "LICENSE").write_text("not a class")


def test_split_follows_the_official_lists_and_never_overlaps(tmp_path):
    make_fake_dataset(tmp_path)
    splits = dataset.split_files(tmp_path)
    assert {name: len(files) for name, files in splits.items()} == {"train": 35, "validation": 35, "test": 35}
    names = {name: {path for path, _ in files} for name, files in splits.items()}
    assert not names["train"] & names["validation"]
    assert not names["train"] & names["test"]
    assert not names["validation"] & names["test"]
    assert all(path.endswith("ccc_nohash_0.wav") for path in names["test"])


def test_labels_are_indices_into_words(tmp_path):
    make_fake_dataset(tmp_path)
    for path, label in dataset.split_files(tmp_path)["train"]:
        assert f"/{WORDS[label]}/" in path.replace("\\", "/")


def test_noise_and_stray_files_are_never_classes(tmp_path):
    make_fake_dataset(tmp_path)
    for files in dataset.split_files(tmp_path).values():
        assert all("_background_noise_" not in path for path, _ in files)
    assert len(dataset.noise_files(tmp_path)) == 1


def test_a_dataset_with_the_wrong_words_is_rejected(tmp_path):
    make_fake_dataset(tmp_path, words=WORDS[:-1] + ("zebra",))
    with pytest.raises(SystemExit):
        dataset.split_files(tmp_path)

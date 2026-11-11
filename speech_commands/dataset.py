"""Read Speech Commands v0.02 from disk using the dataset's own split lists."""
import tarfile
import urllib.request
from pathlib import Path

from .labels import WORDS

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DEFAULT_DATA_DIR = Path("data") / "speech_commands_v0.02"
NOISE_DIR_NAME = "_background_noise_"


def download(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Download (about 2.4 GB) and extract the dataset into data_dir."""
    data_dir = Path(data_dir)
    if (data_dir / "testing_list.txt").exists():
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    # The archive is kept beside the extraction directory, never inside it:
    # everything inside data_dir is treated as dataset content.
    archive = data_dir.parent / "speech_commands_v0.02.tar.gz"
    if not archive.exists():
        print(f"Downloading {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    print(f"Extracting to {data_dir} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=data_dir, filter="data")


def check_words(data_dir: Path) -> None:
    """Fail loudly if the word directories on disk are not exactly WORDS."""
    found = tuple(sorted(
        p.name for p in Path(data_dir).iterdir() if p.is_dir() and p.name != NOISE_DIR_NAME
    ))
    if found != WORDS:
        raise SystemExit(
            f"Dataset at {data_dir} does not match the expected 35 words.\n"
            f"Missing: {sorted(set(WORDS) - set(found))}\nUnexpected: {sorted(set(found) - set(WORDS))}"
        )


def _read_list(data_dir: Path, name: str) -> list:
    return [line.strip() for line in (Path(data_dir) / name).read_text().splitlines() if line.strip()]


def split_files(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Return {'train': [...], 'validation': [...], 'test': [...]} of (path, label index).

    Validation and test files are exactly those named in the dataset's
    validation_list.txt and testing_list.txt. Those lists keep every speaker in
    a single split, so no voice heard in training appears in the test set.
    """
    data_dir = Path(data_dir)
    check_words(data_dir)
    validation = set(_read_list(data_dir, "validation_list.txt"))
    test = set(_read_list(data_dir, "testing_list.txt"))
    splits = {"train": [], "validation": [], "test": []}
    for index, word in enumerate(WORDS):
        for wav in sorted((data_dir / word).glob("*.wav")):
            relative = f"{word}/{wav.name}"
            if relative in test:
                splits["test"].append((str(wav), index))
            elif relative in validation:
                splits["validation"].append((str(wav), index))
            else:
                splits["train"].append((str(wav), index))
    return splits


def noise_files(data_dir: Path = DEFAULT_DATA_DIR) -> list:
    return sorted(str(p) for p in (Path(data_dir) / NOISE_DIR_NAME).glob("*.wav"))

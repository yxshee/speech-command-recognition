from pathlib import Path

import numpy as np
import pytest

from speech_commands.audio import load_clip
from speech_commands.features import INPUT_SHAPE, log_mel_spectrogram
from speech_commands.inference import load_model, predict, probabilities
from speech_commands.labels import WORDS

SAMPLES = Path(__file__).resolve().parent.parent / "assets" / "samples"


class StubModel:
    """Returns fixed logits, so predict() can be tested without trained weights."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)

    def predict(self, batch, verbose=0):
        assert batch.shape == (1,) + INPUT_SHAPE
        return self.logits[np.newaxis, :]


def test_probabilities_sum_to_one_and_keep_the_order_of_the_logits():
    logits = np.linspace(-3.0, 3.0, len(WORDS))
    probs = probabilities(logits)
    assert probs.sum() == pytest.approx(1.0)
    assert int(np.argmax(probs)) == len(WORDS) - 1
    assert (np.diff(probs) > 0).all()


def test_probabilities_do_not_overflow_on_large_logits():
    probs = probabilities(np.array([1000.0] + [0.0] * (len(WORDS) - 1)))
    assert np.isfinite(probs).all()
    assert probs[0] == pytest.approx(1.0)


def test_predict_returns_the_best_word_and_a_ranked_top_three():
    logits = np.full(len(WORDS), -10.0)
    for word, value in (("stop", 4.0), ("up", 3.0), ("go", 2.0)):
        logits[WORDS.index(word)] = value
    prediction = predict(StubModel(logits), np.zeros(INPUT_SHAPE, dtype=np.float32))
    assert prediction.word == "stop"
    assert [word for word, _ in prediction.top] == ["stop", "up", "go"]
    assert prediction.confidence == pytest.approx(prediction.top[0][1])
    assert 0.0 < prediction.confidence <= 1.0


@pytest.fixture(scope="module")
def trained_model():
    return load_model()


def test_committed_model_has_one_output_per_word(trained_model):
    assert trained_model.output_shape == (None, len(WORDS))
    assert trained_model.input_shape == (None,) + INPUT_SHAPE


@pytest.mark.parametrize("word", ["yes", "stop", "left"])
def test_committed_model_recognises_the_sample_clips(trained_model, word):
    clip = load_clip((SAMPLES / f"{word}.wav").read_bytes())
    prediction = predict(trained_model, log_mel_spectrogram(clip))
    assert prediction.word == word

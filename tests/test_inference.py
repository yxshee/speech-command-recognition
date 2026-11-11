from pathlib import Path

import numpy as np
import pytest

from speech_commands.audio import load_clip
from speech_commands.features import INPUT_SHAPE, spectrogram
from speech_commands.inference import load_model, predict, probabilities
from speech_commands.labels import MODEL_LABELS, WORD_MASK

SAMPLES = Path(__file__).resolve().parent.parent / "assets" / "samples"


class StubModel:
    """Returns fixed logits, so predict() can be tested without TensorFlow weights."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)

    def predict(self, batch, verbose=0):
        assert batch.shape == (1,) + INPUT_SHAPE
        return self.logits[np.newaxis, :]


def test_spectrogram_has_the_model_input_shape():
    features = spectrogram(np.zeros(16000, dtype=np.float32))
    assert features.shape == INPUT_SHAPE
    assert np.isfinite(features).all()
    assert (features >= 0).all()


def test_probabilities_ignore_outputs_that_are_not_words():
    logits = np.zeros(len(MODEL_LABELS))
    logits[list(MODEL_LABELS).index("LICENSE")] = 50.0     # a non-word output wins the raw argmax
    logits[list(MODEL_LABELS).index("yes")] = 5.0
    probs = probabilities(logits)
    assert probs[~WORD_MASK].sum() == 0.0
    assert probs.sum() == pytest.approx(1.0)
    assert MODEL_LABELS[int(np.argmax(probs))] == "yes"


def test_predict_returns_the_best_word_and_a_ranked_top_three():
    logits = np.full(len(MODEL_LABELS), -10.0)
    for word, value in (("stop", 4.0), ("up", 3.0), ("go", 2.0)):
        logits[list(MODEL_LABELS).index(word)] = value
    prediction = predict(StubModel(logits), np.zeros(INPUT_SHAPE, dtype=np.float32))
    assert prediction.word == "stop"
    assert [word for word, _ in prediction.top] == ["stop", "up", "go"]
    assert prediction.confidence == pytest.approx(prediction.top[0][1])
    assert 0.0 < prediction.confidence <= 1.0


@pytest.fixture(scope="module")
def trained_model():
    return load_model()


@pytest.mark.parametrize("word", ["yes", "stop", "left"])
def test_committed_model_recognises_the_sample_clips(trained_model, word):
    clip = load_clip((SAMPLES / f"{word}.wav").read_bytes())
    prediction = predict(trained_model, spectrogram(clip))
    assert prediction.word == word

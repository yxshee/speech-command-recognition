"""Load the trained model and turn its output into a word and a confidence."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from .labels import WORDS

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wavmodel.keras"


@dataclass(frozen=True)
class Prediction:
    word: str
    confidence: float
    top: tuple  # ((word, probability), ...) best first


def load_model(path: Path = MODEL_PATH):
    return tf.keras.models.load_model(path, compile=False)


def probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax over the model's outputs."""
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def predict(model, features, top_k: int = 3) -> Prediction:
    """Predict the spoken word for one feature array of shape INPUT_SHAPE."""
    features = np.asarray(features, dtype=np.float32)
    logits = model.predict(features[np.newaxis, ...], verbose=0)[0]
    probs = probabilities(logits)
    order = np.argsort(probs)[::-1][:top_k]
    top = tuple((WORDS[i], float(probs[i])) for i in order)
    return Prediction(word=top[0][0], confidence=top[0][1], top=top)

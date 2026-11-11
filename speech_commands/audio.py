"""Turn an uploaded audio file into the one-second, 16 kHz clip the model expects."""
import io

import librosa
import numpy as np

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)


class AudioDecodeError(ValueError):
    """The uploaded bytes could not be decoded as audio."""


def fit_length(waveform: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """Zero-pad or trim a mono waveform to exactly `length` samples."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < length:
        return np.pad(waveform, (0, length - len(waveform)), 'constant')
    return waveform[:length]


def load_clip(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes to mono float32 at 16 kHz, exactly one second long.

    Only the first second is decoded, so a long upload costs no more than a
    short one. Stereo files are mixed down to mono.
    """
    if not audio_bytes:
        raise AudioDecodeError("The file is empty.")
    try:
        audio, _ = librosa.load(
            io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS,
        )
    except Exception as error:  # soundfile and audioread raise many unrelated types
        raise AudioDecodeError("This file could not be read as audio.") from error
    if audio.size == 0:
        raise AudioDecodeError("The file contains no audio samples.")
    return fit_length(audio)

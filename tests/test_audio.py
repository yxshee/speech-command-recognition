import io
import wave

import numpy as np
import pytest

from speech_commands.audio import CLIP_SAMPLES, AudioDecodeError, fit_length, load_clip


def wav_bytes(seconds: float, rate: int = 16000, channels: int = 1, frequency: float = 440.0) -> bytes:
    """A 16-bit PCM WAV file containing a sine tone."""
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    frames = np.repeat(tone[:, None], channels, axis=1).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(frames)
    return buffer.getvalue()


def test_fit_length_pads_short_audio_with_zeros():
    out = fit_length(np.ones(10, dtype=np.float32), length=16)
    assert out.shape == (16,)
    assert out[:10].tolist() == [1.0] * 10
    assert out[10:].tolist() == [0.0] * 6


def test_fit_length_trims_long_audio():
    out = fit_length(np.arange(20, dtype=np.float32), length=16)
    assert out.tolist() == list(range(16))


@pytest.mark.parametrize("seconds", [0.4, 1.0, 3.0])
def test_clip_is_always_one_second_of_float32(seconds):
    clip = load_clip(wav_bytes(seconds))
    assert clip.shape == (CLIP_SAMPLES,)
    assert clip.dtype == np.float32
    assert np.isfinite(clip).all()


def test_short_clip_is_padded_with_silence():
    clip = load_clip(wav_bytes(0.25))
    assert np.abs(clip[: CLIP_SAMPLES // 8]).max() > 0.1
    assert np.abs(clip[CLIP_SAMPLES // 2:]).max() == 0.0


@pytest.mark.parametrize("rate", [8000, 22050, 44100])
def test_other_sample_rates_are_resampled(rate):
    clip = load_clip(wav_bytes(1.0, rate=rate))
    assert clip.shape == (CLIP_SAMPLES,)
    assert np.abs(clip).max() > 0.1


def test_stereo_is_mixed_down_to_mono():
    clip = load_clip(wav_bytes(1.0, channels=2))
    assert clip.shape == (CLIP_SAMPLES,)
    assert np.abs(clip).max() > 0.1


@pytest.mark.parametrize("payload", [b"", b"this is not a wav file", b"RIFF\x00\x00\x00\x00WAVE"])
def test_bytes_that_are_not_audio_raise_a_clear_error(payload):
    with pytest.raises(AudioDecodeError):
        load_clip(payload)

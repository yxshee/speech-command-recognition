import numpy as np

from speech_commands.features import CLIP_SAMPLES, INPUT_SHAPE, NUM_FRAMES, NUM_MEL_BINS, log_mel_spectrogram


def test_single_clip_has_the_model_input_shape():
    features = log_mel_spectrogram(np.zeros(CLIP_SAMPLES, dtype=np.float32)).numpy()
    assert features.shape == INPUT_SHAPE == (98, 40, 1)
    assert np.isfinite(features).all()


def test_a_batch_keeps_its_batch_dimension():
    features = log_mel_spectrogram(np.zeros((3, CLIP_SAMPLES), dtype=np.float32)).numpy()
    assert features.shape == (3, NUM_FRAMES, NUM_MEL_BINS, 1)


def test_batched_and_single_clips_give_the_same_features():
    rng = np.random.default_rng(0)
    clips = rng.uniform(-0.5, 0.5, size=(2, CLIP_SAMPLES)).astype(np.float32)
    batched = log_mel_spectrogram(clips).numpy()
    single = log_mel_spectrogram(clips[1]).numpy()
    np.testing.assert_allclose(batched[1], single, rtol=1e-5, atol=1e-5)


def test_a_tone_puts_its_energy_in_the_matching_mel_band():
    t = np.arange(CLIP_SAMPLES) / 16000.0
    low = log_mel_spectrogram((0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)).numpy()[..., 0]
    high = log_mel_spectrogram((0.5 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)).numpy()[..., 0]
    assert low.mean(axis=0).argmax() < high.mean(axis=0).argmax()


def test_louder_audio_gives_larger_features():
    t = np.arange(CLIP_SAMPLES) / 16000.0
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    assert log_mel_spectrogram(0.5 * tone).numpy().max() > log_mel_spectrogram(0.05 * tone).numpy().max()

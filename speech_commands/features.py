"""Model input features. Training and inference must both use this module."""
import numpy as np
import tensorflow as tf

STFT_FRAME_LENGTH = 255
STFT_FRAME_STEP = 128
# The STFT of a one-second clip is [124 time frames, 129 frequency bins].
# The training notebook resized it to [161, 101] before the network, so the
# first axis of the model input is time and the second is frequency.
INPUT_TIME_STEPS = 161
INPUT_FREQUENCY_BINS = 101
INPUT_SHAPE = (INPUT_TIME_STEPS, INPUT_FREQUENCY_BINS, 1)


def spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Magnitude STFT resized to the model input shape, with no dB scaling."""
    stft = tf.signal.stft(
        tf.constant(waveform, dtype=tf.float32),
        frame_length=STFT_FRAME_LENGTH,
        frame_step=STFT_FRAME_STEP,
    )
    magnitude = tf.abs(stft)[..., tf.newaxis]
    return tf.image.resize(magnitude, [INPUT_TIME_STEPS, INPUT_FREQUENCY_BINS]).numpy()

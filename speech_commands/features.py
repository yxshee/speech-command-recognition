"""Model input features. Training, evaluation and the app all use this module,
so the network always sees audio prepared in exactly the same way."""
import tensorflow as tf

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
FRAME_LENGTH = 400      # 25 ms window
FRAME_STEP = 160        # 10 ms hop
FFT_LENGTH = 512
NUM_MEL_BINS = 40
LOWER_EDGE_HERTZ = 20.0
UPPER_EDGE_HERTZ = 7600.0

NUM_FRAMES = 1 + (CLIP_SAMPLES - FRAME_LENGTH) // FRAME_STEP   # 98
INPUT_SHAPE = (NUM_FRAMES, NUM_MEL_BINS, 1)                    # [time, mel, 1]

_MEL_MATRIX = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_MEL_BINS,
    num_spectrogram_bins=FFT_LENGTH // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=LOWER_EDGE_HERTZ,
    upper_edge_hertz=UPPER_EDGE_HERTZ,
)


def log_mel_spectrogram(waveforms):
    """Log-mel spectrogram of one-second clips.

    waveforms: float32 tensor or array of shape [CLIP_SAMPLES] or
    [batch, CLIP_SAMPLES], values in [-1, 1].
    Returns a tensor of shape [..., NUM_FRAMES, NUM_MEL_BINS, 1].
    """
    waveforms = tf.convert_to_tensor(waveforms, dtype=tf.float32)
    stft = tf.signal.stft(
        waveforms,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    power = tf.square(tf.abs(stft))
    mel = tf.tensordot(power, _MEL_MATRIX, axes=1)
    log_mel = tf.math.log(mel + 1e-6)
    return log_mel[..., tf.newaxis]

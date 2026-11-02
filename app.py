"""Speech Command Recognition - Streamlit Application."""
import io
import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0
STFT_FRAME_LENGTH = 255
STFT_FRAME_STEP = 128
FREQ_BINS = 161
TIME_FRAMES = 101
MODEL_PATH = Path(__file__).resolve().parent / "model" / "wavmodel.keras"

# The model's 41 outputs follow the order in which model/model.ipynb listed the
# extracted dataset directory, including a few entries that are not words.
# Index i of the model output is MODEL_LABELS[i], so this order must not change.
MODEL_LABELS = np.array([
    'learn', 'on', 'down', 'follow', 'cat', '.DS_Store', 'three', 'forward',
    'up', 'no', 'sheila', 'stop', 'tree', 'zero', 'LICENSE', 'six', 'four',
    'validation_list.txt', 'seven', 'one', 'right', 'off', 'marvin', 'five',
    'dog', 'two', 'go', 'left', 'testing_list.txt', 'wow', 'bird', 'backward',
    'house', 'visual', 'happy', 'nine', '_background_noise_',
    'speech_commands_v0.02.tar.gz', 'eight', 'bed', 'yes',
])
NON_WORD_LABELS = {
    '.DS_Store', 'LICENSE', 'validation_list.txt', 'testing_list.txt',
    '_background_noise_', 'speech_commands_v0.02.tar.gz',
}
WORD_MASK = np.array([label not in NON_WORD_LABELS for label in MODEL_LABELS])


# Page configuration
st.set_page_config(
    page_title="Speech Command Recognition",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    """Load the pre-trained TensorFlow model."""
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded audio file for prediction."""
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    target_length = int(SAMPLE_RATE * AUDIO_DURATION)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    else:
        audio = audio[:target_length]
    return audio


def get_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Compute the model input exactly as model/model.ipynb did during training.

    Magnitude STFT (frame 255, step 128) resized to 161x101, with no dB scaling
    or normalization.
    """
    stft = tf.signal.stft(
        tf.constant(waveform, dtype=tf.float32),
        frame_length=STFT_FRAME_LENGTH,
        frame_step=STFT_FRAME_STEP,
    )
    spectrogram = tf.abs(stft)[..., tf.newaxis]
    return tf.image.resize(spectrogram, [FREQ_BINS, TIME_FRAMES]).numpy()


def predict_command(spectrogram: np.ndarray, model) -> tuple:
    """Predict the spoken word, ignoring outputs that are not words."""
    logits = model.predict(spectrogram[np.newaxis, ...], verbose=0)[0]
    logits = np.where(WORD_MASK, logits, -np.inf)
    probabilities = tf.nn.softmax(logits).numpy()
    predicted_index = int(np.argmax(probabilities))
    return MODEL_LABELS[predicted_index], float(probabilities[predicted_index])


def plot_waveform(waveform: np.ndarray) -> plt.Figure:
    """Plot the waveform of the audio."""
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.linspace(0, len(waveform) / SAMPLE_RATE, num=len(waveform))
    ax.plot(time, waveform, color='steelblue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_spectrogram(spectrogram_db: np.ndarray) -> plt.Figure:
    """Plot the spectrogram of the audio."""
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(spectrogram_db.T, aspect='auto', origin='lower', cmap='magma')
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Frequency Bins")
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    return fig


def main():
    """Main application entry point."""
    st.title("∿ Speech Command Recognition")
    st.image(
        "https://media.giphy.com/media/9Q5fSHyPKfrr2/giphy.gif",
        width=707
    )
    st.markdown(
        "Upload an audio file, and the model will predict the spoken command. "
        "The app displays the waveform and spectrogram along with the prediction."
    )
    
    model = load_model()
    uploaded_file = st.file_uploader("📤 Upload a WAV audio file", type=["wav"])
    
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        
        # Process audio
        waveform = preprocess_audio(audio_bytes)
        spectrogram = get_spectrogram(waveform)
        
        # Predict
        predicted_command, confidence = predict_command(spectrogram, model)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🗣️ Predicted Command", predicted_command.upper())
        with col2:
            st.metric("📊 Confidence", f"{confidence * 100:.1f}%")
        
        # Visualizations
        st.markdown("#### Waveform")
        st.pyplot(plot_waveform(waveform))
        
        st.markdown("#### Spectrogram")
        st.pyplot(plot_spectrogram(librosa.amplitude_to_db(spectrogram.squeeze(axis=-1), ref=np.max)))


if __name__ == "__main__":
    main()

"""Speech Command Recognition - Streamlit Application."""
import io
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0
SPECTROGRAM_FFT = 320
SPECTROGRAM_HOP = 128
FREQ_BINS = 161
TIME_FRAMES = 101

COMMANDS = np.array([
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop',
    'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
])

# Page configuration
st.set_page_config(
    page_title="Speech Command Recognition",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    """Load the pre-trained TensorFlow model."""
    return tf.keras.models.load_model('wavmodel.keras')


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
    """Generate and normalize a spectrogram from the audio waveform."""
    spectrogram = np.abs(librosa.stft(waveform, n_fft=SPECTROGRAM_FFT, hop_length=SPECTROGRAM_HOP))
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    spectrogram_db = librosa.util.fix_length(spectrogram_db, size=TIME_FRAMES, axis=1)
    spectrogram_db = (spectrogram_db - np.mean(spectrogram_db)) / np.std(spectrogram_db)
    return spectrogram_db[:FREQ_BINS, :, np.newaxis]


def predict_command(spectrogram: np.ndarray, model) -> tuple:
    """Predict the command from the spectrogram using the model."""
    predictions = model.predict(spectrogram[np.newaxis, ...], verbose=0)
    predicted_index = np.argmax(predictions[0])
    confidence = tf.nn.softmax(predictions[0])[predicted_index].numpy()
    return COMMANDS[predicted_index], confidence


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
        spectrogram_db = get_spectrogram(waveform)
        
        # Predict
        predicted_command, confidence = predict_command(spectrogram_db, model)
        
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
        st.pyplot(plot_spectrogram(spectrogram_db.squeeze(axis=-1)))


if __name__ == "__main__":
    main()

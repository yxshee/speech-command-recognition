import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
from scipy.io import wavfile

# Suppress warnings for clean output
import warnings
warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Speech Command Recognition",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("🎤 Speech Command Recognition")

# Description
st.markdown("""
Upload an audio file, and the model will predict the spoken command.
The app will display the waveform and spectrogram of the uploaded audio along with the prediction.
""")

@st.cache_resource
def load_model():
    """Load the pre-trained TensorFlow model."""
    model = tf.keras.models.load_model('wavmodel.keras')
    return model

def load_commands():
    """
    Define the list of commands.
    Modify this list according to the commands your model was trained on.
    """
    # Example commands; replace with your actual commands
    commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    return np.array(commands)

def preprocess_audio(audio_bytes, target_sample_rate=16000, target_duration=1.44):
    """
    Preprocess the uploaded audio file.

    Parameters:
        audio_bytes (bytes): The raw bytes of the uploaded audio file.
        target_sample_rate (int): The sample rate to resample the audio.
        target_duration (float): The duration (in seconds) to pad/truncate the audio.

    Returns:
        np.ndarray: The preprocessed audio waveform.
    """
    # Load audio from bytes
    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate

    # Calculate target length
    target_length = int(target_sample_rate * target_duration)

    # Pad or truncate the audio to the target length
    if len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
    else:
        audio = audio[:target_length]

    return audio

def get_spectrogram(waveform, target_size=101):
    """
    Generate a spectrogram from the audio waveform with a fixed number of time frames.

    Parameters:
        waveform (np.ndarray): The audio waveform.
        target_size (int): The number of time frames to fix.

    Returns:
        np.ndarray: The normalized spectrogram with a channel dimension.
    """
    # Generate spectrogram
    spectrogram = librosa.stft(waveform, n_fft=320, hop_length=160)
    spectrogram = np.abs(spectrogram)

    # Convert to log scale (dB)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Fix the number of time frames to target_size (e.g., 101)
    spectrogram_db = librosa.util.fix_length(spectrogram_db, size=target_size, axis=1)

    # Ensure the spectrogram has exactly 'target_size' time frames
    spectrogram_db = spectrogram_db[:, :target_size]

    # Normalize
    spectrogram_db = (spectrogram_db - np.mean(spectrogram_db)) / np.std(spectrogram_db)

    # Add channel dimension
    spectrogram_db = spectrogram_db[..., np.newaxis]

    return spectrogram_db

def predict_command(spectrogram_db, model, commands):
    """
    Predict the command from the spectrogram using the model.

    Parameters:
        spectrogram_db (np.ndarray): The spectrogram of the audio with a channel dimension.
        model (tf.keras.Model): The pre-trained TensorFlow model.
        commands (np.ndarray): Array of command labels.

    Returns:
        tuple: Predicted command and confidence score.
    """
    # Add batch dimension
    spectrogram_db = np.expand_dims(spectrogram_db, axis=0)  # Shape: (1, 161, 101, 1)

    # Make prediction
    predictions = model.predict(spectrogram_db)
    predicted_index = np.argmax(predictions[0])
    if predicted_index >= len(commands):
        predicted_command = "unknown"
        confidence = 0.0
    else:
        predicted_command = commands[predicted_index]
        confidence = tf.nn.softmax(predictions[0])[predicted_index].numpy()

    return predicted_command, confidence

def plot_waveform(waveform, sample_rate=16000):
    """
    Plot the waveform of the audio.

    Parameters:
        waveform (np.ndarray): The audio waveform.
        sample_rate (int): The sample rate of the audio.

    Returns:
        matplotlib.figure.Figure: The plotted waveform figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))
    ax.plot(time, waveform, color='steelblue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_spectrogram(spectrogram_db):
    """
    Plot the spectrogram of the audio.

    Parameters:
        spectrogram_db (np.ndarray): The spectrogram in dB.

    Returns:
        matplotlib.figure.Figure: The plotted spectrogram figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(spectrogram_db.T, aspect='auto', origin='lower', cmap='magma')
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Frequency Bins")
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    return fig

def main():
    # Load the model and commands
    model = load_model()
    commands = load_commands()

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload a WAV audio file", type=["wav"])

    if uploaded_file is not None:
        # Read the uploaded file bytes
        audio_bytes = uploaded_file.read()

        # Display audio player
        st.audio(audio_bytes, format='audio/wav')

        # Preprocess the audio
        waveform = preprocess_audio(audio_bytes)

        # Generate spectrogram
        spectrogram_db = get_spectrogram(waveform)  # Now with channel dimension

        # Make prediction
        predicted_command, confidence = predict_command(spectrogram_db, model, commands)

        # Display prediction
        st.markdown(f"### 🗣️ Predicted Command: **{predicted_command}**")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

        # Plot and display waveform
        st.markdown("#### Waveform")
        fig_waveform = plot_waveform(waveform)
        st.pyplot(fig_waveform)

        # Plot and display spectrogram
        st.markdown("#### Spectrogram")
        # Reshape spectrogram for plotting
        spectrogram_db = spectrogram_db.squeeze(axis=-1)
        fig_spectrogram = plot_spectrogram(spectrogram_db)
        st.pyplot(fig_spectrogram)

if __name__ == "__main__":
    main()

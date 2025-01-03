import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
import warnings

warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Speech Command Recognition",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("∿ Speech Command Recognition")
st.markdown("""Upload an audio file, and the model will predict the spoken command.
The app will display the waveform and spectrogram of the uploaded audio along with the prediction.
""")

@st.cache_resource
def load_model():
    """Load the pre-trained TensorFlow model."""
    return tf.keras.models.load_model('wavmodel.keras')

@st.cache_data
def load_commands():
    """Load the list of commands the model was trained on."""
    return np.array([
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
        'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop',
        'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ])

def preprocess_audio(audio_bytes, target_sample_rate=16000, target_duration=1.0):
    """Preprocess the uploaded audio file for prediction."""
    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    target_length = int(target_sample_rate * target_duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    else:
        audio = audio[:target_length]
    return audio

def get_spectrogram(waveform):
    """Generate and normalize a spectrogram from the audio waveform."""
    spectrogram = librosa.stft(waveform, n_fft=320, hop_length=128)
    spectrogram = np.abs(spectrogram)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    spectrogram_db = librosa.util.fix_length(spectrogram_db, size=101, axis=1)
    spectrogram_db = (spectrogram_db - np.mean(spectrogram_db)) / np.std(spectrogram_db)
    spectrogram_db = spectrogram_db[..., np.newaxis]  # Add channel dimension
    return spectrogram_db[:161, :, :]  # Ensure only 161 frequency bins

def predict_command(spectrogram, model, commands):
    """Predict the command from the spectrogram using the model."""
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Shape: (1, 161, 101, 1)
    predictions = model.predict(spectrogram)
    predicted_index = np.argmax(predictions[0])
    confidence = tf.nn.softmax(predictions[0])[predicted_index].numpy()
    return commands[predicted_index], confidence

def plot_waveform(waveform, sample_rate=16000):
    """Plot the waveform of the audio."""
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
    model = load_model()
    commands = load_commands()
    uploaded_file = st.file_uploader("📤 Upload a WAV audio file", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        # Preprocess audio and generate spectrogram
        waveform = preprocess_audio(audio_bytes)
        spectrogram_db = get_spectrogram(waveform)

        # Validate shape
        st.write("Spectrogram shape:", spectrogram_db.shape)  # Should print (161, 101, 1)

        # Make prediction
        predicted_command, confidence = predict_command(spectrogram_db, model, commands)

        # Display results
        st.markdown(f"### 🗣️ Predicted Command: **{predicted_command}**")
        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

        # Plot waveform and spectrogram
        st.markdown("#### Waveform")
        st.pyplot(plot_waveform(waveform))
        st.markdown("#### Spectrogram")
        st.pyplot(plot_spectrogram(spectrogram_db.squeeze(axis=-1)))

if __name__ == "__main__":
    main()

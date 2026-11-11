"""Speech Command Recognition - Streamlit application."""
import warnings

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from speech_commands.audio import SAMPLE_RATE, AudioDecodeError, load_clip
from speech_commands.features import spectrogram
from speech_commands.inference import load_model, predict

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Speech Command Recognition",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model():
    """Load the pre-trained model once per server process."""
    return load_model()


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
    """Plot the spectrogram of the audio. Input is [time, frequency]."""
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

    model = get_model()
    uploaded_file = st.file_uploader("📤 Upload a WAV audio file", type=["wav"])

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        try:
            waveform = load_clip(audio_bytes)
        except AudioDecodeError as error:
            st.error(f"{error} Please upload a WAV recording of one spoken word.")
            return
        features = spectrogram(waveform)
        prediction = predict(model, features)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🗣️ Predicted Command", prediction.word.upper())
        with col2:
            st.metric("📊 Confidence", f"{prediction.confidence * 100:.1f}%")

        st.markdown("#### Waveform")
        figure = plot_waveform(waveform)
        st.pyplot(figure)
        plt.close(figure)

        st.markdown("#### Spectrogram")
        figure = plot_spectrogram(librosa.amplitude_to_db(features.squeeze(axis=-1), ref=np.max))
        st.pyplot(figure)
        plt.close(figure)


if __name__ == "__main__":
    main()

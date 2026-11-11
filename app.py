"""Speech Command Recognition - Streamlit application."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from speech_commands.audio import SAMPLE_RATE, AudioDecodeError, load_clip
from speech_commands.features import log_mel_spectrogram
from speech_commands.inference import load_model, predict
from speech_commands.labels import WORDS

REPO_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = REPO_ROOT / "assets" / "samples"
METRICS_PATH = REPO_ROOT / "results" / "metrics.json"
LOW_CONFIDENCE = 0.5

st.set_page_config(page_title="Speech Command Recognition", layout="centered")


@st.cache_resource
def get_model():
    """Load the trained model once per server process."""
    return load_model()


def plot_waveform(waveform: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.linspace(0, len(waveform) / SAMPLE_RATE, num=len(waveform))
    ax.plot(time, waveform, color='steelblue')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform (first second, 16 kHz)")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_spectrogram(log_mel: np.ndarray) -> plt.Figure:
    """Input is [time, mel band]: the exact array the model receives."""
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(log_mel.T, aspect='auto', origin='lower', cmap='magma')
    ax.set_xlabel("Time frame (10 ms each)")
    ax.set_ylabel("Mel band")
    ax.set_title("Log-mel spectrogram (the model's input)")
    fig.colorbar(img, ax=ax, label="log energy")
    fig.tight_layout()
    return fig


def show_figure(figure: plt.Figure) -> None:
    st.pyplot(figure)
    plt.close(figure)


def show_result(audio_bytes: bytes) -> None:
    """Decode one clip, run the model and render the prediction."""
    st.audio(audio_bytes, format="audio/wav")
    try:
        waveform = load_clip(audio_bytes)
    except AudioDecodeError as error:
        st.error(f"{error} Use a WAV recording of one spoken word.")
        return

    features = log_mel_spectrogram(waveform).numpy()
    prediction = predict(get_model(), features)

    word_column, confidence_column = st.columns(2)
    word_column.metric("Predicted word", prediction.word.upper())
    confidence_column.metric("Confidence", f"{prediction.confidence * 100:.1f}%")
    if prediction.confidence < LOW_CONFIDENCE:
        st.warning(
            "The model is not sure about this clip. It may be silence, noise, more "
            "than one word, or a word outside the 35 it was trained on. It always "
            "answers with one of those 35 words."
        )

    st.markdown("#### Top three guesses")
    for word, probability in prediction.top:
        st.progress(min(max(probability, 0.0), 1.0), text=f"{word}  {probability * 100:.1f}%")

    st.markdown("#### What the model heard")
    show_figure(plot_waveform(waveform))
    show_figure(plot_spectrogram(features[..., 0]))


def sidebar() -> None:
    st.sidebar.header("About the model")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        st.sidebar.metric("Test accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        st.sidebar.caption(
            f"Measured on the {metrics['clips']:,} clips of the official "
            f"Speech Commands v0.02 test list, {metrics['classes']} words."
        )
    st.sidebar.markdown(
        "A small convolutional network on log-mel spectrograms. "
        "Only the first second of a clip is used."
    )
    with st.sidebar.expander(f"The {len(WORDS)} words it knows"):
        st.write(", ".join(WORDS))


def main() -> None:
    st.title("Speech Command Recognition")
    st.markdown(
        "Say or upload **one word** and a neural network trained on Google's "
        "Speech Commands dataset will tell you which of 35 words it heard."
    )
    sidebar()

    samples = sorted(SAMPLES_DIR.glob("*.wav"))
    if samples:
        st.markdown("#### Try a sample")
        for column, sample in zip(st.columns(len(samples)), samples):
            if column.button(sample.stem, key=f"sample_{sample.stem}"):
                st.session_state["clip"] = sample.read_bytes()
                st.session_state["clip_source"] = f"sample: {sample.stem}"

    record_tab, upload_tab = st.tabs(["Record", "Upload"])
    with record_tab:
        recording = st.audio_input("Record one word", key="recording")
        if recording is not None:
            st.session_state["clip"] = recording.getvalue()
            st.session_state["clip_source"] = "microphone"
    with upload_tab:
        uploaded = st.file_uploader("Upload a WAV file", type=["wav"], key="upload")
        if uploaded is not None:
            st.session_state["clip"] = uploaded.getvalue()
            st.session_state["clip_source"] = f"upload: {uploaded.name}"

    if "clip" in st.session_state:
        st.divider()
        st.caption(f"Source: {st.session_state['clip_source']}")
        show_result(st.session_state["clip"])


if __name__ == "__main__":
    main()

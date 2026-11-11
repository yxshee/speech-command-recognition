"""Evaluate the committed model on the official Speech Commands v0.02 test list.

    python scripts/evaluate.py

Writes results/metrics.json, results/per_class.md and
results/confusion_matrix.png. The README's numbers are copied from these files.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_commands import dataset                      # noqa: E402
from speech_commands.features import CLIP_SAMPLES, log_mel_spectrogram  # noqa: E402
from speech_commands.inference import MODEL_PATH, load_model  # noqa: E402
from speech_commands.labels import WORDS                  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def decode(path):
    audio, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1, desired_samples=CLIP_SAMPLES)
    return tf.squeeze(audio, axis=-1)


def predict_all(model, paths, batch_size=256):
    ds = tf.data.Dataset.from_tensor_slices(paths).map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).map(log_mel_spectrogram, num_parallel_calls=tf.data.AUTOTUNE)
    logits = model.predict(ds.prefetch(tf.data.AUTOTUNE), verbose=0)
    return np.argmax(logits, axis=1)


def confusion_matrix(true, predicted, size):
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def per_class_metrics(matrix):
    rows = {}
    for index, word in enumerate(WORDS):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[word] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return rows


def top_confusions(matrix, count=10):
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    order = np.argsort(off_diagonal, axis=None)[::-1][:count]
    pairs = []
    for flat in order:
        true, predicted = np.unravel_index(flat, off_diagonal.shape)
        if off_diagonal[true, predicted] == 0:
            break
        pairs.append({"spoken": WORDS[true], "heard_as": WORDS[predicted], "count": int(off_diagonal[true, predicted])})
    return pairs


def save_confusion_plot(matrix, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(WORDS)), WORDS, rotation=90)
    ax.set_yticks(range(len(WORDS)), WORDS)
    ax.set_xlabel("Predicted word")
    ax.set_ylabel("Spoken word")
    ax.set_title("Confusion matrix on the official test list (rows sum to 1)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_per_class_table(rows, path):
    lines = ["| Word | Precision | Recall | F1 | Test clips |", "|---|---:|---:|---:|---:|"]
    for word, row in rows.items():
        lines.append(f"| {word} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=dataset.DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()

    files = dataset.split_files(args.data_dir)[args.split]
    paths = [path for path, _ in files]
    true = np.array([label for _, label in files])

    model = load_model(args.model)
    predicted = predict_all(model, paths)

    matrix = confusion_matrix(true, predicted, len(WORDS))
    rows = per_class_metrics(matrix)
    metrics = {
        "dataset": "Speech Commands v0.02",
        "split": args.split,
        "clips": int(len(true)),
        "classes": len(WORDS),
        "accuracy": float((true == predicted).mean()),
        "macro_f1": float(np.mean([row["f1"] for row in rows.values()])),
        "model_parameters": int(model.count_params()),
        "model_bytes": args.model.stat().st_size,
        "top_confusions": top_confusions(matrix),
        "per_class": rows,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_per_class_table(rows, RESULTS_DIR / "per_class.md")
    save_confusion_plot(matrix, RESULTS_DIR / "confusion_matrix.png")
    print(f"{args.split} accuracy: {metrics['accuracy']:.4f} on {metrics['clips']} clips "
          f"(macro F1 {metrics['macro_f1']:.4f})")
    print(f"Wrote {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()

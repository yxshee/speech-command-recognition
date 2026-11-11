"""Train the keyword-spotting model on Speech Commands v0.02.

    python scripts/train.py --download          # first run: fetches about 2.4 GB
    python scripts/train.py                     # later runs

Uses the dataset's official validation and test lists, 35 word classes, log-mel
features, and background-noise and time-shift augmentation on the training set.
Writes the best model (by validation accuracy) to model/wavmodel.keras and the
training curves to results/training_history.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_commands import dataset                      # noqa: E402
from speech_commands.features import CLIP_SAMPLES, INPUT_SHAPE, log_mel_spectrogram  # noqa: E402
from speech_commands.inference import MODEL_PATH          # noqa: E402
from speech_commands.labels import WORDS                  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAX_SHIFT = 1600            # 100 ms
NOISE_PROBABILITY = 0.8
MAX_NOISE_VOLUME = 0.3


def decode(path, label):
    audio, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1, desired_samples=CLIP_SAMPLES)
    return tf.squeeze(audio, axis=-1), label


def load_noise(paths):
    """All background-noise recordings joined into one long waveform."""
    clips = [tf.squeeze(tf.audio.decode_wav(tf.io.read_file(p), desired_channels=1)[0], axis=-1) for p in paths]
    return tf.concat(clips, axis=0)


def make_augment(noise):
    noise_length = tf.shape(noise)[0]

    def augment(waveform, label):
        # Random shift of up to 100 ms either way, padding with silence.
        shift = tf.random.uniform([], -MAX_SHIFT, MAX_SHIFT + 1, dtype=tf.int32)
        padded = tf.pad(waveform, [[MAX_SHIFT, MAX_SHIFT]])
        waveform = tf.slice(padded, [MAX_SHIFT + shift], [CLIP_SAMPLES])
        # Mix in a random second of background noise most of the time.
        start = tf.random.uniform([], 0, noise_length - CLIP_SAMPLES, dtype=tf.int32)
        volume = tf.random.uniform([], 0.0, MAX_NOISE_VOLUME)
        use_noise = tf.cast(tf.random.uniform([]) < NOISE_PROBABILITY, tf.float32)
        waveform = waveform + use_noise * volume * tf.slice(noise, [start], [CLIP_SAMPLES])
        return tf.clip_by_value(waveform, -1.0, 1.0), label

    return augment


def to_features(waveforms, labels):
    return log_mel_spectrogram(waveforms), labels


def make_dataset(files, batch_size, augment=None, shuffle_seed=None):
    paths = [path for path, _ in files]
    labels = [label for _, label in files]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle_seed is not None:
        ds = ds.shuffle(len(files), seed=shuffle_seed, reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if augment is not None:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).map(to_features, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = layers.BatchNormalization()(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(WORDS))(x)
    return tf.keras.Model(inputs, outputs, name="speech_commands_cnn")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=dataset.DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true", help="download and extract the dataset first")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only this many files per split (for a quick smoke run)")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    if args.download:
        dataset.download(args.data_dir)
    splits = dataset.split_files(args.data_dir)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        splits = {name: [files[i] for i in rng.permutation(len(files))[:args.limit]] for name, files in splits.items()}
    print({name: len(files) for name, files in splits.items()})

    noise = load_noise(dataset.noise_files(args.data_dir))
    train_ds = make_dataset(splits["train"], args.batch_size, augment=make_augment(noise), shuffle_seed=args.seed)
    val_ds = make_dataset(splits["validation"], args.batch_size)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint = RESULTS_DIR / "best.weights.h5"
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", mode="max",
                                               save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5),
        ],
    )

    # Save the best epoch as an inference-only model (no optimizer state).
    model.load_weights(checkpoint)
    export = build_model()
    export.set_weights(model.get_weights())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export.save(args.output)
    checkpoint.unlink()

    curves = {key: [float(v) for v in values] for key, values in history.history.items()}
    best_epoch = int(np.argmax(curves["val_accuracy"])) + 1
    summary = {
        "epochs_run": len(curves["loss"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": curves["val_accuracy"][best_epoch - 1],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "train_files": len(splits["train"]),
        "validation_files": len(splits["validation"]),
        "parameters": int(export.count_params()),
        "history": curves,
    }
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {args.output} (best epoch {best_epoch}, val accuracy {summary['best_val_accuracy']:.4f})")


if __name__ == "__main__":
    main()

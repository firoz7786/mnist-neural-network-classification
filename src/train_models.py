from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
NUM_CLASSES = 10


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_data(validation_fraction: float = 0.10):
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=validation_fraction,
        random_state=SEED,
        stratify=y_train_full,
    )

    dense = {
        "train": x_train.reshape((-1, 28 * 28)),
        "val": x_val.reshape((-1, 28 * 28)),
        "test": x_test.reshape((-1, 28 * 28)),
    }
    cnn = {
        "train": x_train[..., np.newaxis],
        "val": x_val[..., np.newaxis],
        "test": x_test[..., np.newaxis],
    }

    labels = {
        "train": keras.utils.to_categorical(y_train, NUM_CLASSES),
        "val": keras.utils.to_categorical(y_val, NUM_CLASSES),
        "test": keras.utils.to_categorical(y_test, NUM_CLASSES),
        "test_int": y_test,
    }

    return dense, cnn, labels


def build_mlp_baseline():
    return keras.Sequential(
        [
            layers.Input(shape=(28 * 28,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mlp_baseline",
    )


def build_mlp_improved():
    return keras.Sequential(
        [
            layers.Input(shape=(28 * 28,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mlp_improved",
    )


def build_cnn_baseline():
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="cnn_baseline",
    )


def build_cnn_improved():
    return keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="cnn_improved",
    )


def compile_model(model, optimizer):
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(
    name,
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    epochs,
    batch_size,
):
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    return {
        "name": name,
        "model": model,
        "history": history.history,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results") / "local_run")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test only: use 1 epoch per model instead of the full assessment schedule.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    set_seeds()

    dense, cnn, labels = prepare_data()

    baseline_epochs = 1 if args.quick else 10
    improved_epochs = 1 if args.quick else 12

    experiments = [
        (
            "MLP Baseline",
            compile_model(build_mlp_baseline(), keras.optimizers.Adam()),
            dense,
            baseline_epochs,
        ),
        (
            "MLP Improved",
            compile_model(build_mlp_improved(), keras.optimizers.Adam()),
            dense,
            improved_epochs,
        ),
        (
            "CNN Baseline",
            compile_model(build_cnn_baseline(), keras.optimizers.Adam()),
            cnn,
            baseline_epochs,
        ),
        (
            "CNN Improved",
            compile_model(build_cnn_improved(), keras.optimizers.Adam()),
            cnn,
            improved_epochs,
        ),
        (
            "CNN + SGD (lr=0.01)",
            compile_model(
                build_cnn_improved(),
                keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            ),
            cnn,
            improved_epochs,
        ),
        (
            "CNN + SGD (lr=0.005)",
            compile_model(
                build_cnn_improved(),
                keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
            ),
            cnn,
            improved_epochs,
        ),
        (
            "CNN + Adam",
            compile_model(build_cnn_improved(), keras.optimizers.Adam()),
            cnn,
            improved_epochs,
        ),
    ]

    results = []

    for name, model, representation, epochs in experiments:
        print("\n" + "=" * 72)
        print(name)
        print("=" * 72)

        result = train_and_evaluate(
            name=name,
            model=model,
            x_train=representation["train"],
            y_train=labels["train"],
            x_val=representation["val"],
            y_val=labels["val"],
            x_test=representation["test"],
            y_test=labels["test"],
            epochs=epochs,
            batch_size=args.batch_size,
        )
        results.append(result)

        print(f"Test accuracy: {result['test_accuracy']:.4f}")
        print(f"Test loss:     {result['test_loss']:.4f}")

    table = pd.DataFrame(
        [
            {
                "model": r["name"],
                "test_accuracy": r["test_accuracy"],
                "test_loss": r["test_loss"],
            }
            for r in results
        ]
    ).sort_values("test_accuracy", ascending=False)

    table.to_csv(args.output / "portfolio_run_results.csv", index=False)

    best = max(results, key=lambda r: r["test_accuracy"])
    x_test = cnn["test"]
    y_prob = best["model"].predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(labels["test_int"], y_pred)
    np.savetxt(args.output / "portfolio_run_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    report = classification_report(
        labels["test_int"],
        y_pred,
        digits=4,
        output_dict=True,
    )
    (args.output / "portfolio_run_classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    summary = {
        "quick_mode": args.quick,
        "seed": SEED,
        "validation_strategy": "10% stratified split from original MNIST training set",
        "best_model": best["name"],
        "best_test_accuracy": best["test_accuracy"],
        "best_test_loss": best["test_loss"],
    }
    (args.output / "portfolio_run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nFinal ranking:")
    print(table.to_string(index=False))
    print(f"\nBest model: {best['name']}")


if __name__ == "__main__":
    main()

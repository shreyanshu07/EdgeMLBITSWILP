"""
train_model.py — LogiEdge Model Training (Task D1, Variant M1: FP32 Baseline)
================================================================================

Trains a 2-hidden-layer MLP (32, 16 units, ReLU) to classify the cargo
compartment state (0=Normal, 1=Warning, 2=Critical) from the 6-value feature
vector produced by data_pipeline/preprocessing.py.

Inputs are z-score normalised using the FIXED training_stats.npy computed by
generate_dataset.py (10 minutes of clean Normal-class data) -- never
recomputed here.

Validation accuracy on a stratified 20% held-out split must exceed 88%
(hard project requirement). The script exits with an error status if it
does not, as instructed in the problem statement.

Usage:
    python train_model.py --data dataset.csv --stats ../data_pipeline/training_stats.npy \
                           --out models/m1_fp32
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import FEATURE_NAMES, load_training_stats, normalise  # noqa: E402

ACCURACY_THRESHOLD = 0.88
RANDOM_STATE = 42


def build_model(n_features: int, n_classes: int = 3) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,), name="features"),
        tf.keras.layers.Dense(32, activation="relu", name="hidden1"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden2"),
        tf.keras.layers.Dense(n_classes, activation="softmax", name="output"),
    ], name="logiedge_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train LogiEdge MLP classifier (M1 FP32 baseline)")
    parser.add_argument("--data", default="dataset.csv")
    parser.add_argument("--stats", default="../data_pipeline/training_stats.npy")
    parser.add_argument("--out", default="models/m1_fp32")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df[FEATURE_NAMES].values.astype("float32")
    y = df["label"].values.astype("int64")

    mean, std = load_training_stats(args.stats)
    X_norm = normalise(X, mean, std).astype("float32")

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"Train: {len(X_train)} windows | Val: {len(X_val)} windows")
    print("Train class balance:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Val class balance  :", dict(zip(*np.unique(y_val, return_counts=True))))

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    tf.config.experimental.enable_op_determinism()
    model = build_model(n_features=X_norm.shape[1])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True, mode="max",
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=16,
        verbose=2,
        callbacks=[early_stop],
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    print(f"\n=== Held-out validation accuracy: {val_acc * 100:.2f}% (threshold: {ACCURACY_THRESHOLD * 100:.0f}%) ===")
    print(classification_report(y_val, y_pred, target_names=["Normal", "Warning", "Critical"], digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, y_pred))

    # Class 2 (Critical) recall -- required to exceed 95% for Task F3.
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    class2_recall = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else float("nan")
    print(f"Class 2 (Critical) recall: {class2_recall * 100:.2f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.export(args.out)  # SavedModel format, used by convert_ptq.py / prune_quantise.py
    print(f"Model saved (SavedModel format): {args.out}")

    if val_acc < ACCURACY_THRESHOLD:
        print(
            f"\n[FAIL] Validation accuracy {val_acc * 100:.2f}% is BELOW the {ACCURACY_THRESHOLD * 100:.0f}% "
            "threshold required for pharmaceutical cold-chain monitoring. "
            "Revisit feature extraction or architecture before proceeding (see problem statement, Task D1).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n[PASS] Validation accuracy meets the 88% threshold.")


if __name__ == "__main__":
    main()

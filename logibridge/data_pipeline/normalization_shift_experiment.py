"""
normalization_shift_experiment.py — Mandatory Experiment (Task C2)
=====================================================================

"Run inference with correct stats, then with stats shifted by 3-sigma;
report accuracy changes."

This demonstrates WHY training_stats.npy must be fixed and loaded (never
recomputed at runtime): if the deployed statistics silently drift away from
the values the model was actually trained against (e.g. a bug that
recomputes mean/std from live data, or stats copied from the wrong truck/
sensor batch), classification accuracy can collapse even though the model
weights themselves are untouched.

Usage:
    python normalization_shift_experiment.py --data ../training/dataset.csv \
        --stats training_stats.npy --model ../inference/model_m2_ptq_int8.tflite
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import FEATURE_NAMES, load_training_stats, normalise  # noqa: E402


def predict_all(model_path: str, X_norm: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    preds = []
    for row in X_norm:
        x = row.reshape(1, -1).astype("float32")
        if in_detail["dtype"] in (np.int8, np.uint8):
            scale, zp = in_detail["quantization"]
            x = (x / scale + zp).round().astype(in_detail["dtype"])
        interpreter.set_tensor(in_detail["index"], x)
        interpreter.invoke()
        out = interpreter.get_tensor(out_detail["index"])[0]
        preds.append(int(np.argmax(out)))
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser(description="Normalisation-shift mandatory experiment (Task C2)")
    parser.add_argument("--data", default="../training/dataset.csv")
    parser.add_argument("--stats", default="training_stats.npy")
    parser.add_argument("--model", default="../inference/model_m2_ptq_int8.tflite")
    parser.add_argument("--shift-sigma", type=float, default=3.0)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df[FEATURE_NAMES].values.astype("float32")
    y = df["label"].values.astype("int64")

    mean, std = load_training_stats(args.stats)

    # --- Condition 1: correct, unshifted statistics -------------------------
    X_correct = normalise(X, mean, std).astype("float32")
    preds_correct = predict_all(args.model, X_correct)
    acc_correct = accuracy_score(y, preds_correct)

    # --- Condition 2: statistics shifted by shift_sigma * std ---------------
    # Simulates a deployment bug: e.g. stats recomputed from a different
    # truck/sensor batch, or a sensor calibration offset not reflected in the
    # saved training_stats.npy.
    shifted_mean = mean + args.shift_sigma * std
    X_shifted = normalise(X, shifted_mean, std).astype("float32")
    preds_shifted = predict_all(args.model, X_shifted)
    acc_shifted = accuracy_score(y, preds_shifted)

    print("=== Normalisation-shift mandatory experiment (Task C2) ===")
    print(f"Model: {args.model}")
    print(f"Shift applied: mean + {args.shift_sigma} x std (per feature)")
    print(f"Accuracy with CORRECT training_stats.npy : {acc_correct * 100:.2f}%")
    print(f"Accuracy with stats shifted by {args.shift_sigma}-sigma   : {acc_shifted * 100:.2f}%")
    print(f"Accuracy drop                              : {(acc_correct - acc_shifted) * 100:.2f} percentage points")
    print()
    print("Interpretation: a 3-sigma shift in the normalisation statistics moves every "
          "feature value by 3 standard deviations relative to what the model was trained "
          "on, pushing inputs far outside the distribution the model learned decision "
          "boundaries for. This is why training_stats.npy is computed ONCE, offline, from "
          "clean data, saved to disk, and loaded verbatim at runtime -- it must never be "
          "recomputed from live (and potentially already-drifting or differently-scaled) "
          "data, or the model's effective accuracy can collapse even though its weights "
          "are completely unchanged.")


if __name__ == "__main__":
    main()

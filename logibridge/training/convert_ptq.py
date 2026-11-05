"""
convert_ptq.py — LogiEdge Model Conversion: M2 (Post-Training Quantisation, Full INT8)
=========================================================================================

Converts the trained M1 FP32 SavedModel into a Full-INT8 TFLite model using
Post-Training Quantisation (PTQ), as required for Variant M2 in Component F.

Uses >= 200 calibration samples drawn from the training dataset (representative
dataset), per the problem statement's requirement ("Full INT8, >=200 calibration
samples").

Usage:
    python convert_ptq.py --saved-model models/m1_fp32 --data ../training/dataset.csv \
                           --stats ../data_pipeline/training_stats.npy \
                           --out ../inference/model_m2_ptq_int8.tflite
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import FEATURE_NAMES, load_training_stats, normalise  # noqa: E402

MIN_CALIBRATION_SAMPLES = 200


def make_representative_dataset(X_norm: np.ndarray):
    def gen():
        for row in X_norm:
            yield [row.reshape(1, -1).astype("float32")]
    return gen


def main():
    parser = argparse.ArgumentParser(description="Convert LogiEdge model to Full-INT8 TFLite (PTQ)")
    parser.add_argument("--saved-model", default="models/m1_fp32")
    parser.add_argument("--data", default="dataset.csv")
    parser.add_argument("--stats", default="../data_pipeline/training_stats.npy")
    parser.add_argument("--out", default="../inference/model_m2_ptq_int8.tflite")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df[FEATURE_NAMES].values.astype("float32")
    mean, std = load_training_stats(args.stats)
    X_norm = normalise(X, mean, std).astype("float32")

    # Oversample with replacement if the dataset has fewer than the minimum
    # required calibration samples (defensive; our dataset has ~294 rows so
    # this path is not normally exercised).
    if len(X_norm) < MIN_CALIBRATION_SAMPLES:
        idx = np.random.default_rng(0).integers(0, len(X_norm), MIN_CALIBRATION_SAMPLES)
        calib = X_norm[idx]
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_norm), size=max(MIN_CALIBRATION_SAMPLES, len(X_norm)), replace=False) \
            if len(X_norm) >= MIN_CALIBRATION_SAMPLES else np.arange(len(X_norm))
        calib = X_norm[idx]
    print(f"Calibration samples used: {len(calib)} (minimum required: {MIN_CALIBRATION_SAMPLES})")

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(calib)
    # Full INT8: force integer-only ops, including input/output tensors.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"M2 (PTQ INT8) model written: {args.out}  ({size_kb:.1f} KB)")

    # Sanity check: run one inference through the quantised interpreter.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    scale, zero_point = in_detail["quantization"]
    sample = X_norm[0:1]
    q_input = (sample / scale + zero_point).astype(in_detail["dtype"])
    interpreter.set_tensor(in_detail["index"], q_input)
    interpreter.invoke()
    out = interpreter.get_tensor(out_detail["index"])
    print(f"Sanity check inference OK. Input dtype={in_detail['dtype']}, output dtype={out_detail['dtype']}, "
          f"raw output={out}")


if __name__ == "__main__":
    main()

"""
prune_quantise.py — LogiEdge Model Optimisation: M3 (Structured Pruning + PTQ INT8)
======================================================================================

Produces Variant M3: 35% magnitude-based pruning (PolynomialDecay schedule,
via the TensorFlow Model Optimization Toolkit) followed by fine-tuning and
Full-INT8 Post-Training Quantisation.

IMPORTANT TECHNICAL NOTE (read before your viva):
    Module 6 distinguishes *unstructured* pruning (individual weights set to
    zero -> high sparsity but NO speed-up on standard, non-sparse hardware)
    from *structured* pruning (whole filters/channels/neurons removed ->
    the weight tensor actually shrinks, which DOES speed up standard
    hardware). `tfmot.sparsity.keras.prune_low_magnitude` -- the toolkit
    this project uses, exactly as named in the problem statement -- performs
    MAGNITUDE-BASED, PER-WEIGHT pruning. On a Dense (fully-connected) layer
    like ours, this is technically "unstructured" sparsity in the strict
    Module 6 sense: the pruned weights are zeroed but the weight matrix
    keeps its original shape, so TFLite's standard (non-sparse) kernels see
    no FLOP reduction from pruning alone -- the real benefit for this
    project comes from the SUBSEQUENT Full-INT8 quantisation step, and from
    pruning improving quantisation robustness (zeroed small-magnitude
    weights quantise more cleanly). We still target 35% final sparsity via
    PolynomialDecay exactly as specified, and we report model-FILE-SIZE and
    latency improvements empirically in benchmark.py rather than assuming
    pruning alone speeds up inference on this non-sparse-aware runtime.
    Be ready to explain this distinction if asked in the viva.

Usage:
    python prune_quantise.py --saved-model models/m1_fp32 --data dataset.csv \
                              --stats ../data_pipeline/training_stats.npy \
                              --out ../inference/model_m3_pruned_int8.tflite
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import FEATURE_NAMES, load_training_stats, normalise  # noqa: E402

FINAL_SPARSITY = 0.35
RAMP_EPOCHS = 15        # epochs over which sparsity ramps 0% -> 35%
RECOVERY_EPOCHS = 25    # additional epochs fine-tuning AT the fixed 35% sparsity
FINE_TUNE_EPOCHS = RAMP_EPOCHS + RECOVERY_EPOCHS
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser(description="LogiEdge M3: 35% pruning + Full-INT8 PTQ")
    parser.add_argument("--saved-model", default="models/m1_fp32")
    parser.add_argument("--data", default="dataset.csv")
    parser.add_argument("--stats", default="../data_pipeline/training_stats.npy")
    parser.add_argument("--pruned-keras-out", default="models/m3_pruned.keras")
    parser.add_argument("--out", default="../inference/model_m3_pruned_int8.tflite")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    tf.config.experimental.enable_op_determinism()

    df = pd.read_csv(args.data)
    X = df[FEATURE_NAMES].values.astype("float32")
    y = df["label"].values.astype("int64")
    mean, std = load_training_stats(args.stats)
    X_norm = normalise(X, mean, std).astype("float32")

    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )

    # Load the M1 baseline and rebuild an identical, freshly-instantiated
    # Keras model so we can wrap its Dense layers with pruning. (A loaded
    # SavedModel's inference-only signature cannot be wrapped directly, so
    # we reconstruct the architecture and transfer weights.)
    base = tf.keras.layers.TFSMLayer(args.saved_model, call_endpoint="serving_default") \
        if hasattr(tf.keras.layers, "TFSMLayer") else None

    # Reconstruct architecture (must match train_model.py exactly) and load weights
    # via the SavedModel's concrete function, since pruning wrappers need a
    # standard tf.keras.Model built from layers, not a generic SavedModel.
    loaded = tf.saved_model.load(args.saved_model)
    serve_fn = loaded.signatures["serving_default"]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_norm.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu", name="hidden1"),
        tf.keras.layers.Dense(16, activation="relu", name="hidden2"),
        tf.keras.layers.Dense(3, activation="softmax", name="output"),
    ])
    # Transfer weights from the SavedModel's variables, matched by shape/order.
    sm_vars = {v.name: v for v in loaded.variables}
    keras_weights = []
    sm_var_list = list(loaded.variables)
    for layer in model.layers:
        w, b = layer.get_weights()
        # find matching kernel/bias by shape among remaining SavedModel vars
        match_w = next(v for v in sm_var_list if tuple(v.shape) == tuple(w.shape))
        sm_var_list.remove(match_w)
        match_b = next(v for v in sm_var_list if tuple(v.shape) == tuple(b.shape))
        sm_var_list.remove(match_b)
        layer.set_weights([match_w.numpy(), match_b.numpy()])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    base_acc = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"Reconstructed M1 model val accuracy (sanity check): {base_acc * 100:.2f}%")

    # --- Apply magnitude-based pruning with a PolynomialDecay schedule -----
    # `end_step` is deliberately set to RAMP_EPOCHS (not the full training
    # run): sparsity ramps 0% -> 35% over the first RAMP_EPOCHS epochs, then
    # stays fixed at 35% for RECOVERY_EPOCHS more epochs so the model can
    # fine-tune and recover accuracy AT the final sparsity level. Setting
    # end_step == total epochs (an earlier bug in this script) leaves no
    # recovery phase and destabilises training right as it ends.
    # `frequency` (mask-recompute interval, in steps) is lowered from the
    # tfmot default of 100 to 2, because this tiny dataset has only ~15
    # steps/epoch -- the default would barely update the mask at all.
    steps_per_epoch = len(X_train) // 16 + 1
    end_step = RAMP_EPOCHS * steps_per_epoch
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=FINAL_SPARSITY,
        begin_step=0, end_step=end_step, frequency=2,
    )
    prune_params = {"pruning_schedule": pruning_schedule}

    pruned_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_norm.shape[1],)),
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(32, activation="relu", name="hidden1"), **prune_params),
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(16, activation="relu", name="hidden2"), **prune_params),
        tfmot.sparsity.keras.prune_low_magnitude(
            tf.keras.layers.Dense(3, activation="softmax", name="output"), **prune_params),
    ])
    # Each tfmot prune-wrapper owns the original layer via `.layer` plus
    # extra mask/threshold variables, so we must copy weights into the
    # wrapped inner layer -- a flat set_weights() on the wrapper model
    # would mismatch (kernel+bias+mask+threshold per layer vs kernel+bias).
    for wrapped_layer, base_layer in zip(pruned_model.layers, model.layers):
        wrapped_layer.layer.set_weights(base_layer.get_weights())
    pruned_model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
                          loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # No EarlyStopping here deliberately: restore_best_weights would likely
    # pick an early, low-sparsity epoch (before the ramp completes) and
    # silently defeat the 35% sparsity target. We train the full fixed
    # schedule (ramp + recovery) and report the FINAL epoch's metrics.
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    pruned_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=FINE_TUNE_EPOCHS, batch_size=16, verbose=2, callbacks=callbacks)

    val_acc_pruned = pruned_model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"Pruned (35% target sparsity) + fine-tuned val accuracy (final epoch): {val_acc_pruned * 100:.2f}%")

    # Strip pruning wrappers -> plain Dense layers with (mostly) zeroed weights.
    stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # Report actual achieved sparsity per layer.
    for layer in stripped.layers:
        if hasattr(layer, "get_weights") and layer.get_weights():
            w = layer.get_weights()[0]
            sparsity = np.mean(np.abs(w) < 1e-8)
            print(f"  layer '{layer.name}': achieved sparsity = {sparsity * 100:.1f}%")

    os.makedirs(os.path.dirname(args.pruned_keras_out) or ".", exist_ok=True)
    stripped.save(args.pruned_keras_out)
    print(f"Pruned Keras model saved: {args.pruned_keras_out}")

    # --- Full INT8 PTQ on the pruned model ---------------------------------
    rng = np.random.default_rng(0)
    calib = X_norm[rng.choice(len(X_norm), size=min(len(X_norm), 250), replace=False)]

    def rep_gen():
        for row in calib:
            yield [row.reshape(1, -1).astype("float32")]

    converter = tf.lite.TFLiteConverter.from_keras_model(stripped)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"M3 (pruned 35% + Full-INT8) model written: {args.out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

"""
benchmark.py — LogiEdge Five-Metric Benchmarking (Task F2) + Pareto Analysis
================================================================================

Benchmarks all three model variants (M1 FP32, M2 PTQ-INT8, M3 Pruned+INT8)
on the five required metrics, following the course's Lab 2 methodology:

    1. Mean inference latency (ms)  -- 200 timed runs after 10 warm-up runs
    2. p95 inference latency (ms)
    3. Model file size (KB)
    4. Classification accuracy on the held-out validation split (%)
    5. Energy per inference (mJ)    -- E = P x t, P estimated from psutil
                                        CPU% x an assumed device TDP

Writes optimisation/results/benchmark_results.csv and a Pareto-frontier
scatter chart (optimisation/results/pareto_chart.png: accuracy vs latency,
with Pareto-optimal points annotated and model size encoded by marker size).

Usage:
    python benchmark.py --models-dir ../inference --data ../training/dataset.csv \
                         --stats ../data_pipeline/training_stats.npy \
                         --out results/benchmark_results.csv \
                         --chart results/pareto_chart.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import FEATURE_NAMES, load_training_stats, normalise  # noqa: E402

WARMUP_RUNS = 10
TIMED_RUNS = 200
RANDOM_STATE = 42

# Assumed device TDP for the energy-per-inference estimate (Task F2 formula:
# E = P(W) x t(s) x 1000, using psutil CPU% as a proxy for instantaneous
# power draw relative to the device's rated TDP). This benchmark runs on the
# development machine (a laptop/WSL VM), NOT the target edge hardware -- the
# absolute mJ values are therefore illustrative; the RELATIVE ordering across
# M1/M2/M3 (which is what the Pareto analysis needs) is the valid takeaway.
ASSUMED_DEVICE_TDP_W = 15.0  # representative of a Raspberry Pi 5-class SoC under load

VARIANTS = [
    ("M1_FP32", "model_m1_fp32.tflite"),
    ("M2_PTQ_INT8", "model_m2_ptq_int8.tflite"),
    ("M3_Pruned_INT8", "model_m3_pruned_int8.tflite"),
]


def run_single_inference(interpreter, x_row: np.ndarray) -> float:
    """Runs one inference, returns latency in milliseconds."""
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    x = x_row.reshape(1, -1).astype("float32")
    if in_detail["dtype"] in (np.int8, np.uint8):
        scale, zp = in_detail["quantization"]
        x = (x / scale + zp).round().astype(in_detail["dtype"])
    else:
        x = x.astype(in_detail["dtype"])

    t0 = time.perf_counter()
    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()
    _ = interpreter.get_tensor(out_detail["index"])
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def predict_class(interpreter, x_row: np.ndarray) -> int:
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    x = x_row.reshape(1, -1).astype("float32")
    if in_detail["dtype"] in (np.int8, np.uint8):
        scale, zp = in_detail["quantization"]
        x = (x / scale + zp).round().astype(in_detail["dtype"])
    else:
        x = x.astype(in_detail["dtype"])
    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(out_detail["index"])[0]
    if out_detail["dtype"] in (np.int8, np.uint8):
        scale, zp = out_detail["quantization"]
        out = (out.astype("float32") - zp) * scale
    return int(np.argmax(out))


def benchmark_variant(name: str, model_path: str, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    size_kb = os.path.getsize(model_path) / 1024
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # --- accuracy over the FULL held-out validation set ---
    preds = [predict_class(interpreter, X_val[i]) for i in range(len(X_val))]
    acc = accuracy_score(y_val, preds)
    cm = confusion_matrix(y_val, preds, labels=[0, 1, 2])
    class2_recall = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else float("nan")

    # --- latency: warm-up, then timed runs, cycling through validation rows ---
    rng = np.random.default_rng(0)
    idx_pool = rng.integers(0, len(X_val), WARMUP_RUNS + TIMED_RUNS)
    for i in idx_pool[:WARMUP_RUNS]:
        run_single_inference(interpreter, X_val[i])

    cpu_before = psutil.cpu_percent(interval=None)
    latencies = []
    t_start = time.perf_counter()
    for i in idx_pool[WARMUP_RUNS:]:
        latencies.append(run_single_inference(interpreter, X_val[i]))
    t_end = time.perf_counter()
    cpu_after = psutil.cpu_percent(interval=None)

    latencies = np.array(latencies)
    mean_latency_ms = float(latencies.mean())
    p95_latency_ms = float(np.percentile(latencies, 95))

    # --- energy per inference: E = P x t, P estimated from CPU utilisation ---
    avg_cpu_pct = max(cpu_before, cpu_after, 1.0) / 100.0  # crude utilisation proxy
    power_w = ASSUMED_DEVICE_TDP_W * avg_cpu_pct
    total_time_s = (t_end - t_start)
    total_energy_mj = power_w * total_time_s * 1000.0
    energy_per_inference_mj = total_energy_mj / TIMED_RUNS

    return {
        "variant": name,
        "model_file": os.path.basename(model_path),
        "size_kb": round(size_kb, 2),
        "mean_latency_ms": round(mean_latency_ms, 4),
        "p95_latency_ms": round(p95_latency_ms, 4),
        "accuracy_pct": round(acc * 100, 2),
        "class2_recall_pct": round(class2_recall * 100, 2),
        "energy_per_inference_mj": round(energy_per_inference_mj, 4),
    }


def plot_pareto(df: pd.DataFrame, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = 200 * (df["size_kb"] / df["size_kb"].max())
    scatter = ax.scatter(df["mean_latency_ms"], df["accuracy_pct"], s=sizes + 80,
                          c=range(len(df)), cmap="viridis", edgecolors="black", zorder=3)

    # Pareto frontier: a point is Pareto-optimal if no other point has BOTH
    # lower latency AND higher-or-equal accuracy.
    is_pareto = []
    for i, row in df.iterrows():
        dominated = (
            (df["mean_latency_ms"] < row["mean_latency_ms"]) & (df["accuracy_pct"] >= row["accuracy_pct"])
        ).any() or (
            (df["mean_latency_ms"] <= row["mean_latency_ms"]) & (df["accuracy_pct"] > row["accuracy_pct"])
        ).any()
        is_pareto.append(not dominated)
    df["pareto_optimal"] = is_pareto

    for i, row in df.iterrows():
        label = f"{row['variant']}\n{row['size_kb']:.1f} KB"
        marker = "*" if row["pareto_optimal"] else "o"
        if row["pareto_optimal"]:
            ax.scatter(row["mean_latency_ms"], row["accuracy_pct"], marker="*",
                       s=400, facecolors="none", edgecolors="red", linewidths=2, zorder=4)
        ax.annotate(label, (row["mean_latency_ms"], row["accuracy_pct"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=8)

    ax.set_xlabel("Mean inference latency (ms)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("LogiEdge Model Variants — Pareto Frontier", pad=28)
    fig.text(0.5, 0.925, "(marker size ~ model file size; red star = Pareto-optimal)",
              ha="center", fontsize=8, style="italic")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    print(f"Pareto chart saved: {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="LogiEdge 5-metric benchmark + Pareto analysis")
    parser.add_argument("--models-dir", default="../inference")
    parser.add_argument("--data", default="../training/dataset.csv")
    parser.add_argument("--stats", default="../data_pipeline/training_stats.npy")
    parser.add_argument("--out", default="results/benchmark_results.csv")
    parser.add_argument("--chart", default="results/pareto_chart.png")
    args = parser.parse_args()

    df_data = pd.read_csv(args.data)
    X = df_data[FEATURE_NAMES].values.astype("float32")
    y = df_data["label"].values.astype("int64")
    mean, std = load_training_stats(args.stats)
    X_norm = normalise(X, mean, std).astype("float32")

    _, X_val, _, y_val = train_test_split(
        X_norm, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"Benchmarking on {len(X_val)} held-out validation windows")

    results = []
    for name, filename in VARIANTS:
        model_path = os.path.join(args.models_dir, filename)
        if not os.path.exists(model_path):
            print(f"[WARN] missing {model_path}, skipping {name}")
            continue
        print(f"Benchmarking {name} ({filename}) ...")
        result = benchmark_variant(name, model_path, X_val, y_val)
        results.append(result)
        print(f"  {result}")

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults written: {args.out}")
    print(df.to_string(index=False))

    df = plot_pareto(df, args.chart)
    df.to_csv(args.out, index=False)  # rewrite with pareto_optimal column included

    best = df[df["pareto_optimal"]].sort_values("mean_latency_ms")
    print("\nPareto-optimal variant(s):")
    print(best[["variant", "size_kb", "mean_latency_ms", "accuracy_pct", "class2_recall_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
generate_dataset.py — LogiEdge Labelled Dataset Generator (Task D1)
=====================================================================

Runs the ColdChainSimulator (from data_pipeline/simulator.py) in each of the
three required modes, pushes the raw samples through the same
WindowedFeatureExtractor used at inference time (data_pipeline/preprocessing.py),
and writes a single labelled CSV dataset:

    Class 0 (Normal)   <- --anomaly none       x 20 minutes  (~120 windows)
    Class 1 (Warning)  <- --anomaly temp_drift x 15 minutes  (~90 windows)
    Class 2 (Critical) <- --anomaly combined   x 15 minutes  (~90 windows)

It also computes and saves `training_stats.npy` (per-feature mean/std) from
the FIRST 10 minutes of the Class-0 (Normal) run only, per the mandatory
normalisation rule in Task C2: statistics are computed once, offline, from
clean data, and never recomputed at runtime.

Usage:
    python generate_dataset.py --out dataset.csv --stats ../data_pipeline/training_stats.npy
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from simulator import ColdChainSimulator           # noqa: E402
from preprocessing import (                         # noqa: E402
    WindowedFeatureExtractor, FEATURE_NAMES, compute_training_stats, save_training_stats,
)

RUNS = [
    # (class_label, class_name, anomaly_mode, duration_minutes, seed)
    (0, "Normal",   "none",       20, 101),
    (1, "Warning",  "temp_drift", 15, 102),
    (2, "Critical", "combined",   15, 103),
]

STATS_WINDOW_MINUTES = 10  # per Task C2: compute stats from 10 min of clean Normal output


def run_one(class_label: int, class_name: str, anomaly: str, minutes: float, seed: int, truck_id: str):
    """Run the simulator for `minutes` of simulated time and return (features, feature_times)."""
    duration_s = minutes * 60
    sim = ColdChainSimulator(truck_id=truck_id, anomaly=anomaly, seed=seed)
    extractor = WindowedFeatureExtractor()

    feature_rows = []
    feature_times = []
    last_t = 0.0
    for sample in sim.generate(duration_s):
        last_t = sample.t
        if sample.stream == "temperature":
            extractor.add_temperature(sample.t, sample.value)
        elif sample.stream == "vibration":
            extractor.add_vibration(sample.t, sample.value)
        # door events are not part of the 6-value feature vector (Task C2 spec);
        # they are published separately for chain-of-custody logging only.

        feats = extractor.poll(sample.t)
        if feats is not None:
            feature_rows.append(feats)
            feature_times.append(sample.t)

    print(f"  [{class_name}] anomaly={anomaly} duration={minutes}min -> {len(feature_rows)} windows "
          f"(last sample t={last_t:.1f}s)")
    return np.array(feature_rows), np.array(feature_times)


def main():
    parser = argparse.ArgumentParser(description="Generate LogiEdge labelled training dataset")
    parser.add_argument("--out", default="dataset.csv")
    parser.add_argument("--stats", default="../data_pipeline/training_stats.npy")
    args = parser.parse_args()

    all_features, all_labels, all_class_names = [], [], []
    normal_features_for_stats = None
    normal_times_for_stats = None

    print("Generating LogiEdge labelled dataset ...")
    for class_label, class_name, anomaly, minutes, seed in RUNS:
        feats, times = run_one(class_label, class_name, anomaly, minutes, seed, truck_id="sim-truck")
        all_features.append(feats)
        all_labels.extend([class_label] * len(feats))
        all_class_names.extend([class_name] * len(feats))

        if class_label == 0:
            # Keep only the first STATS_WINDOW_MINUTES of the Normal run for
            # computing training_stats.npy (Task C2 requirement).
            mask = times <= STATS_WINDOW_MINUTES * 60
            normal_features_for_stats = feats[mask]
            normal_times_for_stats = times[mask]

    X = np.vstack(all_features)
    y = np.array(all_labels)

    # --- write dataset CSV --------------------------------------------------
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label", "class_name"])
        for row, label, name in zip(X, all_labels, all_class_names):
            writer.writerow(list(row) + [label, name])
    print(f"Dataset written: {args.out}  ({len(X)} rows, {X.shape[1]} features)")

    # --- compute & save training_stats.npy from clean Normal data only -----
    mean, std = compute_training_stats(normal_features_for_stats)
    save_training_stats(args.stats, mean, std)
    print(f"training_stats.npy written: {args.stats}  "
          f"(from {len(normal_features_for_stats)} windows, "
          f"first {STATS_WINDOW_MINUTES} min of Normal-class run, "
          f"t<= {normal_times_for_stats.max():.0f}s)")
    print("Per-feature mean:", dict(zip(FEATURE_NAMES, mean.round(4))))
    print("Per-feature std :", dict(zip(FEATURE_NAMES, std.round(4))))

    # --- class balance summary ----------------------------------------------
    unique, counts = np.unique(y, return_counts=True)
    print("Class balance:", dict(zip(unique.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()

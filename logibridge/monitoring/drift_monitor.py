"""
drift_monitor.py — LogiEdge PSI Drift Monitoring (Task E1)
==============================================================

Monitors the model's output CONFIDENCE SCORE distribution (not accuracy --
we have no ground truth in production) using the Population Stability Index
(PSI), exactly as taught in Module 5:

    PSI = sum( (actual% - expected%) * ln(actual% / expected%) )   over 4 bins
    bins: [0, 0.25), [0.25, 0.50), [0.50, 0.75), [0.75, 1.0]
    thresholds: PSI < 0.1 stable | 0.1-0.25 moderate (investigate) | > 0.25 significant (retrain)

Two modes:

  --mode build-reference
        Run the model on 300 CLEAN Normal-class windows and save the
        confidence-score bin distribution as reference_dist.json. Run this
        ONCE after training, before any drift monitoring.

  --mode simulate   (recommended for the recorded demo video)
        Self-contained, deterministic, simulated-time replay: normal traffic
        primes a rolling window of 100 inferences, --anomaly combined is
        injected at a chosen simulated time, and clean traffic resumes
        afterwards to show recovery. Because it runs on simulated time (not
        wall-clock sleep), the ENTIRE demo -- prime, drift, recover -- plays
        out in a few real seconds while still printing faithfully-timed PSI
        checks every 60 SIMULATED seconds, so the "PSI crosses 0.25 within
        5 minutes of drift injection" requirement is directly visible and
        reproducible on camera without a 15+ minute real-time recording.

  --mode live
        Production-realistic design: subscribes to the already-running
        inference_service's MQTT output topic
        (logibridge/trucks/{truck_id}/inference) and checks PSI on real
        wall-clock cadence. Use this if you have time for a longer, fully
        real-time recording instead of --mode simulate.

Usage:
    python drift_monitor.py --mode build-reference --model ../inference/model_m2_ptq_int8.tflite
    python drift_monitor.py --mode simulate --model ../inference/model_m2_ptq_int8.tflite
    python drift_monitor.py --mode live --truck-id truck-01
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from simulator import ColdChainSimulator                                   # noqa: E402
from preprocessing import WindowedFeatureExtractor, load_training_stats, normalise  # noqa: E402

BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]
N_BINS = 4
ROLLING_WINDOW = 100
CHECK_INTERVAL_S = 60       # seconds (simulated or real, per mode)
PSI_ALERT_THRESHOLD = 0.25
PSI_STABLE_THRESHOLD = 0.10
BUILD_REFERENCE_N_WINDOWS = 300
REFERENCE_DATASET_MINUTES = (BUILD_REFERENCE_N_WINDOWS * 10 + 30) / 60.0  # windows every 10s, +30s warm-up


# ---------------------------------------------------------------------------
# Core PSI math (single source of truth, used by all three modes)
# ---------------------------------------------------------------------------

def bin_confidences(confidences) -> np.ndarray:
    """Return bin COUNTS (length N_BINS) for a list of confidence scores."""
    counts, _ = np.histogram(confidences, bins=BIN_EDGES)
    return counts


def compute_psi(reference_props: np.ndarray, actual_counts: np.ndarray) -> float:
    actual_props = actual_counts / actual_counts.sum() if actual_counts.sum() > 0 else actual_counts
    psi = 0.0
    # Epsilon floor of 1e-3 (not a much smaller value like 1e-6): with only
    # 4 bins and a confident classifier, the reference distribution can be
    # highly concentrated (near-zero probability in 1-2 bins). An
    # excessively small floor makes PSI blow up from a single stray sample
    # landing in an near-empty bin -- a known PSI stability pitfall with
    # sparse bins, not a real distribution shift. 1e-3 is a standard,
    # commonly-used floor in production PSI implementations for this reason.
    EPS = 1e-3
    for exp, act in zip(reference_props, actual_props):
        exp = max(float(exp), EPS)
        act = max(float(act), EPS)
        psi += (act - exp) * np.log(act / exp)
    return float(psi)


def psi_verdict(psi: float) -> str:
    if psi > PSI_ALERT_THRESHOLD:
        return "SIGNIFICANT (retrain)"
    if psi > PSI_STABLE_THRESHOLD:
        return "MODERATE (investigate)"
    return "STABLE"


# ---------------------------------------------------------------------------
# Model inference helper (shared with inference_service.py's logic)
# ---------------------------------------------------------------------------

def load_interpreter(model_path: str):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def infer_confidence(interpreter, features_norm: np.ndarray) -> float:
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    x = features_norm.reshape(1, -1).astype("float32")
    if in_detail["dtype"] in (np.int8, np.uint8):
        scale, zp = in_detail["quantization"]
        x = (x / scale + zp).round().astype(in_detail["dtype"])
    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(out_detail["index"])[0]
    if out_detail["dtype"] in (np.int8, np.uint8):
        scale, zp = out_detail["quantization"]
        out = (out.astype("float32") - zp) * scale
    # Clip-and-renormalise only (no min-subtraction -- see inference_service.py
    # for why that distorts genuinely low-confidence predictions).
    probs = np.clip(out.astype("float64"), 0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    return float(probs.max())  # confidence of the predicted (argmax) class


def generate_feature_windows(anomaly: str, duration_s: float, seed: int, truck_id: str = "drift-sim"):
    """Yield (t, feature_vector) for `duration_s` of simulated sensor data."""
    sim = ColdChainSimulator(truck_id=truck_id, anomaly=anomaly, seed=seed)
    extractor = WindowedFeatureExtractor()
    for sample in sim.generate(duration_s):
        if sample.stream == "temperature":
            extractor.add_temperature(sample.t, sample.value)
        elif sample.stream == "vibration":
            extractor.add_vibration(sample.t, sample.value)
        feats = extractor.poll(sample.t)
        if feats is not None:
            yield sample.t, feats


# ---------------------------------------------------------------------------
# Mode 1: build-reference
# ---------------------------------------------------------------------------

def mode_build_reference(args):
    mean, std = load_training_stats(args.stats)
    interpreter = load_interpreter(args.model)

    confidences = []
    for t, feats in generate_feature_windows("none", REFERENCE_DATASET_MINUTES * 60, seed=555):
        feats_norm = normalise(feats, mean, std)
        confidences.append(infer_confidence(interpreter, feats_norm))
        if len(confidences) >= BUILD_REFERENCE_N_WINDOWS:
            break

    counts = bin_confidences(confidences)
    props = (counts / counts.sum()).tolist()

    reference = {
        "bin_edges": BIN_EDGES,
        "bin_proportions": props,
        "n_samples": len(confidences),
        "model_path": args.model,
    }
    with open(args.reference, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"Reference distribution built from {len(confidences)} clean Normal-class windows.")
    print(f"Bin proportions {BIN_EDGES}: {[round(p, 4) for p in props]}")
    print(f"Saved to {args.reference}")


# ---------------------------------------------------------------------------
# Mode 2: simulate (recommended for the recorded demo)
# ---------------------------------------------------------------------------

def mode_simulate(args):
    with open(args.reference) as f:
        reference = json.load(f)
    reference_props = np.array(reference["bin_proportions"])

    mean, std = load_training_stats(args.stats)
    interpreter = load_interpreter(args.model)

    rolling = collections.deque(maxlen=ROLLING_WINDOW)
    next_check_t = CHECK_INTERVAL_S

    print(f"=== LogiEdge drift monitor (simulated time) ===")
    print(f"Reference distribution: {len(reference['bin_proportions'])} bins from "
          f"{reference['n_samples']} clean windows")
    print(f"Timeline: 0-{args.prime_s}s NORMAL (priming) | "
          f"{args.prime_s}-{args.prime_s + args.drift_s}s DRIFT (--anomaly combined) | "
          f"{args.prime_s + args.drift_s}-{args.prime_s + args.drift_s + args.recover_s}s RECOVERY (clean)\n")

    schedule = [
        ("none", args.prime_s),
        ("combined", args.drift_s),
        ("none", args.recover_s),
    ]

    drift_injected_at = args.prime_s
    first_alert_t = None
    recovered_t = None

    t_offset = 0.0
    for anomaly, phase_duration in schedule:
        for t_local, feats in generate_feature_windows(anomaly, phase_duration, seed=hash(anomaly) % 1000):
            t = t_offset + t_local
            feats_norm = normalise(feats, mean, std)
            conf = infer_confidence(interpreter, feats_norm)
            rolling.append(conf)

            if t >= next_check_t:
                next_check_t += CHECK_INTERVAL_S
                if len(rolling) < ROLLING_WINDOW:
                    print(f"[t={t:6.0f}s] rolling window warming up ({len(rolling)}/{ROLLING_WINDOW})")
                    continue
                counts = bin_confidences(list(rolling))
                psi = compute_psi(reference_props, counts)
                verdict = psi_verdict(psi)
                print(f"[t={t:6.0f}s] PSI={psi:.3f}  ({verdict})")
                if psi > PSI_ALERT_THRESHOLD:
                    print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")
                    if first_alert_t is None and t >= drift_injected_at:
                        first_alert_t = t
                elif psi < PSI_STABLE_THRESHOLD and first_alert_t is not None and recovered_t is None:
                    recovered_t = t
        t_offset += phase_duration

    print("\n=== Summary ===")
    print(f"Drift injected at simulated t = {drift_injected_at:.0f}s")
    if first_alert_t is not None:
        delay = first_alert_t - drift_injected_at
        print(f"First PSI>0.25 alert at t = {first_alert_t:.0f}s "
              f"({delay:.0f}s = {delay / 60:.1f} min after injection)")
        if delay <= 300:
            print("Requirement MET: PSI crossed 0.25 within 5 minutes of drift injection.")
        else:
            print("Requirement NOT MET: increase --drift-s or review feature sensitivity.")
    else:
        print("No alert fired during the drift phase -- investigate feature sensitivity.")
    if recovered_t is not None:
        print(f"PSI returned below 0.10 at t = {recovered_t:.0f}s during recovery phase.")


# ---------------------------------------------------------------------------
# Mode 3: live (production-realistic, real MQTT + wall clock)
# ---------------------------------------------------------------------------

def mode_live(args):
    import paho.mqtt.client as mqtt

    with open(args.reference) as f:
        reference = json.load(f)
    reference_props = np.array(reference["bin_proportions"])

    rolling = collections.deque(maxlen=ROLLING_WINDOW)
    state = {"last_check": time.time()}

    def on_connect(c, userdata, flags, rc, *a):
        topic = f"logibridge/trucks/{args.truck_id}/inference"
        print(f"[drift_monitor] connected, subscribing to {topic}")
        c.subscribe(topic, qos=1)

    def on_message(c, userdata, msg):
        payload = json.loads(msg.payload.decode("utf-8"))
        rolling.append(payload["confidence"])
        now = time.time()
        if now - state["last_check"] >= CHECK_INTERVAL_S:
            state["last_check"] = now
            if len(rolling) < ROLLING_WINDOW:
                print(f"[drift_monitor] warming up ({len(rolling)}/{ROLLING_WINDOW})")
                return
            counts = bin_confidences(list(rolling))
            psi = compute_psi(reference_props, counts)
            print(f"[drift_monitor] PSI={psi:.3f} ({psi_verdict(psi)})")
            if psi > PSI_ALERT_THRESHOLD:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")

    client = mqtt.Client(client_id=f"drift-monitor-{args.truck_id}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, keepalive=30)
    client.loop_forever()


def main():
    parser = argparse.ArgumentParser(description="LogiEdge PSI drift monitor")
    parser.add_argument("--mode", choices=["build-reference", "simulate", "live"], required=True)
    parser.add_argument("--model", default="../inference/model_m2_ptq_int8.tflite")
    parser.add_argument("--stats", default="../data_pipeline/training_stats.npy")
    parser.add_argument("--reference", default="reference_dist.json")
    parser.add_argument("--truck-id", default="truck-01")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    # --mode simulate timeline controls (simulated seconds)
    parser.add_argument("--prime-s", type=float, default=1100, help="clean traffic before drift (fills rolling window)")
    parser.add_argument("--drift-s", type=float, default=400, help="--anomaly combined duration")
    parser.add_argument("--recover-s", type=float, default=1400,
                         help="clean traffic after drift; needs >= ROLLING_WINDOW*10s to fully flush the rolling window")
    args = parser.parse_args()

    if args.mode == "build-reference":
        mode_build_reference(args)
    elif args.mode == "simulate":
        mode_simulate(args)
    elif args.mode == "live":
        mode_live(args)


if __name__ == "__main__":
    main()

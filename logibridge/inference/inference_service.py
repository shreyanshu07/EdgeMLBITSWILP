"""
inference_service.py — LogiEdge On-Truck Inference Service (Task D2)
========================================================================

The service that runs inside the Docker container on the truck's edge node:

    MQTT sensor streams -> preprocessing (filter, window, feature-extract,
    normalise) -> TFLite inference -> classify -> publish result + alerts
    -> append to local, append-only alert log (offline-safe).

Model-agnostic: the model file is selected purely via the MODEL_PATH
environment variable, so swapping M1/M2/M3 (or a future OTA-pushed model)
requires NO container rebuild -- only a new file at MODEL_PATH and a
container restart (see deployment/logibridge_deploy.yml).

Environment variables:
    MODEL_PATH      path to a .tflite model file (required)
    TRUCK_ID        logical truck identifier used in MQTT topics (default: truck-01)
    MQTT_BROKER     MQTT broker host (default: localhost)
    MQTT_PORT       MQTT broker port (default: 1883)
    STATS_PATH      path to training_stats.npy (default: /app/data/training_stats.npy)
    ALERT_LOG_PATH  path to the local append-only alert log (default: /app/data/alert_log.jsonl)

Course context: AIML ZG535 Mini-Project, Component D (Module 4).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import WindowedFeatureExtractor, load_training_stats, normalise  # noqa: E402

CLASS_NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}


def get_interpreter(model_path: str):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter, features_norm: np.ndarray):
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    x = features_norm.reshape(1, -1).astype("float32")
    if in_detail["dtype"] in (np.int8, np.uint8):
        scale, zero_point = in_detail["quantization"]
        x = (x / scale + zero_point).round().astype(in_detail["dtype"])
    else:
        x = x.astype(in_detail["dtype"])

    interpreter.set_tensor(in_detail["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(out_detail["index"])[0]

    if out_detail["dtype"] in (np.int8, np.uint8):
        scale, zero_point = out_detail["quantization"]
        out = (out.astype("float32") - zero_point) * scale

    pred_class = int(np.argmax(out))
    # The model's final layer is already softmax, so `out` is already a
    # (dequantised, slightly noisy) probability vector. We only clip tiny
    # negative dequantisation artefacts to 0 and renormalise to sum to 1 --
    # we must NOT subtract the min before normalising, since that distorts
    # genuinely low-confidence (uncertain) predictions into appearing
    # artificially high-confidence.
    probs = np.clip(out.astype("float64"), 0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    confidence = float(probs[pred_class])
    return pred_class, confidence


def append_alert_log(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        print("[inference_service] FATAL: MODEL_PATH environment variable is required", file=sys.stderr)
        sys.exit(1)

    truck_id = os.environ.get("TRUCK_ID", "truck-01")
    broker = os.environ.get("MQTT_BROKER", "localhost")
    port = int(os.environ.get("MQTT_PORT", "1883"))
    stats_path = os.environ.get("STATS_PATH", "/app/data/training_stats.npy")
    alert_log_path = os.environ.get("ALERT_LOG_PATH", "/app/data/alert_log.jsonl")

    print(f"[inference_service] starting: truck_id={truck_id} model={model_path} broker={broker}:{port}")

    interpreter = get_interpreter(model_path)
    mean, std = load_training_stats(stats_path)
    extractor = WindowedFeatureExtractor()

    import paho.mqtt.client as mqtt

    client = mqtt.Client(client_id=f"inference-{truck_id}")
    sensor_base = f"logibridge/trucks/{truck_id}/sensors"
    result_topic = f"logibridge/trucks/{truck_id}/inference"
    alert_topics = {
        1: f"logibridge/trucks/{truck_id}/alerts/warning",
        2: f"logibridge/trucks/{truck_id}/alerts/critical",
    }

    def on_connect(c, userdata, flags, rc, *args):
        print(f"[inference_service] connected to broker, rc={rc}")
        c.subscribe(f"{sensor_base}/temperature", qos=0)
        c.subscribe(f"{sensor_base}/vibration", qos=0)

    def on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        t = payload.get("t", time.time())
        value = payload.get("value")
        if msg.topic.endswith("/temperature"):
            extractor.add_temperature(t, value)
        elif msg.topic.endswith("/vibration"):
            extractor.add_vibration(t, value)

        feats = extractor.poll(t)
        if feats is None:
            return

        feats_norm = normalise(feats, mean, std)
        pred_class, confidence = run_inference(interpreter, feats_norm)
        class_name = CLASS_NAMES[pred_class]
        now_iso = datetime.now(timezone.utc).isoformat()

        result = {
            "truck_id": truck_id, "t": t, "timestamp": now_iso,
            "class": pred_class, "class_name": class_name, "confidence": round(confidence, 4),
        }
        c.publish(result_topic, json.dumps(result), qos=1)
        print(f"[inference_service] {class_name} (conf={confidence:.3f}) at t={t:.1f}s")

        # Local, offline-safe alert log -- written regardless of MQTT/cellular
        # connectivity; this is what the store-and-forward bridge replays
        # upstream once the connection returns.
        append_alert_log(alert_log_path, result)

        if pred_class in alert_topics:
            c.publish(alert_topics[pred_class], json.dumps(result), qos=2 if pred_class == 2 else 1)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=30)
    client.loop_forever()


if __name__ == "__main__":
    main()

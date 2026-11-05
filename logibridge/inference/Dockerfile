# LogiEdge — Edge Inference Service Dockerfile
# ================================================
# Component D (Module 4), Task D2.
#
# IMPORTANT: build context is the REPO ROOT (logibridge/), not this folder,
# because the image needs files from both inference/ and data_pipeline/:
#
#   docker build -f inference/Dockerfile -t logibridge-inference:latest .
#
# Layer-cache design (OTA demo, see demo/ota_layer_cache_demo.sh):
#   1. requirements.txt + `pip install` is the EXPENSIVE layer (~300-500 MB,
#      several minutes to resolve/download). It only changes if a dependency
#      version changes.
#   2. Application code (inference_service.py, preprocessing.py) changes
#      occasionally.
#   3. The three .tflite model variants are the CHEAP, FREQUENTLY-CHANGING
#      layer (each only a few KB). They are copied LAST, so a routine model
#      update (retrain -> new .tflite -> rebuild) only invalidates this one
#      small layer -- Docker reuses the cached pip-install layer unchanged.
#      This is exactly what Task D2 asks you to demonstrate and measure.
#
# Production note: tensorflow-cpu is used here (not the lighter
# `tflite-runtime` / `ai-edge-litert` wheel) to guarantee the image builds
# identically on any grading machine regardless of CPU architecture. A real
# fleet deployment on Raspberry Pi / Jetson would swap this for the
# architecture-matched tflite-runtime wheel to cut image size drastically.

FROM python:3.11-slim

WORKDIR /app

# ---- Layer 1: dependencies (cached; rebuilt only if requirements.txt changes) ----
COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Layer 2: application code + fixed normalisation statistics ----
COPY data_pipeline/preprocessing.py ./data_pipeline/preprocessing.py
COPY inference/inference_service.py .
COPY data_pipeline/training_stats.npy ./data/training_stats.npy

# ---- Layer 3: model variants (THE layer that changes on a routine OTA update) ----
# All three benchmarked variants (M1/M2/M3) are bundled so MODEL_PATH can
# switch between them at `docker run` time with ZERO rebuild.
COPY inference/model_m1_fp32.tflite       ./models/model_m1_fp32.tflite
COPY inference/model_m2_ptq_int8.tflite   ./models/model_m2_ptq_int8.tflite
COPY inference/model_m3_pruned_int8.tflite ./models/model_m3_pruned_int8.tflite

ENV MODEL_PATH=/app/models/model_m2_ptq_int8.tflite
ENV STATS_PATH=/app/data/training_stats.npy
ENV ALERT_LOG_PATH=/app/data/alert_log.jsonl
ENV TRUCK_ID=truck-01
ENV MQTT_BROKER=host.docker.internal
ENV MQTT_PORT=1883

ENTRYPOINT ["python", "inference_service.py"]

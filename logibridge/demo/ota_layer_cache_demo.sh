#!/usr/bin/env bash
# demo/ota_layer_cache_demo.sh
# =============================
# LogiEdge Task D2 — OTA layer-cache demonstration.
#
# Run this on a machine with Docker installed (Docker Desktop on Windows/Mac,
# or Docker Engine on Linux/WSL2). This script is NOT executed automatically
# by the assistant that built this repo (no Docker daemon was available in
# that build environment) -- YOU must run it yourself, capture the terminal
# output shown below, and include it in your demo video + Final Report
# Section 3, replacing the illustrative numbers in ota_bandwidth_calc.md
# with your own measured values.
#
# What it demonstrates:
#   1. First build: all layers build from scratch (slow).
#   2. Change ONLY a model file, rebuild: the (slow) pip-install layer is
#      reused from cache ("CACHED"), only the (fast) model-copy layer and
#      anything after it re-executes.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (logibridge/)

echo "=== Build 1: initial build (expect NO cache hits, all layers execute) ==="
time docker build -f inference/Dockerfile -t logibridge-inference:v1 . 

echo
echo "=== Simulating a routine model update: overwrite M2 with a 'retrained' file ==="
# In real life this would be the output of a fresh training run. Here we
# just touch the file with new bytes so its content hash changes.
python3 -c "
import struct, time
with open('inference/model_m2_ptq_int8.tflite', 'ab') as f:
    f.write(struct.pack('<d', time.time()))  # append 8 bytes -> new content hash
"

echo
echo "=== Build 2: rebuild after ONLY the model file changed ==="
echo "    (Watch for 'CACHED' next to the pip install step, and note the much"
echo "     shorter wall-clock time reported by 'time' below.)"
time docker build -f inference/Dockerfile -t logibridge-inference:v2 .

echo
echo "=== Layer size comparison (for the bandwidth-saving calculation) ==="
docker history logibridge-inference:v2 --no-trunc --format "table {{.Size}}\t{{.CreatedBy}}" | head -20
docker images logibridge-inference --format "table {{.Tag}}\t{{.Size}}"

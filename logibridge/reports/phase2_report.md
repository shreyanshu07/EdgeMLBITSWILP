# LogiEdge — Phase 2 Progress Report

**AIML ZG535 — Machine Learning on Edge | Mini-Project (EC-I)**
Covers: Component C (Sensor Pipeline & MQTT Architecture, Module 3) and
Component D (Model Training, Conversion, and Docker Deployment, Module 4).

## 1. Scope completed this phase

- **Task C1 — Sensor simulator:** `data_pipeline/simulator.py` generates the
  three required streams (temperature 1 Hz, vibration_rms 0.5 Hz, discrete
  door events) with a `--anomaly {none|temp_drift|vibration|combined}` CLI
  flag, publishing to a local MQTT broker. Verified: a 120-second
  `--anomaly combined` run shows temperature drifting from 4.0 degC to
  ~13.2 degC exactly as the +0.08 degC/reading linear-drift specification
  predicts.
- **Task C2 — Preprocessing pipeline:** `data_pipeline/preprocessing.py`
  implements the 5-sample moving average, the 30s/10s sliding-window
  6-value feature extractor, and fixed-statistics normalisation.
  **Mandatory experiment result:** inference accuracy with the correct
  `training_stats.npy` was **99.32%**; with statistics shifted by 3 sigma,
  accuracy collapsed to **57.82%** — a 41.50-point drop that concretely
  demonstrates why normalisation statistics must be fixed and versioned,
  never recomputed live.
- **Task C3 — Data fusion justification:** implemented feature-level fusion
  and justified it against data-level and decision-level alternatives in
  `data_pipeline/mqtt_architecture.md` (section 4).
- **Task D1 — Dataset + training:** `training/generate_dataset.py` produced
  294 labelled windows (118 Normal / 88 Warning / 88 Critical) exactly
  matching the specified simulator-mode/duration table.
  `training/train_model.py` trained the 2-hidden-layer MLP (32, 16 units),
  achieving **98.31%** held-out validation accuracy (threshold: 88%) and
  **100%** Class-2 (Critical) recall.
- **Task D2 — Conversion + Docker:** `training/convert_ptq.py` produced the
  Full-INT8 PTQ variant (M2, 3.40 KB, 100% accuracy — see Phase
  3/Final Report for the complete benchmark). `inference/Dockerfile` and
  `inference/inference_service.py` package the preprocessing -> inference ->
  MQTT-publish chain, with `MODEL_PATH` selecting between model variants
  with zero rebuild, and the model-file COPY instructions placed after the
  `pip install` layer for OTA layer-cache efficiency.
- **Task D3 — Pipeline mapping:** mapped LogiEdge onto all ten stages of
  Module 4's Edge ML lifecycle (Data -> Train -> Validate -> Optimise ->
  Convert -> Deploy -> Infer -> Monitor -> Update -> feedback loop); see
  Final Report Section 3.

## 2. End-to-end verification performed

The full chain — simulator publishing `--anomaly combined` sensor data over
a real local MQTT broker, consumed by `inference_service.py` running the
preprocessing + TFLite inference chain — correctly classified the truck as
**Critical** within 40 seconds of the anomaly starting, comfortably inside
the 90-second SLA, with the local alert log (`alert_log.jsonl`) recording
every classification regardless of any simulated connectivity loss.

## 3. Plan for Phase 3 / Final Report

- Build the PSI drift monitor (Task E1) and Ansible OTA playbook (Task E2),
  and complete the OTA strategy cost/risk analysis (Task E3).
- Produce the third model variant (35% pruning + Full-INT8 PTQ, Task F1) and
  run the five-metric benchmark + Pareto analysis (Task F2), then finalise
  the deployment recommendation (Task F3).
- Assemble the Final Report, record the demo video, and complete the two
  personal-reflection sections based on first-hand experience running the
  full pipeline.

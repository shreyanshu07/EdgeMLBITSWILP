# LogiEdge — Intelligent Edge AI Platform for Cold-Chain Logistics

**AIML ZG535 — Machine Learning on Edge | Mini-Project (EC-I) | 30 Marks**
BITS Pilani WILP | Instructor: Pravin Yashwant Pawar

LogiEdge is an end-to-end Edge AI pipeline that classifies the cargo
compartment state (`Normal` / `Warning` / `Critical`) of FreightBridge
Logistics' refrigerated trucks in real time, entirely on-device, so that
cold-chain pharmaceutical shipments remain monitored even when cellular
connectivity is unavailable.

> **Before you submit:** read [`NOTES_FOR_STUDENT.md`](./NOTES_FOR_STUDENT.md)
> — it explains exactly what you still need to do yourself (run the demo,
> record the video, write the two personal-reflection paragraphs) before
> this is submission-ready. Per the course's academic-integrity policy, you
> must be able to explain and re-run every part of this repository.

## Repository Structure

```
logibridge/
├── README.md                          <- you are here
├── scenario_architecture/             Component A (Module 1)
│   ├── constraint_analysis.md         Task A1: latency/bandwidth/connectivity/privacy
│   ├── generate_architecture_diagram.py
│   └── system_architecture.png        Task A2
├── hardware/                          Component B (Module 2)
│   └── hardware_justification.md      Task B1 (Constraint Triangle) + B2 (Roofline)
├── data_pipeline/                     Component C (Module 3)
│   ├── simulator.py                   Task C1: 3-stream sensor simulator + MQTT publisher
│   ├── preprocessing.py               Task C2: filter -> window -> feature-extract -> normalise
│   ├── normalization_shift_experiment.py   Task C2 mandatory experiment
│   ├── training_stats.npy             fixed normalisation statistics (generated)
│   └── mqtt_architecture.md           Task C1/C3: topic tree, QoS, data-fusion justification
├── training/                          Component D (Module 4)
│   ├── generate_dataset.py            Task D1: labelled dataset from the simulator
│   ├── train_model.py                 Task D1: M1 FP32 baseline MLP
│   ├── convert_ptq.py                 Task D2 input: M2 Full-INT8 PTQ conversion
│   ├── prune_quantise.py              Task F1 input: M3 35% pruning + INT8 PTQ
│   ├── dataset.csv                    generated labelled dataset
│   └── models/                        saved Keras models
├── inference/                         Component D (Module 4)
│   ├── Dockerfile                     Task D2: layer-cached container build
│   ├── requirements.txt
│   ├── inference_service.py           Task D2: preprocessing -> inference -> MQTT publish
│   └── model_*.tflite                 M1/M2/M3 variants (generated)
├── monitoring/                        Component E (Module 5)
│   ├── drift_monitor.py               Task E1: PSI drift detection (build-reference/simulate/live)
│   └── reference_dist.json            generated reference confidence distribution
├── deployment/                        Component E (Module 5)
│   ├── logibridge_deploy.yml          Task E2: 7-task idempotent Ansible playbook
│   ├── inventory.ini                  example fleet inventory
│   └── ota_strategy_analysis.md       Task E3: OTA strategy cost/risk comparison
├── optimisation/                      Component F (Module 6)
│   ├── benchmark.py                   Task F2: 5-metric benchmark + Pareto analysis
│   └── results/
│       ├── benchmark_results.csv
│       └── pareto_chart.png
├── demo/
│   ├── ota_layer_cache_demo.sh        Task D2: layer-cache rebuild demonstration
│   └── ota_bandwidth_calc.md          bandwidth-saving calculation (Task D2)
└── reports/
    ├── phase1_report.md
    ├── phase2_report.md
    └── final_report.md                -> export to PDF before submission
```

## How To Run Everything, In Order

All commands assume a Linux shell (WSL2 on Windows works) with Python 3.10+,
and are run from inside the `logibridge/` directory unless stated otherwise.
A `requirements-dev.txt` at the repo root installs everything needed except
Docker and Ansible (see below).

```bash
pip install --user tensorflow-cpu==2.15.0 tensorflow-model-optimization==0.7.5 \
    paho-mqtt==1.6.1 scikit-learn matplotlib psutil pandas

# 1. Sensor simulator self-test (no broker needed)
python3 data_pipeline/preprocessing.py
python3 data_pipeline/simulator.py --help

# 2. Generate the labelled dataset + fixed normalisation statistics
cd training
python3 generate_dataset.py --out dataset.csv --stats ../data_pipeline/training_stats.npy

# 3. Train the M1 FP32 baseline (must exceed 88% validation accuracy)
python3 train_model.py --data dataset.csv --stats ../data_pipeline/training_stats.npy --out models/m1_fp32

# 4. Convert to M2 (Full-INT8 PTQ) and M3 (35% pruning + Full-INT8 PTQ)
python3 convert_ptq.py --saved-model models/m1_fp32 --data dataset.csv \
    --stats ../data_pipeline/training_stats.npy --out ../inference/model_m2_ptq_int8.tflite
python3 prune_quantise.py --saved-model models/m1_fp32 --data dataset.csv \
    --stats ../data_pipeline/training_stats.npy --out ../inference/model_m3_pruned_int8.tflite
python3 -c "import tensorflow as tf; open('../inference/model_m1_fp32.tflite','wb').write(tf.lite.TFLiteConverter.from_saved_model('models/m1_fp32').convert())"

# 5. Mandatory Task C2 normalisation-shift experiment
cd ../data_pipeline
python3 normalization_shift_experiment.py --data ../training/dataset.csv \
    --stats training_stats.npy --model ../inference/model_m2_ptq_int8.tflite

# 6. Build the PSI reference distribution, then run the drift demo
cd ../monitoring
python3 drift_monitor.py --mode build-reference --model ../inference/model_m2_ptq_int8.tflite \
    --stats ../data_pipeline/training_stats.npy --reference reference_dist.json
python3 drift_monitor.py --mode simulate --model ../inference/model_m2_ptq_int8.tflite \
    --stats ../data_pipeline/training_stats.npy --reference reference_dist.json

# 7. Five-metric benchmark + Pareto chart
cd ../optimisation
python3 benchmark.py --models-dir ../inference --data ../training/dataset.csv \
    --stats ../data_pipeline/training_stats.npy --out results/benchmark_results.csv \
    --chart results/pareto_chart.png

# 8. Live end-to-end demo (needs a running MQTT broker, e.g. `mosquitto` or
#    `pip install amqtt` for a pure-Python stand-in)
mosquitto -d                                              # terminal 1 (or an amqtt broker)
cd ../inference && MODEL_PATH=model_m2_ptq_int8.tflite \
    STATS_PATH=../data_pipeline/training_stats.npy \
    python3 inference_service.py                           # terminal 2
cd ../data_pipeline && python3 simulator.py --anomaly combined --duration 120  # terminal 3

# 9. Docker build + OTA layer-cache demo (needs Docker installed)
cd .. && bash demo/ota_layer_cache_demo.sh

# 10. Ansible idempotency demo (needs Ansible + Docker on the target host)
cd deployment
ansible-playbook -i inventory.ini logibridge_deploy.yml \
    -e model_src=../inference/model_m2_ptq_int8.tflite \
    -e reference_src=../monitoring/reference_dist.json -e truck_id=truck-07
ansible-playbook -i inventory.ini logibridge_deploy.yml \
    -e model_src=../inference/model_m2_ptq_int8.tflite \
    -e reference_src=../monitoring/reference_dist.json -e truck_id=truck-07   # run again: changed=0
```

## What Has Been Verified By Re-Running End-To-End (see `reports/phase2_report.md`)

| Component | Result |
|---|---|
| Dataset generation | 294 labelled windows (118 Normal / 88 Warning / 88 Critical) |
| M1 FP32 training | **98.31%** held-out validation accuracy, **100%** Class-2 recall |
| M2 PTQ-INT8 conversion | 5.22 KB -> 3.40 KB, **98.31%** accuracy, **100%** Class-2 recall (Pareto-optimal) |
| M3 pruning (35%) + INT8 | 35.0-35.4% achieved sparsity per layer, **93.22%** accuracy but only **82.35%** Class-2 recall (fails the 95% gate -- NOT recommended) |
| Normalisation-shift experiment | 99.32% -> 57.82% accuracy under a 3-sigma stats shift |
| End-to-end MQTT pipeline | Simulator -> broker -> inference_service correctly classified `Critical` within 40s of a combined-anomaly injection (well inside the 90s SLA) |
| PSI drift monitor | Crossed PSI > 0.25 within **3.7 minutes** of drift injection (requirement: <= 5 min); stayed below the alert threshold during normal operation and recovered to PSI < 0.10 after the drift phase |
| Ansible playbook | Run 1: `changed=6`. Run 2 (no changes): **`changed=0`** |
| Benchmark + Pareto | M2 (PTQ-INT8) is Pareto-optimal (smallest + fastest, ties M1 on accuracy at 98.31%, 100% Class-2 recall); M3 (pruned) is smallest in size but fails the Class-2 recall gate (82.35% < 95%) and is NOT recommended |

## What You Still Need To Do Yourself

See [`NOTES_FOR_STUDENT.md`](./NOTES_FOR_STUDENT.md).

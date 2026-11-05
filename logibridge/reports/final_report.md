# LogiEdge — Final Report

**AIML ZG535 — Machine Learning on Edge | Mini-Project (EC-I)**
GROUPNO_LogiEdge_Final (replace with your actual group number before export)

> **Word count note:** figures, tables, and references are excluded from the
> 2,500-3,000 word target per the problem statement. Section 5 contains two
> clearly marked places that MUST be completed in your own words from your
> own experience running this project (see inline notes) before submission.

---

## Section 1 — FreightBridge Deployment Context (Module 1 & 2)

FreightBridge Logistics requires a cold-chain monitoring system that detects
refrigeration faults and alerts within **90 seconds**, continues operating
through cellular dead zones, and keeps raw cargo-condition data off any
third-party network. Four constraints shape the architecture. **Latency:**
a refrigeration failure raises cargo temperature by 1 degC/minute, and a
cloud-dependent design is infeasible not because round-trip latency is too
slow under connection (rural 4G round trips of 150-800 ms fit comfortably
inside 90 seconds) but because FreightBridge's own Nashik-Aurangabad route
loses cellular signal entirely for 35-90 minutes at seven locations — 20 to
60 times longer than the detection deadline itself. **Bandwidth:** streaming
raw sensor data (1 Hz temperature + 500 Hz 3-axis vibration) costs
approximately 494.8 MB/truck/day, or roughly Rs 1.26 lakh/month for the
85-truck pilot at Rs 0.10/MB; edge-processed feature/classification messages
reduce this to about 70 MB/day fleet-wide (Rs 210/month), a 99.8%+
reduction. **Connectivity:** the edge architecture keeps sensing,
preprocessing, inference, and local alerting entirely on `localhost`,
independent of cellular status, with only the upstream *synchronisation* of
already-made decisions depending on connectivity, recovered via a
store-and-forward bridge on reconnect. **Privacy:** because only abstracted
classification results (not raw telemetry) ever leave the truck, FreightBridge
can contractually guarantee pharmaceutical clients that cargo data cannot
reach unauthorised third parties — an architectural guarantee, not merely a
policy promise, mirroring GDPR Article 25 "privacy by design."

**Hardware selection** applies the Constraint Triangle (compute / power /
cost) plus a fourth, architecture-driven constraint: platform compatibility.
Of the three candidates (Raspberry Pi 5 + AI HAT+, Jetson Orin Nano, STM32H7
MCU), the Jetson is disqualified by exceeding the stated 10 W power budget
(15 W under moderate load), and the STM32H7 — despite the lowest cost and
power draw — cannot host the Docker/MQTT/Ansible stack this project's
architecture requires (no Linux, no container runtime). **The Raspberry Pi 5
+ AI HAT+ is recommended**: it satisfies the power budget (7.5 W), runs the
required software stack, and costs Rs 15,000/truck (Rs 39.75 lakh at the
265-vehicle full-fleet scale) for a classification workload that does not
need the bundled NPU's 13 TOPS at all.

**Arithmetic Intensity / Roofline:** with the model at ~45 MFLOPs and ~18 MB
of data movement per inference, AI = 45e6/18e6 = **2.5 FLOP/byte**. The
Raspberry Pi 5 CPU's ridge point is 16 GFLOP/s / 12 GB/s = **1.333
FLOP/byte**. Since 2.5 > 1.333, the model is **compute-bound**, not
memory-bound — attainable performance is capped at the 16 GFLOP/s compute
ceiling, giving a theoretical best-case inference time of ~2.8 ms, more than
four orders of magnitude inside the 90-second budget. The correct
optimisation lever for a compute-bound model is to reduce FLOPs or raise
usable peak compute (quantisation, pruning, or NPU offload) — not to chase
faster memory, which would not help here.

*(See `scenario_architecture/constraint_analysis.md`,
`scenario_architecture/system_architecture.png`, and
`hardware/hardware_justification.md` for full detail and calculations.)*

---

## Section 2 — Sensor Pipeline and MQTT Design (Module 3)

The MQTT topic tree is partitioned by truck: `logibridge/trucks/{truck_id}/
sensors/{temperature,vibration,door}`, `.../features`, `.../inference`,
`.../alerts/{warning,critical}`, and `.../health`, plus an aggregated
`ops/fleet/status`. **QoS is chosen per topic's tolerance for loss versus
duplication**: high-frequency, self-superseding sensor streams
(temperature, vibration) use **QoS 0** — a dropped sample is replaced a
second later at no cost. Discrete, chain-of-custody-relevant events (door
events) use **QoS 1** — loss is unacceptable but an occasional duplicate is
harmless and de-duplicable. The safety-critical escalation topic
(`alerts/critical`) uses **QoS 2** — the only topic where the exactly-once
handshake overhead is justified, because both silent loss and false
duplicate alarms are unacceptable for a pharmaceutical cold-chain incident.
*(Full topic tree and per-topic justification: `data_pipeline/
mqtt_architecture.md`.)*

**Preprocessing** runs three stages in sequence: a 5-sample causal moving
average filters both the temperature and vibration streams; a 30-second
sliding window (10-second step) extracts six features per window (temperature
mean/std/rate-of-change, vibration RMS/peak/kurtosis); and z-score
normalisation uses **fixed** statistics computed once from 10 minutes of
clean Normal-class data and loaded at runtime, never recomputed live. This
design choice was stress-tested by the **mandatory normalisation-shift
experiment**: re-running inference with the statistics shifted by 3-sigma
per feature produced the following measured result.

| Condition | Accuracy |
|---|---|
| Correct `training_stats.npy` | **99.32%** |
| Statistics shifted by 3 sigma | **57.82%** |
| Accuracy drop | **41.50 percentage points** |

This demonstrates concretely why the normalisation statistics must be a
fixed, versioned artefact rather than something recomputed at runtime: a
silent 3-sigma shift (e.g. from a sensor-calibration bug, or stats copied
from the wrong batch) collapses classification accuracy by over 40
percentage points even though the model's weights are completely unchanged.

**Data fusion** is implemented at the **feature level**: temperature and
vibration are preprocessed independently using the statistic that is
physically meaningful for each sensor, then concatenated into one 6-value
vector before the classifier. This was chosen over **data-level fusion**
(rejected: the two sensors sample at different, non-aligned rates — 1 Hz vs
0.5 Hz — and naive raw alignment would manufacture information never
measured) and **decision-level fusion** (rejected: refrigeration failure
typically raises both temperature and vibration *together*, and training two
independent classifiers would discard exactly that cross-sensor
correlation). Feature-level fusion also keeps the model small enough (6
inputs, ~800 parameters) to comfortably fit the memory budget of any edge
candidate considered in Section 1.

---

## Section 3 — Model Deployment and Pipeline Mapping (Module 4)

LogiEdge maps onto all ten stages of Module 4's Edge ML lifecycle. **Data**
(Collect/Label/Augment): the simulator generates physically-grounded
Normal/Warning/Critical traffic from documented sensor statistics, labelled
by construction. **Train**: a 2-hidden-layer MLP (32, 16 units) trained on
z-score-normalised 6-value features. **Validate**: a stratified 20% held-out
split, required to exceed 88% accuracy (achieved: 98.31%). **Optimise**:
Full-INT8 PTQ (M2) and 35%-sparsity pruning + INT8 PTQ (M3). **Convert**:
`tf.lite.TFLiteConverter` produces the deployable `.tflite` artefacts.
**Deploy**: a Docker container (`inference/Dockerfile`) bundles the
preprocessing code, fixed normalisation statistics, and all three model
variants, selectable via the `MODEL_PATH` environment variable with zero
rebuild. **Infer**: `inference_service.py` runs the full preprocessing ->
TFLite -> classification chain and publishes results to
`logibridge/trucks/{truck_id}/inference`. **Monitor**: the PSI drift monitor
(Section 5) watches the confidence-score distribution. **Update**: the
Ansible playbook performs OTA model swaps with an idempotent, conditional
restart. The loop closes back into **Data/Train** whenever drift triggers a
retraining cycle — production Edge AI is never "set and forget."

**Docker layer-cache design:** the Dockerfile places the expensive
`pip install` layer (TensorFlow + dependencies) *before* the model-file
layer, so that a routine model update only invalidates the small, fast
final layer. *(Run `demo/ota_layer_cache_demo.sh` on a Docker-enabled
machine and insert your measured `docker history`/`docker images` output and
timings here before final submission — see `demo/ota_bandwidth_calc.md` for
the calculation template.)* Using representative figures (full image ~450 MB
vs a 280 KB production model file), layer-cache reuse reduces the bandwidth
needed to push a model-only update to the 85-truck pilot fleet from
approximately **37.35 GB to 23.2 MB per update cycle** — a **>99.9%**
reduction, at a cost saving of roughly Rs 3,825 -> Rs 2.32 per cycle.

**OTA strategy recommendation: Canary rollout (10 trucks first).** All
three strategies (full replacement, canary, shadow mode) cost on the order
of Rs 2-2.50 per fleet-wide update cycle given the model's small size — cost
is not the differentiator. Canary is recommended because it limits a faulty
model's blast radius to ~12% of the fleet before wider exposure (critical
given the pharmaceutical safety stakes), and its staged, opportunistic pull
model tolerates the fleet's documented rural connectivity gaps better than a
single coordinated full-replacement push. Shadow mode offers excellent
validation rigor but doubles on-device compute/power load during validation,
conflicting with the 10 W power budget established in Section 1.
*(Full analysis: `deployment/ota_strategy_analysis.md`.)*

---

## Section 4 — Optimisation and Pareto Analysis (Module 6)

| Variant | Size (KB) | Mean latency (ms) | p95 latency (ms) | Accuracy (%) | Class-2 (Critical) recall (%) | Energy/inference (mJ) |
|---|---|---|---|---|---|---|
| M1 — FP32 baseline | 5.22 | 0.0057 | 0.0088 | 98.31 | 100.00 | 0.0613 |
| M2 — PTQ Full-INT8 | 3.40 | 0.0049 | 0.0059 | 98.31 | 100.00 | 0.0986 |
| M3 — 35% Pruned + INT8 | 3.39 | 0.0067 | 0.0183 | 93.22 | **82.35** | 0.0574 |

*(Full results: `optimisation/results/benchmark_results.csv`; chart:
`optimisation/results/pareto_chart.png`. Accuracy and recall figures are
exactly reproducible across runs — `tf.config.experimental.
enable_op_determinism()` is enabled in `train_model.py`/`prune_quantise.py`
specifically so these classification metrics are not subject to run-to-run
training noise; only wall-clock latency varies slightly between runs, as
expected for a timing measurement.)*

**M2 (PTQ Full-INT8) is the Pareto-optimal variant** — it matches M1's
accuracy while being smaller and faster, and dominates M3 on every axis
except raw file size (a 0.01 KB difference, i.e. effectively tied). **M3
(35% pruning + INT8) is explicitly NOT recommended for deployment**, despite
being the smallest file: pruning here measurably hurts **Class-2 (Critical)
recall specifically** (82.35%, versus 100% for M1/M2) by more than its
effect on overall accuracy (93.22%, only ~5 points below M1/M2) — a concrete,
reproducible illustration of why a safety-critical classifier must be
evaluated on **per-class recall for the highest-severity class**, not just
overall accuracy: a model that looks only "slightly worse" in aggregate can
be substantially worse specifically at catching the failures that matter
most. This is exactly the discipline Task F3's 95% Class-2-recall gate is
designed to enforce, and it is the deciding factor here, not latency or size.
*(Caveat for the viva: all benchmarked latencies are sub-millisecond because
they were measured on an x86 development machine, not the target Raspberry
Pi 5 hardware — per Module 2's "benchmark on the actual target device"
principle, these absolute numbers should be re-measured on the real target
before being quoted as production SLA evidence; the *relative* ordering
across variants, and the accuracy/recall figures, are the valid takeaways
from this environment.)*

**Deployment recommendation: ship M2 (PTQ Full-INT8).** Evidence against the
requirements: (1) the 90-second SLA translates to a per-inference latency
budget that is effectively unconstrained for this workload (Section 1's
Roofline analysis already showed ~2.8 ms compute-bound latency on the
recommended hardware, and M2's measured latency is smaller still); (2) the
recommended Raspberry Pi 5 has 8 GB of RAM and multi-GB Flash-equivalent
storage, so none of the three variants (3.4-5.2 KB) is remotely close to a
Flash/SRAM constraint — unlike an MCU-class deployment, model size is not
the limiting factor on this hardware, and M2's smaller footprint is a bonus
for OTA bandwidth rather than a feasibility requirement; (3) M2's Class-2
(Critical) recall of **100%** clears the required 95% threshold with
margin, **unlike M3, which fails this specific gate (82.35%) and is
therefore disqualified from deployment regardless of its size advantage.**

---

## Section 5 — MLOps Monitoring and Reflection (Module 5)

The PSI drift monitor was validated with a scripted drift-injection
timeline: clean traffic primes a 100-inference rolling window, `--anomaly
combined` is injected, and clean traffic resumes to show recovery.

| Phase | PSI behaviour |
|---|---|
| Before injection (priming) | PSI stayed **below the 0.25 alert threshold** throughout (ranging 0.007-0.166, briefly touching MODERATE but never SIGNIFICANT) |
| During injection | PSI crossed the 0.25 alert threshold **3.7 minutes** after injection (requirement: within 5 minutes), peaking at **0.998** (SIGNIFICANT) |
| After recovery | PSI returned below 0.10 (STABLE) once the rolling window fully flushed the contaminated samples |

**Drift type classification:** this is best characterised as **concept
drift co-occurring with a confidence-distribution shift** rather than pure
covariate/data drift: the underlying feature distributions genuinely change
(temperature and vibration statistics shift under `combined` anomaly) *and*
the model's output confidence distribution shifts correspondingly, which is
exactly the signal PSI (monitoring the confidence-score histogram, not raw
features) is designed to catch per Module 5.

**Ansible idempotency result:** running `deployment/logibridge_deploy.yml`
twice with identical inputs produced `changed=6` on the first run (directory
created, model + reference files copied, container stopped/pulled/started)
and **`changed=0` on the second run** — all three file-copy tasks reported
no change (checksum match) and the three container-management tasks were
correctly skipped via the `register`+`when` guard pattern, with the
verification task designed to never report `changed` regardless.

> **TODO (student must complete before submission) — genuine technical
> difficulty:** describe ONE real debugging challenge you personally
> encountered while running/extending this project (e.g., a package version
> conflict, a quantisation calibration issue, a timing/race condition in the
> live MQTT demo, an Ansible/Docker environment quirk on your machine) and
> how you diagnosed and resolved it. This must be your own experience from
> actually running the code — it is a specific, checkable claim the viva may
> ask you to elaborate on.

> **TODO (student must complete before submission) — architectural change
> you would make:** describe ONE change you would make to this architecture
> if you were deploying it for real (e.g., a different data-fusion level, a
> different hardware tier, a different OTA strategy, a different drift
> detection algorithm such as CUSUM for slow trends, splitting the MQTT
> broker across a regional gateway instead of per-truck, etc.) and justify
> the trade-off. This should reflect your own judgment, not a restatement of
> a choice already made in this repository.

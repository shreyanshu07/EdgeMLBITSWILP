# LogiEdge — Phase 1 Progress Report

**AIML ZG535 — Machine Learning on Edge | Mini-Project (EC-I)**
Covers: Component A (System Architecture & Deployment Justification, Module 1)
and Component B (Hardware Selection & Justification, Module 2).

## 1. Scope completed this phase

- **Task A1 — Constraint analysis:** quantified all four Edge AI constraints
  (latency, bandwidth, connectivity, privacy) for the FreightBridge
  cold-chain deployment, with concrete numbers drawn from the problem
  statement (90-second detection deadline, 1 Hz/500 Hz sensor rates, 35-90
  minute connectivity gaps, Rs 0.10/MB transmission cost). Full writeup:
  `scenario_architecture/constraint_analysis.md`.
- **Task A2 — System architecture diagram:** produced
  `scenario_architecture/system_architecture.png`, showing the truck edge
  node (sensors -> local MQTT broker -> preprocessing -> Docker inference
  pipeline -> local alert log -> store-and-forward bridge), the intermittent
  cellular uplink, and the operations-centre backend.
- **Task B1 — Constraint Triangle:** evaluated all three candidate hardware
  options (Raspberry Pi 5 + AI HAT+, Jetson Orin Nano, STM32H7 MCU) against
  compute/power/cost, plus a fourth constraint this project surfaced:
  Docker/Linux platform compatibility, which rules out the otherwise
  cheapest MCU option. Recommended: Raspberry Pi 5 + AI HAT+. Full
  reasoning: `hardware/hardware_justification.md`.
- **Task B2 — Arithmetic Intensity / Roofline:** computed AI = 2.5 FLOP/byte
  against a ridge point of 1.333 FLOP/byte for the recommended hardware,
  classifying the model as compute-bound, with a theoretical best-case
  latency of ~2.8 ms — far inside the 90-second SLA.

## 2. Key findings so far

1. Cloud-only inference is infeasible not primarily because of round-trip
   latency physics, but because of FreightBridge's documented, multi-hour
   connectivity dead zones — a forcing constraint the Edge AI decision
   framework (Module 1) predicts clearly.
2. Streaming raw sensor data would cost roughly Rs 1.26 lakh/month for the
   pilot fleet; edge preprocessing is projected to cut this by >99.8%
   (confirmed numerically once Component C's simulator/preprocessing
   pipeline is built in Phase 2).
3. The cheapest hardware option (STM32H7, ~Rs 3,500/truck) is disqualified
   not by the Constraint Triangle but by this project's own architectural
   dependency on Docker — an instructive example of software-architecture
   decisions constraining hardware choice.

## 3. Plan for Phase 2

- Build the sensor simulator (Task C1), preprocessing pipeline (Task C2),
  and run the mandatory normalisation-shift experiment.
- Justify the feature-level data-fusion choice (Task C3).
- Generate the labelled dataset and train the M1 FP32 baseline model (Task
  D1), targeting >88% held-out validation accuracy.

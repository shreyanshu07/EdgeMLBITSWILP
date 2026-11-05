# LogiEdge — Hardware Selection & Justification (Component B, Module 2)

## Task B1 — Constraint Triangle Application

### Candidate hardware

| Option | Hardware | India price (per truck) | TDP | Linux/Docker capable? |
|---|---|---|---|---|
| 1 | Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS Hailo-8L) | ~Rs 15,000 | 7.5 W | **Yes** (Debian-based Raspberry Pi OS) |
| 2 | Jetson Orin Nano Super Developer Kit (67 TOPS) | ~Rs 45,000 | 15 W (moderate load) | Yes |
| 3 | STM32H7-based custom MCU with sensor ICs | ~Rs 3,500 | 0.4 W | **No** (bare-metal / RTOS only, no container runtime) |

### Applying the Constraint Triangle (Compute / Power / Cost)

LogiEdge's classification workload is genuinely tiny: a 6-feature, 2-hidden-layer
MLP (~800 parameters, ~45 MFLOPs/inference — see Task B2). **None of the three
options is compute-constrained for this specific model.** The Constraint
Triangle therefore resolves to the other two vertices plus a fourth,
architecture-driven constraint that the course material flags as able to
override the triangle entirely: **certification / platform compatibility**.

1. **Power — hard cutoff, not a trade-off.** The problem statement fixes a
   **10 W AI power budget** (12 V truck supply via DC-DC converter). Option 2
   (Jetson Orin Nano, 15 W under moderate load) **exceeds this budget outright**
   and is disqualified regardless of its compute headroom — 67 TOPS is
   irrelevant to a workload that needs none of it, and the course's own
   "TOPS is a lie" lesson applies directly here: buying more compute than the
   workload needs, at a power cost the platform cannot absorb, is a pure loss.
2. **Platform compatibility — a forcing constraint from the project's own
   architecture.** Components C–E of this project (MQTT broker, the
   `inference_service.py` Docker container, and the Ansible-orchestrated OTA
   deployment in `deployment/logibridge_deploy.yml`) all assume a **Linux
   host capable of running a Docker container runtime**. The STM32H7 MCU
   option runs bare-metal or an RTOS directly on silicon — it **cannot run a
   Docker container, a local MQTT broker process, or be managed by the
   Ansible playbook's `docker` module tasks** as specified. This is not a
   soft preference; it is a hard architectural incompatibility. Option 3 is
   therefore disqualified for *this specific deployment design*, despite
   being the cheapest and lowest-power candidate by a wide margin — a clear
   example of a software/architecture decision (made in Components C–E)
   overriding what would otherwise be the economically obvious hardware
   choice.
3. **Cost — the deciding vertex among the remaining option.** With Option 2
   disqualified on power and Option 3 disqualified on platform compatibility,
   **Option 1 (Raspberry Pi 5 + AI HAT+) is the only remaining candidate**,
   and it comfortably satisfies the 90-second latency requirement (see B2) at
   a fleet-appropriate cost: Rs 15,000 x 85 trucks (pilot) = **Rs 12.75 lakh**,
   scaling to Rs 15,000 x 265 (fleet) = **Rs 39.75 lakh** at full rollout —
   roughly a third of the cost of standardising on Jetson-class hardware
   fleet-wide, for a workload that cannot use the extra compute anyway.

### Recommendation

**Deploy the Raspberry Pi 5 (8 GB) + AI HAT+.** It is the only option that
simultaneously satisfies the 10 W power budget, runs the Docker/MQTT/Ansible
architecture this project depends on, meets the 90-second latency requirement
with enormous margin (see B2), and keeps fleet-scale cost proportionate to
the simplicity of the actual classification workload. The bundled 13-TOPS
Hailo-8L NPU is not required by today's MLP classifier, but it is a legitimate
hedge for FreightBridge's likely next step — adding camera-based cargo
inspection — without a second hardware refresh across 265 trucks.

**Against Jetson Orin Nano:** exceeds the 10 W power budget by 50%, and its
67 TOPS of accelerator compute is entirely wasted on a sub-megaflop-class
classifier — 3x the unit cost for capability the workload does not use.

**Against the STM32H7 custom MCU:** despite the lowest cost and power draw by
a wide margin, it cannot host the Docker container, MQTT broker, or
Ansible-managed deployment this project's architecture requires. (If
FreightBridge later redesigns toward a bare-metal TFLite-Micro firmware with
a separate cellular gateway module instead of an on-truck Docker/MQTT stack,
the STM32H7 option would become the stronger long-term economic choice —
but that is a different architecture than the one built in Components C–E.)

---

## Task B2 — Arithmetic Intensity & Roofline Analysis

### Given

- Model compute: **~45 MFLOPs per inference**
- Model data movement: **~18 MB accessed per inference** (weights + activations)
- Target device CPU (Raspberry Pi 5, NEON SIMD): **~16 GFLOP/s** peak compute
- Target device memory (LPDDR4X): **~12 GB/s** peak bandwidth

### Arithmetic Intensity

```
AI = FLOPs / Bytes accessed = 45,000,000 / 18,000,000 = 2.5 FLOP/byte
```

### Ridge point

```
Ridge point = Peak compute / Peak bandwidth = 16 GFLOP/s / 12 GB/s = 1.333 FLOP/byte
```

### Classification

Because **AI (2.5 FLOP/byte) > Ridge point (1.333 FLOP/byte)**, the model sits
to the **right** of the ridge point on the Roofline chart — it is in the
**compute-bound region**, not the memory-bandwidth-bound region.

```
Attainable performance = min(Peak compute, AI x Peak bandwidth)
                        = min(16 GFLOP/s, 2.5 x 12 GFLOP/s)
                        = min(16 GFLOP/s, 30 GFLOP/s)
                        = 16 GFLOP/s   <- capped by peak COMPUTE, confirming compute-bound

Theoretical best-case latency = FLOPs / Attainable performance
                               = 45,000,000 / 16,000,000,000
                               = 2.8 ms  (compute-only estimate, excludes software/OS overhead)
```

### What this means for optimisation

Since the model is **compute-bound** (not memory-bound), the Roofline model
says the correct lever is to **reduce FLOPs or increase usable peak compute
— NOT to chase faster memory**:

- Faster/wider LPDDR would not help, because the bottleneck is compute, not
  data movement — this is the opposite conclusion from most large vision
  models (which Module 2 notes are typically memory-bound), and is a useful,
  concrete illustration of why the Roofline classification must be checked
  per-model rather than assumed.
- Valid levers, in order of what this project already applies: (1) **INT8
  quantisation** (Task D2, Variant M2) increases *effective* usable compute
  throughput on hardware with INT8 SIMD paths, shifting the compute ceiling
  upward; (2) **pruning** (Task F1, Variant M3) directly reduces the FLOP
  count of the model itself; (3) offloading to the bundled **Hailo-8L NPU**
  instead of the CPU would raise the peak-compute ceiling far above 16
  GFLOP/s, though for a workload this small the NPU's dispatch overhead may
  exceed the compute time saved.

### Why this does not threaten the 90-second SLA

Even the unoptimised, compute-bound theoretical estimate (2.8 ms per
inference) is **more than four orders of magnitude** inside the 90-second
fault-to-alert deadline. Combined with the preprocessing pipeline's 10-second
windowing cadence (Component C), the dominant term in the end-to-end latency
budget is **not** inference time at all — it is the 30-second feature window
plus MQTT/Docker/application overhead, all of which remain a small fraction
of the 90-second budget (see `scenario_architecture/constraint_analysis.md`
Task A1 for the full latency decomposition). The Roofline analysis confirms
that hardware compute is not, and will not become, the limiting factor for
this specific workload on the recommended Raspberry Pi 5 platform.

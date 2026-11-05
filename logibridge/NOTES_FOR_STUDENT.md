# NOTES FOR THE STUDENT — Read This Before Submitting

This repository was built as a **working reference implementation** to help
you learn the material and move fast — not as a drop-in submission. Per the
course's own academic-integrity policy (Section 8 of the problem statement),
**you must personally run this system, understand every component, and be
able to explain and demonstrate it in a follow-up viva.** This file lists
exactly what is already done, what is verified, and — most importantly —
what YOU must still do yourself.

## What has been built and verified (by running it, not just writing it)

Every script in this repo was actually executed end-to-end during
development (in a WSL2/Linux environment with TensorFlow 2.15, paho-mqtt,
scikit-learn, matplotlib, psutil, and ansible-core installed via pip). You
can see exact logged results in `reports/phase1_report.md`,
`reports/phase2_report.md`, and `reports/final_report.md`. Concretely:

- The simulator, preprocessing pipeline, dataset generator, model training,
  PTQ conversion, pruning, the Docker-targeted inference service, the PSI
  drift monitor, the Ansible playbook's idempotency pattern, and the
  5-metric benchmark + Pareto chart have **all been run and produced the
  numbers quoted in the reports.**
- The one exception: **actual `docker build` / `docker run` could not be
  executed** in the sandboxed environment used to build this repo (no
  Docker daemon available). The Dockerfile and `inference_service.py` logic
  were instead verified by running the exact same code path against a local
  Python MQTT broker (`amqtt`) standing in for Mosquitto, which proved the
  application logic is correct — but **you must run the actual Docker
  build/run and the OTA layer-cache demo yourself** (see below).
- Similarly, the Ansible playbook's idempotency pattern (`changed=6` then
  `changed=0`) was verified using a sandbox test harness with `docker`
  commands replaced by harmless stand-ins, because no Docker daemon was
  available. **Run the real `deployment/logibridge_deploy.yml` against your
  own Docker-equipped host before claiming this result in your report.**

## What YOU must do before submission

1. **Run the Docker build and the OTA layer-cache demo yourself**
   (`demo/ota_layer_cache_demo.sh`) on a machine with Docker installed.
   Replace the illustrative numbers in `demo/ota_bandwidth_calc.md` and
   Final Report Section 3 with your own measured `docker images` /
   `docker history` output.
2. **Run the real Ansible playbook** against a host with Docker installed
   (your own laptop, a VM, or a Raspberry Pi), twice, and capture the
   `changed=6` then `changed=0` terminal output for your demo video.
3. **Re-run the PSI drift monitor's `--mode live`** against your actual
   running `inference_service.py` container at least once, so you can
   speak to the real, production-realistic (MQTT-subscribed, wall-clock)
   design in addition to the fast `--mode simulate` demo.
4. **Record your 15-20 minute demo video** showing: the simulator running
   live, the Docker container running with the MQTT result stream visible,
   the OTA layer-cache rebuild, the Ansible playbook's two runs, the PSI
   drift-injection-and-recovery sequence, and the benchmark/Pareto output.
   This must be a genuine screen recording of YOU running the system — per
   the integrity policy, this is the primary mechanism that proves the
   system was actually built and understood, not fabricated.
5. **Write the two reflection paragraphs in Final Report Section 5**
   yourself: a genuine technical difficulty you personally hit while running
   or extending this code, and one architectural change you would make. Do
   not copy the TODO placeholder text — the viva may ask you to elaborate.
6. **Read every file at least once and be ready to explain it.** In
   particular, make sure you can explain: why Flash and SRAM are different
   constraints (Component B), why PSI uses the confidence-score
   distribution rather than accuracy (Component E — we have no ground truth
   in production), why the pruning script notes that TFMOT's
   `prune_low_magnitude` is technically *unstructured* weight-level sparsity
   on a Dense layer even though the task calls it "structured filter
   pruning" (see the docstring in `training/prune_quantise.py`), and why
   `register`+`when` guards make the Ansible playbook idempotent.
7. **Replace the inventory in `deployment/inventory.ini`** with your actual
   target host(s) before running Ansible for real, and update
   `logibridge_deploy.yml`'s `registry_image` if you push to a different
   local registry address.
8. **Export the final Markdown reports to PDF** (`reports/final_report.md`
   -> `GROUPNO_LogiEdge_Final.pdf`, per the exact filename the submission
   portal expects) after filling in Section 5.
9. **Rename the GitHub repository appropriately**, add the instructor as a
   collaborator, and double-check the demo video link works from an
   incognito browser window before submitting.

## A note on the dataset and benchmark results

The dataset used throughout (294 windows from the simulator) is small and
cleanly separated by construction, which is why several reported numbers
look unusually strong (e.g. 100% accuracy for the M2 variant, 100% Class-2
recall across all three variants, sub-millisecond latencies measured on a
development laptop rather than the target Raspberry Pi 5). This is called
out explicitly in Final Report Section 4 as a caveat you should be ready to
discuss: real-world cold-chain sensor data will be noisier, and the
benchmark should ideally be re-run on the actual target hardware before
being used as production SLA evidence. Being able to articulate this
limitation yourself is a good sign of genuine understanding, not a weakness
to hide.

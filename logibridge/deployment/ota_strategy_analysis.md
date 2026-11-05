# LogiEdge — OTA Strategy Selection (Task E3)

## Given

- Model updated every **6 weeks**.
- Model file: **280 KB** (INT8 TFLite).
- M2M SIM cost: **Rs 0.10/MB**.
- Pilot fleet: **85 trucks**.

## Bandwidth Cost Per Update Cycle, By Strategy

| Strategy | What is transmitted | Trucks reached this cycle | Data this cycle | Cost this cycle (Rs 0.10/MB) |
|---|---|---|---|---|
| **Full replacement** | 280 KB to every truck simultaneously | 85 / 85 | 85 x 280 KB = 23,800 KB = **23.24 MB** | **Rs 2.32** |
| **Canary (10 trucks first)** | Phase 1: 280 KB to 10 trucks. After a validation window, Phase 2: 280 KB to the remaining 75. | 10, then 75 (same 85 total) | Phase 1: 2.734 MB. Phase 2: 20.51 MB. Total: **23.24 MB** | Phase 1: Rs 0.27. Phase 2: Rs 2.05. **Total Rs 2.32** |
| **Shadow mode** | New model (280 KB) downloaded to ALL 85 trucks, but runs *alongside* the existing production model (old model is NOT removed) for a validation window, plus a small telemetry payload comparing the two models' outputs. | 85 / 85 (download) | Model: 23.24 MB. Telemetry (est. ~10 KB/truck over the validation window): + 0.83 MB | **~24.07 MB** (~Rs 2.41) |

**Key finding: bandwidth cost is NOT the differentiator.** Because the
model file is tiny (280 KB) relative to a typical M2M data plan, all three
strategies cost on the order of **Rs 2–2.50 per fleet-wide update cycle** —
a rounding error for a logistics company already spending lakhs per month on
fleet operations. The real differentiator is **risk (blast radius) and
validation rigor**, not cost.

## Strategy Comparison on Risk & Operational Fit

| Criterion | Full replacement | Canary (10 trucks) | Shadow mode |
|---|---|---|---|
| Blast radius of a bad model | 100% of fleet (85/85) at once | 12% of fleet (10/85) until validated | 0% — old model still drives every safety decision during validation |
| Validation before full exposure | None | Field validation on 10 real trucks before wider rollout | Strongest — new model's predictions are compared against the live, trusted model with zero live risk |
| Tolerance of rural connectivity gaps | Weak — trucks that miss the push window diverge from the fleet-wide version until they next connect | Strong — staged, opportunistic pull model; a truck simply updates whenever it next has connectivity, regardless of phase | Strong, same reasoning as canary |
| On-device compute/power load during validation | Normal (one model) | Normal (one model) | **Doubled** — running two models simultaneously conflicts with the 10W AI power budget on the 12V truck supply (Component B) |
| Time to full fleet confidence | Immediate exposure, no time to detect a bad model before damage | Fast (field data from 10 trucks informs the Phase 2 go/no-go decision) | Slowest — requires accumulating sufficient shadow-mode evidence (course material: >=50,000 inferences, 24-72 hours) before cutover, and the safety benefit is not realised until cutover finally happens |

## Recommendation: Canary Rollout (10 trucks first)

**Canary is recommended.** Reasoning, referencing cold-chain
safety-criticality and rural connectivity:

1. **Cold-chain safety-criticality -> limit blast radius.** A pharmaceutical
   shipment that silently loses refrigeration monitoring is a patient-safety
   and regulatory-compliance incident (see the Rs 28 lakh vaccine-spoilage
   event in the background). Canary caps the exposure of a faulty model to
   10 of 85 trucks (~12%) while field performance is validated, instead of
   risking the entire fleet simultaneously (Full replacement) or delaying a
   safety-relevant model improvement to the entire fleet for days while a
   parallel shadow model merely watches (Shadow mode delays benefit without
   eliminating eventual cutover risk).
2. **Rural connectivity characteristics favour opportunistic, staged
   pull.** FreightBridge's Nashik-Aurangabad route alone loses cellular
   signal for 35-90 minutes at seven locations. A full-replacement push
   assumes every truck is reachable in a single coordinated window; trucks
   mid-dead-zone will simply miss it and run a stale model version until
   their next connection — fleet version drift either way. A canary/staged
   rollout has no "single window" assumption: each phase's trucks update
   opportunistically whenever they have signal, which is exactly how a
   fleet with unreliable connectivity should be updated.
3. **Negligible cost difference.** Because the model is only 280 KB, the
   extra validation rigor of canary costs virtually nothing extra in
   bandwidth (Rs 2.32 vs Rs 2.32 for full replacement) — there is no
   economic reason to accept the higher blast-radius risk of full
   replacement.

**Argument against Full replacement:** it offers no opportunity to catch a
regression (e.g., a retrained model with a lower Class-2/Critical recall)
before it reaches the entire pharmaceutical cold-chain fleet simultaneously
— unacceptable for a safety-critical application given the stated 95%
Critical-recall requirement (Task F3).

**Argument against Shadow mode:** validation rigor is excellent, but (a) it
doubles on-device compute/power load during the validation window, directly
conflicting with the 10W AI power budget established in Component B, and
(b) the safety-decision risk is not actually *reduced* relative to canary —
it is merely *deferred* until cutover, at which point the new model still
goes live fleet-wide in one step unless combined with a canary cutover
(in which case you are effectively paying for both mechanisms). For a
small, well-understood MLP classifier (as opposed to, say, a large vision
model with a much higher risk of subtle regressions), canary alone already
provides adequate validation at lower operational cost.

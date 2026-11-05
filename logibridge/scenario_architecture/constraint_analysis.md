# LogiEdge — Constraint Analysis (Component A, Module 1)

## Task A1 — The Four Edge AI Constraints for FreightBridge Cold-Chain

### 1. Latency

A refrigeration failure raises cargo temperature by **1 degC/minute**. The
system must detect a fault signature and alert **within 90 seconds**.

**Is cloud inference feasible on India's rural cellular latency profile?**
A naive answer looks only at round-trip time: typical rural 3G/4G round
trips in India run roughly **150-800 ms** under an active connection — which
*individually* fits inside a 90-second budget with enormous margin. If
latency-under-connection were the only concern, cloud inference would look
acceptable. **This is the trap the problem statement is testing.** The real
constraint is not the latency of a connected round trip; it is the
**probability and duration of having no connection at all**. FreightBridge's
own Nashik-Aurangabad route loses cellular signal entirely for
**35-90 minutes at seven documented locations** — a complete connectivity
outage **20-60x longer than the entire 90-second detection deadline**. A
cloud-dependent architecture does not degrade gracefully here; it simply
**cannot function** for the duration of each dead zone, regardless of how
fast the network is the rest of the time. Conclusion: cloud inference is
**not feasible as the primary detection path** for this requirement — this
is a forcing constraint (Module 1's Decision Framework) that mandates
on-device (edge) inference for the safety-critical classification itself,
with the cloud relegated to analytics/retraining (see Component D's 10-stage
pipeline mapping).

### 2. Bandwidth

Per truck: temperature at 1 Hz, vibration at 500 Hz (3-axis), door events
(discrete). Assuming 4-byte (float32) samples per channel:

```
Temperature:  1 Hz x 4 bytes                      = 4 bytes/s
Vibration:    500 Hz x 3 axes x 4 bytes            = 6,000 bytes/s
Door events:  ~20 events/day x 20 bytes            = negligible (~400 bytes/day)

Raw rate (temp + vibration)        = 6,004 bytes/s
Raw volume per truck per day       = 6,004 x 86,400 s / 1,048,576 (MB conversion)
                                    ~= 494.8 MB/day per truck

Fleet (85 trucks) raw volume       ~= 494.8 x 85 = 42,061 MB/day ~= 41.08 GB/day
Cost at Rs 0.10/MB (raw, per truck) = 494.8 x 0.10 ~= Rs 49.5 / truck / day
Fleet raw transmission cost         ~= Rs 4,206 / day ~= Rs 1,26,000+ / month
```

**Edge-processed alert/feature volume** (6-value feature vector + predicted
class + timestamp, published every 10 s per the sliding-window step in
Component C, ~100 bytes per JSON message including field-name overhead):

```
Messages/day per truck  = 86,400 / 10 = 8,640
Processed volume/truck  = 8,640 x 100 bytes / 1,048,576 ~= 0.82 MB/day
Fleet processed volume  ~= 0.82 x 85 ~= 70 MB/day
Cost at Rs 0.10/MB (fleet) ~= Rs 7/day ~= Rs 210/month

Reduction = 1 - (0.82 / 494.8) ~= 99.83%
```

Streaming raw sensor data would cost roughly **Rs 1.26 lakh/month** for the
pilot fleet alone — edge preprocessing cuts this to **roughly Rs 210/month**,
a reduction of **>99.8%**, while still delivering everything the operations
centre needs (classification result + summary features) for monitoring and
chain-of-custody documentation.

### 3. Connectivity

During a 35-90 minute dead zone on the Nashik-Aurangabad route, a
**cloud-only system loses all monitoring capability**: no inference, no
alerting, no logging — the truck is invisible to the operations centre for
up to an hour and a half, precisely when a refrigeration fault is most
likely to go undetected (matching the real Rs 28 lakh vaccine-spoilage
incident described in the background). **LogiEdge's edge architecture**
keeps the entire detection path (sensors -> MQTT -> preprocessing ->
on-device TFLite inference -> local alert log) running **entirely on
`localhost`**, independent of cellular connectivity (see
`data_pipeline/mqtt_architecture.md` section 3). Only the upstream *sync* of
already-made decisions depends on cellular connectivity; that sync uses a
store-and-forward bridge that buffers the local, append-only alert log and
replays it — in original timestamp order — once connectivity returns. No
detection capability is lost during a dead zone; only the *notification* to
the operations centre is delayed, and even that is automatically recovered.

### 4. Privacy

FreightBridge's pharmaceutical clients require proof that cargo condition
data cannot be accessed by unauthorised third parties. **On-device inference
supports this contractually** in two concrete ways: (1) the raw
temperature/vibration/door-event streams **never leave the truck** — only a
derived classification result (`Normal`/`Warning`/`Critical`) and a small
feature summary are ever transmitted, so even a compromised cellular link or
an unauthorised party intercepting the uplink gains no raw cargo-condition
telemetry, only an already-abstracted status label; (2) this is an
**architectural guarantee, not a policy promise** — FreightBridge can state
contractually to pharmaceutical clients that raw physiological/cargo data
"cannot" (not merely "should not") reach third parties, because the system
design makes it physically impossible for that data to be transmitted at
all, mirroring the GDPR Article 25 "privacy by design" principle covered in
Module 1.

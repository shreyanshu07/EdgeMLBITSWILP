# LogiEdge — MQTT Topic Tree & Data Fusion Architecture

Component C (Module 3) · Tasks C1–C3

## 1. Full MQTT Topic Tree

```
logibridge/
└── trucks/
    └── {truck_id}/                       e.g. truck-07
        ├── sensors/
        │   ├── temperature                 raw 1 Hz cargo temperature (deg C)
        │   ├── vibration                    raw 0.5 Hz compressor vibration RMS (g)
        │   └── door                         discrete OPEN/CLOSE events
        ├── features                        6-value feature vector, every 10 s (sliding window output)
        ├── inference                        classification result {class, confidence, t}
        ├── alerts/
        │   ├── warning                      Class 1 escalation (ops dashboard, non-urgent)
        │   └── critical                     Class 2 escalation (immediate operator page)
        └── health                           heartbeat: uptime, signal strength, battery/12V rail voltage
└── ops/
    └── fleet/
        └── status                          gateway-aggregated summary across all connected trucks (regional gateway -> backend)
```

Rationale for the hierarchy: `{truck_id}` is the partitioning key so that a
single shared broker (or broker-per-region) can scale from the 85-truck
pilot to the 265-vehicle fleet without any topic redesign -- a subscriber
can wildcard-subscribe to `logibridge/trucks/+/alerts/critical` to watch the
whole fleet, or to `logibridge/trucks/truck-07/#` to inspect one vehicle.

## 2. QoS Justification per Topic

| Topic | QoS | Justification |
|---|---|---|
| `.../sensors/temperature` | **0** (at most once) | 1 Hz stream; a dropped sample is immediately superseded by the next one one second later. Retrying a stale reading wastes the limited cellular/radio budget for no benefit. |
| `.../sensors/vibration` | **0** | Same reasoning as temperature — high frequency, self-healing stream. |
| `.../sensors/door` | **1** (at least once) | A discrete, low-frequency event (chain-of-custody evidence for pharma shipments). Losing it silently is unacceptable, but an occasional duplicate delivery is harmless because the preprocessing layer can de-duplicate on `(truck_id, t, event)`. |
| `.../features` | **1** | Feeds the on-device inference step; losing one 10 s window degrades confidence briefly but the next window arrives in 10 s. At-least-once avoids silent gaps feeding into drift monitoring. |
| `.../inference` | **1** | Result is logged locally regardless of delivery (see §4 offline design), so MQTT delivery failure does not lose the safety decision — QoS 1 is for the *synchronisation* copy sent to the ops centre, not the local alert path. |
| `.../alerts/critical` | **2** (exactly once) | Safety-critical escalation for a pharmaceutical shipment. Must not be lost (QoS 0/1 risk) **and** must not be duplicated into a false double-alert that could cause operator alarm fatigue or confused chain-of-custody logs. This is the one topic where the QoS-2 handshake overhead is justified. |
| `.../alerts/warning` | **1** | Important but not safety-critical within seconds; at-least-once is sufficient. |
| `.../health` | **0** | Heartbeat; a missed heartbeat is itself meaningful (treated as "truck went quiet"), so retransmission isn't needed. |
| `ops/fleet/status` | **1** | Aggregated dashboard feed; occasional loss is tolerable since it's refreshed continuously, but should not silently stop entirely. |

## 3. Local Broker & Offline Design

The MQTT broker (Mosquitto in production; `amqtt`/any MQTT-3.1.1-compatible
broker for local development) runs **on the truck's edge node itself**,
not in the cloud. This is the architectural key to offline operation:

- The simulator → preprocessing → inference → local alert chain communicates
  entirely over `localhost` MQTT and **never depends on cellular connectivity**.
- A **bridge process** (the only cellular-dependent component) subscribes to
  the local broker and forwards `inference`, `alerts/*`, and `health` topics
  upstream to the operations-centre backend *when connectivity is available*.
- During a connectivity gap (e.g. the Nashik–Aurangabad dead zones), the
  bridge simply cannot publish upstream — but the local broker, the
  inference pipeline, and the **on-device alert log (append-only file)**
  keep running unaffected. On reconnect, the bridge replays everything
  appended to the local alert log since the last successful sync
  (store-and-forward), tagged with original timestamps so ordering is
  preserved centrally.

## 4. Task C3 — Data Fusion Level Justification

LogiEdge uses **feature-level fusion**: temperature and vibration are
preprocessed independently (filtering → per-stream feature extraction) and
then concatenated into one 6-value joint feature vector before the single
MLP classifier sees them.

| Fusion level | What it means here | Why rejected / accepted |
|---|---|---|
| **Data-level (raw) fusion** | Feed raw 1 Hz temperature samples and 0.5 Hz vibration samples directly into one model, aligned sample-by-sample. | Rejected: the two sensors have different, non-aligned sampling rates (1 Hz vs 0.5 Hz) and different noise characteristics; naive raw alignment would require upsampling/interpolation that manufactures information that was never measured, and increases the input dimensionality (and therefore SRAM/Flash footprint) for no accuracy benefit on an MCU-class or Pi-class device. |
| **Feature-level fusion (chosen)** | Extract domain-meaningful statistics per stream (mean, std, rate-of-change for temperature; RMS, peak, kurtosis for vibration) over a common 30 s window, then concatenate into one 6-value vector. | **Accepted.** Each stream is summarised using the statistic that is physically meaningful for *that* sensor (a temperature "rate of change" vs a vibration "kurtosis" capture different failure signatures), the two streams are naturally time-aligned by the shared 30 s/10 s windowing scheme, and the fused vector is tiny (6 floats = 24 bytes), which keeps the classifier small enough to fit comfortably within the SRAM/Flash budget of any of the three candidate hardware options in Component B. |
| **Decision-level fusion** | Train two separate classifiers (one on temperature features, one on vibration features) and combine their two class predictions with a voting/arbitration rule. | Rejected for this task: refrigeration failure frequently manifests as a *correlated* pattern across both sensors (e.g., a failing compressor raises both temperature and vibration together), and a decision-level scheme discards exactly that cross-sensor correlation because each sub-model never sees the other sensor's values. Decision-level fusion would be reconsidered if the two sensors needed independent redundancy paths (e.g., if a camera and a thermal sensor could each *individually* claim "fire" in the wildfire use case from the course), which is not the situation here — both sensors must agree that routes through one physical refrigeration subsystem.

Feature-level fusion is therefore the best fit for LogiEdge: it preserves
cross-sensor correlation, keeps the on-device compute/memory footprint
minimal, and matches the natural windowing cadence already required by the
latency budget (90 s SLA, 30 s window, 10 s step → a detection is possible
within at most 40 s of a fault signature first appearing, comfortably inside
budget).

"""
simulator.py — LogiEdge Cold-Chain Sensor Simulator
=====================================================

Simulates the three sensor streams produced by a refrigerated truck's edge
node and publishes them to a local MQTT broker:

    1. temperature   (1 Hz)    - cargo compartment temperature, deg C
    2. vibration_rms (0.5 Hz)  - refrigeration-unit compressor vibration RMS, g
    3. door_event    (discrete) - OPEN / CLOSE events with timestamps

Anomaly modes (--anomaly):
    none         - healthy operation (Class 0: Normal)
    temp_drift   - linear temperature drift, +0.08 C per reading (Class 1: Warning)
    vibration    - step change in compressor vibration, bearing wear (Class 1/2)
    combined     - both anomalies simultaneously (Class 2: Critical)

The statistical model lives in `ColdChainSimulator` so that both the live
MQTT publisher (this CLI) and the offline dataset generator
(training/generate_dataset.py) share a single source of truth for sensor
behaviour.

Usage (live MQTT publishing):
    python simulator.py --anomaly none --truck-id truck-07 --duration 1200
    python simulator.py --anomaly combined --broker localhost --port 1883

Course context: AIML ZG535 Mini-Project, Component C (Module 3).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Physical / statistical constants (from the mini-project specification)
# ---------------------------------------------------------------------------

TEMP_SETPOINT_C = 4.0
TEMP_NOISE_STD = 0.3          # N(4.0, 0.3)
TEMP_DRIFT_PER_READING = 0.08  # linear drift, deg C per 1 Hz reading

VIBRATION_NORMAL_MEAN = 0.45   # g, N(0.45, 0.05)
VIBRATION_NORMAL_STD = 0.05
VIBRATION_ANOMALY_MEAN = 1.20  # g, step change -> N(1.2, 0.15)
VIBRATION_ANOMALY_STD = 0.15

TEMPERATURE_HZ = 1.0
VIBRATION_HZ = 0.5

# Average one door event every ~8 minutes of normal operation (loading stops).
DOOR_EVENT_MEAN_INTERVAL_S = 480


@dataclass
class SensorSample:
    """A single timestamped sensor reading."""
    stream: str            # "temperature" | "vibration" | "door"
    t: float                # simulated elapsed seconds since start
    value: object           # float for temperature/vibration, str for door


@dataclass
class ColdChainSimulator:
    """
    Reusable generator for FreightBridge cold-chain truck sensor data.

    This class performs NO sleeping and NO MQTT I/O -- it is a pure sample
    generator so it can be reused unmodified by generate_dataset.py (fast,
    offline dataset creation) and by the live MQTT CLI below (real-time
    publishing, used for the demo video).
    """
    truck_id: str
    anomaly: str = "none"              # none | temp_drift | vibration | combined
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _door_open: bool = field(default=False, init=False)
    _next_door_event_t: float = field(default=0.0, init=False)

    VALID_ANOMALIES = ("none", "temp_drift", "vibration", "combined")

    def __post_init__(self):
        if self.anomaly not in self.VALID_ANOMALIES:
            raise ValueError(f"--anomaly must be one of {self.VALID_ANOMALIES}, got {self.anomaly!r}")
        self._rng = random.Random(self.seed)
        self._next_door_event_t = self._rng.expovariate(1.0 / DOOR_EVENT_MEAN_INTERVAL_S)

    # -- individual stream models -----------------------------------------

    def temperature_at(self, reading_index: int) -> float:
        """Reading at 1 Hz. `reading_index` = number of 1 Hz ticks since start."""
        mean = TEMP_SETPOINT_C
        if self.anomaly in ("temp_drift", "combined"):
            # Linear drift: refrigeration unit losing capacity.
            mean = TEMP_SETPOINT_C + TEMP_DRIFT_PER_READING * reading_index
        return round(self._rng.gauss(mean, TEMP_NOISE_STD), 3)

    def vibration_at(self, reading_index: int) -> float:
        """Reading at 0.5 Hz. Anomaly is a STEP change (bearing wear), not gradual."""
        if self.anomaly in ("vibration", "combined"):
            mean, std = VIBRATION_ANOMALY_MEAN, VIBRATION_ANOMALY_STD
        else:
            mean, std = VIBRATION_NORMAL_MEAN, VIBRATION_NORMAL_STD
        return round(max(0.0, self._rng.gauss(mean, std)), 4)

    def maybe_door_event(self, t: float) -> Optional[str]:
        """Poisson-process door events; returns 'OPEN'/'CLOSE' or None."""
        if t < self._next_door_event_t:
            return None
        event = "CLOSE" if self._door_open else "OPEN"
        self._door_open = not self._door_open
        # Schedule next event: opens/closes happen close together (a stop),
        # then a long gap until the next stop.
        gap = self._rng.uniform(20, 90) if event == "OPEN" else self._rng.expovariate(1.0 / DOOR_EVENT_MEAN_INTERVAL_S)
        self._next_door_event_t = t + gap
        return event

    # -- sample stream generator -------------------------------------------

    def generate(self, duration_s: float) -> Iterator[SensorSample]:
        """
        Yield SensorSample objects for `duration_s` simulated seconds, in
        correct chronological (t) order across all three streams, with NO
        real-time delay. Used by generate_dataset.py for fast dataset build.
        """
        temp_period = 1.0 / TEMPERATURE_HZ
        vib_period = 1.0 / VIBRATION_HZ
        temp_idx = vib_idx = 0
        t = 0.0
        # Merge-sort the three periodic/aperiodic streams by timestamp.
        while t < duration_s:
            next_temp_t = temp_idx * temp_period
            next_vib_t = vib_idx * vib_period
            # advance to earliest pending event
            t = min(next_temp_t, next_vib_t)
            if next_temp_t <= next_vib_t:
                yield SensorSample("temperature", t, self.temperature_at(temp_idx))
                temp_idx += 1
            else:
                yield SensorSample("vibration", t, self.vibration_at(vib_idx))
                vib_idx += 1
            door = self.maybe_door_event(t)
            if door is not None:
                yield SensorSample("door", t, door)

    # -- live MQTT publishing -----------------------------------------------

    def run_mqtt(self, broker: str, port: int, duration_s: Optional[float], qos_map: dict) -> None:
        """Publish samples to MQTT in real time (used for live demo)."""
        import paho.mqtt.client as mqtt

        client = mqtt.Client(client_id=f"simulator-{self.truck_id}")
        client.connect(broker, port, keepalive=30)
        client.loop_start()

        topic_base = f"logibridge/trucks/{self.truck_id}/sensors"
        start = time.time()
        temp_idx = vib_idx = 0
        print(f"[simulator] truck={self.truck_id} anomaly={self.anomaly} broker={broker}:{port}")
        try:
            while True:
                now = time.time() - start
                if duration_s is not None and now >= duration_s:
                    break

                if now >= temp_idx * (1.0 / TEMPERATURE_HZ):
                    reading = self.temperature_at(temp_idx)
                    payload = json.dumps({"truck_id": self.truck_id, "t": now, "value": reading})
                    client.publish(f"{topic_base}/temperature", payload, qos=qos_map.get("temperature", 0))
                    temp_idx += 1

                if now >= vib_idx * (1.0 / VIBRATION_HZ):
                    reading = self.vibration_at(vib_idx)
                    payload = json.dumps({"truck_id": self.truck_id, "t": now, "value": reading})
                    client.publish(f"{topic_base}/vibration", payload, qos=qos_map.get("vibration", 0))
                    vib_idx += 1

                door = self.maybe_door_event(now)
                if door is not None:
                    payload = json.dumps({"truck_id": self.truck_id, "t": now, "event": door})
                    client.publish(f"{topic_base}/door", payload, qos=qos_map.get("door", 1))
                    print(f"[simulator] door event: {door} at t={now:.1f}s")

                time.sleep(0.05)  # 20 Hz polling loop, fine enough for 1 Hz/0.5 Hz streams
        except KeyboardInterrupt:
            print("\n[simulator] stopped by user")
        finally:
            client.loop_stop()
            client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="LogiEdge cold-chain sensor simulator")
    parser.add_argument("--anomaly", choices=ColdChainSimulator.VALID_ANOMALIES, default="none")
    parser.add_argument("--truck-id", default="truck-01")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--duration", type=float, default=None, help="seconds; omit to run until Ctrl+C")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    qos_map = {"temperature": 0, "vibration": 0, "door": 1}
    sim = ColdChainSimulator(truck_id=args.truck_id, anomaly=args.anomaly, seed=args.seed)
    sim.run_mqtt(args.broker, args.port, args.duration, qos_map)


if __name__ == "__main__":
    main()

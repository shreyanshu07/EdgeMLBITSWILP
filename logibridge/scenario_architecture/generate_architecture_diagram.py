"""
generate_architecture_diagram.py — builds system_architecture.png (Task A2)

Draws the LogiEdge truck edge node architecture: sensors -> local MQTT
broker -> preprocessing/inference pipeline -> local alert log -> cellular
uplink (store-and-forward) -> operations-centre backend.

Run once to (re)generate scenario_architecture/system_architecture.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, w, h, text, fc="#dbe9f6", ec="#2b5f8a", fontsize=9.5, style="round,pad=0.02"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.6,
                        facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, wrap=True)
    return (x, y, w, h)


def arrow(b1, b2, text="", side1="right", side2="left", color="#333333", style="-|>", linestyle="-"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {
        "right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
        "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1),
    }
    pts2 = {
        "right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
        "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2),
    }
    a = FancyArrowPatch(pts[side1], pts2[side2], arrowstyle=style, mutation_scale=14,
                         linewidth=1.4, color=color, zorder=1, linestyle=linestyle)
    ax.add_patch(a)
    if text:
        mx, my = (pts[side1][0] + pts2[side2][0]) / 2, (pts[side1][1] + pts2[side2][1]) / 2
        ax.text(mx, my + 0.15, text, ha="center", fontsize=7.5, color="#333333", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


# ---- Truck Edge Node (big outer box) ----
truck = box(0.3, 1.0, 8.0, 6.5, "", fc="#f4f8fb", ec="#555555", style="round,pad=0.03")
ax.text(0.5, 7.2, "TRUCK EDGE NODE  (Raspberry Pi 5 + AI HAT+, 12V DC-DC powered)",
        fontsize=10.5, fontweight="bold", color="#1a3f5c")

sensors = box(0.6, 5.6, 2.1, 1.3, "Sensors\n\nTemperature (1Hz)\nVibration (0.5Hz)\nDoor (OPEN/CLOSE)")
broker = box(3.1, 5.6, 2.1, 1.3, "Local MQTT\nBroker\n(Mosquitto,\nlocalhost:1883)")
prep = box(5.6, 5.6, 2.3, 1.3, "Preprocessing\n(filter, window,\nfeature-extract,\nnormalise)")

infer = box(5.6, 3.6, 2.3, 1.3, "Inference Pipeline\n(Docker container:\nTFLite model,\nclass + confidence)")
alertlog = box(3.1, 3.6, 2.1, 1.3, "Local Alert Log\n(append-only\n.jsonl, survives\noffline gaps)")
bridge = box(0.6, 3.6, 2.1, 1.3, "Store-and-Forward\nBridge\n(cellular-dependent,\nreplays on reconnect)")

siren = box(0.6, 1.4, 3.6, 1.3, "On-truck actions:\nlocal siren / driver alert, chain-of-custody log entry\n(ALWAYS available, no cellular dependency)")
ansible_box = box(5.6, 1.4, 2.3, 1.3, "Ansible-managed\nOTA updates\n(model.tflite,\nreference_dist.json)")

# flow within truck
arrow(sensors, broker, "sensor\nreadings")
arrow(broker, prep, "subscribe")
arrow(prep, infer, side1="bottom", side2="top", text="6-value\nfeature vec")
arrow(infer, alertlog, side1="left", side2="right", text="class+\nconfidence")
arrow(alertlog, bridge, side1="left", side2="right", text="buffered\nrecords")
arrow(alertlog, siren, side1="bottom", side2="top", text="Critical/\nWarning")

# ---- Cellular link (dashed, shown as unreliable) ----
# Routed along a horizontal lane BELOW the middle row of boxes (y=3.35) so it
# does not cross through any box or label.
ax.annotate("", xy=(9.3, 3.35), xytext=(1.65, 3.35),
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8, linestyle=(0, (5, 3))))
ax.text(5.5, 3.1, "Cellular uplink (intermittent -- 35-90 min dead zones on rural routes)",
        ha="center", fontsize=8.5, color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c0392b", alpha=0.9))

# ---- Operations Centre Backend (outside truck) ----
ops = box(9.3, 3.6, 3.2, 2.3,
          "OPERATIONS CENTRE BACKEND\n\nFleet dashboard\nAlert escalation (Warning/Critical)\nChain-of-custody archive\nModel registry + retraining\nAnsible control node")
arrow(bridge, ops, side1="right", side2="left")

# Legend
legend_elems = [
    Line2D([0], [0], color="#333333", lw=1.4, label="Always-on local path (no cellular dependency)"),
    Line2D([0], [0], color="#c0392b", lw=1.6, linestyle=(0, (5, 3)), label="Intermittent cellular path"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=8, frameon=True)

ax.set_title("LogiEdge System Architecture — Component A, Task A2", fontsize=13, fontweight="bold", pad=14)

fig.tight_layout()
fig.savefig("system_architecture.png", dpi=160)
print("Saved system_architecture.png")

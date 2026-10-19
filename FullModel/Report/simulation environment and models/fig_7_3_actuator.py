#!/usr/bin/env python
"""Actuator model, demonstrated on a step command rather than on a descent.

NOT TAKEN FROM A TRAJECTORY, deliberately. On the nominal descent neither limit saturates -- the
commanded slew peaks under half the rate the engine can deliver -- so a plot of the flown profile
shows the limits never being reached, which is a statement about that trajectory and not about the
actuator. Driving the model with a step exposes both responses directly.
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TH = "/Users/rodrigoalves/Desktop/Thesis"
sys.path.insert(0, os.path.join(TH, "FullModel", "SCvx"))
sys.path.insert(0, os.path.join(TH, "FullModel", "Guidance and Control"))
HERE = os.path.dirname(os.path.abspath(__file__))
import config as C
from actuator_model import Actuator

act = Actuator(tau=0.5, slew_max_deg_s=C.SLEW_MAX_DEG_S, t_min=C.TMIN, t_max=C.TMAX)
DT = 0.02

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})


def save(fig, stem):
    # FIXED MARGINS, NOT A TIGHT BOX. bbox_inches="tight" crops each figure to its own content, so
    # a panel with a longer y-label comes out physically wider and is then scaled differently when
    # the two are placed side by side at the same width in LaTeX. Identical figsize and identical
    # axes margins make the panels the same size on the page.
    fig.subplots_adjust(left=0.215, right=0.965, bottom=0.185, top=0.965)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"{stem}.{ext}"), dpi=300)
    print("wrote", stem, flush=True)


# ---------------------------------------------------------------- magnitude: first-order lag
t = np.arange(0, 8.0, DT)
e = np.array([0.0, 0.0, 1.0])
cmd = np.where(t < 1.0, C.TMIN, np.where(t < 4.5, C.TMAX, C.TMIN))
mag, d = act.clip_command(cmd[0] * e)
out = np.empty_like(t)
for i in range(t.size):
    mag, d, _ = act.step(mag, d, cmd[i] * e, DT)
    out[i] = mag
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.step(t, cmd, where="post", color="0.35", lw=1.2, ls="--", label="commanded")
ax.plot(t + DT, out, color="#1d3557", lw=1.6, label="delivered")
for k in (1.0, 4.5):
    ax.axvline(k, color="0.85", lw=0.6, zorder=0)
ax.set_xlabel("time  [s]")
ax.set_ylabel("thrust  [N]")
ax.set_xlim(0, 8); ax.set_ylim(C.TMIN - 2, C.TMAX + 2)
ax.grid(alpha=.25, lw=.5)
# NO LEGEND: the two series are the same ones as the direction panel, which carries it for the pair.
save(fig, "fig_act_lag"); plt.close(fig)

# ---------------------------------------------------------------- direction: rate limit
t2 = np.arange(0, 60.0, DT)
d0 = np.array([0.0, 0.0, 1.0])
d_cmd = np.array([np.sin(np.radians(90.0)), 0.0, np.cos(np.radians(90.0))])
mag, d = act.clip_command(C.TMAX * d0)
ang = np.empty_like(t2); rate = np.empty_like(t2)
for i in range(t2.size):
    tgt = d0 if t2[i] < 1.0 else d_cmd
    mag, d, w = act.step(mag, d, C.TMAX * tgt, DT)
    ang[i] = np.degrees(np.arccos(np.clip(d @ d0, -1, 1)))
    rate[i] = np.degrees(w)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.step(t2, np.where(t2 < 1.0, 0.0, 90.0), where="post", color="0.35", lw=1.2, ls="--",
        label="commanded")
ax.plot(t2 + DT, ang, color="#1d3557", lw=1.6, label="delivered")
ax.axvline(1.0, color="0.85", lw=0.6, zorder=0)
ax.annotate(rf"${C.SLEW_MAX_DEG_S:g}^\circ$/s", xy=(24, 46), xytext=(33, 28), fontsize=8,
            color="#d94801",
            arrowprops=dict(arrowstyle="->", color="#d94801", lw=0.9))
ax.set_xlabel("time  [s]")
ax.set_ylabel("thrust direction  [deg]")
ax.set_xlim(0, 60); ax.set_ylim(-5, 100)
ax.grid(alpha=.25, lw=.5)
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
save(fig, "fig_act_slew"); plt.close(fig)

i90 = int(np.argmax(ang > 89.9))
print(f"a {90:.0f} deg reorientation takes {t2[i90]-1.0:.1f} s at {C.SLEW_MAX_DEG_S:g} deg/s",
      flush=True)

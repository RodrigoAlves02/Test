#!/usr/bin/env python
"""Figures: the solved descent, and the fuel-vs-flight-time curve it has to sit on."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "Figures&data")
INK, ACC, WARN, GRID = "#2B2B2B", "#1B5E9C", "#B8351F", "#D5D5D5"


def main(tag="_K60"):
    z = np.load(os.path.join(OUT, f"descent{tag}.npz"))
    X, U, s, L, T, A = z["X"], z["U"], float(z["s"]), float(z["L"]), float(z["T"]), float(z["A"])
    rf, n, K = z["rf"], z["n_hat"], int(z["K"])
    tf = s * T
    t = np.linspace(0, tf, K)
    alt = (n @ (X[0:3] - (rf / L)[:, None])) * L
    Pl = np.eye(3) - np.outer(n, n)
    lat = np.linalg.norm(Pl @ (X[0:3] - (rf / L)[:, None]), axis=0) * L
    v = np.linalg.norm(X[3:6], axis=0) * (L / T)
    # THRUST IS A STATE: X[7:10], not U[0:3] (which is the thrust RATE). Because the rate is held,
    # the thrust is piecewise LINEAR in time, so it is drawn as a line rather than as a step.
    m = np.exp(X[6]); Aacc = X[7:10]; an = np.linalg.norm(Aacc, axis=0)
    Th = m * an * A
    dh = Aacc / np.maximum(an, 1e-12)
    ang = np.degrees(np.arccos(np.clip(dh.T @ n, -1, 1)))
    slew = np.degrees(np.arccos(np.clip(np.sum(dh[:, :-1] * dh[:, 1:], axis=0), -1, 1))) / (tf / (K - 1))

    fig, ax = plt.subplots(2, 3, figsize=(14.5, 7.6))
    ax[0, 0].plot(t, alt / 1e3, color=ACC, lw=2); ax[0, 0].set_ylabel("altitude [km]")
    ax[0, 0].set_title("Altitude above the site", fontsize=10)
    ax[0, 1].plot(t, v, color=ACC, lw=2)
    ax[0, 1].axhline(C.VTOUCH, color=WARN, ls="--", lw=1.2)
    ax[0, 1].text(t[-1], C.VTOUCH, f" {C.VTOUCH} m/s", color=WARN, fontsize=8, va="bottom", ha="right")
    ax[0, 1].set_ylabel("speed [m/s]"); ax[0, 1].set_title("Speed (rotating frame)", fontsize=10)
    ax[0, 2].plot(lat, alt, color=ACC, lw=2, label="trajectory")
    hh = np.linspace(0, alt.max(), 50)
    ax[0, 2].plot(np.tan(np.radians(C.GLIDESLOPE_DEG)) * hh, hh, color=WARN, ls="--", lw=1.4,
                  label=f"{C.GLIDESLOPE_DEG:g}$^\\circ$ glideslope")
    ax[0, 2].set_xlabel("lateral offset [m]"); ax[0, 2].set_ylabel("altitude [m]")
    ax[0, 2].set_title("Approach cone", fontsize=10); ax[0, 2].legend(fontsize=8)
    ax[1, 0].plot(t, Th, color=ACC, lw=2)
    for lv, lb in ((C.TMIN, "$T_{min}$"), (C.TMAX, "$T_{max}$")):
        ax[1, 0].axhline(lv, color=WARN, ls=":", lw=1.2)
        ax[1, 0].text(t[-1], lv, f" {lb}", color=WARN, fontsize=8, va="center", ha="right")
    ax[1, 0].set_ylabel("thrust [N]"); ax[1, 0].set_ylim(0, C.TMAX * 1.18)
    ax[1, 0].set_title("Thrust: max-min-max (continuous, a is a state)", fontsize=10)
    ax[1, 1].plot(t, ang, color=ACC, lw=2)
    ax[1, 1].axhline(C.POINTING_DEG, color=WARN, ls="--", lw=1.2)
    ax[1, 1].text(t[-1], C.POINTING_DEG, f" limit {C.POINTING_DEG:g}$^\\circ$", color=WARN,
                  fontsize=8, va="top", ha="right")
    ax[1, 1].set_ylabel("angle from local vertical [deg]")
    ax[1, 1].set_title("Thrust pointing", fontsize=10)
    ax[1, 2].plot(t[:-1], slew, color=ACC, lw=2, zorder=3)
    if C.SLEW_MAX_DEG_S is not None:
        ax[1, 2].axhline(C.SLEW_MAX_DEG_S, color=WARN, ls="--", lw=1.3)
        ax[1, 2].text(t[-1], C.SLEW_MAX_DEG_S, f" limit {C.SLEW_MAX_DEG_S:g}$^\\circ$/s",
                      color=WARN, fontsize=8, va="top", ha="right")
        ax[1, 2].set_ylim(0, C.SLEW_MAX_DEG_S * 1.25)
    ax[1, 2].set_ylabel("thrust slew rate [deg/s]")
    ax[1, 2].set_title(f"Slew rate (peak {slew.max():.2f}$^\\circ$/s)", fontsize=10)

    for a_ in ax.ravel():
        a_.grid(alpha=0.25, lw=0.6)
        if a_ in (ax[1, 0], ax[1, 1]) or a_ in (ax[0, 0], ax[0, 1]):
            a_.set_xlabel("time [s]")
    miss_m = float(np.linalg.norm(Pl @ (X[0:3, -1] - (rf / L)))) * L
    ang_end = ang[K - 1]          # the terminal THRUST state, not the rate
    fig.suptitle(f"Eros powered descent, two-stage SCvx  —  K={K}, {tf:.0f} s, "
                 f"{m[0]-m[-1]:.3f} kg, miss {miss_m*1e3:.1f} mm, "
                 f"touchdown {ang_end:.1f}$^\\circ$ from vertical", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"descent{tag}.{ext}"), dpi=300, bbox_inches="tight")
    print("wrote " + os.path.join(OUT, f"descent{tag}.{{pdf,png}}"))


if __name__ == "__main__":
    main()

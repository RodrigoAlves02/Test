#!/usr/bin/env python
"""Closed-loop MPC descent: what guidance asked for against what the actuator delivered.

Reads Figures&data/closed_loop.json, written by closed_loop.py. The thrust magnitude relaxes toward
the command with time constant tau; the direction slews at up to slew_max. Both are exact zero-order
-hold updates applied at the actuator sub-step, so the lag visible here is the modelled hardware,
not integration error.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "Figures&data", "closed_loop.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()
    d = json.load(open(a.json))
    tr = d["trace"]
    t = np.asarray(tr["t"], float)
    Tc, Ta = np.asarray(tr["T_cmd"], float), np.asarray(tr["T_act"], float)
    ac, aa = np.asarray(tr["ang_cmd"], float), np.asarray(tr["ang_act"], float)
    ae = np.asarray(tr["ang_err"], float)
    cyc = d["cycles"]
    tg = np.array([c["t"] for c in cyc])
    tl = float(d["t_land"])

    CMD, ACT, ERR, GRID = "#8A8A8A", "#1B5E9C", "#B8351F", "#D8D8D8"
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True)

    for ax in (a1, a2):
        for g in tg:
            ax.axvline(g, color=GRID, lw=0.7, zorder=0)
        ax.axvline(tl, color="#3E3E3E", lw=1.2, ls="--", zorder=1)
        ax.grid(alpha=0.25, lw=0.6)

    a1.step(t, Tc, where="post", color=CMD, lw=2.0, zorder=2, label="commanded (SCvx plan, ZOH on its own node grid)")
    a1.plot(t, Ta, color=ACT, lw=2.0, zorder=3, label=f"actual (first-order lag, $\\tau$={d['tau']:g} s)")
    tmin, tmax = float(np.min(Tc)), float(np.max(Tc))
    for lvl, lab in ((tmin, f"$T_{{min}}$ = {tmin:.0f} N"), (tmax, f"$T_{{max}}$ = {tmax:.0f} N")):
        a1.axhline(lvl, color="#B0B0B0", lw=0.9, ls=":", zorder=1)
        a1.text(t[-1], lvl, "  " + lab, fontsize=7.5, color="#7A7A7A", va="center", ha="left")
    a1.set_ylim(0.0, tmax * 1.12)
    a1.set_xlim(t[0] - 8, t[-1] + 40)
    a1.set_ylabel("thrust magnitude  [N]")
    a1.set_title(f"Closed-loop descent: guidance command vs delivered thrust "
                 f"— touchdown {tl:.1f} s, miss {d['miss']:.2f} m, "
                 f"{d['touchdown_speed']:.2f} m/s", fontsize=10.5)
    a1.legend(fontsize=8.5, loc="center left", framealpha=0.95)
    a1.text(0.02, 0.05, "thin vertical lines: guidance cycles", transform=a1.transAxes,
            ha="left", va="bottom", fontsize=7.5, color="#7A7A7A")
    # The run picks up an existing descent, so the actuator starts where the nominal had it --
    # already lit and already pointed. There is no start-up transient to explain away.
    a2.text(0.03, 0.30, f"actuator initialised from the nominal at the handover node:\n"
            f"{Ta[0]:.2f} N at {aa[0]:.1f} deg from local vertical",
            transform=a2.transAxes, ha="left", va="bottom", fontsize=7.5, color="#7A7A7A")

    a2.step(t, ac, where="post", color=CMD, lw=2.0, zorder=2, label="commanded angle from local vertical (SCvx plan)")
    a2.plot(t, aa, color=ACT, lw=2.0, zorder=3, label=f"actual (slew limit {d['slew']:g} deg/s)")
    a2.plot(t, ae, color=ERR, lw=1.4, zorder=4, label="pointing error |actual - commanded|")
    a2.set_ylabel("angle  [deg]")
    a2.set_xlabel("time since start of the closed-loop run  [s]")
    a2.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    a2.text(tl, a2.get_ylim()[1], "  touchdown", fontsize=8, color="#3E3E3E",
            va="top", ha="left")

    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"closed_loop.{ext}"), dpi=300, bbox_inches="tight")
    print(f"landed {d['landed']}, {len(cyc)} cycles, peak pointing error {ae.max():.2f} deg, "
          f"max |T_cmd - T_act| {np.abs(Tc-Ta).max():.3f} N")
    print("wrote " + os.path.join(a.out, "closed_loop.{pdf,png}"))


if __name__ == "__main__":
    main()

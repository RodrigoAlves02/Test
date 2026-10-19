#!/usr/bin/env python
"""max error against dt for the actuator-as-a-state RK4 study.

Reads Figures&Data/actuator_state_convergence.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figures&Data")


def main():
    J = json.load(open(os.path.join(OUT, "actuator_state_convergence.json")))
    rows = sorted(J["rows"], key=lambda r: r["dt"])          # ascending
    dts = np.array([r["dt"] for r in rows])
    mx = np.array([r["max_r"] for r in rows])
    wall = 2.785 * J["tau"]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    # slope-4 reference, anchored on the coarsest point so the deficit opens downward
    xs = np.array([dts.min() * 0.85, dts.max() * 1.1])
    ax.loglog(xs, mx[-1] * (xs / dts[-1]) ** 4.0, "--", color="#b0b6bd", lw=1.5, zorder=1,
              label="slope 4")

    ax.loglog(dts, mx, "o-", color="#1b4f72", lw=2.2, ms=8, zorder=3, label="measured")

    # the stability wall -- the lag eigenvalue, not accuracy, is what caps the step
    ax.axvline(wall, color="#9b2226", ls="--", lw=1.5, zorder=2)
    ax.text(wall * 0.95, mx.max() * 1.9,
            f"RK4 goes unstable for dt > {wall:.2f} s\n"
            f"(2.785 x the {J['tau']:g} s engine lag)",
            color="#9b2226", fontsize=9, ha="right", va="top")

    # local order between neighbouring rungs
    for i in range(len(dts) - 1):
        od = np.log(mx[i + 1] / mx[i]) / np.log(dts[i + 1] / dts[i])
        ax.annotate(f"{od:.2f}", xy=(np.sqrt(dts[i] * dts[i + 1]), np.sqrt(mx[i] * mx[i + 1])),
                    xytext=(9, -4), textcoords="offset points", fontsize=9, color="#1b4f72")

    ax.set_xlim(dts.min() * 0.75, wall * 1.25)
    ax.set_ylim(mx.min() * 0.45, mx.max() * 3.2)
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("max |r(dt) - r(dt/2)|   [m]")
    ax.set_title("RK4, engine lag carried as a state\n"
                 f"thrust response time constant {J['tau']:g} s", fontsize=11.5)
    ax.grid(alpha=.22, which="both")
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    f = os.path.join(OUT, "actuator_state_error_vs_dt.png")
    fig.savefig(f, dpi=150, bbox_inches="tight")
    print(f"wrote {f}")


if __name__ == "__main__":
    main()

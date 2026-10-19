#!/usr/bin/env python
"""Convergence figure for the parking-orbit step-halving benchmark.

Same construction as the earlier convergence figures: position error on the left axis, velocity
error on its own right axis in its own units, and CPU cost on a secondary axis parallel to x, since
cost is a function of the step and therefore of x. No tolerance line and no selected step: this
figure establishes the convergence behaviour and marks the step selected against the tolerance.

No cost axis. This run only has to produce the benchmark step that the later integrator comparisons
are measured against; what each of THOSE costs is the question those figures answer, not this one.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "..", "scvx_landing"))
import thesis_style as ts                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--tol", type=float, default=1e-5, help="model tolerance [m]")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()
    a.json = a.json or sorted(glob.glob(os.path.join(HERE, "Figures&data", "parking_benchmark_*.json")))[0]
    d = json.load(open(a.json))

    FS = ts.use()
    rows = [r for r in d["rows"] if r["dr"] > 0]
    dt = np.array([r["dt"] for r in rows])
    dr = np.array([r["dr"] for r in rows])
    dv = np.array([r["dv"] for r in rows])
    cpu = np.array([r["cpu"] for r in rows])
    ev = np.array([r["evals"] for r in rows], float)
    order = np.array([r["order"] for r in rows], float)

    # Fit only where the scheme is genuinely converging. Once the halving difference sits on the
    # round-off floor, halving again returns nothing and the apparent order collapses; including
    # those points drags an honest p = 4 downwards.
    FLOOR = 10.0 * np.nanmin(dr)
    ok = np.isfinite(order) & (order > 3.0) & (dr > FLOOR)
    p_fit = np.polyfit(np.log(dt[ok]), np.log(dr[ok]), 1)[0] if ok.sum() >= 2 else np.nan

    fig, ax = plt.subplots(figsize=ts.figsize())
    C1, C2 = ts.BLUE, ts.ORANGE

    # The floor comes from the STEP being too small, so it marks a region of the x axis.
    floor_dt = dt[dr <= FLOOR]
    dt_lo = float(floor_dt.max() * np.sqrt(2.0)) if floor_dt.size else float(dt.min())
    if floor_dt.size:
        ax.axvspan(dt.min() / 1.6, dt_lo, color="0.90", zorder=0, lw=0)
        ax.text(np.sqrt(dt.min() / 1.6 * dt_lo), 0.50, "round-off\nlimited",
                transform=ax.get_xaxis_transform(), fontsize=FS, color="0.35",
                ha="center", va="center", linespacing=1.3)

    h = ax.loglog(dt, dr, "o-", c=C1, label="Position error")
    if np.isfinite(p_fit):
        ref = dr[ok][-1] * (dt / dt[ok][-1]) ** 4
        h += ax.loglog(dt, ref, ":", c=ts.GREY,
                       label=rf"$\propto\Delta t^{{4}}$  (fitted $p={p_fit:.2f}$)")

    _tl = (rf"{a.tol*1e3:g} mm tolerance" if a.tol < 1e-2 else rf"{a.tol:g} m tolerance")
    h.append(ax.axhline(a.tol, c=ts.GREEN, ls="-.", label=_tl))
    i = np.where(dr <= a.tol)[0]
    if len(i):
        k = i[0]                       # coarsest step meeting the tolerance
        h.append(ax.scatter([dt[k]], [dr[k]], s=130, facecolors="none", edgecolors=ts.RED,
                            lw=2.0, zorder=6, label=rf"selected $\Delta t={dt[k]:.0f}$ s"))

    ax2 = ax.twinx()
    h_vel = ax2.loglog(dt, dv, "s--", c=C2, ms=4.5, label="Velocity error")
    ax2.set_ylabel(r"Velocity Error  [m/s]", color=C2)
    ax2.tick_params(axis="y", colors=C2)
    ax2.spines["right"].set_color(C2)
    ax2.set_ylim(10 ** np.floor(np.log10(dv[dv > 0].min())) / 3,
                 10 ** np.ceil(np.log10(dv.max())) * 3)
    ax2.grid(False)

    ax.set_xlabel(r"Step Size $\Delta t$  [s]")
    ax.set_ylabel(r"Position Error  [m]", color=C1)
    ax.tick_params(axis="y", colors=C1)
    ax.spines["left"].set_color(C1)
    ax.set_ylim(10 ** np.floor(np.log10(dr.min())) / 3, 10 ** np.ceil(np.log10(dr.max())) * 3)
    ax.set_xlim(dt.min() / 1.6, dt.max() * 2.6)
    ax.grid(True, alpha=.25, which="both")

    ax.legend(handles=[h[0]] + h_vel + h[1:], loc="lower right", ncol=1)
    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(a.json))[0]
    for ext in ("pdf", "png"):
        f = os.path.join(a.out, f"{stem}.{ext}")
        fig.savefig(f, dpi=200, bbox_inches="tight")
        print("wrote", f)
    print(f"fitted order {p_fit:.2f}; round-off floor {np.nanmin(dr):.2e} m")
    if len(i):
        print(f"selected dt = {dt[i[0]]:.0f} s -> {dr[i[0]]:.3e} m, "
              f"{ev[i[0]]:.0f} evaluations, {cpu[i[0]]:.2f} s")


if __name__ == "__main__":
    main()

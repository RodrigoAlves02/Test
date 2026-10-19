#!/usr/bin/env python
"""The two plots for the truth-propagator step selection study.

    step_error_vs_time.png   time on x, error on y, one curve per dt
    step_error_vs_dt.png     dt on x, max error on y

Reads Figures&Data/step_convergence.json, written by step_convergence.py.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Figures&Data")


# CHAPTER FIGURE FORMAT, as elsewhere in the report: fixed margins rather than a tight box so
# every panel is the same size on the page, and the chapter palette rather than a colormap.
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
NAVY, ORANGE = "#1d3557", "#d94801"


def save(fig, stem):
    fig.subplots_adjust(left=0.215, right=0.965, bottom=0.185, top=0.965)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300)
    print("wrote", stem, flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_actx",
                    help="which convergence run to plot; _actx is the actuator case with the "
                         "exact in-hold thrust, which is what the truth propagator flies")
    a = ap.parse_args()
    J = json.load(open(os.path.join(OUT, f"step_convergence{a.tag}.json")))
    rows = sorted(J["rows"], key=lambda r: -r["dt"])
    # one hue, light to dark with refinement: the curves are a sequence, not categories
    cm = [plt.matplotlib.colors.to_rgba(NAVY, alpha=al)
          for al in np.linspace(0.30, 1.0, len(rows))]

    # ---------------------------------------------------------------- error against time
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for r, c in zip(rows, cm):
        ax.plot(r["t"], r["err_t"], lw=1.3, color=c,
                label=rf"$\Delta t = {r['dt']:g}$ s")
    ax.set_yscale("log")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel(r"$\|\mathbf{r}(\Delta t) - \mathbf{r}(\Delta t/2)\|$   [m]")
    ax.set_xlim(0, J["t_end"])
    ax.grid(alpha=.25, lw=.5)
    ax.legend(frameon=False, loc="lower right", fontsize=6.4, labelspacing=0.25,
              borderpad=0.2, handlelength=1.6)
    save(fig, "step_error_vs_time"); plt.close(fig)

    # ---------------------------------------------------------------- max error against dt
    dts = np.array([r["dt"] for r in rows])
    mx = np.array([r["max_r"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(dts, mx, "o-", color=NAVY, lw=1.3, ms=4)
    ref = mx[0] * (dts / dts[0]) ** 8
    ax.loglog(dts, ref, "--", color="0.55", lw=0.9)
    ax.text(dts[len(dts)//2], ref[len(dts)//2] * 3, r"$\Delta t^{8}$", fontsize=7, color="0.45")
    ax.set_xlabel(r"$\Delta t$  [s]")
    ax.set_ylabel(r"max $\|\mathbf{r}(\Delta t) - \mathbf{r}(\Delta t/2)\|$   [m]")
    ax.grid(alpha=.25, lw=.5, which="both")
    save(fig, "step_error_vs_dt"); plt.close(fig)

    print(f"\n{'dt [s]':>9}{'dt/2':>9}{'max |dr| [m]':>16}")
    for r in rows:
        print(f"{r['dt']:>9g}{r['dt']/2:>9g}{r['max_r']:>16.4e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Cost and flight-time sensitivity to one state component. One panel per quantity.

Reads the shards written by sweep.py, so it plots whatever has finished rather than requiring the
whole sweep. Non-converged draws are drawn at the axis as crosses rather than dropped silently: a
sensitivity plot that hides the cases the solver could not do is making the problem look smoother
than it is.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Figures&data")
UNIT = {"r_x": "m", "r_y": "m", "r_z": "m", "v_x": "m/s", "v_y": "m/s", "v_z": "m/s", "m": "kg"}
LBL = {"r_x": r"$\Delta r_x$", "r_y": r"$\Delta r_y$", "r_z": r"$\Delta r_z$",
       "v_x": r"$\Delta v_x$", "v_y": r"$\Delta v_y$", "v_z": r"$\Delta v_z$",
       "m": r"$\Delta m_0$"}


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "sweep_shard_*.json"))):
        rows += json.load(open(f))["rows"]
    nom = json.load(open(os.path.join(OUT, "sweep_nominal.json")))
    return rows, nom


def main():
    chans = sys.argv[1:] or None
    rows, nom = load()
    if not rows:
        raise SystemExit("no shards in " + OUT)
    present = [c for c in UNIT if any(r["channel"] == c for r in rows)]
    chans = [c for c in (chans or present) if c in present]
    J0, tf0 = nom["J_kg"], nom["tf_s"]

    for ch in chans:
        s = [r for r in rows if r["channel"] == ch]
        ok = [r for r in s if r["converged"]]
        bad = [r for r in s if not r["converged"]]
        d = np.array([r["delta"] for r in ok]); o = np.argsort(d); d = d[o]
        dJ = (np.array([r["J_kg"] for r in ok])[o] - J0) * 1e3        # grams
        dt = np.array([r["tf_s"] for r in ok])[o] - tf0

        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
        for a, y, yl in ((ax[0], dJ, r"$\Delta J$ [g]"), (ax[1], dt, r"$\Delta t_f$ [s]")):
            a.axhline(0.0, lw=0.6, color="0.75", zorder=0)
            a.axvline(0.0, lw=0.6, color="0.75", zorder=0)
            a.plot(d, y, "o", ms=3.4, mfc="#1f5fa8", mec="none", zorder=3)
            # a straight line through the origin, FITTED TO THE CONVERGED POINTS ONLY, as the
            # reference a linear sensitivity would follow. Drawn so curvature is visible as
            # departure from it rather than having to be judged by eye against nothing.
            if len(d) > 2:
                k = float(np.sum(d * y) / np.sum(d * d))
                a.plot(d, k * d, "-", lw=0.9, color="#c2541b", zorder=2)
                a.text(0.03, 0.94, f"slope {k:+.4g} {yl.split('[')[1][:-1]}/{UNIT[ch]}",
                       transform=a.transAxes, fontsize=7.0, va="top", color="#c2541b")
            for r in bad:
                a.plot([r["delta"]], [0.0], "x", ms=5, mew=1.2, color="#b00020", zorder=4)
            a.set_xlabel(f"{LBL[ch]} [{UNIT[ch]}]", fontsize=8.5)
            a.set_ylabel(yl, fontsize=8.5)
            a.tick_params(labelsize=7.5)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        n = f"{len(ok)} converged" + (f", {len(bad)} not" if bad else "")
        ax[0].text(0.03, 0.06, n, transform=ax[0].transAxes, fontsize=7.0, color="0.35")
        fig.subplots_adjust(left=0.095, right=0.985, bottom=0.21, top=0.96, wspace=0.33)
        stem = f"sensitivity_{ch}"
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=200)
        plt.close(fig)
        print(f"{ch:>4}  {len(ok):>3} ok {len(bad):>2} bad   "
              f"dJ {dJ.min():+8.2f} .. {dJ.max():+8.2f} g   "
              f"dtf {dt.min():+7.2f} .. {dt.max():+7.2f} s   -> {stem}.pdf")


if __name__ == "__main__":
    main()

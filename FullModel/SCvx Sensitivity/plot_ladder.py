#!/usr/bin/env python
"""How the cost and flight-time sensitivities change down the descent.

One panel per (quantity, input group), because the units do not mix: g/m for position, g per m/s
for velocity, g/kg for mass. Altitude runs high-to-low left to right, so the panels read in the
direction the vehicle flies.

The zero line is drawn in every panel and it matters: the near-null direction is set by the primer
vector, which rotates along the trajectory, so individual channels CROSS zero between rungs. A
channel's sensitivity changing sign is the signal, not an artefact.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Figures&data")
GROUPS = [("position", ["r_x", "r_y", "r_z"], "m", r"$\Delta r$"),
          ("velocity", ["v_x", "v_y", "v_z"], "m/s", r"$\Delta v$"),
          ("mass", ["m"], "kg", r"$\Delta m_0$")]
COL = {"r_x": "#1f5fa8", "r_y": "#c2541b", "r_z": "#2e7d32",
       "v_x": "#1f5fa8", "v_y": "#c2541b", "v_z": "#2e7d32", "m": "#5b3a8e"}
LBL = {"r_x": "$x$", "r_y": "$y$", "r_z": "$z$",
       "v_x": "$v_x$", "v_y": "$v_y$", "v_z": "$v_z$", "m": "$m_0$"}


def slopes(path):
    """Linear coefficient of a quadratic fit through the origin, per channel: dJ and dtf."""
    Z = json.load(open(path))
    J0, tf0 = Z["nominal"]["J_kg"], Z["nominal"]["tf_s"]
    out = {}
    for ch in LBL:
        # FIRST-PASS DRAWS ONLY. A draw that converged on a fallback was solved by a different
        # trust-region schedule from its neighbours, and the two schedules were measured
        # disagreeing by ~3 g while two near-identical draws under ONE schedule agree to 0.19 g.
        # Mixing them puts the schedule difference straight into the fitted slope.
        s = [r for r in Z["rows"]
             if r["channel"] == ch and r["converged"] and r["status"] == "converged"]
        if len(s) < 4:
            continue
        d = np.array([r["delta"] for r in s])
        M = np.vstack([d, d * d]).T
        (aJ, _), *_ = np.linalg.lstsq(M, (np.array([r["J_kg"] for r in s]) - J0) * 1e3, rcond=None)
        (at, _), *_ = np.linalg.lstsq(M, np.array([r["tf_s"] for r in s]) - tf0, rcond=None)
        out[ch] = (float(aJ), float(at), len(s))
    return out, Z


def main():
    rungs = []
    # ONLY THE 1 mm SET (tag _m*). The earlier _a* runs used the 0.1 m label-convention slack,
    # which let the optimiser pick a landing point anywhere in a 75 mm ball and put ~4 g of
    # ambiguity into every draw. Mixing the two conventions on one axis would be comparing
    # gradients measured under different problems.
    for p in sorted(glob.glob(os.path.join(OUT, "sweep_n*.json"))):
        b = os.path.basename(p)
        if "nominal" in b or "costate" in b or b.startswith("sweep_shard"):
            continue
        sl, Z = slopes(p)
        if not sl:
            continue
        alt = Z.get("alt_km")
        if alt is None:
            alt = Z["altitude_km"]
        # the rung's TRUE altitude, from the base state rather than the requested one
        r0 = np.array(Z["rows"][0]["r0"]); site = np.array(Z["site"].get("r_site", [0, 0, 0]))
        rungs.append((float(alt), sl, Z))
    rungs.sort(key=lambda t: -t[0])
    if not rungs:
        raise SystemExit("no rungs in " + OUT)
    alts = np.array([r[0] for r in rungs])
    print("rungs: " + ", ".join(f"{a:.3f} km" for a in alts))

    fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.3))
    for col, (name, chans, unit, _s) in enumerate(GROUPS):
        for row, (key, qty, qunit) in enumerate(
                ((0, r"$\partial J/\partial$", "g"), (1, r"$\partial t_f/\partial$", "s"))):
            a = ax[row][col]
            a.axhline(0.0, lw=0.7, color="0.7", zorder=0)
            for ch in chans:
                y = [r[1][ch][row] for r in rungs if ch in r[1]]
                x = [r[0] for r in rungs if ch in r[1]]
                if not y:
                    continue
                a.plot(x, y, "o-", ms=3.6, lw=1.2, color=COL[ch], label=LBL[ch], zorder=3)
            a.invert_xaxis()                      # high altitude first: the descent reads left to right
            a.set_xlabel("altitude above the site [km]", fontsize=7.6)
            a.set_ylabel(f"{qty} {name.split()[0]}  [{qunit}/{unit}]", fontsize=7.6)
            a.tick_params(labelsize=7.0)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            if len(chans) > 1:
                a.legend(fontsize=6.6, frameon=False, ncol=3, loc="best",
                         handlelength=1.1, columnspacing=0.8)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.115, top=0.985,
                        wspace=0.42, hspace=0.46)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"sensitivity_ladder.{ext}"), dpi=200)
    print(f"wrote {OUT}/sensitivity_ladder.{{pdf,png}}")
    print("\nfirst-pass draws behind each slope (of 50 per channel):")
    print(f"     {'alt [km]':>9} " + "".join(f"{LBL[c].strip('$'):>10}" for g in GROUPS for c in g[1]))
    for alt, sl, _ in rungs:
        print(f"     {alt:>9.3f} " + "".join(
            f"{sl[c][2]:>10d}" if c in sl else f"{'-':>10}" for g in GROUPS for c in g[1]))
    hdr = f"{'alt [km]':>9} " + "".join(f"{LBL[c].strip('$'):>10}" for g in GROUPS for c in g[1])
    print("\ndJ:  " + hdr)
    for alt, sl, _ in rungs:
        print(f"     {alt:>9.3f} " + "".join(
            f"{sl[c][0]:>10.4f}" if c in sl else f"{'-':>10}" for g in GROUPS for c in g[1]))
    print("\ndtf: " + hdr)
    for alt, sl, _ in rungs:
        print(f"     {alt:>9.3f} " + "".join(
            f"{sl[c][1]:>10.4f}" if c in sl else f"{'-':>10}" for g in GROUPS for c in g[1]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Work-precision comparison of the three propagator back ends.

Cost is GRAVITY EVALUATIONS. All three back ends call the same polyhedral field through the same
right-hand side, so an evaluation costs the same wherever it is issued (measured: 356-396 us across
every run) and the count is the only thing that separates them. Wall clock would add the library's
own overhead, which is reported separately because it turns out to be ~2.5% of the total.

Colour is the library, shape is the scheme: solid circles for the order-5 pair, dashed triangles for
the order-8 pair, stars for fixed-step RK4 at dt = 64 s.
"""
from __future__ import annotations

import argparse
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

LIB = {"scipy": ts.BLUE, "tudat": ts.ORANGE, "local": ts.GREEN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "Figures&data", "library_test.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()
    d = json.load(open(a.json))
    rows = d["rows"]
    floor = d["ref_err_estimate"]
    ts.use()

    fig, ax = plt.subplots(figsize=ts.figsize())

    # The reference is the dt = 32 s benchmark, whose own error is about 1e-6 m. Anything below that
    # is not a measurement of the scheme, it is the reference's own error showing through.
    ax.axhspan(1e-9, floor, color="0.90", zorder=0, lw=0)
    ax.text(0.5, 0.055, "below the benchmark's own error", transform=ax.transAxes,
            ha="center", va="center", color="0.35")

    def pick(lib, scheme):
        r = [x for x in rows if x["lib"] == lib and x["scheme"] == scheme]
        return (np.array([x["evals"] for x in r], float),
                np.array([max(x["err"], 1e-12) for x in r], float))

    series = [
        ("scipy", "RK45",    "-",  "o", "SciPy RK45"),
        ("tudat", "rkf_45",  "-",  "o", "tudat RKF45"),
        ("scipy", "DOP853",  "--", "^", "SciPy DOP853"),
        ("tudat", "rkdp_87", "--", "^", "tudat RKDP87"),
    ]
    for lib, scheme, lsty, mk, lbl in series:
        x, y = pick(lib, scheme)
        if not len(x):
            continue
        o = np.argsort(x)
        ax.plot(x[o], y[o], lsty, marker=mk, color=LIB[lib], mfc="none", mew=1.5,
                label=lbl, zorder=3)

    # The two fixed-step RK4 runs land on the same point -- that agreement is the result -- so the
    # local marker is drawn larger and encircles the tudat one rather than hiding it.
    for lib, ms, lbl in (("local", 22, r"local RK4, $\Delta t$ = 64 s"),
                         ("tudat", 12, r"tudat RK4, $\Delta t$ = 64 s")):
        r = [x for x in rows if x["lib"] == lib and x["scheme"] == "rk4"
             and "64" in x["setting"]]
        if r:
            ax.plot([r[0]["evals"]], [r[0]["err"]], "*", ms=ms, color=LIB[lib],
                    mfc="none", mew=1.8, label=lbl, zorder=5 if lib == "tudat" else 4)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Gravity Evaluations")
    ax.set_ylabel("Position Error [m]")
    ax.set_ylim(2e-7, 2e0)
    ax.grid(True, alpha=.25, which="both")
    ax.legend(loc="upper right", ncol=2, columnspacing=1.2, handletextpad=0.5)

    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        f = os.path.join(a.out, f"library_comparison.{ext}")
        fig.savefig(f, dpi=200, bbox_inches="tight")
        print("wrote", f)

    print(f"\n{'lib':>7}{'scheme':>9}{'setting':>14}{'evals':>8}"
          f"{'total s':>9}{'grav s':>9}{'integ s':>9}{'integ %':>9}")
    for r in rows:
        print(f"{r['lib']:>7}{r['scheme']:>9}{r['setting']:>14}{r['evals']:8d}"
              f"{r['total_s']:9.2f}{r['grav_s']:9.2f}{r['integ_s']:9.3f}"
              f"{100*r['integ_s']/r['total_s']:9.1f}")


if __name__ == "__main__":
    main()

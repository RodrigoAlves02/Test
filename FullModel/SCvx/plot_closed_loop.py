#!/usr/bin/env python
"""Closed-loop descent: what guidance asked for, what the actuator delivered, where it landed."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "Figures&data")
INK, ACC, CMD, WARN, GRID = "#2B2B2B", "#1B5E9C", "#9A9A9A", "#B8351F", "#DCDCDC"


def main():
    d = json.load(open(os.path.join(OUT, "closed_loop.json")))
    tr = d["trace"]; cy = d["cycles"]
    t = np.array([r["t"] for r in tr]); alt = np.array([r["alt"] for r in tr])
    # REPAIR DOUBLE-COUNTED TIMESTAMPS from runs made before the closed_loop.py fix. There the
    # recorded time was t_start_of_cycle + 2*i*dt_act instead of + i*dt_act, so each cycle overshot
    # and then reset backward at the boundary. The true time is (t_cycle_start + recorded)/2. A
    # corrected run is strictly increasing and skips this untouched.
    jumps = np.where(np.diff(t) < 0)[0]
    if len(jumps):
        starts = np.concatenate([[0], jumps + 1])
        ends = np.concatenate([jumps + 1, [len(t)]])
        t = t.copy()
        for a0, b0 in zip(starts, ends):
            t[a0:b0] = 0.5 * (t[a0] + t[a0:b0])
        assert np.all(np.diff(t) >= -1e-9), "timestamps still not monotonic"
        print(f"  repaired {len(jumps)} backward jumps in the stored trace "
              f"(pre-fix run); t now spans {t[0]:.1f}..{t[-1]:.1f} s")
    Tc = np.array([r["T_cmd"] for r in tr]); Ta = np.array([r["T_act"] for r in tr])
    ac = np.array([r["ang_cmd"] for r in tr]); aa = np.array([r["ang_act"] for r in tr])
    ct = np.array([c["t"] for c in cy]); calt = np.array([c["alt"] for c in cy])
    # The solve for cycle i is made BEFORE that cycle is flown, so its t_go is measured from the
    # PREVIOUS cycle's timestamp. Pairing t_go with the post-flight t (as this first did) reads the
    # predicted landing one whole cycle too late, and reconstructing it as t - dt_guid is wrong for
    # a final cycle cut short by touchdown. Newer runs record t_solve directly.
    t_solve = np.array([c.get("t_solve", 0.0 if i == 0 else cy[i - 1]["t"])
                        for i, c in enumerate(cy)])
    cspd = np.array([c["speed"] for c in cy]); ctgo = np.array([c["t_go"] for c in cy])
    tL = d["t"]

    fig, ax = plt.subplots(2, 3, figsize=(15.0, 7.8))
    for a_ in ax.ravel():
        for x in ct:
            a_.axvline(x, color=GRID, lw=0.6, zorder=0)
        a_.axvline(tL, color=INK, lw=1.2, ls="--", zorder=1)
        a_.grid(alpha=0.22, lw=0.6); a_.set_xlabel("time [s]")

    ax[0, 0].plot(t, alt / 1e3, color=ACC, lw=2, zorder=3)
    ax[0, 0].set_ylabel("altitude [km]"); ax[0, 0].set_title("Altitude above the site", fontsize=10)
    ax[0, 0].text(0.03, 0.06, "thin lines: guidance cycles", transform=ax[0, 0].transAxes,
                  fontsize=7.5, color="#7A7A7A")

    ax[0, 1].plot(ct, cspd, "o-", color=ACC, lw=1.8, ms=4, zorder=3)
    ax[0, 1].axhline(C.VTOUCH, color=WARN, ls="--", lw=1.2)
    ax[0, 1].text(0.03, C.VTOUCH, f" {C.VTOUCH} m/s limit", color=WARN, fontsize=8, va="bottom")
    ax[0, 1].set_ylabel("speed [m/s]"); ax[0, 1].set_title("Speed (rotating frame)", fontsize=10)

    # t + t_go should be flat if the plan is self-consistent
    ax[0, 2].plot(t_solve, t_solve + ctgo, "o-", color=ACC, lw=1.8, ms=4, zorder=3,
                  label="predicted landing, $t_{solve} + t_{go}$")
    ax[0, 2].axhline(tL, color=WARN, ls="--", lw=1.2, label=f"actual touchdown {tL:.0f} s")
    ax[0, 2].set_ylabel("predicted touchdown time [s]")
    ax[0, 2].set_title("When does it think it will land?", fontsize=10)
    ax[0, 2].legend(fontsize=8, loc="lower left")
    _e = (t_solve + ctgo)[-2] - tL
    ax[0, 2].text(0.5, 0.92, f"converges to within {abs(_e):.1f} s of the truth",
                  transform=ax[0, 2].transAxes, ha="center", fontsize=8, color="#7A7A7A")

    ax[1, 0].step(t, Tc, where="post", color=CMD, lw=2.4, zorder=2, label="commanded (SCvx plan)")
    ax[1, 0].plot(t, Ta, color=ACC, lw=1.5, zorder=3,
                  label=f"delivered ($\\tau$={d['tau']:g} s lag)")
    for lv, lb in ((C.TMIN, "$T_{min}$"), (C.TMAX, "$T_{max}$")):
        ax[1, 0].axhline(lv, color=WARN, ls=":", lw=1.1)
        ax[1, 0].text(t[-1], lv, f" {lb}", color=WARN, fontsize=8, va="center", ha="right")
    ax[1, 0].set_ylabel("thrust [N]"); ax[1, 0].set_ylim(0, C.TMAX * 1.2)
    ax[1, 0].set_title("Thrust: commanded vs delivered", fontsize=10)
    ax[1, 0].legend(fontsize=8, loc="center left")

    ax[1, 1].step(t, ac, where="post", color=CMD, lw=2.4, zorder=2, label="commanded")
    ax[1, 1].plot(t, aa, color=ACC, lw=1.5, zorder=3,
                  label=f"delivered (slew {d['slew']:g}$^\\circ$/s)")
    ax[1, 1].set_ylabel("thrust angle from local vertical [deg]")
    ax[1, 1].set_title("Thrust pointing", fontsize=10); ax[1, 1].legend(fontsize=8)

    err = np.abs(Ta - Tc)
    ax[1, 2].semilogy(t, np.maximum(err, 1e-6), color=WARN, lw=1.4, zorder=3,
                      label=f"|T$_{{act}}$ - T$_{{cmd}}$|, max {err.max():.3f} N")
    ax[1, 2].set_ylabel("thrust tracking error [N]")
    ax[1, 2].set_title("Actuator tracking", fontsize=10); ax[1, 2].legend(fontsize=8)

    fig.suptitle(f"Closed-loop Eros descent, K={d['K']}, {d['dt_guid']:.0f} s guidance cycle  —  "
                 f"touchdown {tL:.1f} s, miss {d['miss']:.2f} m, {d['speed']:.3f} m/s, "
                 f"{d['fuel']:.3f} kg", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"closed_loop.{ext}"), dpi=300, bbox_inches="tight")
    print(f"landed {d['landed']}, {len(cy)} cycles, miss {d['miss']:.4f} m, "
          f"max thrust tracking error {err.max():.4f} N, max pointing error "
          f"{np.abs(aa-ac).max():.2f} deg")
    print("wrote " + os.path.join(OUT, "closed_loop.{pdf,png}"))


if __name__ == "__main__":
    main()

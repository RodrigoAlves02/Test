#!/usr/bin/env python
"""Figures for the SCvx verification run.

Reproduces Fig. 6 of arXiv:2001.03572 -- position, velocity and thrust acceleration against time --
and adds the three panels that decide the verdict: the thrust magnitude against its bounds with the
published switch time marked, the propellant against the published value, and a REFINEMENT STUDY.

The refinement panel is the one that matters. A correct discretisation converges to the reference
as K grows; a formulation error does not. With the thrust carried as a STATE (zero-order hold on
its rate, hence first-order hold on the thrust itself) the propellant error sat at 0.65 / 0.58 /
0.60 % for K = 30 / 60 / 120 and the intermediate-thrust arc GREW with K -- flat error and a
structure that refuses to sharpen is the signature of a representation that cannot express the
optimum, which here is bang-bang. With the thrust as a zero-order-held CONTROL the same sweep
converges.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark as B                                        # noqa: E402
import run_verification as RV                                # noqa: E402

OUT = os.path.join(HERE, "Figures&data")


def traj(r, sc):
    K = r.X.shape[1]
    t = np.linspace(0.0, sc.t_dim(r.s), K)
    m = np.exp(r.X[6, :])
    # THE THRUST IS Gamma*d: magnitude from the control U[3], direction from the state X[7:10].
    # U[0:3] is omega, the direction's ANGULAR RATE -- plotting it as an acceleration plots a
    # rotation rate on an m/s^2 axis.
    acc = r.U[3, :] * r.X[7:10, :] * sc.A
    # TWO angles, because a direction in 3D needs two. The paper's 0..200 deg panel holds both:
    # the cone angle off the vertical and the azimuth on the 0..360 branch, which here only reaches
    # 191 deg. One angle alone says nothing about which way the thrust is leaning.
    na = np.linalg.norm(acc, axis=0)
    u = acc / np.maximum(na, 1e-12)
    elev = np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0)))
    azim = np.mod(np.degrees(np.arctan2(u[1], u[0])), 360.0)
    ang = np.vstack([elev, azim])
    # The paper's Fig. 6 third panel is vdot -- thrust PLUS gravity, the derivative of the velocity
    # it plots beside it -- not the thrust acceleration alone. Its axis runs -4 to 4, ours ran -4 to
    # 7 for exactly that reason. |a| itself is right: 2.610 = Tmin/m0 and 7.679 = Tmax/m at the ends.
    vdot = acc + B.G_MARS[:, None]
    return t, r.X[0:3, :] * sc.L, r.X[3:6, :] * sc.v, m, vdot, na * m, ang


def main(case="min-max", Ks=(30, 60, 120)):
    cs = B.CASES[case]
    runs = {}
    for K in Ks:
        r, sc, _ = RV.run(case, K)
        runs[K] = (r, sc, traj(r, sc))
    Kmain = Ks[-1]
    r, sc, (t, pos, vel, m, acc, T, ang) = runs[Kmain]

    CP = "#444444"
    fig = plt.figure(figsize=(18.5, 8.6))
    gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.30, top=0.85)
    fig.suptitle(
        f"SCvx VERIFICATION  -  case '{case}', zero-order-held thrust, K = {Kmain}\n"
        f"benchmark: Johnston et al. arXiv:2001.03572 flying the Acikmese-Ploen lander;  "
        f"m_used = {cs['m_used']} kg, tf = {cs['tf']} s  (TFC and GPOPS-II agree to 6 s.f.)",
        fontsize=11.5, y=0.965)

    # AXES EXACTLY AS THE PAPER'S FIG. 6, read off its tick labels, so the two figures can be laid
    # side by side without rescaling: position -1000..1500 by 500, velocity -100..50 by 50,
    # acceleration -4..4 by 2, time 0..30 by 5. The acceleration panel is vdot, not thrust alone --
    # that is what makes -4..4 the right range.
    AX = B.AXES[case]
    tk = np.arange(AX["t"][0], AX["t"][1] + 1, AX["t"][2])
    def lims(key):
        lo, hi, st = AX[key]
        return dict(ylim=(lo, hi), yticks=np.arange(lo, hi + 0.1 * st, st))
    AXES = (lims("pos"), lims("vel"), lims("acc"))
    for j, (dat, name, unit, sym) in enumerate(
            ((pos, "position", "m", "r"), (vel, "velocity", "m/s", "v"),
             (acc, r"acceleration  $\dot{v}$ = thrust + gravity", "m/s$^2$", r"\dot{v}"))):
        ax = fig.add_subplot(gs[0, j])
        for i, c in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
            lbl = f"${sym}_{i+1}$"
            if j == 2:
                ax.step(t, dat[i], where="post", color=c, lw=1.7, label=lbl)
            else:
                ax.plot(t, dat[i], color=c, lw=1.7, label=lbl)
        ax.set_xlabel("time [s]"); ax.set_ylabel(f"{name.split('  ')[0]} [{unit}]")
        # the paper's TICKS, but never clip the trajectory: it runs to tf = 31.26 s, past 30
        ax.set_xlim(0, t[-1]); ax.set_xticks(tk)
        ax.set_ylim(*AXES[j]["ylim"]); ax.set_yticks(AXES[j]["yticks"])
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3, loc="best")
        ax.set_title(name, fontsize=9.5)

    ax = fig.add_subplot(gs[0, 3])
    ax.step(t, ang[0], where="post", color="#6a3d9a", lw=1.8,
            label=r"cone, from vertical  $\arccos(u_3)$")
    ax.step(t, ang[1], where="post", color="#e377c2", lw=1.8,
            label=r"azimuth  $\mathrm{atan2}(u_2,u_1)$")
    ax.set_xlabel("time [s]"); ax.set_ylabel("thrust direction [deg]")
    ax.set_xlim(0, t[-1]); ax.set_xticks(tk)
    ax.set_ylim(*lims("ang")["ylim"]); ax.set_yticks(lims("ang")["yticks"])
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="upper left")
    for sw in cs["switches"]:
        ax.axvline(sw, color=CP, ls=":", lw=1.4)
    ax.set_title("thrust direction: both angles", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    # THROTTLE, as the paper plots it: Tmin and Tmax ARE 0.3 and 0.8 of the cant-corrected stack
    # thrust, so dividing by T_REF puts the bang-bang between those two on a 0..1 axis.
    for b in (0.3, 0.8):
        ax.axhline(b, color="crimson", ls="--", lw=1.3)
    ax.annotate("0.8 = $T_{max}$", (t[-1], 0.8), (-4, 5), textcoords="offset points",
                fontsize=8, color="crimson", ha="right")
    ax.annotate("0.3 = $T_{min}$", (t[-1], 0.3), (-4, -14), textcoords="offset points",
                fontsize=8, color="crimson", ha="right")
    ax.step(t, T / B.T_REF, where="post", color="#1f4e79", lw=2.0, label=f"K = {Kmain}")
    for i, sw in enumerate(cs["switches"]):
        ax.axvline(sw, color=CP, ls=":", lw=1.6)
        ax.annotate(f"{sw:.4f} s", (sw, 0.62 - 0.12 * i), (-6, 0), ha="right",
                    textcoords="offset points", fontsize=8, color=CP)
    ax.annotate("published switches", (0.03, 0.93), xycoords="axes fraction",
                fontsize=8, color=CP)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"throttle  $\||T\|| / (T\,N_T \cos\phi)$")
    ax.set_xlim(0, t[-1]); ax.set_xticks(tk)
    ax.set_ylim(0, 1); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center right")
    ax.set_title("throttle: bang-bang, held across each segment", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    # the three K curves lie on top of each other -- that IS the result, so say so rather than
    # letting the top one hide the rest
    for K, ls in zip(Ks, ("-", "--", ":")):
        tt, _, _, mk, _, _, _ = runs[K][2]        # NOT tk: that holds the x ticks
        ax.plot(tt, B.M_WET - mk, lw=1.8, ls=ls, label=f"K = {K}")
    ax.axhline(cs["m_used"], color=CP, ls="-.", lw=1.6)
    ax.annotate(f"published {cs['m_used']} kg", (0.97, cs["m_used"]), (0, -14),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                fontsize=8.5, color=CP, ha="right")
    ax.set_xlabel("time [s]"); ax.set_ylabel("propellant used [kg]")
    ax.set_xlim(0, t[-1]); ax.set_xticks(tk)       # ours, not the paper's: it has no fuel panel
    ax.set_ylim(*lims("fuel")["ylim"]); ax.set_yticks(lims("fuel")["yticks"])
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("propellant against the published value", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    err = [100 * ((B.M_WET - float(np.exp(runs[K][0].X[6, -1]))) - cs["m_used"]) / cs["m_used"]
           for K in Ks]
    terr = [100 * (runs[K][1].t_dim(runs[K][0].s) - cs["tf"]) / cs["tf"] for K in Ks]
    ax.plot(Ks, err, "o-", color="#1f4e79", lw=1.8, ms=7, mfc="none", mew=1.8, label="propellant")
    ax.plot(Ks, terr, "s-", color="#c1570c", lw=1.8, ms=7, mfc="none", mew=1.8, label="flight time")
    # the first-order-hold-on-thrust result, measured before the change, for contrast
    ax.plot(Ks, [0.65, 0.58, 0.60][:len(Ks)], "^--", color="0.6", lw=1.5, ms=7, mfc="none",
            mew=1.5, label="propellant, thrust as a STATE")
    ax.axhline(0, color="k", lw=1.0)
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(Ks))); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("nodes K"); ax.set_ylabel("error vs the benchmark [%]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("REFINEMENT: does the error go to zero?", fontsize=10)

    ax = fig.add_subplot(gs[1, 3])
    for K in Ks:
        rk = runs[K][0]
        h = rk.history
        if h:
            d = [it.get("defect_m", np.nan) for it in h]
            ax.semilogy(range(1, len(d) + 1), d, "o-", ms=4, mfc="none", lw=1.4, label=f"K = {K}")
    ax.set_xlabel("PTR iteration"); ax.set_ylabel("max nonlinear defect [m]")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("defect (our analogue of their residual)", fontsize=10)

    os.makedirs(OUT, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"verification_{case}.{e}"), dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}/verification_{case}.png")
    print(f"{'K':>6} {'m_used':>10} {'err':>8} {'tf':>10} {'err':>8}")
    for K, e, te in zip(Ks, err, terr):
        used = B.M_WET - float(np.exp(runs[K][0].X[6, -1]))
        print(f"{K:6d} {used:10.4f} {e:+7.2f}% {runs[K][1].t_dim(runs[K][0].s):10.4f} {te:+7.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "min-max")

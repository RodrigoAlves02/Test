#!/usr/bin/env python
"""Reproduce Figure 21 of Malyuta et al. arXiv:2106.09125 with our solver.

Their Fig. 21 shows (a) the trajectory against the glideslope cone, (c) speed against vmax, and
(f) the thrust magnitude, where the minimum-thrust arc makes lossless convexification visible --
||Tc|| < rho_min is FEASIBLE for the relaxed problem, and the fact that it never happens is the
LCvx guarantee holding with equality. The remaining panels are the constraint margins and the
LCvx gap itself, which is the quantity that decides whether the relaxation was lossless.
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
import run_malyuta as RM                                     # noqa: E402

M, OUT = B.MAL, os.path.join(HERE, "Figures&data")
CL, CR, CG = "#1f4e79", "crimson", "#444444"


def main(K=151):
    # SOLVED ON THE MESH LADDER, not cold at K. Cold, this case converges only at K=38; K=76 runs to
    # maxiter and K=151 fails outright with the SOCP unbounded, because the flight time has to climb
    # from the 25.7 s seed to ~75 s and cannot do it fast enough once the mesh is fine. Warm-started
    # from the previous rung every mesh converges, and the flight-time error finally REFINES
    # (+0.82% -> +0.81% -> +0.67%). Plotting the finest rung is therefore plotting the best solution
    # we have, not the only one that happened to run.
    for Kr, r, sc in RM.solve_ladder(tuple(k for k in (38, 76, 151) if k <= K) or (K,)):
        pass
    if r.status != "converged":
        print(f"  WARNING: final rung K={Kr} status {r.status} -- plotting it anyway")
    a = RM.audit(r, sc)
    t, pos, vel = a["t"], a["pos"], a["vel"]

    fig = plt.figure(figsize=(18.5, 8.6))
    gs = fig.add_gridspec(2, 4, hspace=0.36, wspace=0.30, top=0.85)
    fig.suptitle(
        f"SCvx VERIFICATION, CONSTRAINED CASE  -  reproducing Fig. 21 of Malyuta et al. "
        f"arXiv:2106.09125 (K = {K})\n"
        f"glideslope {M['gamma_gs']}$^\\circ$, thrust tilt {M['gamma_p']}$^\\circ$, rotating frame "
        f"$\\omega$ = (3.5, 0, 2)$\\times10^{{-3}}$ deg/s;  published $t_f^*$ = {M['tf']} s",
        fontsize=11.5, y=0.965)

    # (a) trajectory against the glideslope cone, side view
    ax = fig.add_subplot(gs[0, 0])
    horiz = np.linalg.norm(pos[0:2], axis=0)
    ax.plot(horiz, pos[2], color=CL, lw=2.0, label="trajectory")
    hmax = horiz.max() * 1.05
    ax.plot([0, hmax], [0, hmax / np.tan(np.radians(M["gamma_gs"]))], color=CR, ls="--", lw=1.5,
            label=f"glideslope {M['gamma_gs']}$^\\circ$")
    ax.plot(horiz[0], pos[2][0], "o", color=CL, ms=7); ax.plot(0, 0, "x", color=CR, ms=10, mew=2)
    ax.set_xlabel("horizontal range [m]"); ax.set_ylabel("altitude [m]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title("(a) trajectory vs the cone", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    for i, c in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        ax.plot(t, pos[i], color=c, lw=1.7, label=f"$r_{i+1}$")
    ax.set_xlabel("time [s]"); ax.set_ylabel("position [m]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3); ax.set_title("(b) position", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, a["sp"], color=CL, lw=2.0)
    ax.axhline(M["v_max"], color=CR, ls="--", lw=1.5)
    ax.annotate(f"$v_{{max}}$ = {M['v_max']:.1f} m/s", (t[-1], M["v_max"]), (-5, -14),
                textcoords="offset points", fontsize=8, color=CR, ha="right")
    ax.set_ylim(0, M["v_max"] * 1.15)
    ax.set_xlabel("time [s]"); ax.set_ylabel("speed [m/s]")
    ax.grid(alpha=0.3); ax.set_title("(c) speed vs $v_{max}$ (never active)", fontsize=10)

    ax = fig.add_subplot(gs[0, 3])
    for i, c in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        ax.plot(t, vel[i], color=c, lw=1.7, label=f"$v_{i+1}$")
    ax.set_xlabel("time [s]"); ax.set_ylabel("velocity [m/s]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3); ax.set_title("(d) velocity", fontsize=10)

    # (f) the panel that shows LCvx is lossless
    ax = fig.add_subplot(gs[1, 0])
    for b, lb in ((M["rho_min"], r"$\rho_{min}$"), (M["rho_max"], r"$\rho_{max}$")):
        ax.axhline(b, color=CR, ls="--", lw=1.4)
        ax.annotate(lb, (t[-1], b), (-5, 5), textcoords="offset points",
                    fontsize=8, color=CR, ha="right")
    ax.step(t, a["nT"], where="post", color=CL, lw=2.0)
    ax.set_xlabel("time [s]"); ax.set_ylabel(r"$\|T_c\|$ [N]")
    ax.set_ylim(M["rho_min"] * 0.9, M["rho_max"] * 1.05)
    ax.grid(alpha=0.3); ax.set_title("(f) thrust: the min-thrust arc", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    ax.step(t, a["tilt"], where="post", color=CL, lw=2.0)
    ax.axhline(M["gamma_p"], color=CR, ls="--", lw=1.5)
    ax.annotate(f"$\\gamma_p$ = {M['gamma_p']}$^\\circ$", (t[-1], M["gamma_p"]), (-5, 5),
                textcoords="offset points", fontsize=8, color=CR, ha="right")
    ax.set_ylim(0, M["gamma_p"] * 1.3)
    ax.set_xlabel("time [s]"); ax.set_ylabel("thrust tilt from vertical [deg]")
    ax.grid(alpha=0.3); ax.set_title("(e) pointing constraint: ACTIVE", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    # The cone's APEX sits at the achieved landing point, which this solution reaches to 1e-11 m, so
    # the cone ANGLE there is 0/0 -- read as an angle from the origin over all nodes it returned
    # 90.39 deg against an 86 deg limit and looked like a violation. The constraint residual is the
    # quantity the constraint actually states, it stays finite at the apex, and it says the cone is
    # satisfied and tight: max -3.9e-03 m. audit() returns both, already restricted to the nodes the
    # cone is imposed on (k = 0..K-2).
    ng = a["gs"].size
    ax.plot(t[:ng], a["gs_res_m"], color=CL, lw=2.0)
    ax.axhline(0.0, color=CR, ls="--", lw=1.5)
    ax.annotate(f"constraint boundary ({M['gamma_gs']}$^\\circ$)", (t[0], 0.0), (5, -14),
                textcoords="offset points", fontsize=8, color=CR)
    ax.set_xlabel("time [s]"); ax.set_ylabel("glideslope residual [m]")
    ax.grid(alpha=0.3)
    ax.set_title(f"(g) glideslope: max {a['gs_res_m'].max():+.1e} m "
                 f"(angle {a['gs'].max():.2f}$^\\circ$/{M['gamma_gs']:.0f}$^\\circ$)", fontsize=10)

    ax = fig.add_subplot(gs[1, 3])
    ax.semilogy(t, np.maximum(np.abs(a["sig"] - a["nT"]), 1e-16), color=CL, lw=1.8)
    ax.set_xlabel("time [s]"); ax.set_ylabel(r"$|\Gamma - \|\Gamma d\||$ [N]")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"(h) charge vs delivery: max {a['lcvx']:.1e} N of "
                 f"{M['rho_max']:.0f}", fontsize=10)

    os.makedirs(OUT, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"verification_malyuta.{e}"), dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}/verification_malyuta.png")
    print(f"  tf {a['tf']:.4f} s (published {M['tf']}, {100*(a['tf']-M['tf'])/M['tf']:+.2f}%)   "
          f"propellant {a['used']:.3f} kg")
    print(f"  tilt max {a['tilt'].max():.4f} / {M['gamma_p']}   "
          f"glideslope max {a['gs'].max():.3f} / {M['gamma_gs']} "
          f"(residual {a['gs_res_m'].max():+.2e} m)   "
          f"speed max {a['sp'].max():.2f} / {M['v_max']:.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 151)

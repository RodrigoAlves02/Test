#!/usr/bin/env python
"""How wrong is the onboard PINN gravity, and where?

WHY THIS EXISTS.  The closed-loop Monte Carlo (job 865910, 330 cases) found a systematic altitude
error in the navigation solution: the filter believes it is -0.43 +/- 0.47 m LOWER than it really
is at contact, and the ~5% of runs whose error reaches about -1.2 m stop their descent above the
surface and fly back up, because minimum thrust is 1.66x weight and the vehicle cannot hold a
hover.  The lidar and camera innovations are persistently POSITIVE below 500 m (+0.24 to +0.62 m,
"you are higher than you think"), so the sensors are not the liars -- they are fighting a drift.
The EKF propagates on PINN gravity while the truth runs on the 49k polyhedral, which makes the
onboard gravity model the suspect.  This script measures it.

WHAT IS MEASURED, and why two metrics.
  rel   = |g_pinn - g_poly| / |g_poly|        the usual PINN-GM figure of merit
  dg_v  = n . (g_pinn - g_poly)               the SIGNED component along the local vertical
The second is the one that matters here: a relative error that averages out in direction is
harmless to an altitude estimate, while a small persistent vertical component integrates twice
into a position bias.  Reporting only `rel` would hide the mechanism.

The sampling answers three separate questions:
  A  along the descent corridor   -- what the vehicle actually flies through
  B  laterally at fixed altitude  -- is the error a property of this site or of this altitude?
  C  over the whole body          -- is the near-field degradation global?
  D  integrated                   -- does the measured error actually account for the -0.43 m?
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.join(HERE, "..", "SCvx")
sys.path.insert(0, SCVX)
OUT = os.path.join(HERE, "Figures&data")


def main():
    import gravity as GR
    import config as C
    poly, pinn = GR.field(), GR.pinn_field()
    # the checkpoint this replaced, kept as the comparison curve: same architecture and same loss
    # (Eq 12 unmodified), trained on radii 0-10R instead of the paper's Appendix-B 0-3R
    prev = GR.PinnField(npz_path=os.path.join(C.THESIS, "scvx_landing", "data",
                                              "pinn_eros_np.npz"))
    r_site, n_hat = GR.site_and_normal()
    east = np.cross([0.0, 0.0, 1.0], n_hat); east /= np.linalg.norm(east)
    north = np.cross(n_hat, east)
    X_STAR = 17629.185527976042          # the PINN's own length normalisation, from the npz

    def err(P, model=None):
        """rel error and signed vertical error at an (N,3) array of points."""
        f = model or pinn
        gp = np.array([poly.g(p) for p in P])
        gn = np.array([f.g(p) for p in P])
        rel = np.linalg.norm(gn - gp, axis=1) / np.linalg.norm(gp, axis=1)
        return rel, (gn - gp) @ n_hat, np.linalg.norm(gp, axis=1)

    t0 = time.perf_counter()

    # -- A: the descent corridor, straight up from the site -------------------------------------
    hA = np.unique(np.concatenate([np.linspace(1, 100, 60), np.logspace(2, np.log10(12000), 240)]))
    corridor = np.array([r_site + n_hat * h for h in hA])
    relA, dgA, _ = err(corridor)
    relA0, dgA0, _ = err(corridor, prev)

    # -- B: laterally, at the altitudes that matter ---------------------------------------------
    HB = [100.0, 300.0, 1000.0, 3000.0]
    gridB = {}
    q = np.linspace(-2000.0, 2000.0, 41)
    E, N = np.meshgrid(q, q)
    for h in HB:
        P = (r_site + n_hat * h)[None, :] + E.ravel()[:, None] * east + N.ravel()[:, None] * north
        r, d, _ = err(P)
        gridB[h] = (r.reshape(E.shape), d.reshape(E.shape))

    # -- C: is the near-field degradation global? ------------------------------------------------
    # Facet centroids give surface points with an outward normal already defined by the mesh, so
    # "altitude above local terrain" is well posed everywhere, not just over the landing site.
    v, f = GR._mesh(GR.C.MESH_TAB)
    tri = v[f]
    cen = tri.mean(axis=1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    nrm *= np.sign((nrm * cen).sum(1))[:, None]          # force outward
    rng = np.random.default_rng(7)
    pick = rng.choice(len(cen), size=250, replace=False)
    HC = np.array([10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
    relC = np.zeros((len(HC), len(pick)))
    relC0 = np.zeros_like(relC)
    for i, h in enumerate(HC):
        relC[i], _, _ = err(cen[pick] + nrm[pick] * h)
        relC0[i], _, _ = err(cen[pick] + nrm[pick] * h, prev)

    # -- D: does it account for the observed -0.43 m? --------------------------------------------
    # Integrate the vertical gravity error twice along a real descent. This is the drift the filter
    # would take if NOTHING corrected it, i.e. the forcing the sensors have to fight.
    import json
    mcf = os.path.join(SCVX, "Figures&data", "mc", "closed_loop_mc_p0_v0_s901.json")
    tD = altD = driftD = None
    if os.path.exists(mcf):
        d = json.load(open(mcf))
        tr = d["trace"]
        tD = np.array([p["t"] for p in tr])
        R = np.array([p["r_true"] for p in tr])
        altD = (R - r_site) @ n_hat
        _, dgD, _ = err(R)
        _, dgD0, _ = err(R, prev)
        def _drift(dg):
            v = np.concatenate([[0.0], np.cumsum(0.5 * (dg[1:] + dg[:-1]) * np.diff(tD))])
            return np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tD))])
        driftD, driftD0 = _drift(dgD), _drift(dgD0)

    print(f"sampled {len(hA) + sum(g[0].size for g in gridB.values()) + relC.size + (0 if tD is None else len(tD))}"
          f" points in {time.perf_counter() - t0:.1f} s\n")
    print("A  descent corridor above the landing site       REPLACED (0-10R) -> DEPLOYED (0-3R)")
    print(f"   {'altitude':>12} {'rel err':>19} {'dg_vert [m/s2]':>30}")
    for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, 1000),
                   (1000, 2000), (2000, 5000), (5000, 12001)):
        m = (hA >= lo) & (hA < hi)
        if m.sum():
            print(f"   {str(lo)+'-'+str(hi)+' m':>12} {100*relA0[m].mean():8.3f}% ->{100*relA[m].mean():8.3f}%"
                  f"   {dgA0[m].mean():+13.3e} ->{dgA[m].mean():+13.3e}")
    print("\nC  the same altitudes, 250 sites over the whole body (median rel err)")
    print(f"   {'altitude':>12} {'replaced':>10} {'deployed':>10}")
    for i, h in enumerate(HC):
        print(f"   {str(int(h))+' m':>12} {100*np.median(relC0[i]):9.3f}% {100*np.median(relC[i]):9.3f}%")
    if driftD is not None:
        k = np.argmin(np.abs(altD - 2.0))
        print(f"\nD  unopposed drift from the gravity error alone, by 2 m altitude:"
              f"   replaced {driftD0[k]:+.3f} m   ->   deployed {driftD[k]:+.3f} m")
        print(f"   the replaced model produced a -0.429 m median nav altitude bias at contact")

    # ============================== figure ======================================================
    fig = plt.figure(figsize=(14.5, 9.0))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.34, top=0.86)
    fig.suptitle(
        "Onboard PINN gravity against the 49k-facet polyhedral truth\n"
        "retraining on the paper's Appendix-B 0-3R range instead of 0-10R, same loss and "
        "architecture, removes the near-field error that biased the navigation",
        fontsize=12, y=0.965)
    C_COR, C_LAT, C_GLB = "#1f4e79", "#c1570c", "#2f7a3e"

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hA, 100 * relA0, color="0.62", lw=1.6, ls="--", label="0-10R training (replaced)")
    ax.plot(hA, 100 * relA, color=C_COR, lw=1.9, label="0-3R training (deployed)")
    lo = [100 * np.percentile(gridB[h][0], 5) for h in HB]
    hi = [100 * np.percentile(gridB[h][0], 95) for h in HB]
    ax.fill_between(HB, lo, hi, color=C_LAT, alpha=0.22, lw=0,
                    label="5-95% over a 4 km box at that altitude")
    ax.plot(HC, 100 * np.median(relC, axis=1), "o-", color=C_GLB, lw=1.5, ms=6, mfc="none",
            mew=1.6, label="deployed, median over 250 sites")
    ax.plot(HC, 100 * np.median(relC0, axis=1), "s--", color="0.62", lw=1.3, ms=5, mfc="none",
            mew=1.3, label="replaced, same 250 sites")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("altitude above local terrain [m]"); ax.set_ylabel("|dg| / |g|  [%]")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5, loc="lower center")
    ax.set_title("relative acceleration error", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color="0.4", lw=1.0)
    ax.plot(hA, dgA0, color="0.62", lw=1.6, ls="--", label="replaced")
    ax.plot(hA, dgA, color=C_COR, lw=1.9, label="deployed")
    ax.legend(fontsize=8, loc="lower right")
    ax.axvspan(30, 500, color="crimson", alpha=0.10, lw=0)
    ax.annotate("the filter's blind descent:\nlidar dark below 30 m,\nbias builds through here",
                (0.04, 0.06), xycoords="axes fraction", fontsize=8, color="crimson", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("altitude above the site [m]")
    ax.set_ylabel("vertical dg  [m/s^2]")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("VERTICAL component, signed\nnegative = the PINN pulls down too hard",
                 fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.pcolormesh(q / 1e3, q / 1e3, 100 * gridB[300.0][0], cmap="magma", shading="auto")
    ax.plot(0, 0, "x", ms=11, mew=2.2, color="white")
    ax.annotate("site", (0, 0), (7, 5), textcoords="offset points", color="white", fontsize=8.5)
    ax.set_aspect("equal"); ax.set_xlabel("east [km]"); ax.set_ylabel("north [km]")
    fig.colorbar(im, ax=ax, label="|dg| / |g|  [%]")
    ax.set_title("300 m altitude, 4 km box\nsmooth, large-scale - not terrain roughness",
                 fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    for i, h in enumerate(HC):
        ax.scatter(np.full(relC.shape[1], h) * np.exp(np.linspace(-0.12, 0.12, relC.shape[1])),
                   100 * relC[i], s=5, color=C_GLB, alpha=0.30, lw=0)
    ax.plot(HC, 100 * np.median(relC0, axis=1), "s--", color="0.62", lw=1.5, ms=6, mfc="none",
            mew=1.5, label="replaced (0-10R)")
    ax.plot(HC, 100 * np.median(relC, axis=1), "o-", color="black", lw=1.8, ms=7, mfc="none",
            mew=1.8, label="deployed (0-3R)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("altitude above local terrain [m]"); ax.set_ylabel("|dg| / |g|  [%]")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("250 sites over the whole body - the degradation is global", fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    rnd = np.linalg.norm(np.array([r_site + n_hat * h for h in hA]), axis=1) / X_STAR
    ax.plot(rnd, 100 * relA, color=C_COR, lw=1.8)
    r0 = float(np.linalg.norm(r_site)) / X_STAR
    ax.axvspan(r0, (float(np.linalg.norm(r_site)) + 1000.0) / X_STAR, color="crimson",
               alpha=0.12, lw=0)
    ax.annotate("the whole last 1 km\nlives in this strip", (r0, 1.1), (14, 0),
                textcoords="offset points", fontsize=8, color="crimson", va="center")
    ax.set_xlabel("radius / x_star  (the PINN's own normalisation)")
    ax.set_ylabel("|dg| / |g|  [%]"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_title("against the network's input scale\n(x_star = 17629 m; the site sits at 0.37)",
                 fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    if driftD is not None:
        ax.plot(altD, driftD0, color="0.62", lw=1.6, ls="--", label="replaced (0-10R)")
        ax.plot(altD, driftD, color=C_COR, lw=1.9, label="deployed (0-3R)")
        ax.axhline(-0.429, color="crimson", ls=":", lw=1.5,
                   label="nav bias the replaced model produced")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("altitude [m]  (descending to the right)")
        ax.set_ylabel("altitude error injected [m]")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("does the gravity error account for the bias?", fontsize=10)

    os.makedirs(OUT, exist_ok=True)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"pinn_gravity_error.{e}"), dpi=140, bbox_inches="tight")
    np.savez(os.path.join(OUT, "pinn_gravity_error.npz"),
             h_corridor=hA, rel_corridor=relA, dgv_corridor=dgA,
             h_global=HC, rel_global=relC, rel_global_prev=relC0,
             rel_corridor_prev=relA0, dgv_corridor_prev=dgA0, lat_q=q,
             rel_300=gridB[300.0][0], dgv_300=gridB[300.0][1])
    print(f"\nwrote {OUT}/pinn_gravity_error.png + .pdf + .npz")


if __name__ == "__main__":
    main()

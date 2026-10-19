#!/usr/bin/env python
"""The PTR loop as it actually steps: where it is, how far it may go, where it went, and why.

ONE PICTURE OF THE WHOLE IDEA. The flight time is used as the axis because it is a single number
that the algorithm genuinely has to search over, and because the trust region on it (Rs) is an
explicit bound the subproblem carries -- so "how far am I allowed to move" can be drawn to scale
rather than sketched.

  (a) per iteration: the REFERENCE flight time, the BAND the trust region allows around it, and
      the point the subproblem CHOSE inside that band. Green = the ratio test accepted it and the
      reference moves there. Red = rejected, the reference stays put and the band shrinks.
  (b) the two radii as staircases. Every shrink is a rejection, every growth is a step the linear
      model predicted well. This is the whole update rule, visible.

Run on the real handover case, not a toy.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Figures&data")
V1 = os.path.abspath(os.path.join(HERE, "..", "SCvx", "_v1_zoh_backup"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "SCvx")))
sys.path.insert(0, V1)
import config as C
C.POINTING_DEG = 180.0
C.LAT_DEG, C.LON_DEG = -36.975, 95.401
import gravity as GR, ptr
from dynamics import Dynamics

r_site, n_hat = GR.site_and_normal(lat_deg=C.LAT_DEG, lon_deg=C.LON_DEG)
east = np.cross([0, 0, 1.0], n_hat); east /= np.linalg.norm(east)
north = np.cross(n_hat, east)
r0 = r_site + 11771.0 * n_hat
vl = np.asarray(C.V0_LOCAL, float)
v0 = vl[0] * n_hat + vl[1] * east + vl[2] * north
sc = C.Scales.build(r0, r_site, m_wet=C.MWET)
dyn = Dynamics(sc, field=GR.pinn_field())
x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(C.MWET)]])
cm = dict(dyn=dyn, scales=sc, x_init=x_init, rf_nd=r_site / sc.L, n_hat=n_hat, K=100,
          verbose=False, iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4)

r1 = ptr.run(mode="error", warm=None, **cm)
r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.001 / sc.L, warm=(r1.X, r1.U, r1.s),
             ratio_test=True, R_bounds=(1e-10, 1.0), **cm)
h = r2.history
print(f"stage 2: {r2.status} in {r2.iters} it, {sum(1 for e in h if not e['accepted'])} rejected")

T = sc.T
fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                       gridspec_kw=dict(height_ratios=[2.0, 1.0]))
for e in h:
    i, ref, cho = e["it"], e["tf_ref"], e["tf_s"]
    band = e["Rs"] * T
    ok = e["accepted"]
    col = "#2e7d32" if ok else "#b00020"
    ax[0].add_patch(plt.Rectangle((i - 0.33, ref - band), 0.66, 2 * band,
                                  facecolor="#9fb8d4", edgecolor="none", alpha=0.45, zorder=1))
    ax[0].plot([i], [ref], "_", ms=14, mew=2.0, color="#1f5fa8", zorder=3)
    ax[0].annotate("", xy=(i, cho), xytext=(i, ref), zorder=4,
                   arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
    ax[0].plot([i], [cho], "o" if ok else "X", ms=5.5 if ok else 7,
               color=col, zorder=5)
ax[0].plot([], [], "_", ms=12, mew=2, color="#1f5fa8", label="reference $t_f$ this iteration")
ax[0].add_patch(plt.Rectangle((0, 0), 0, 0, facecolor="#9fb8d4", alpha=0.45,
                              label=r"band the trust region allows, $\pm R_s$"))
ax[0].plot([], [], "o", ms=5.5, color="#2e7d32", label="chosen, ACCEPTED: reference moves here")
ax[0].plot([], [], "X", ms=7, color="#b00020", label="chosen, REJECTED: reference stays, band shrinks")
ax[0].set_ylabel("flight time $t_f$ [s]", fontsize=9)
ax[0].legend(fontsize=7.0, frameon=False, loc="upper right")

ax[1].step([e["it"] for e in h], [e["R"] for e in h], where="mid", lw=1.4,
           color="#1f5fa8", label="$R$  (state radius)")
ax[1].step([e["it"] for e in h], [e["Rs"] for e in h], where="mid", lw=1.4,
           color="#c2541b", label="$R_s$  (flight-time radius)")
for e in h:
    if not e["accepted"]:
        ax[1].axvline(e["it"], color="#b00020", lw=0.7, alpha=0.35, zorder=0)
ax[1].set_yscale("log")
ax[1].set_xlabel("PTR iteration", fontsize=9)
ax[1].set_ylabel("trust radius", fontsize=9)
ax[1].legend(fontsize=7.0, frameon=False, loc="best")
for a in ax:
    a.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.095, right=0.985, bottom=0.10, top=0.985, hspace=0.08)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"trust_region.{ext}"), dpi=200)
print(f"wrote {OUT}/trust_region.{{pdf,png}}")

#!/usr/bin/env python
"""Figure 7.2 -- the effect of the gravity model on a descent.

ONE TRAJECTORY, REPLAYED UNDER EACH FIELD WITH THE CONTROL HISTORY FROZEN. The plan is solved
once; every curve here is the SAME (X_0, U, s) propagated through the SAME ten-state dynamics and
the SAME integrator, so the only difference between the runs is the gravity model and the
divergence is attributable to it alone.

NO TERMINAL EVENT, DELIBERATELY. This is a model-attribution test, not a flight: the comparison
domain must be a fixed time span common to every field. Stopping each run at its own touchdown
would make the domain model-dependent and the curves would no longer be comparable. The
Werner-Scheeres potential is valid and continuous across the surface, so the final seconds carry
no singularity to avoid.

Panel (a) is the instantaneous acceleration error, the model error itself. Panel (b) is the
position error it integrates into, which is what the guidance has to absorb.
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TH = "/Users/rodrigoalves/Desktop/Thesis"
sys.path.insert(0, os.path.join(TH, "FullModel", "SCvx"))
HERE = os.path.dirname(os.path.abspath(__file__))
import gravity as GR, problem_setup as PS
from dynamics import Dynamics
sys.path.insert(0, HERE)
import sph_harm as SH

# THE OLD-SITE NOMINAL TRAJECTORY, the one the label set was generated around. config.py already
# points at this site, so nothing is overridden here.
z = np.load(os.path.join(TH, "FullModel", "SCvx", "Figures&data", "pinn", "nominal.npz"),
            allow_pickle=True)
X, U, s = z["X"], z["U"], float(z["s"])
K = X.shape[1]
sc = PS.build(r0_override=z["r0"], v0_override=z["v0"], m_wet=float(z["m0"]))["scales"]
tf = sc.t_dim(s)
N_SUB = 20

POL = GR.field()
# THE REFERENCE IS A FINER MESH THAN THE ONE THAT FLIES. Measuring ver64q against itself reports
# zero and says nothing; the question is how far every model in use -- the flight mesh included --
# departs from the best shape available. ver256q has sixteen times the facets.
#
# THE TWO MESHES CARRY THE SAME TOTAL MASS. A finer mesh stops cutting corners off the surface and
# so encloses slightly more volume (2506.39 against 2507.38 km^3), which at fixed density would be
# a different mass and would put a constant offset on every curve that has nothing to do with
# resolution. The reference density is therefore set so that rho*V matches the mass ver64q carries
# at its calibrated 2670 kg/m^3.
import trimesh as _tm
def _vol(path):
    _v, _f = GR._mesh(path)
    return float(_tm.Trimesh(vertices=_v, faces=_f, process=False).volume)
import config as _C
_MESH256 = os.path.join(os.path.dirname(_C.MESH_TAB), "ver256q.tab")
_MASS = _C.BULK_DENSITY * _vol(_C.MESH_TAB)
REF = GR.field(_MESH256, _MASS / _vol(_MESH256))
print(f"reference ver256q, mass matched to ver64q at {_C.BULK_DENSITY} kg/m^3", flush=True)

GM_MESH, RREF = 4.466476e5, 16.0e3
# THE HARMONICS ARE FITTED TO THE POLYHEDRAL FIELD ITSELF, so every model here describes ONE mass
# distribution and the comparison is about the representation alone. Fitted over several radii in
# the exterior, then evaluated along a descent that lies INSIDE the circumscribing sphere.
print("fitting spherical harmonics to the polyhedral field ...", flush=True)
SHF = {N: SH.fit(REF, N, RREF, GM_MESH, radii_km=(18.0, 22.0, 30.0, 45.0), n_dir=160)
       for N in (2, 4, 8)}
_MESH128 = os.path.join(os.path.dirname(_C.MESH_TAB), "ver128q.tab")
POL128 = GR.field(_MESH128, _MASS / _vol(_MESH128))
FIELDS = [("Polyhedral, 49 152 facets", POL, "#1d3557", "-"),
          ("Polyhedral, 196 608 facets", POL128, "#1d3557", (0, (4, 2))),
          ("Harmonics, $N=2$",  SHF[2], "#9ecae1", "-"),
          ("Harmonics, $N=4$",  SHF[4], "#4292c6", "-"),
          ("Harmonics, $N=8$",  SHF[8], "#08519c", "-"),
          ("PINN-GM-III",       GR.pinn_field(), "#d94801", "-")]


def replay(field):
    dyn = Dynamics(sc, field=field)
    h = s / (K - 1) / N_SUB
    x = X[:, 0].copy()
    out = [x.copy()]
    for k in range(K - 1):
        for _ in range(N_SUB):
            x = dyn.rk4(x, U[:, k], h)
            out.append(x.copy())
    return np.array(out)


traj, accel = {}, {}
for name, fld, _c, _ls in FIELDS:
    traj[name] = replay(fld)
    print(f"replayed {name}", flush=True)

# THE REFERENCE TRAJECTORY IS THE REFERENCE FIELD'S, not the flight mesh's. Both panels must
# measure against the same model or the figure compares two different things: the acceleration
# error against ver256q and the position error against ver64q.
traj["__ref__"] = replay(REF)
print("replayed the reference field", flush=True)
t = np.arange(traj[FIELDS[0][0]].shape[0]) * (tf / ((K - 1) * N_SUB))
ref = traj["__ref__"]

# the model error itself, evaluated ALONG THE REFERENCE PATH so it is a property of the field and
# not of the trajectory each field happens to fly
step = 10
idx = np.arange(0, ref.shape[0], step)
pos = ref[idx, 0:3] * sc.L
g_ref = np.array([REF.g(p) for p in pos])
gn = np.linalg.norm(g_ref, axis=1)
for name, fld, _c, _ls in FIELDS:
    g_oth = np.array([fld.g(p) for p in pos])
    accel[name] = 100.0 * np.linalg.norm(g_oth - g_ref, axis=1) / gn
print("acceleration error evaluated along the reference path", flush=True)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})


def save(fig, stem):
    # FIXED MARGINS, NOT A TIGHT BOX. bbox_inches="tight" crops each figure to its own content, so
    # a panel with a longer y-label comes out physically wider and is then scaled differently when
    # the two are placed side by side at the same width in LaTeX. Identical figsize and identical
    # axes margins make the panels the same size on the page.
    fig.subplots_adjust(left=0.215, right=0.965, bottom=0.185, top=0.965)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"{stem}.{ext}"), dpi=300)
    print("wrote", stem, flush=True)


R_BRILL = 17.629e3
rmag = np.linalg.norm(ref[:, 0:3], axis=1) * sc.L
cross = np.where(rmag < R_BRILL)[0]
t_cross = t[cross[0]] if cross.size else None

# ---- model error along the descent, as a fraction of the local gravity
step = 10
idx = np.arange(0, ref.shape[0], step)
pos = ref[idx, 0:3] * sc.L
g_ref = np.array([REF.g(p) for p in pos])
gn = np.linalg.norm(g_ref, axis=1)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for name, fld, c, ls in FIELDS:
    g_oth = np.array([np.asarray(fld.g(p), float) for p in pos])
    e = 100.0 * np.linalg.norm(g_oth - g_ref, axis=1) / gn
    ax.semilogy(t[idx], e, color=c, ls=ls, lw=1.4, label=name)
    print(f"{name}: {e[0]:.3g} % at handover -> {e[-1]:.3g} % at touchdown", flush=True)
if t_cross is not None:
    ax.axvline(t_cross, color="0.35", lw=0.9, ls="--")
    ax.text(t_cross + 25, 1.2e4, "circumscribing sphere", rotation=90, ha="left", va="top",
            fontsize=7, color="0.35")
ax.set_xlabel("time  [s]")
ax.set_ylabel("relative acceleration error   [%]")
ax.set_xlim(0, tf)
ax.grid(alpha=.25, lw=.5)
save(fig, "fig_gravity_accel"); plt.close(fig)

# ---- the trajectory error it produces
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for name, fld, c, ls in FIELDS:
    dr = np.linalg.norm(traj[name][:, 0:3] - ref[:, 0:3], axis=1) * sc.L
    ax.semilogy(t, np.maximum(dr, 1e-6), color=c, ls=ls, lw=1.4, label=name)
    print(f"{name}: |dr| final {dr[-1]:.4g} m", flush=True)
if t_cross is not None:
    ax.axvline(t_cross, color="0.35", lw=0.9, ls="--")
ax.set_xlabel("time  [s]")
ax.set_ylabel(r"$\|\Delta \mathbf{r}\|$   [m]")
ax.set_xlim(0, tf)
ax.set_ylim(1e-4, 1e5)
ax.grid(alpha=.25, lw=.5)
# ONE LEGEND FOR THE PAIR, and it goes here: the curves rise to the top right of this panel, so the
# bottom right is clear, whereas the acceleration panel has no comparable gap.
ax.legend(frameon=False, loc="upper left", fontsize=6.4, labelspacing=0.25,
          borderpad=0.2, handlelength=1.8)
save(fig, "fig_gravity_pos"); plt.close(fig)

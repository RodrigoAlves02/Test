#!/usr/bin/env python
"""Figure 7.2 -- what the choice of gravity model costs a descent.

ONE TRAJECTORY, THREE FIELDS. The SCvx solution is computed once and then REPLAYED under each
gravity model with the control history frozen. Nothing else differs between the runs, so the
divergence between them is attributable to the field alone.

THE REPLAY USES THE TEN-STATE DYNAMICS, not a reconstructed thrust history. The direction d is a
STATE obeying d-dot = omega x d, so with omega held zero-order over a segment the thrust direction
ROTATES CONTINUOUSLY across it -- first-order-hold behaviour on the thrust vector, obtained
structurally (see dynamics.py). Flattening the plan into a piecewise-constant thrust vector would
therefore not be the same trajectory. Propagating the ten-state system with the ZOH control (omega,
Gamma) reproduces the planner's continuous dynamics exactly, and it also gets the magnitude right:
what is held constant over a segment is the thrust ACCELERATION Gamma*A, not the force in newtons,
which falls with the mass.

THE PROPAGATION IS CONTINUOUS, not per segment. dynamics.defect restarts at every node, which is
what makes it a linearisation diagnostic; here the state must be carried through all K-1 segments
without restarting, or the error has no chance to accumulate and the figure shows nothing.

Stage 1 is cached in data/traj_7_2.npz so the plotting can be iterated without re-solving.
"""
import os, sys, time, argparse
import numpy as np

TH = "/Users/rodrigoalves/Desktop/Thesis"
SCVX = os.path.join(TH, "FullModel", "SCvx")
sys.path.insert(0, SCVX)
HERE = os.path.dirname(os.path.abspath(__file__))

import config as C
import gravity as GR
import problem_setup as PS
import ptr
from dynamics import Dynamics

SITE = (-0.1993, 176.1639)          # the site selected in Section 7.1
CACHE = os.path.join(HERE, "data", "traj_7_2.npz")

ap = argparse.ArgumentParser()
ap.add_argument("--resolve", action="store_true", help="re-run the SCvx solve")
ap.add_argument("--n-sub", type=int, default=100, help="RK4 sub-steps per segment in the replay")
ap.add_argument("--K", type=int, default=100)
a = ap.parse_args()

# THE SITE IS OVERRIDDEN IN PROCESS, not by editing config.py. site_and_normal binds C.LAT_DEG and
# C.LON_DEG as DEFAULT ARGUMENTS, which are evaluated at import, so reassigning them on the module
# would not take effect; wrapping the function is what actually moves the site, and it leaves the
# solver's own configuration untouched.
_orig_site = GR.site_and_normal
GR.site_and_normal = lambda *_a, **_k: _orig_site(lat_deg=SITE[0], lon_deg=SITE[1],
                                                  **{k: v for k, v in _k.items()
                                                     if k not in ("lat_deg", "lon_deg")})

# ---------------------------------------------------------------- stage 1: the trajectory
if a.resolve or not os.path.exists(CACHE):
    P = PS.build()
    sc = P["scales"]
    # PLANNED ON THE PINN FIELD, which is both the faster solve and the field the flight guidance
    # plans on. The comparison below is about PROPAGATION, so the planning field only has to be a
    # legitimate plan, not the truth.
    dyn_plan = Dynamics(sc, field=GR.pinn_field())
    cm = dict(dyn=dyn_plan, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=P["n_hat"],
              K=a.K, verbose=False, slew_max_deg_s=C.SLEW_MAX_DEG_S,
              term_point_deg=C.TERMINAL_POINT_DEG, term_point_nodes=C.TERMINAL_POINT_NODES,
              point_deg=180.0, align_theta0_deg=45.0, align_h_m=100.0,
              iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4, tol_vc_m=1e-4)
    t0 = time.perf_counter()
    r1 = ptr.run(mode="error", warm=None, **cm)
    print(f"stage 1 {r1.status} {r1.iters:>3} it   D {r1.D_nd*sc.L:.4f} m   "
          f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L, warm=(r1.X, r1.U, r1.s), **cm)
    print(f"stage 2 {r2.status} {r2.iters:>3} it   fuel {r2.fuel_kg:.5f} kg   tf {r2.tf_s:.2f} s   "
          f"miss {r2.miss_m:.4f} m   [{time.perf_counter()-t0:.0f}s]", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=r2.X, U=r2.U, s=r2.s, tf_s=r2.tf_s, K=a.K,
                        L=sc.L, V=sc.v, A=sc.A, T=sc.T, alpha_nd=sc.alpha_nd,
                        r_site=P["r_site"], n_hat=P["n_hat"], site=np.array(SITE),
                        fuel_kg=r2.fuel_kg, miss_m=r2.miss_m, status=str(r2.status),
                        defect_m=r2.defect_m)
    print(f"cached {CACHE}", flush=True)

z = np.load(CACHE, allow_pickle=True)
X, U, s = z["X"], z["U"], float(z["s"])
K, L, tf = int(z["K"]), float(z["L"]), float(z["tf_s"])
print(f"trajectory: K={K}  tf={tf:.1f} s  fuel={float(z['fuel_kg']):.4f} kg  "
      f"miss={float(z['miss_m']):.4f} m  site={z['site']}", flush=True)

# ---------------------------------------------------------------- stage 2: replay under each field
P = PS.build()
sc = P["scales"]
FIELDS = {"polyhedral": GR.field(), "pinn": GR.pinn_field()}


def replay(field, n_sub):
    """Carry the state through ALL segments without restarting, recording every sub-step."""
    dyn = Dynamics(sc, field=field)
    dtau = 1.0 / (K - 1)
    h = s * dtau / n_sub
    x = X[:, 0].copy()
    xs = [x.copy()]
    for k in range(K - 1):
        for _ in range(n_sub):
            x = dyn.rk4(x, U[:, k], h)
            xs.append(x.copy())
    t = np.arange(len(xs)) * h * sc.T
    return t, np.array(xs)


out = {}
for name, fld in FIELDS.items():
    t0 = time.perf_counter()
    t, xs = replay(fld, a.n_sub)
    out[name] = xs
    wall = time.perf_counter() - t0
    print(f"replay {name:>11}: {len(xs):,} steps  [{wall:.1f}s]  "
          f"final r {np.linalg.norm(xs[-1, 0:3])*L/1e3:.4f} km", flush=True)

ref = out["polyhedral"]
for name, xs in out.items():
    if name == "polyhedral":
        continue
    dr = np.linalg.norm(xs[:, 0:3] - ref[:, 0:3], axis=1) * L
    dv = np.linalg.norm(xs[:, 3:6] - ref[:, 3:6], axis=1) * sc.v
    print(f"{name:>11} vs polyhedral:  |dr| final {dr[-1]:.3f} m   max {dr.max():.3f} m   "
          f"|dv| final {dv[-1]:.5f} m/s", flush=True)

np.savez_compressed(os.path.join(HERE, "data", "replay_7_2.npz"),
                    t=t, L=L, V=sc.v, n_sub=a.n_sub, **out)
print("wrote data/replay_7_2.npz", flush=True)

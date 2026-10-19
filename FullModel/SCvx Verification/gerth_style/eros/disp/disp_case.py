"""Trust-region variants on dispersed Eros handovers, both formulations.

usage: disp_case.py <10|7> <seed> <variant> [core]
  seed 0  : the undispersed handover (11.771 km, V0_LOCAL, 600 kg)
  seed>0  : pos sigma 1000 m, vel sigma 1 m/s, mass sigma 2 kg (pinn_case.py draw)
  variant : default | ratio | fixedR (R 0.35 frozen) | R0.7 (R 0.7 frozen) | noslew (10-state, slew limit off)
Gravity: grid_gravity.GridField (polyhedral field tabulated in the cone; exact fallback outside).
Settings as pinn_case.py: tol_obj 1e-7, tol_defect 1e-3 m / 1e-4 m/s, 200 iterations, pointing cone
off (180 deg), slew 2 deg/s, align corridor 45 deg / 100 m (10-state). d(0) free (no pinning chain):
this is the cold two-stage solve, the hardest part for the trust region.
"""
import os, sys, json, time
which, seed, variant = sys.argv[1], int(sys.argv[2]), sys.argv[3]
core = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.sched_setaffinity(0, {core})                     # before gravity is imported: TBB sizes its pool here
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "SCvx")); V1 = os.path.join(SCVX, "_v1_zoh_backup")
sys.path.insert(0, SCVX)
if which == "7":
    sys.path.insert(0, V1)
sys.path.insert(0, HERE)
import config as C
if which == "7":
    assert os.path.dirname(os.path.abspath(C.__file__)) == V1
    C.TERMINAL_POINT_DEG = 1.0; C.TERMINAL_POINT_NODES = 1; C.ALTITUDE_KM = 11.771
    C.POINTING_DEG = 180.0                          # v1 ptr.run has no point_deg argument
import problem_setup as PS, ptr
# keep the control ptr's convergence test judged: ptr.run re-gauges omega AFTER the test and returns
# the gauged copy (ptr.py:575-579). The last _gauge call of a run is on the final accepted (Xn, Un).
_last = {}
if hasattr(ptr, "_gauge"):
    _g0 = ptr._gauge
    def _keep(X, U):
        _last["X"], _last["U"] = np.array(X), np.array(U); return _g0(X, U)
    ptr._gauge = _keep
from dynamics import Dynamics, NX
from grid_gravity import GridField

P0 = PS.build()
r0, v0, m0 = P0["r0"], P0["v0"], C.MWET
if seed > 0:
    rng = np.random.default_rng(seed)
    r0 = r0 + 1000.0 * rng.standard_normal(3); v0 = v0 + 1.0 * rng.standard_normal(3); m0 = m0 + 2.0 * rng.standard_normal()
if which == "7":
    P = PS.build(m_wet=m0)
    # v1 build has no r0/v0 override: rebuild the initial state by hand, same layout as v1 problem_setup
    sc = C.Scales.build(r0, P["r_site"], m_wet=m0)
    x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(m0)]])
    rf_nd = P["r_site"] / sc.L
else:
    P = PS.build(r0_override=r0, v0_override=v0, m_wet=m0); sc = P["scales"]
    x_init, rf_nd = P["x_init"], P["rf_nd"]
fld = GridField()
os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
dyn = Dynamics(sc, field=fld)

VAR = {"default": {}, "ratio": dict(ratio_test=True), "fixedR": dict(w_growth=1.0),
       "R0.7": dict(R0=0.7, Rs0=1.0, R_bounds=(1e-4, 1.4), w_growth=1.0), "noslew": {}}[variant]
cm = dict(dyn=dyn, scales=sc, x_init=x_init, rf_nd=rf_nd, n_hat=P["n_hat"], K=60, verbose=False,
          iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4, **VAR)
if which == "10":
    cm.update(tol_vc_m=1e-4, slew_max_deg_s=None if variant == "noslew" else C.SLEW_MAX_DEG_S, point_deg=180.0,
              align_theta0_deg=45.0, align_h_m=100.0)
t = time.time()
r1 = ptr.run(mode="error", warm=None, **cm); t1 = time.time() - t
r2 = None
if not r1.status.startswith("subproblem"):
    r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L, warm=(r1.X, r1.U, r1.s), **cm)
wall = time.time() - t


def flown(r, U_override=None, pre=""):
    """Open-loop flights of the returned control (DOP853, rtol 1e-12, same field), three schemes:
      rate  : (omega_k, Gamma_k) held, d integrated from d_0 the whole way (verify.fly)
      att   : same, but d re-anchored to the plan's d_k at every node (closed_loop.plan_thrust)
      attT  : att with the thrust in NEWTONS held at Gamma_k m_k (constant-thrust engine)
    plus the per-segment direction defect, which ptr's convergence test does not look at.
    7-state: one flight (a held), reported under all three keys' first."""
    from scipy.integrate import solve_ivp
    X, U, s_ = r.X, (r.U if U_override is None else U_override), r.s; K = X.shape[1]; h = s_ / (K - 1)
    n = P["n_hat"]; Pl = np.eye(3) - np.outer(n, n); o = {}
    def fT(y, uk, zk):
        u = uk.copy(); u[3] = uk[3] * np.exp(zk - y[6]); return dyn.f(y, u)
    for mode in (("rate", "att", "attT") if NX == 10 else ("rate",)):
        x = X[:, 0].copy()
        for k in range(K - 1):
            if mode != "rate": x[7:10] = X[7:10, k]
            uk = U[:, k]; zk = X[6, k]
            rhs = (lambda t, y: fT(y, uk, zk)) if mode == "attT" else (lambda t, y: dyn.f(y, uk))
            x = solve_ivp(rhs, (0.0, h), x, method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
        rr = x[0:3] * sc.L - P["r_site"]
        o[f"{pre}{mode}_miss_m"] = float(np.linalg.norm(Pl @ rr)); o[f"{pre}{mode}_alt_m"] = float(n @ rr)
        o[f"{pre}{mode}_speed"] = float(np.linalg.norm(x[3:6]) * sc.v)
    if NX == 10:
        from scipy.spatial.transform import Rotation as Rot
        o[f"{pre}d_defect_max_rad"] = float(max(np.linalg.norm(Rot.from_rotvec(U[0:3, k] * h).apply(X[7:10, k]) - X[7:10, k + 1])
                                          for k in range(K - 1)))
    return o


def summ(r):
    H = r.history
    d = dict(status=r.status, it=r.iters, fuel=r.fuel_kg, tf=r.tf_s, miss=r.miss_m, defect_m=r.defect_m,
             rejected=sum(1 for h in H if not h.get("accepted", True)),
             R_min=min((h["R"] for h in H if "R" in h), default=None), R_max=max((h["R"] for h in H if "R" in h), default=None))
    if NX == 10:
        d["dnorm"] = [r.dnorm_lo, r.dnorm_hi]
    return d


out = dict(formulation=which, seed=seed, variant=variant, r0_off_m=float(np.linalg.norm(r0 - P0["r0"])),
           m0=m0, wall_s=wall, wall_s1=t1, grav_calls=fld.n_calls, grav_exact=fld.n_exact,
           stage1=summ(r1), stage2=summ(r2) if r2 is not None else None)
if r2 is not None:
    out["stage2"].update(flown(r2))
    if NX == 10 and "U" in _last and np.allclose(_last["X"], r2.X):
        from dynamics import defect as _defect
        out["stage2"].update(flown(r2, _last["U"], pre="j_"))
        dp, dv, _, _, _ = _defect(dyn, r2.X, _last["U"], r2.s, n_sub=20)
        out["stage2"].update(j_defect_m=float(dp * sc.L), j_defect_v=float(dv * sc.v))
    np.savez(os.path.join(HERE, "res", f"{which}_{seed}_{variant}.npz"), X=r2.X, U=r2.U, s=r2.s, L=sc.L, T=sc.T, A=sc.A)
print(json.dumps(out, default=float), flush=True)
json.dump(out, open(os.path.join(HERE, "res", f"{which}_{seed}_{variant}.json"), "w"), indent=1, default=float)

"""Is the post-convergence re-gauge what breaks direction consistency?
Seed 0, 10-state, dataset settings, grid field. ptr._gauge is wrapped to keep the control it was
handed (Un, the solution the convergence test judged); run returns the gauged one. Fly both."""
import os, sys, json
os.sched_setaffinity(0, {1})
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "SCvx"))); sys.path.insert(0, HERE)
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rot
import config as C, problem_setup as PS, ptr
from dynamics import Dynamics, defect
from grid_gravity import GridField
last = {}
_g = ptr._gauge
def keep(X, U):
    last["X"], last["U"] = np.array(X), np.array(U); return _g(X, U)
ptr._gauge = keep
P = PS.build(); sc = P["scales"]; dyn = Dynamics(sc, field=GridField())
cm = dict(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=P["n_hat"], K=60, verbose=False, iters=200,
          tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4, tol_vc_m=1e-4, slew_max_deg_s=C.SLEW_MAX_DEG_S,
          point_deg=180.0, align_theta0_deg=45.0, align_h_m=100.0)
r1 = ptr.run(mode="error", warm=None, **cm)
r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L, warm=(r1.X, r1.U, r1.s), **cm)
assert np.allclose(last["X"], r2.X)
n = P["n_hat"]; Pl = np.eye(3) - np.outer(n, n); K = 60; h = r2.s / (K - 1)
def check(X, U, tag):
    dd = max(np.linalg.norm(Rot.from_rotvec(U[0:3, k] * h).apply(X[7:10, k]) - X[7:10, k + 1]) for k in range(K - 1))
    dp, dv, _, _, _ = defect(dyn, X, U, r2.s, n_sub=20)
    o = dict(tag=tag, d_defect_rad=float(dd), r_defect_m=float(dp * sc.L), v_defect_ms=float(dv * sc.v),
             max_omega_dot_d=float(np.max(np.abs(np.sum(U[0:3, :-1] * X[7:10, :-1], 0)))))
    for mode in ("rate", "att"):
        x = X[:, 0].copy()
        for k in range(K - 1):
            if mode == "att": x[7:10] = X[7:10, k]
            uk = U[:, k]
            x = solve_ivp(lambda t, y: dyn.f(y, uk), (0.0, h), x, method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
        rr = x[0:3] * sc.L - P["r_site"]
        o[f"{mode}_miss_m"] = float(np.linalg.norm(Pl @ rr)); o[f"{mode}_alt_m"] = float(n @ rr)
    print(json.dumps(o), flush=True); return o
out = dict(status=r2.status, it=r2.iters, fuel=r2.fuel_kg, tf=r2.tf_s,
           returned=check(r2.X, r2.U, "returned (gauged)"), judged=check(last["X"], last["U"], "judged (pre-gauge Un)"))
json.dump(out, open(os.path.join(HERE, "regauge_test.json"), "w"), indent=1)

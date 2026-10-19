"""Gerth Sec. 8.4 analogue: where does one PTR iteration spend its time? (min-max benchmark, K=60)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.argv = ["x"]
import run_verification as RV
import ptr, subproblem as SP
from dynamics import discretize, defect
import config as C, benchmark as B
from dynamics import Dynamics
cs = B.CASES["min-max"]; r0, v0 = cs["r0"], cs["v0"]; rf = np.zeros(3); n = np.array([0, 0, 1.0])
for K in (30, 60, 120):
    sc = C.Scales.build(r0, rf, m_wet=B.M_WET, t_max=B.TMAX)
    dyn = Dynamics(sc, field=B.ConstantField())
    x0 = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(B.M_WET)], n])
    t_lo, t_hi = C.tf_bounds(np.linalg.norm(v0), np.linalg.norm(r0))
    Xr, Ur, sr = ptr.initial_guess(dyn, x0, rf, n, sc, K, sc.t_nd(t_lo), sc.t_nd(t_hi), term_point_deg=180.0, term_point_nodes=0)
    T = dict(disc=0, build=0, compile=0, solve=0, defect=0)
    for it in range(3):
        t = time.perf_counter(); Ad, Bd, Sd, zd, _ = discretize(dyn, Xr, Ur, sr, n_sub=10); T["disc"] += time.perf_counter() - t
        t = time.perf_counter()
        prob, V = SP.build(Ad, Bd, Sd, zd, Xr, Ur, sr, x0, rf, n, sc, mode="fuel", D_nd=0.0, w_tr=0, w_ts=0,
                           s_lo=sc.t_nd(t_lo), s_hi=sc.t_nd(t_hi), vtouch_nd=0.0, R=0.35, Rs=0.5,
                           term_point_deg=180.0, term_point_nodes=0, slew_max_nd=None)
        T["build"] += time.perf_counter() - t
        t = time.perf_counter(); prob.solve(solver="CLARABEL"); tot = time.perf_counter() - t
        T["compile"] += prob.compilation_time; T["solve"] += tot - prob.compilation_time
        Xn, Un, sn = V["X"].value, V["U"].value, float(V["s"].value)
        t = time.perf_counter(); defect(dyn, Xn, Un, sn, n_sub=20); T["defect"] += time.perf_counter() - t
        Xr, Ur, sr = Xn, ptr._gauge(Xn, Un), sn
    tot = sum(T.values())
    print(f"K {K:4d}  per iteration: " + "  ".join(f"{k} {v/3:6.3f} s ({100*v/tot:4.1f}%)" for k, v in T.items())
          + f"   solver-internal {prob.solver_stats.solve_time:.3f} s", flush=True)

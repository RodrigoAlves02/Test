#!/usr/bin/env python
"""Does SCvx's own propagation agree with a high-fidelity one?

SCvx does not integrate its solution the way a truth propagator would. It divides each zero-order-
hold segment into a fixed number of RK4 substeps -- sub_dt = 4 s, so about 15 substeps per node --
and never lets a step cross a node. That is switch detection by construction: the segment boundaries
ARE the switches, and the integrator is told where they are.

If the trajectory SCvx returns is to be trusted as a reference for anything, that propagation has to
agree with one that makes no such assumption. Three propagations of the identical control history,
all through the identical right-hand side (scvx's own Dyn.f, rotating frame):

  A  RK4, sub_dt = 4 s, per segment          exactly what SCvx does internally
  B  DOP853 at tight tolerance, per segment  high fidelity, still told where the switches are
  C  DOP853 at tight tolerance, straight     high fidelity, NOT told -- steps may straddle a switch

A against B answers the question asked: is SCvx's fixed substep good enough, or is the trajectory it
reports subtly not a solution of its own dynamics?

C is included because it isolates what the switch detection is worth. It is the same integrator as B
at the same tolerance, differing only in whether the thrust discontinuities are respected.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.join(HERE, "..", "..", "..", "scvx_landing")
sys.path.insert(0, SCVX)


class Counter:
    """Count gravity calls at the source so the three runs can be priced on the same scale."""

    def __init__(self, dyn):
        self.dyn, self.n, self._g = dyn, 0, dyn.g

    def __enter__(self):
        outer = self

        def counted(r):
            outer.n += 1
            return outer._g(r)
        self.dyn.g = counted
        return self

    def __exit__(self, *a):
        self.dyn.g = self._g


def run_A(dyn, x0, U, sigma, sub_dt):
    """SCvx's own scheme: RK4 with a fixed substep, restarted at every node."""
    K = U.shape[1]
    h = sigma / (K - 1)
    keep, dyn.sub_dt = dyn.sub_dt, sub_dt
    try:
        x = np.asarray(x0, float).copy()
        out = [x.copy()]
        for k in range(K - 1):
            x = dyn.rk4_segment(x, U[:, k], U[:, k], h)
            out.append(x.copy())
    finally:
        dyn.sub_dt = keep
    return np.array(out)


def run_B(dyn, x0, U, sigma, rtol, atol):
    """High fidelity, segment by segment: the solver is restarted at every switch."""
    K = U.shape[1]
    h = sigma / (K - 1)
    x = np.asarray(x0, float).copy()
    out = [x.copy()]
    for k in range(K - 1):
        uk = U[:, k]
        s = solve_ivp(lambda t, y: dyn.f(y, uk), (0.0, h), x, method="DOP853",
                      rtol=rtol, atol=atol)
        if not s.success:
            raise RuntimeError(s.message)
        x = s.y[:, -1]
        out.append(x.copy())
    return np.array(out)


def run_C(dyn, x0, U, sigma, rtol, atol):
    """High fidelity, one call for the whole arc: the solver is never told where the thrust jumps."""
    K = U.shape[1]
    h = sigma / (K - 1)
    t_nodes = np.linspace(0.0, sigma, K)

    def f(t, y):
        k = min(int(t // h), K - 2)
        return dyn.f(y, U[:, k])

    s = solve_ivp(f, (0.0, sigma), np.asarray(x0, float), method="DOP853",
                  rtol=rtol, atol=atol, t_eval=t_nodes)
    if not s.success:
        raise RuntimeError(s.message)
    return s.y.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(SCVX, "descent_nominal.npz"))
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--atol", type=float, default=1e-14)
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    import tune
    import gen_dataset as G
    from scvx.core import Dyn
    from scvx.gravity import make_polyhedral, load_mesh

    z = np.load(a.npz)
    X, U, sigma = z["X"], z["U"], float(z["sigma"])
    x0 = X[:, 0].copy()
    K = X.shape[1]
    h = sigma / (K - 1)

    # scvx's own dynamics object, configured exactly as build_case does
    dyn = Dyn(tune.ISP, poly=make_polyhedral(load_mesh(), density=2670.0),
              sub_dt=tune.SUB_DT, hold=tune.HOLD)
    dyn.disc = "rk4"

    print(f"solution : {K} nodes, t_f {sigma:.2f} s, segment {h:.3f} s")
    print(f"scvx     : sub_dt {tune.SUB_DT:g} s -> {max(1, int(round(h/tune.SUB_DT)))} substeps "
          f"per segment, hold {tune.HOLD}")
    print(f"reference: DOP853 rtol {a.rtol:.0e}, atol {a.atol:.0e}\n")

    runs = {}
    for name, fn in (("A_scvx_rk4", lambda: run_A(dyn, x0, U, sigma, tune.SUB_DT)),
                     ("B_dop853_segmented", lambda: run_B(dyn, x0, U, sigma, a.rtol, a.atol)),
                     ("C_dop853_straight", lambda: run_C(dyn, x0, U, sigma, a.rtol, a.atol))):
        with Counter(dyn) as c:
            t0 = time.perf_counter()
            Y = fn()
            cpu = time.perf_counter() - t0
        runs[name] = dict(Y=Y, evals=c.n, cpu=cpu)
        print(f"  {name:20s} {c.n:8d} evals  {cpu:7.2f} s")

    YA, YB, YC = (runs[k]["Y"] for k in ("A_scvx_rk4", "B_dop853_segmented", "C_dop853_straight"))
    t = np.linspace(0.0, sigma, K)

    dAB = np.linalg.norm(YA[:, 0:3] - YB[:, 0:3], axis=1)
    dAC = np.linalg.norm(YA[:, 0:3] - YC[:, 0:3], axis=1)
    dBC = np.linalg.norm(YB[:, 0:3] - YC[:, 0:3], axis=1)
    vAB = np.linalg.norm(YA[:, 3:6] - YB[:, 3:6], axis=1)

    print(f"\nfinal position difference")
    print(f"  A vs B  (scvx RK4   vs high fidelity, switches respected) : {dAB[-1]:.6e} m")
    print(f"  A vs C  (scvx RK4   vs high fidelity, switches ignored)   : {dAC[-1]:.6e} m")
    print(f"  B vs C  (high fidelity, with vs without switch detection) : {dBC[-1]:.6e} m")
    print(f"  A vs B final velocity difference                          : {vAB[-1]:.6e} m/s")
    print(f"\nmax over the trajectory:  A-B {dAB.max():.3e} m   A-C {dAC.max():.3e} m   "
          f"B-C {dBC.max():.3e} m")

    # where the SCvx solution says it lands, and where each propagation actually puts it
    with open(os.path.join(SCVX, "descent_nominal.npz"), "rb"):
        pass
    rf = z["rf"] if "rf" in z else None
    if rf is not None:
        print(f"\nmiss against the target site")
        for nm, Y in (("scvx solution nodes", X.T), ("A scvx RK4", YA),
                      ("B high fidelity", YB), ("C no switch detection", YC)):
            print(f"  {nm:22s} {np.linalg.norm(Y[-1, 0:3] - rf):.6e} m")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "scvx_propagation_check.json")
    json.dump(dict(nodes=K, tf=sigma, seg=h, sub_dt=tune.SUB_DT, hold=tune.HOLD,
                   rtol=a.rtol, atol=a.atol, t=t.tolist(),
                   evals={k: runs[k]["evals"] for k in runs},
                   cpu={k: runs[k]["cpu"] for k in runs},
                   d_AB=dAB.tolist(), d_AC=dAC.tolist(), d_BC=dBC.tolist(),
                   v_AB=vAB.tolist()), open(f, "w"), indent=1)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()

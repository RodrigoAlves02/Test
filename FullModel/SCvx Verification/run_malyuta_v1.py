#!/usr/bin/env python
"""The Malyuta benchmark against the SEVEN-STATE formulation.

Same case as run_malyuta.py -- Malyuta et al. arXiv:2106.09125, Problem (95), parameters (108),
published optimum tf = 75.0 s -- solved with SCvx/_v1_zoh_backup instead of SCvx/. The ten-state
formulation is already verified on it (tf +0.82 / +0.81 / +0.67 % at K = 38 / 76 / 151, cold then
warm-laddered). This file is what lets the seven-state formulation carry the same claim, which it
must before anything is built on top of it.

WHAT DIFFERS FROM run_malyuta.py, AND NOTHING ELSE DOES. The benchmark parameters come from the
same benchmark.MAL, so the problem is identical; only the solver changes.

  x_init    7 elements, [r/L, v T/L, ln m].  No direction row -- that is the whole point.
  thrust    the CONTROL, U[0:3] * A * m, not Gamma * d with d read off the state.
  kwargs    v1's ptr.run has no slew_max_deg_s and no a_init. Malyuta imposes no rate limit, so
            there is nothing to pass; a_init has no seven-state meaning.
  gs/point  v1's ptr.run does not forward gs_deg or point_deg, so they arrive through the config
            constants. THEY ARE PATCHED BEFORE ptr IS IMPORTED, because subproblem.build takes them
            as DEFAULT arguments and a default is evaluated once, at definition time -- patching the
            module afterwards would silently leave the glideslope at 30 deg and the pointing cone at
            140, i.e. verify a different problem and report a pass.
  tf_bounds v1's signature has no v_touch argument; the ten-state one gained it later.

THE IMPORT ORDER IS LOAD-BEARING. _v1_zoh_backup goes on sys.path FIRST so config, ptr, dynamics
and subproblem all resolve to the seven-state copies; SCvx goes on AFTER it, only so that v1's
dynamics.py can find gravity, which this case never calls (ConstantField).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.abspath(os.path.join(HERE, "..", "SCvx", "_v1_zoh_backup"))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
sys.path.insert(0, SCVX)                 # third: gravity only
sys.path.insert(0, HERE)                 # second: benchmark
sys.path.insert(0, V1)                   # first: the seven-state solver wins every name

import benchmark as B                                                        # noqa: E402

M = B.MAL
import config as C                                                           # noqa: E402

assert os.path.dirname(os.path.abspath(C.__file__)) == V1, \
    f"config resolved to {C.__file__}, not the seven-state copy"

C.TMIN, C.TMAX = M["rho_min"], M["rho_max"]
C.MWET, C.MDRY = M["m_wet"], M["m_dry"]
C.ISP, C.G0, C.ALPHA = M["isp"], M["ge"], M["alpha"]
C.W_BODY = M["omega"]                       # the ROTATING FRAME is active in this case
C.VTOUCH = 0.0                              # r(tf) = v(tf) = 0 exactly
C.GLIDESLOPE_DEG = M["gamma_gs"]
C.POINTING_DEG = M["gamma_p"]
C.TERMINAL_POINT_DEG = 180.0                # no terminal pointing in the benchmark
C.TERMINAL_POINT_NODES = 0
_tf0 = C.tf_bounds
C.tf_bounds = lambda v0n, rng, **kw: _tf0(v0n, rng, m_wet=M["m_wet"], m_dry=M["m_dry"],
                                          t_max=M["rho_max"], t_min=M["rho_min"],
                                          g_max=abs(M["g"][2]))

import ptr                                                                   # noqa: E402
import subproblem as SP                                                      # noqa: E402
from dynamics import NX, Dynamics                                            # noqa: E402

assert NX == 7, f"this is the seven-state runner but dynamics reports NX = {NX}"
# the defaults really did pick up the patch
import inspect                                                               # noqa: E402
_d = inspect.signature(SP.build).parameters
assert abs(_d["gs_deg"].default - M["gamma_gs"]) < 1e-12, "glideslope default not patched"
assert abs(_d["point_deg"].default - M["gamma_p"]) < 1e-12, "pointing default not patched"


def resample(warm, K):
    """Put a warm start onto K nodes. v1's ptr.run does NOT do this and the ten-state's does.

    Both discretize() and build() read the node count from Xref.shape[1], so handing v1 a warm
    start of a different length silently solves the OLD K and reports the new one. Measured here
    first time out: K = 76 and K = 151 returned bit-identical tf, defect, miss and LCvx gap,
    because both were re-solving the K = 38 problem. Linear interpolation in normalised time is
    what the ten-state ptr.run does; this reproduces it outside the solver so v1 stays untouched.
    """
    if warm is None:
        return None
    Xr, Ur, sr = (np.array(warm[0], float), np.array(warm[1], float), float(warm[2]))
    if Xr.shape[1] != K:
        tw = np.linspace(0.0, 1.0, Xr.shape[1])
        tq = np.linspace(0.0, 1.0, K)
        Xr = np.array([np.interp(tq, tw, Xr[i]) for i in range(Xr.shape[0])])
        Ur = np.array([np.interp(tq, tw, Ur[i]) for i in range(Ur.shape[0])])
    return Xr, Ur, sr


def run(K=76, fix_tf=None, verbose=False, iters=400, warm=None):
    """Solve the Malyuta case with the seven-state solver. Returns (Result, scales)."""
    warm = resample(warm, K)
    r0, v0, rf = M["r0"], M["v0"], np.zeros(3)
    n_hat = np.array([0.0, 0.0, 1.0])
    sc = C.Scales.build(r0, rf, m_wet=M["m_wet"], t_max=M["rho_max"])
    dyn = Dynamics(sc, field=B.ConstantField(M["g"]))
    x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(M["m_wet"])]])
    r = ptr.run(dyn=dyn, scales=sc, x_init=x_init, rf_nd=rf / sc.L, n_hat=n_hat,
                K=K, mode="fuel", D_nd=0.0, iters=iters, verbose=verbose,
                term_point_deg=180.0, term_point_nodes=0,
                warm=warm, fix_tf_s=fix_tf, tol_obj=1e-6, tol_defect_m=1e-2)
    # THE MESH THAT WAS ASKED FOR IS THE MESH THAT WAS SOLVED. Without this the ladder reports
    # mesh convergence it never performed.
    assert r.X.shape[1] == K, f"asked for K = {K}, solver returned {r.X.shape[1]} nodes"
    return r, sc


def audit(r, sc):
    """The quantities the benchmark is judged on, in SI."""
    m = np.exp(r.X[6, :])
    K = r.X.shape[1]
    pos = r.X[0:3, :] * sc.L
    vel = r.X[3:6, :] * sc.v
    # THRUST IS THE CONTROL HERE. U[0:3] is the thrust ACCELERATION, U[3] the LCvx magnitude slack.
    Tc = r.U[0:3, :] * sc.A * m
    sig = r.U[3, :] * sc.A * m
    nT = np.linalg.norm(Tc, axis=0)
    tilt = np.degrees(np.arccos(np.clip(Tc[2] / np.maximum(nT, 1e-12), -1.0, 1.0)))
    # LCvx is lossless exactly when the slack is tight: ||T|| == sigma at every node.
    lcvx = float(np.max(np.abs(nT[:K - 1] - sig[:K - 1])))
    e3 = np.array([0.0, 0.0, 1.0])
    P = np.eye(3) - np.outer(e3, e3)
    # glideslope with its apex at the ACHIEVED landing point, on k = 0..K-2, as written
    dr = pos[:, :K - 1] - pos[:, K - 1:K]
    gs = np.degrees(np.arctan2(np.linalg.norm(P @ dr, axis=0), np.maximum(e3 @ dr, 1e-12)))
    return dict(tf_s=float(sc.t_dim(r.s)), fuel_kg=float(m[0] - m[-1]),
                m_final=float(m[-1]), status=str(r.status), iters=int(r.iters),
                defect_m=float(r.defect_m), miss_m=float(np.linalg.norm(pos[:, -1] - 0.0)),
                v_final_ms=float(np.linalg.norm(vel[:, -1])),
                T_min_N=float(nT[:K - 1].min()), T_max_N=float(nT[:K - 1].max()),
                tilt_max_deg=float(tilt[:K - 1].max()), gs_max_deg=float(gs.max()),
                lcvx_gap_N=lcvx, wall_s=float(r.wall_s))


def main():
    import argparse, json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--Ks", default="38,76,151")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    Ks = [int(x) for x in a.Ks.split(",")]

    print(f"MALYUTA Problem (95), parameters (108) -- SEVEN-STATE solver (NX = {NX})")
    print(f"published optimum tf = {M['tf']:.1f} s")
    print(f"glideslope {M['gamma_gs']:g} deg   pointing {M['gamma_p']:g} deg   "
          f"thrust {M['rho_min']:g}-{M['rho_max']:g} N   omega {M['omega_deg_s']} deg/s")
    print("warm-laddered: the coarsest mesh cold, each finer one from the previous solution\n")
    print(f"{'K':>5} {'status':<14} {'it':>4} {'tf [s]':>9} {'err %':>7} {'fuel [kg]':>10} "
          f"{'m_f [kg]':>9} {'defect':>9} {'miss [m]':>9} {'|v_f|':>8} {'tilt':>6} {'gs':>6} "
          f"{'LCvx [N]':>9} {'wall':>7}")
    rows, warm = [], None
    for K in Ks:
        r, sc = run(K, verbose=a.verbose, warm=warm, iters=a.iters)
        warm = (r.X, r.U, r.s)
        d = audit(r, sc)
        d["K"] = K
        d["tf_err_pct"] = 100.0 * (d["tf_s"] - M["tf"]) / M["tf"]
        rows.append(d)
        print(f"{K:>5} {d['status']:<14} {d['iters']:>4} {d['tf_s']:>9.4f} "
              f"{d['tf_err_pct']:>+7.2f} {d['fuel_kg']:>10.3f} {d['m_final']:>9.3f} "
              f"{d['defect_m']:>9.2e} {d['miss_m']:>9.2e} {d['v_final_ms']:>8.2e} "
              f"{d['tilt_max_deg']:>6.2f} {d['gs_max_deg']:>6.2f} {d['lcvx_gap_N']:>9.2e} "
              f"{d['wall_s']:>7.1f}", flush=True)

    out = os.path.join(HERE, "Figures&data", "malyuta_v1_7state.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(case="Malyuta arXiv:2106.09125 Problem (95), params (108)",
                   formulation="7-state (SCvx/_v1_zoh_backup)", NX=int(NX),
                   tf_published_s=float(M["tf"]), rows=rows), open(out, "w"), indent=1)
    print(f"\nreference, 10-state: tf +0.82 / +0.81 / +0.67 % at K = 38 / 76 / 151")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

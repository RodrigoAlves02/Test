"""Checks on the EKF. Properties the physics requires, not regressions against recorded numbers.

The one that matters is test 2: the filter is handed the TRUE state and a PERFECT IMU, and asked to
reproduce the recorded truth trajectory. If a fictitious-force term were left in from the rotating
frame, or the gravity rotation were wrong or missing, this is where it shows -- those errors are
O(1e-3) m/s^2 against a gravity of 6e-3, so they cannot hide.

Run: python test_ekf.py [--seed N] [--steps N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import config_nav as CN
import ekf as EKF
import frames as NFR

sys.path.insert(0, CN.SCVX_DIR)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400, help="sub-steps for the propagation test")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    import gravity as GR
    fld = GR.field()
    wz = CN.w_body_z()
    ip = dict(CN.imu_params()); ip["vrw"] = 0.0; ip["bias_inst"] = 0.0
    ok = True
    print(f"EKF checks   seed {a.seed}   bias frame '{CN.EKF_BIAS_FRAME}'   wz {wz:.6e} rad/s")

    # -- 1. the gravity rotation --------------------------------------------------------------------
    print("\n1. gravity: g_I(Rz r, t) == Rz g_body(r)")
    f0 = EKF.EKF(np.zeros(CN.NX), np.eye(CN.NX), fld, ip, wz=wz)
    worst = 0.0
    for t in (0.0, 300.0, 1049.0, 2098.0):
        r_b = np.array([3.0e3, -7.0e3, -6.0e3]) + rng.normal(0, 200.0, 3)
        Rz = NFR.Rz(wz * t)
        lhs = f0.g_inertial(Rz @ r_b, t)
        rhs = Rz @ fld.g(r_b)
        worst = max(worst, np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))
    ok &= check("rotation is exact", worst < 1e-12, f"max relative {worst:.2e}")
    r_b = np.array([3.0e3, -7.0e3, -6.0e3])
    ok &= check("t = 0 is the identity",
                np.allclose(f0.g_inertial(r_b, 0.0), fld.g(r_b), rtol=0, atol=1e-18))
    ok &= check("|g| is rotation invariant",
                abs(np.linalg.norm(f0.g_inertial(NFR.Rz(wz*997.0) @ r_b, 997.0))
                    - np.linalg.norm(fld.g(r_b))) < 1e-15)

    # -- 2. propagate the recorded truth with a perfect IMU -----------------------------------------
    print("\n2. propagation reproduces the flown truth (perfect IMU, truth gravity)")
    path = os.path.join(CN.SCVX_DIR, "Figures&data", "closed_loop_imuplot.json")
    tr = json.load(open(path))["trace"]
    k0 = next(i for i, r in enumerate(tr) if r["T_act"] > 0.1)
    rows = tr[k0:k0 + a.steps + 1]
    dt = rows[1]["t"] - rows[0]["t"]

    def truth_I(row):
        Rz = NFR.Rz(wz * row["t"])
        r_I = Rz @ np.array(row["r"], float)
        v_I = Rz @ np.array(row["v"], float) + np.cross([0, 0, wz], r_I)
        return r_I, v_I

    r0, v0 = truth_I(rows[0])
    f = EKF.EKF(np.concatenate([r0, v0, np.zeros(6)]), np.eye(CN.NX) * 1e-6, fld, ip, wz=wz)
    per_step = []
    for i in range(len(rows) - 1):
        row = rows[i]
        a_rot = row["T_act"] / row["mass"] * np.array(row["d_act"], float)
        a_I = NFR.vec_rot_to_inertial(a_rot, row["t"], wz)
        f.propagate(a_I, dt, row["t"], d_I=NFR.vec_rot_to_inertial(row["d_act"], row["t"], wz))
        rt, vt = truth_I(rows[i + 1])
        per_step.append((np.linalg.norm(f.x[CN.IR] - rt), np.linalg.norm(f.x[CN.IV] - vt)))
        f.x[CN.IR], f.x[CN.IV] = rt, vt              # reset: isolate ONE step of integration error
    per_step = np.array(per_step)
    ok &= check(f"single-step position error over {len(per_step)} steps of {dt:g} s",
                per_step[:, 0].max() < 1e-4,
                f"max {per_step[:,0].max():.3e} m, mean {per_step[:,0].mean():.3e} m")
    ok &= check("single-step velocity error", per_step[:, 1].max() < 1e-6,
                f"max {per_step[:,1].max():.3e} m/s")

    # FREE-RUNNING, no reset -- where a missing term accumulates instead of cancelling.
    #
    # The measurement has to be re-rotated at every sample. The actuator holds the thrust constant
    # in ACAF over each 0.5 s sub-step, so in ACI that same physical thrust ROTATES at w: over one
    # sub-step the direction moves w*dt = 1.7e-4 rad. Freezing a_I at the start of the sub-step is
    # a zero-order hold applied in the wrong frame, and it costs 1.3e-2 m over 200 s no matter how
    # finely the propagation is stepped. Re-rotating per sample makes the drift first order in dt,
    # which is what the numbers below show. The closed loop samples the IMU at 100 Hz with a_I
    # rotated at each sample, so this is the regime that actually flies.
    span = rows[-1]["t"] - rows[0]["t"]
    dt_imu = 1.0 / CN.IMU_RATE_HZ
    n_sub = max(1, int(round(dt / dt_imu)))
    drifts = {}
    for label, rotate in (("frozen a_I", False), ("rotated per sample", True)):
        f = EKF.EKF(np.concatenate([r0, v0, np.zeros(6)]), np.eye(CN.NX) * 1e-6, fld, ip, wz=wz)
        for i in range(len(rows) - 1):
            row = rows[i]
            a_rot = row["T_act"] / row["mass"] * np.array(row["d_act"], float)
            for j in range(n_sub):
                tj = row["t"] + j * dt_imu
                f.propagate(NFR.vec_rot_to_inertial(a_rot, tj if rotate else row["t"], wz),
                            dt_imu, tj)
        rt, vt = truth_I(rows[-1])
        drifts[label] = (np.linalg.norm(f.x[CN.IR] - rt), np.linalg.norm(f.x[CN.IV] - vt))
    dr, dv = drifts["rotated per sample"]
    ok &= check(f"free-running drift over {span:g} s at {CN.IMU_RATE_HZ:g} Hz", dr < 1e-3,
                f"|dr| {dr:.3e} m, |dv| {dv:.3e} m/s")
    fr = drifts["frozen a_I"][0]
    ok &= check("re-rotating the measurement is what buys that", fr > 10 * dr,
                f"frozen a_I gives {fr:.3e} m, {fr/dr:.0f}x worse")
    # what a leftover Coriolis term would have cost, for scale
    vbar = np.linalg.norm(np.array(rows[0]["v"], float))
    cor = 2 * wz * vbar
    print(f"         for scale: a stray Coriolis 2*w*|v| = {cor:.3e} m/s^2 would give "
          f"~{0.5*cor*span**2:.1f} m over {span:g} s -- {0.5*cor*span**2/dr:.0f}x the drift above")

    # -- 3. the Jacobian against finite differences --------------------------------------------------
    print("\n3. F matches a finite difference of f")
    f = EKF.EKF(np.concatenate([r0, v0, rng.normal(0, 3e-3, 3), np.zeros(3)]), np.eye(CN.NX), fld, ip,
                wz=wz, use_grad=True)
    t0 = rows[0]["t"]
    a_I = np.array([1e-3, -2e-3, 3e-3])
    x = f.x.copy()
    Fa = f._F(x, t0)
    Fn = np.zeros((CN.NX, CN.NX))
    for j in range(CN.NX):
        h = max(1e-4, abs(x[j]) * 1e-7)
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        Fn[:, j] = (f._f(xp, a_I, t0) - f._f(xm, a_I, t0)) / (2 * h)
    err = np.abs(Fa - Fn)
    ok &= check("F[r,v] = I", np.allclose(Fa[CN.IR, CN.IV], np.eye(3)))
    ok &= check("F[v,b] = -B", np.allclose(Fa[CN.IV, CN.IB], -f.B))
    ok &= check("F[b,b] = 0 (bias is a constant)", np.all(Fa[CN.IB, CN.IB] == 0.0))
    ok &= check("F[v,a_u] = the DMC frame rotation",
                np.allclose(Fa[CN.IV, CN.IA], f._dmc_rot(t0))
                and err[CN.IV, CN.IA].max() < 1e-9,
                f"max |dF| {err[CN.IV, CN.IA].max():.2e}")
    ok &= check("F[a_u,a_u] = 0 (DMC is a random walk, no self-dynamics)",
                np.all(Fa[CN.IA, CN.IA] == 0.0))
    ok &= check("F[v,r] matches the numerical gravity gradient",
                err[CN.IV, CN.IR].max() < 1e-9, f"max |dF| {err[CN.IV, CN.IR].max():.2e}")
    ok &= check("no spurious F[v,v] (no Coriolis in an inertial frame)",
                np.all(Fa[CN.IV, CN.IV] == 0.0) and err[CN.IV, CN.IV].max() < 1e-12,
                f"numerical F[v,v] max {err[CN.IV, CN.IV].max():.2e}")

    # -- 4. the Joseph update keeps P symmetric and positive definite ---------------------------------
    print("\n4. Joseph update over many measurements")
    f = EKF.EKF(np.concatenate([r0, v0, np.zeros(6)]),
                np.diag(np.concatenate([np.full(3, 100.0), np.full(3, 1e-4),
                                        np.full(3, CN.imu_params()["bias_sigma"] ** 2),
                                        np.full(3, CN.EKF_DMC_P0 ** 2)])),
                fld, ip, wz=wz)
    H = np.zeros((3, CN.NX)); H[:, CN.IR] = np.eye(3)
    R = np.eye(3) * 4.0
    worst_sym, min_eig, traces = 0.0, np.inf, []
    for i in range(2000):
        z = f.x[CN.IR] + rng.normal(0, 2.0, 3)
        f.update(z, H, R, f.x[CN.IR].copy())
        worst_sym = max(worst_sym, np.abs(f.P - f.P.T).max())
        ev = np.linalg.eigvalsh(f.P)
        min_eig = min(min_eig, ev.min())
        traces.append(np.trace(f.P[CN.IR, CN.IR]))
    ok &= check("P stays symmetric", worst_sym < 1e-18, f"max |P - P'| {worst_sym:.2e}")
    ok &= check("P stays positive definite", min_eig > 0, f"min eigenvalue {min_eig:.3e}")
    ok &= check("position covariance shrinks monotonically",
                np.all(np.diff(traces) <= 1e-18), f"trace {traces[0]:.3e} -> {traces[-1]:.3e}")

    # -- 5. tau = inf, and the bias frames ------------------------------------------------------------
    print("\n5. bias handling")
    f = EKF.EKF(np.zeros(CN.NX), np.eye(CN.NX), fld, CN.imu_params(), wz=wz)
    Q = f._Q(0.01)
    ok &= check("Q_bb = 0 (random constant)", np.all(Q[CN.IB, CN.IB] == 0.0))
    ok &= check("Q_vv = N^2 dt", abs(Q[CN.IV, CN.IV][0, 0] - CN.imu_params()["vrw"] ** 2 * 0.01)
                < 1e-30)
    fi = EKF.EKF(np.zeros(CN.NX), np.eye(CN.NX), fld, ip, wz=wz, bias_frame="inertial",
                 d0_I=np.array([0., 0., 1.]))
    ft = EKF.EKF(np.zeros(CN.NX), np.eye(CN.NX), fld, ip, wz=wz, bias_frame="thrust",
                 d0_I=np.array([0., 0., 1.]))
    th = np.radians(np.linspace(0, 130, 400))
    for q in th:
        d = np.array([np.sin(q), 0.0, np.cos(q)])
        fi.set_boresight(d); ft.set_boresight(d)
    ok &= check("bias_frame='inertial' leaves B = I", np.allclose(fi.B, np.eye(3)))
    ok &= check("bias_frame='thrust' transports B", not np.allclose(ft.B, np.eye(3)))
    ok &= check("B stays a proper rotation",
                np.allclose(ft.B.T @ ft.B, np.eye(3), atol=1e-12)
                and abs(np.linalg.det(ft.B) - 1) < 1e-12,
                f"|B'B - I| {np.abs(ft.B.T @ ft.B - np.eye(3)).max():.2e}")
    bt = np.array([1.0, 2.0, 3.0]) * 1e-3
    ang = np.degrees(np.arccos(np.clip((ft.B @ bt) @ bt / (bt @ bt), -1, 1)))
    ok &= check("B rotates the bias by the boresight sweep, magnitude preserved",
                abs(np.linalg.norm(ft.B @ bt) - np.linalg.norm(bt)) < 1e-15,
                f"bias turned {ang:.2f} deg while the boresight turned 130.00 deg")

    # -- the chi-square innovation gate -----------------------------------------------------------
    # The case this exists for, verbatim from the Monte Carlo: a lidar range of 11 112 m against a
    # predicted 0.33 m at sigma 3 m, which the ungated filter accepted and turned into a 414 m
    # position jump while reporting sigma 1.3 m throughout.
    print("\n-- innovation gate --")
    def fresh(gate):
        f = EKF.EKF(np.concatenate([np.array([7000.0, 0.0, 0.0]), np.zeros(9)]),
                     np.diag(np.concatenate([np.full(3, 1.3 ** 2), np.full(3, 0.01 ** 2),
                                             np.full(3, 3e-5 ** 2), np.full(3, 1e-9 ** 2)])),
                     fld, CN.imu_params(), wz=0.0, gate_nis=gate)
        return f
    H = np.zeros((1, CN.NX)); H[0, 0] = 1.0
    R = np.array([[3.0 ** 2]])
    f = fresh(CN.EKF_GATE_NIS)
    x0, P0 = f.x.copy(), f.P.copy()
    f.update(np.array([11112.5]), H, R, np.array([0.33]))
    ok &= check("gate rejects an 11 km lidar outlier",
                f.n_rej == 1 and np.array_equal(f.x, x0) and np.array_equal(f.P, P0),
                f"n_rej {f.n_rej}, state moved {np.linalg.norm(f.x - x0):.3e} m")
    f = fresh(CN.EKF_GATE_NIS)
    f.update(np.array([0.33 + 1.0]), H, R, np.array([0.33]))
    ok &= check("gate passes a 1-sigma measurement",
                f.n_rej == 0 and f.n_upd == 1 and np.linalg.norm(f.x - x0) > 0,
                f"n_rej {f.n_rej}, n_upd {f.n_upd}, state moved "
                f"{np.linalg.norm(f.x - x0):.3f} m")
    # gate_nis=None means "take the configured value", so inf is how a caller actually turns the
    # gate off -- nothing exceeds inf, and it needs no sentinel branch in update().
    f = fresh(np.inf)
    f.update(np.array([11112.5]), H, R, np.array([0.33]))
    ok &= check("gate=inf reproduces the old (ungated) behaviour",
                f.n_rej == 0 and np.linalg.norm(f.x - x0) > 100.0,
                f"ungated, the state moves {np.linalg.norm(f.x - x0):.1f} m")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

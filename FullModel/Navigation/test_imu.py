"""Checks on the IMU block. Every assertion here is a property the physics requires, not a
regression against a recorded number. Run: python test_imu.py [--seed N]
"""
from __future__ import annotations

import argparse
import numpy as np

import config_nav as CN
import imu as NIMU


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1, help="RNG seed; recorded in the output")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    g0, ug = CN.G0, CN.UG
    ok = True
    print(f"IMU block checks   seed {a.seed}   grade {CN.IMU_GRADE}   bias frame {CN.IMU_BIAS_FRAME}")

    # -- 1. the Rodrigues rotation ------------------------------------------------------------------
    print("\n1. rotation_between")
    worst_map = worst_orth = worst_det = 0.0
    for _ in range(2000):
        u = NIMU._unit(rng.normal(size=3)); v = NIMU._unit(rng.normal(size=3))
        R = NIMU.rotation_between(u, v)
        worst_map = max(worst_map, np.linalg.norm(R @ u - v))
        worst_orth = max(worst_orth, np.linalg.norm(R.T @ R - np.eye(3)))
        worst_det = max(worst_det, abs(np.linalg.det(R) - 1.0))
    ok &= check("maps d_from onto d_to", worst_map < 1e-12, f"max err {worst_map:.2e}")
    ok &= check("orthogonal", worst_orth < 1e-12, f"max |R'R - I| {worst_orth:.2e}")
    ok &= check("proper (det = +1)", worst_det < 1e-12, f"max |det-1| {worst_det:.2e}")
    u = np.array([0.0, 0.0, 1.0])
    R = NIMU.rotation_between(u, -u)
    ok &= check("anti-parallel case", np.linalg.norm(R @ u + u) < 1e-12 and abs(np.linalg.det(R) - 1) < 1e-12)
    R = NIMU.rotation_between(u, u)
    ok &= check("identical case -> I", np.allclose(R, np.eye(3)))

    # -- 2. the bias is a constant in the thrust frame, and transport preserves its norm -----------
    print("\n2. bias: constant magnitude, transported direction")
    dev = NIMU.IMU(seed=a.seed)
    b0 = dev.bias0.copy()
    n0 = np.linalg.norm(b0)
    # sweep the boresight through the descent's real 129.5 deg in many small steps
    th = np.radians(np.linspace(130.94, 1.43, 20000))
    dirs = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1)
    norms, first, last = [], None, None
    for k, d in enumerate(dirs):
        z = dev.measure(0.00833 * d, 0.01, d_I=d)   # 5 N / 600 kg along the boresight
        norms.append(np.linalg.norm(dev.bias))
        if k == 0:
            first = dev.bias.copy()
        last = dev.bias.copy()
    norms = np.array(norms)
    rel = (norms.max() - norms.min()) / n0
    ok &= check("|b| constant under transport", rel < 1e-12,
                f"relative spread {rel:.2e} over {len(norms)} accumulated rotations")
    ok &= check("|b| equals the draw", abs(norms[0] - n0) < 1e-15, f"{norms[0]:.6e} vs {n0:.6e}")
    ok &= check("b_taf untouched", np.allclose(dev.bias0, b0))
    # The boresight sweeps in the x-z plane, so every transport rotates about y. Only the
    # component PERPENDICULAR to that axis turns; the along-axis component is untouched. So the
    # swing is 2|b_perp|sin(dth/2), and 2|b|sin(dth/2) is only its upper bound (b_perp = b).
    k_ax = np.array([0.0, 1.0, 0.0])
    b_perp = np.linalg.norm(first - (first @ k_ax) * k_ax)
    swing = np.linalg.norm(last - first)
    pred = 2.0 * b_perp * np.sin(np.radians(130.94 - 1.43) / 2.0)
    ok &= check("ACI bias swing = 2|b_perp|sin(dth/2)", abs(swing - pred) < 1e-9 * max(pred, 1e-30),
                f"{swing/ug:.1f} ug measured vs {pred/ug:.1f} ug predicted "
                f"(upper bound 2|b|sin = {2*n0*np.sin(np.radians(129.51)/2)/ug:.1f} ug)")
    ok &= check("along-slew-axis component untouched",
                abs(first @ k_ax - last @ k_ax) < 1e-15 * n0,
                f"drift {abs(first @ k_ax - last @ k_ax):.2e}")

    # -- 3. parallel transport is path dependent, and returning closes the loop --------------------
    print("\n3. transport properties")
    dev = NIMU.IMU(seed=a.seed)
    d_a = NIMU._unit([0.0, 0.0, 1.0]); d_b = NIMU._unit([1.0, 0.0, 1.0])
    dev.measure(0.01 * d_a, 0.01, d_I=d_a); b_start = dev.bias.copy()
    dev.measure(0.01 * d_b, 0.01, d_I=d_b)
    dev.measure(0.01 * d_a, 0.01, d_I=d_a)      # straight back along the same arc
    ok &= check("out-and-back returns the bias", np.linalg.norm(dev.bias - b_start) < 1e-14,
                f"residual {np.linalg.norm(dev.bias - b_start):.2e}")
    dev = NIMU.IMU(seed=a.seed, bias_frame="inertial")
    b_i = dev.bias.copy()
    for d in dirs[::100]:
        dev.measure(0.00833 * d, 0.01, d_I=d)
    ok &= check("bias_frame='inertial' holds b fixed in ACI", np.allclose(dev.bias, b_i))

    # -- 4. white noise: sigma_a = N/sqrt(dt), and the velocity increment is step invariant --------
    print("\n4. white noise scaling")
    N = CN.imu_params()["vrw"]
    d = np.array([0.0, 0.0, 1.0]); a_true = 0.0333 * d
    for dt in (0.5, 0.05, 0.005):
        dev = NIMU.IMU(seed=a.seed)
        dev.measure(a_true, dt, d_I=d)                      # fixes the frame
        b = dev.bias.copy()
        res = np.array([dev.measure(a_true, dt, d_I=d) - a_true - b for _ in range(200000)])
        s_emp, s_th = res.std(), N / np.sqrt(dt)
        ok &= check(f"sigma_a at dt={dt:g} s", abs(s_emp / s_th - 1.0) < 0.01,
                    f"{s_emp:.4e} vs {s_th:.4e} ({100*(s_emp/s_th-1):+.2f}%)")
        # accumulated velocity error over a fixed 100 s, which must not depend on dt
        nstep = int(round(100.0 / dt))
        dv = np.array([np.sum(np.array([dev.measure(a_true, dt, d_I=d) - a_true - b
                                        for _ in range(nstep)]), axis=0) * dt for _ in range(60)])
        ok &= check(f"  sigma_dv over 100 s at dt={dt:g}", abs(dv.std() / (N * 10.0) - 1.0) < 0.25,
                    f"{dv.std():.4e} vs N*sqrt(100) = {N*10.0:.4e}")

    # -- 5. reproducibility and the datasheet numbers ----------------------------------------------
    print("\n5. reproducibility and datasheet values")
    x = NIMU.IMU(seed=7); y = NIMU.IMU(seed=7); z = NIMU.IMU(seed=8)
    ok &= check("same seed -> same bias", np.allclose(x.bias0, y.bias0))
    ok &= check("different seed -> different bias", not np.allclose(x.bias0, z.bias0))
    ok &= check("same seed -> same noise", np.allclose(x.measure(a_true, 0.01), y.measure(a_true, 0.01)))
    p = CN.imu_params("ln200s")
    ok &= check("bias_sigma = 300 ug", abs(p["bias_sigma"] - 300 * ug) < 1e-18,
                f"{p['bias_sigma']:.4e} m/s^2")
    ok &= check("vrw = 35 ug/rtHz", abs(p["vrw"] - 35 * ug) < 1e-18, f"{p['vrw']:.4e}")
    ok &= check("tau = inf (random constant)", np.isinf(p["tau"]))

    # -- 6. tau = inf must feed the EKF without producing nan --------------------------------------
    print("\n6. tau = inf through the EKF's bias terms")
    dtq = 0.01
    beta = np.exp(-dtq / p["tau"])
    qbb = p["bias_inst"] ** 2 * max(1.0 - beta ** 2, 0.0)
    bdot = -np.ones(3) / p["tau"]
    Fbb = -np.eye(3) / p["tau"]
    ok &= check("beta = 1", beta == 1.0)
    ok &= check("Q_bb = 0", qbb == 0.0)
    ok &= check("b_dot = 0, finite", np.all(bdot == 0.0) and np.all(np.isfinite(bdot)))
    ok &= check("F[b,b] = 0, finite", np.all(Fbb == 0.0) and np.all(np.isfinite(Fbb)))

    # -- 7. what the bias costs if unestimated ------------------------------------------------------
    print("\n7. error budget at these numbers (t = 2052 s)")
    T = 2052.0
    print(f"  bias      0.5*b*T^2 = {0.5*p['bias_sigma']*T*T:8.0f} m     (1-sigma draw magnitude)")
    print(f"  noise   N*sqrt(T^3/3) = {N*np.sqrt(T**3/3.0):8.1f} m")
    print(f"  OD vel       dv0*T   = {CN.OD_VEL_SIGMA*T:8.1f} m")
    print(f"  OD pos         dr0   = {CN.OD_POS_SIGMA:8.1f} m")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

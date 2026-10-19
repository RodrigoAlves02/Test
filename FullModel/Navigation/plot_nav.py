"""The EKF run against the flown descent: what it estimates, and how wrong it is.

OPEN LOOP ON PURPOSE. The recorded run is the perfect-states descent -- guidance saw the truth, so
the trajectory is fixed. The filter is flown alongside it: the IMU samples the true specific force,
the camera and lidar take measurements off the true state, and the EKF estimates. Nothing feeds
back into guidance. That isolates the navigation chain from the control loop, so an error here is
the filter's and nothing else's.

WHAT IS ON BOARD versus WHAT IS TRUE, which is the whole point:
    gravity    truth 49 152-facet polyhedral   |   filter PINN-GM-III
    shape      truth 49 152-facet mesh         |   filter the same mesh (survey-phase product)
    bias       one 300 ug draw, thrust-frame   |   filter estimates it, thrust-frame
    position   propagated by the truth         |   filter starts with the OD error and drifts

Run: python plot_nav.py [--seed N] [--no-cam] [--no-lidar]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config_nav as CN
import ekf as NEKF
import frames as NFR
import imu as NIMU
import perception as PERC

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "SCvx", "Figures&data"))


def hermite(r0, v0, r1, v1, h, s):
    """Cubic Hermite on [0, h], evaluated at s in [0, 1]. Returns (position, velocity).

    The recorded trace carries BOTH position and velocity at every 0.5 s node, which is exactly
    what Hermite needs -- so the truth between nodes is reconstructed rather than held. Holding it
    (a zero-order hold) manufactures a sawtooth of amplitude |v|*dt in the error, 4.3 m at
    handover falling to 0.15 m at touchdown, which is the recording and not the filter.

    The interpolation error is h^4/384 * max|D4|, where D4 is the fourth time derivative of the
    position. The thrust is constant across a sub-step and gravity varies on a kilometre scale, so
    |D4| ~ 4e-11 m/s^4 and the error is ~7e-15 m -- fifteen orders below the metre-level
    quantities being plotted.

    Done in ACAF and rotated afterwards, not the other way round: the body-frame trajectory has
    lower curvature (no w x r sweep added to it) and the rotation is exact at every instant.
    """
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    r = h00 * r0 + h10 * h * v0 + h01 * r1 + h11 * h * v1
    d00 = 6 * s2 - 6 * s
    d10 = 3 * s2 - 4 * s + 1
    d01 = -6 * s2 + 6 * s
    d11 = 3 * s2 - 2 * s
    v = (d00 * r0 + d01 * r1) / h + d10 * v0 + d11 * v1
    return r, v


def bands(t, mask):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    w, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != p + 1:
            w.append((t[s], t[p])); s = i
        p = i
    w.append((t[s], t[p]))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="closed_loop_imuplot.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-cam", action="store_true")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--gravity", choices=("pinn", "poly"), default="pinn",
                    help="the ON-BOARD field. truth is always the 49k polyhedral")
    ap.add_argument("--record-every", type=int, default=5,
                help="store every Nth IMU sample (the truth is Hermite-interpolated to each)")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    sys.path.insert(0, CN.SCVX_DIR)
    import gravity as GR
    fld = GR.pinn_field() if a.gravity == "pinn" else GR.field()
    r_site, n_hat = GR.site_and_normal()
    wz = CN.w_body_z()

    path = a.run if os.path.isabs(a.run) else os.path.join(DATA, a.run)
    tr = json.load(open(path))["trace"]
    dt_act = tr[1]["t"] - tr[0]["t"]
    dt_imu = 1.0 / CN.IMU_RATE_HZ
    n_sub = max(1, int(round(dt_act / dt_imu)))

    rng = np.random.default_rng(a.seed)
    imu = NIMU.IMU(seed=a.seed)
    x0, P0, dr0, dv0 = NEKF.initial_state_and_cov(
        np.array(tr[0]["r"], float), np.array(tr[0]["v"], float), rng, t=tr[0]["t"], wz=wz)
    nav = NEKF.EKF(x0, P0, fld, imu.p, wz=wz)

    sensors = []
    if not a.no_cam:
        sensors += [("cam+", PERC.Camera(sign=+1, name="cam+")),
                    ("cam-", PERC.Camera(sign=-1, name="cam-"))]
    if not a.no_lidar:
        sensors += [("lidar", PERC.Lidar(name="lidar"))]
    nxt = {k: tr[0]["t"] for k, _ in sensors}
    hits = {k: 0 for k, _ in sensors}

    print(f"seed {a.seed}   on-board gravity '{a.gravity}'   IMU {CN.IMU_GRADE} @ "
          f"{CN.IMU_RATE_HZ:g} Hz   bias frame '{CN.EKF_BIAS_FRAME}'")
    print(f"OD handover error: |dr| {np.linalg.norm(dr0):.3f} m, |dv| {np.linalg.norm(dv0):.5f} m/s")
    print(f"true IMU bias |b| {np.linalg.norm(imu.bias0):.4e} m/s^2 "
          f"({np.linalg.norm(imu.bias0)/CN.UG:.1f} ug)")
    print(f"sensors: {[k for k, _ in sensors]}   propagating {len(tr)*n_sub} IMU samples\n")

    rec = dict(t=[], er=[], ev=[], eb=[], sp=[], sv=[], sb=[], alt=[], bt=[], bh=[])
    meas = []
    t0 = time.perf_counter()
    for i, row in enumerate(tr):
        r_rot = np.array(row["r"], float)
        v_rot = np.array(row["v"], float)
        d_rot = np.array(row["d_act"], float)
        a_rot = row["T_act"] / row["mass"] * d_rot
        for j in range(n_sub):
            tj = row["t"] + j * dt_imu
            d_I = NFR.vec_rot_to_inertial(d_rot, tj, wz)
            z_imu = imu.measure(NFR.vec_rot_to_inertial(a_rot, tj, wz), dt_imu, d_I=d_I)
            nav.propagate(z_imu, dt_imu, tj, d_I=d_I)

            for key, dev in sensors:
                if tj < nxt[key] or not dev.active(r_rot, n_hat, r_site, d_rot):
                    continue
                nxt[key] = tj + 1.0 / dev.rate
                z, H, R = dev.measure(r_rot, n_hat, r_site, rng, thrust_dir=d_rot, t=tj)
                h = dev.predict(nav.x, n_hat, r_site, thrust_dir=d_rot, t=tj) \
                    if isinstance(dev, PERC.Lidar) else dev.predict(nav.x, n_hat, r_site)
                if not np.all(np.isfinite(h)):
                    continue                       # the on-board shape sees nothing there
                y = float(np.linalg.norm(np.atleast_1d(z) - np.atleast_1d(h)))
                meas.append(dict(t=tj, s=key, y=y, sig=float(np.sqrt(R[0, 0]))))
                nav.update(z, H, R, h)
                hits[key] += 1

            if (i * n_sub + j) % a.record_every:
                continue
            # THE TRUTH AT THIS INSTANT, reconstructed by cubic Hermite from the bracketing nodes
            # and then rotated into ACI. Never held -- see hermite() for why that matters.
            if i + 1 < len(tr):
                rb, vb = hermite(r_rot, v_rot, np.array(tr[i + 1]["r"], float),
                                 np.array(tr[i + 1]["v"], float), dt_act, j * dt_imu / dt_act)
            else:
                rb, vb = r_rot, v_rot
            Rz = NFR.Rz(wz * tj)
            rt = Rz @ rb
            vt = Rz @ vb + np.cross([0, 0, wz], rt)
            rec["t"].append(tj)
            rec["er"].append(nav.x[CN.IR] - rt)
            rec["ev"].append(nav.x[CN.IV] - vt)
            rec["eb"].append(nav.bias_inertial - imu.bias)
            rec["sp"].append(nav.sigma[CN.IR].copy())
            rec["sv"].append(nav.sigma[CN.IV].copy())
            rec["sb"].append(nav.sigma[CN.IB].copy())
            rec["alt"].append(row["alt"])
            rec["bt"].append(imu.bias.copy())
            rec["bh"].append(nav.bias_inertial.copy())
    wall = time.perf_counter() - t0
    for k in rec:
        rec[k] = np.array(rec[k])
    t = rec["t"]

    print(f"propagated in {wall:.1f} s   updates: " +
          "  ".join(f"{k} {v}" for k, v in hits.items()))
    for lab, e, s in (("position [m]", rec["er"], rec["sp"]),
                      ("velocity [m/s]", rec["ev"], rec["sv"]),
                      ("bias [m/s^2]", rec["eb"], rec["sb"])):
        n_e = np.linalg.norm(e, axis=1)
        print(f"  {lab:16s} final |err| {n_e[-1]:12.5e}   max {n_e.max():12.5e}   "
              f"final sigma {np.linalg.norm(s[-1]):12.5e}")

    # ---- figure ---------------------------------------------------------------------------------
    axc = ("#1f77b4", "#d62728", "#2ca02c"); lab = ("x", "y", "z")
    scol = {"cam+": "#1f77b4", "cam-": "#d62728", "lidar": "#2ca02c"}
    fig = plt.figure(figsize=(13.5, 15.5))
    gs = fig.add_gridspec(5, 2, hspace=0.45, wspace=0.24, top=0.935)
    fig.suptitle(
        f"EKF on the flown descent (open loop)   -   {os.path.basename(path)}\n"
        f"ACI frame, on-board gravity '{a.gravity}', IMU {CN.IMU_GRADE} "
        f"({np.linalg.norm(imu.bias0)/CN.UG:.0f} ug bias) @ {CN.IMU_RATE_HZ:g} Hz, "
        f"bias frame '{CN.EKF_BIAS_FRAME}', seed {a.seed}", fontsize=11.5, y=0.985)

    for k, (name, e, s, unit) in enumerate((("position", rec["er"], rec["sp"], "m"),
                                            ("velocity", rec["ev"], rec["sv"], "m/s"))):
        ax = fig.add_subplot(gs[k, :])
        for i in range(3):
            ax.plot(t, e[:, i], "-", lw=0.7, color=axc[i], label=f"$e_{lab[i]}$")
            ax.plot(t, 3 * s[:, i], "--", lw=0.6, color=axc[i], alpha=0.45)
            ax.plot(t, -3 * s[:, i], "--", lw=0.6, color=axc[i], alpha=0.45)
        ax.set_ylabel(f"{name} error [{unit}]"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
        ax.set_title(f"{name} error against the filter's own $\\pm3\\sigma$ (dashed)", fontsize=10)

    ax = fig.add_subplot(gs[2, :])
    for i in range(3):
        ax.plot(t, rec["bt"][:, i] / CN.UG, "-", lw=0.9, color=axc[i], label=f"true $b_{lab[i]}$")
        ax.plot(t, rec["bh"][:, i] / CN.UG, "--", lw=0.8, color=axc[i], alpha=0.85,
                label=f"est $\\hat b_{lab[i]}$")
    ax.set_ylabel("bias, ACI [ug]"); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=6)
    ax.set_title("the accelerometer bias the filter is chasing: truth (solid) and estimate (dashed)",
                 fontsize=10)

    ax = fig.add_subplot(gs[3, 0])
    ax.semilogy(t, np.linalg.norm(rec["er"], axis=1), "-", lw=0.7, color="k", label="|position err|")
    ax.semilogy(t, np.linalg.norm(rec["sp"], axis=1), "--", lw=0.8, color="0.45", label="filter 1σ")
    for key, _ in sensors:
        m = np.array([mm["t"] for mm in meas if mm["s"] == key])
        if len(m):
            ax.plot(m, np.full(len(m), ax.get_ylim()[0] * 1.6), "|", ms=7, color=scol[key],
                    label=f"{key} ({hits[key]})")
    ax.set_xlabel("t [s]"); ax.set_ylabel("[m]"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5); ax.set_title("position error and what the filter thinks", fontsize=9.5)

    ax = fig.add_subplot(gs[3, 1])
    ax.semilogy(t, np.linalg.norm(rec["eb"], axis=1) / CN.UG, "-", lw=0.7, color="k",
                label="|bias err|")
    ax.semilogy(t, np.linalg.norm(rec["sb"], axis=1) / CN.UG, "--", lw=0.8, color="0.45",
                label="filter 1σ")
    ax.axhline(np.linalg.norm(imu.bias0) / CN.UG, color="#d62728", ls=":", lw=1.4,
               label=f"|b| = {np.linalg.norm(imu.bias0)/CN.UG:.0f} ug (unestimated)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("[ug]"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5); ax.set_title("does the bias become observable?", fontsize=9.5)

    ax = fig.add_subplot(gs[4, 0])
    for key, _ in sensors:
        mm = [m for m in meas if m["s"] == key]
        if mm:
            ax.plot([m["t"] for m in mm], [m["y"] / m["sig"] for m in mm], "o", ms=3.2,
                    mfc="none", mew=1.0, color=scol[key], label=key)
    for k2, ls in ((1, "--"), (3, ":")):
        ax.axhline(k2, color="k", ls=ls, lw=1.0); ax.axhline(-k2, color="k", ls=ls, lw=1.0)
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$|y| / \sigma_{meas}$"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("pre-fit innovations: large means the filter and the sensor disagree", fontsize=9.5)

    ax = fig.add_subplot(gs[4, 1])
    for name, e, s, c in (("pos", rec["er"], rec["sp"], "#1f77b4"),
                          ("vel", rec["ev"], rec["sv"], "#d62728"),
                          ("bias", rec["eb"], rec["sb"], "#2ca02c")):
        ax.semilogy(t, np.linalg.norm(e, axis=1) / np.maximum(np.linalg.norm(s, axis=1), 1e-30),
                    "-", lw=0.7, color=c, label=name)
    ax.axhline(1.0, color="k", ls="--", lw=1.2); ax.axhline(3.0, color="k", ls=":", lw=1.0)
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$|err| / \sigma$"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("consistency: above 3 means the filter is overconfident", fontsize=9.5)

    os.makedirs(a.out, exist_ok=True)
    tag = f"_s{a.seed}" + ("" if not a.no_cam else "_nocam") + ("" if not a.no_lidar else "_nolidar")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"ekf_open{tag}.{ext}"), dpi=140, bbox_inches="tight")
    print(f"\nwrote {os.path.join(a.out, f'ekf_open{tag}.png')}")


if __name__ == "__main__":
    main()

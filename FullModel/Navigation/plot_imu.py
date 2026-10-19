"""What the IMU block does to a real thrust history.

Reads a closed-loop run, rebuilds the TRUE specific force in inertial axes from the recorded
delivered thrust and mass, feeds it through the IMU at its own rate, and plots the result.

The thrust is a zero-order hold over each actuator sub-step -- that is how the truth propagator
applies it -- so the IMU is driven with the recorded value held constant across each sub-step and
sampled n = dt_act/dt_imu times inside it. Nothing here re-runs the solver; it is pure
post-processing of a recorded run, so the seed and the bias frame can be swept for free.

The two panels that are a TEST rather than a picture are the bottom pair:
  * angle(b, d) must be CONSTANT -- parallel transport carries the bias with the boresight, so
    their included angle cannot change. Any drift means the transport is wrong.
  * |b| must be FLAT -- the bias is a random constant, only its direction moves.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config_nav as CN
import frames as NFR
import imu as NIMU

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "SCvx", "Figures&data"))


def build(run, seed, frame, dt_imu, dt_act):
    """Drive the IMU with the recorded thrust history. Returns per-sample arrays."""
    tr = run["trace"]
    need = ("d_act", "mass")
    missing = [k for k in need if k not in tr[0]]
    if missing:
        raise SystemExit(f"trace lacks {missing}: this run predates the extended trace. "
                         f"Re-run closed_loop.py to record the 3D thrust direction.")
    wz = _wz()
    dev = NIMU.IMU(seed=seed, bias_frame=frame)
    n_sub = max(1, int(round(dt_act / dt_imu)))
    T, A_true, A_meas, B, D, ANG = [], [], [], [], [], []
    for row in tr:
        t0 = float(row["t"])
        d_rot = np.asarray(row["d_act"], float)
        a_rot = float(row["T_act"]) / float(row["mass"]) * d_rot      # specific force, ACAF
        for j in range(n_sub):
            t = t0 + j * dt_imu
            a_I = NFR.vec_rot_to_inertial(a_rot, t, wz)
            d_I = NFR.vec_rot_to_inertial(d_rot, t, wz)
            # d_I is passed EXPLICITLY rather than inferred from a_I: the recorded direction is
            # known even when the engine is off, and inferring it would silently freeze the frame
            # through any zero-thrust stretch.
            z_I = dev.measure(a_I, dt_imu, d_I=d_I)
            T.append(t); A_true.append(a_I); A_meas.append(z_I)
            # the boresight comes from the RECORDED trajectory, not from dev.d: it exists whatever
            # frame the bias is modelled in, and with bias_frame='inertial' the device never
            # tracks it at all (which is the whole point of that comparison).
            B.append(dev.bias); D.append(d_I)
            ANG.append(float(row["ang_act"]))
    return (np.array(T), np.array(A_true), np.array(A_meas), np.array(B),
            np.array(D), np.array(ANG), dev)


def _wz():
    """Eros spin rate about +z, from the SCvx config so there is one definition of it."""
    import sys
    sys.path.insert(0, CN.SCVX_DIR)
    import config as C
    return float(C.W_BODY[2])


def ang_between(u, v):
    num = np.einsum("ij,ij->i", u, v)
    den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    return np.degrees(np.arccos(np.clip(num / np.maximum(den, 1e-300), -1.0, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="closed_loop_imuplot.json")
    ap.add_argument("--seed", type=int, default=1, help="IMU seed, recorded in the title")
    ap.add_argument("--frame", default=None, choices=(None, "thrust", "inertial"))
    ap.add_argument("--dt-imu", type=float, default=None, help="default 1/IMU_RATE_HZ")
    ap.add_argument("--dt-act", type=float, default=None, help="default: from the run")
    ap.add_argument("--every", type=int, default=20, help="decimation for the scatter layers")
    ap.add_argument("--zoom", type=float, nargs=2, default=None,
                    help="window [s] for the zoom panel; default is 60 s at the 20 N pulse")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    path = a.run if os.path.isabs(a.run) else os.path.join(DATA, a.run)
    run = json.load(open(path))
    frame = a.frame or CN.IMU_BIAS_FRAME
    dt_imu = a.dt_imu or 1.0 / CN.IMU_RATE_HZ
    dt_act = a.dt_act or float(run["trace"][1]["t"] - run["trace"][0]["t"]) or 0.5

    t, a_true, a_meas, b, d, ang, dev = build(run, a.seed, frame, dt_imu, dt_act)
    err = a_meas - a_true
    mag_t = np.linalg.norm(a_true, axis=1)
    mag_m = np.linalg.norm(a_meas, axis=1)
    nb = np.linalg.norm(b, axis=1)
    bd = ang_between(b, d)
    turn_d = ang_between(d, np.broadcast_to(d[0], d.shape))
    turn_b = ang_between(b, np.broadcast_to(b[0], b.shape))
    e = max(1, a.every)
    ug = CN.UG

    if a.zoom is None:
        k = int(np.argmax(mag_t)); t0 = max(t[0], t[k] - 30.0)
        zoom = (t0, min(t[-1], t0 + 60.0))
    else:
        zoom = tuple(a.zoom)
    zm = (t >= zoom[0]) & (t <= zoom[1])

    fig = plt.figure(figsize=(13.5, 15.0))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0],
                          hspace=0.42, wspace=0.22)
    col = ("#1f77b4", "#d62728", "#2ca02c")
    lab = ("x", "y", "z")

    # -- 1. specific-force magnitude, true vs measured -------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t[::e], mag_m[::e], ".", ms=1.0, color="0.72", rasterized=True,
            label=f"measured  |a_meas|  (every {e}th sample)")
    ax.plot(t, mag_t, "-", lw=1.6, color="k", label="true  |a_true| = |T|/m")
    ax.set_ylabel("specific force  [m/s$^2$]")
    ax.set_title(f"IMU on the flown thrust history   -   {os.path.basename(path)}   "
                 f"grade {CN.IMU_GRADE}, bias frame '{frame}', seed {a.seed}, "
                 f"{1/dt_imu:.0f} Hz\n"
                 f"|b| = {nb[0]/ug:.1f} ug = {nb[0]:.3e} m/s$^2$    "
                 f"$\\sigma_a$ = N/$\\sqrt{{dt}}$ = {CN.imu_params()['vrw']/np.sqrt(dt_imu):.3e} m/s$^2$",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3)
    ax.axvspan(zoom[0], zoom[1], color="#ff7f0e", alpha=0.12, lw=0)

    # -- 2. the zoom: at 100 Hz the per-sample noise is comparable to the signal -----------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t[zm], mag_m[zm], ".", ms=2.0, color="0.62", label="measured")
    ax.plot(t[zm], mag_t[zm], "-", lw=2.0, color="k", label="true")
    ax.set_xlabel("t [s]"); ax.set_ylabel("|a|  [m/s$^2$]")
    ax.set_title(f"zoom {zoom[0]:.0f}-{zoom[1]:.0f} s: per-sample noise vs signal", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # -- 3. error = bias + white noise, per inertial axis ----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for i in range(3):
        ax.plot(t[::e], err[::e, i] / ug, ".", ms=0.8, color=col[i], alpha=0.25, rasterized=True)
        ax.plot(t, b[:, i] / ug, "-", lw=1.8, color=col[i], label=f"bias$_{lab[i]}$ (ACI)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("a_meas - a_true  [ug]")
    ax.set_title("measurement error: bias (bold) inside the white-noise cloud", fontsize=9.5)
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)

    # -- 4. the bias in inertial axes, which is what the question is about -----------------------
    ax = fig.add_subplot(gs[2, :])
    for i in range(3):
        ax.plot(t, b[:, i] / ug, "-", lw=1.7, color=col[i], label=f"$b_{lab[i]}$ (ACI)")
    ax.plot(t, nb / ug, "--", lw=1.4, color="k", label="$|b|$ (must be flat)")
    ax.set_ylabel("bias  [ug]"); ax.grid(alpha=0.3)
    ax.set_title("the accelerometer bias in inertial axes: " + (
                 "constant magnitude, direction carried by the boresight" if frame == "thrust"
                 else "frozen in ACI -- the previous model, shown for contrast"), fontsize=10)
    ax.legend(fontsize=8.5, ncol=4)

    # -- 5. boresight vs bias direction ----------------------------------------------------------
    ax = fig.add_subplot(gs[3, :])
    for i in range(3):
        ax.plot(t, d[:, i], "-", lw=1.5, color=col[i], label=f"$\\hat d_{lab[i]}$ (ACI)")
        ax.plot(t, b[:, i] / nb, "--", lw=1.2, color=col[i], alpha=0.75,
                label=f"$\\hat b_{lab[i]}$")
    ax.set_ylabel("unit-vector components"); ax.grid(alpha=0.3)
    ax.set_title("boresight (solid) and bias direction (dashed) in ACI: " + (
                 "the bias follows the thrust" if frame == "thrust"
                 else "the bias ignores the thrust"), fontsize=10)
    ax.legend(fontsize=7.5, ncol=6)

    # -- 6. the two tests ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(t, bd, "-", lw=1.6, color="#9467bd")
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\angle(b,\hat d)$  [deg]")
    ax.set_title((f"TEST: included angle must be CONSTANT" if frame == "thrust"
                  else "included angle is NOT constant (bias frozen in ACI)")
                 + f"\nspread {bd.max()-bd.min():.2e} deg about {bd.mean():.3f} deg", fontsize=9.5)
    ax.grid(alpha=0.3)
    lo, hi = bd.min(), bd.max()
    pad = max(1e-9, 0.05 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)

    ax = fig.add_subplot(gs[4, 1])
    ax.plot(t, turn_d, "-", lw=1.7, color="k", label=r"$\hat d$ turned from $\hat d(0)$")
    ax.plot(t, turn_b, "-", lw=1.7, color="#9467bd", label=r"$\hat b$ turned from $\hat b(0)$")
    ax.plot(t, ang, ":", lw=1.3, color="0.5", label="thrust angle from local vertical (ACAF)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("[deg]")
    ax.set_title((r"rotation carried: $\hat b$ turns with $\hat d$, by $\leq$ as much"
                  "\n(only the part perpendicular to the slew axis turns)") if frame == "thrust"
                 else "no rotation carried: $\\hat b$ stays put while $\\hat d$ slews", fontsize=9.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    os.makedirs(a.out, exist_ok=True)
    tag = f"_{frame}_s{a.seed}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"imu_block{tag}.{ext}"), dpi=140, bbox_inches="tight")
    print(f"wrote {os.path.join(a.out, f'imu_block{tag}.png')}")

    print(f"\nrun            : {os.path.basename(path)}   tf {t[-1]:.1f} s, {len(t)} IMU samples")
    print(f"seed           : {a.seed}   bias frame '{frame}'   rate {1/dt_imu:.0f} Hz")
    print(f"|b|            : {nb[0]/ug:.2f} ug = {nb[0]:.4e} m/s^2   "
          f"flatness {nb.max()-nb.min():.2e} ({(nb.max()-nb.min())/nb[0]:.1e} rel)")
    print(f"angle(b, d)    : {bd.mean():.4f} deg   spread {bd.max()-bd.min():.3e} deg")
    print(f"boresight turn : {turn_d.max():.2f} deg      bias direction turn: {turn_b.max():.2f} deg")
    print(f"sigma_a        : {CN.imu_params()['vrw']/np.sqrt(dt_imu):.4e} m/s^2   "
          f"empirical {err.std():.4e}   (bias removed: {(err-b).std():.4e})")
    print(f"|a_true|       : {mag_t.min():.4e} .. {mag_t.max():.4e} m/s^2   "
          f"bias/|a| = {100*nb[0]/mag_t.min():.1f} % .. {100*nb[0]/mag_t.max():.1f} %")
    print(f"mean error     : {(err.mean(axis=0)/ug)} ug   (should be the time-mean bias, "
          f"{b.mean(axis=0)/ug} ug)")


if __name__ == "__main__":
    main()

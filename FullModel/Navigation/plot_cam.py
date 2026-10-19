"""What the camera block does on a real descent.

Reads a closed-loop run, evaluates the FOV geometry against the REAL 49 152-facet shape at every
recorded sub-step, and takes a position fix from whichever camera can see the surface at its own
update rate. Nothing re-runs the solver.

A NOTE ON FRAMES, because it runs the opposite way to the IMU's. Landmarks are fixed to the body,
so the sensor itself works in BODY-FIXED (ACAF) axes: the raw fix is already a position in ACAF,
with no rotation anywhere. The IMU is the mirror image -- it senses in ACI and has to be rotated in.
The BLOCK OUTPUT is still ACI, because that is the frame the filter runs in: Camera.measure takes a
time and rotates the fix (and its covariance) on the way out. This plot deliberately stays in ACAF,
because it is comparing the sensor against the ACAF truth the run recorded.

The panel that is a TEST rather than a picture is the error-vs-sigma one: |z - r_true| must sit
inside the 1-sigma envelope about 68 % of the time per axis, and the envelope itself must track the
imaging range, not the altitude.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config_nav as CN
import perception as PERC

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "SCvx", "Figures&data"))


def site():
    sys.path.insert(0, CN.SCVX_DIR)
    import gravity as GR
    return GR.site_and_normal()


def sweep(trace, cams, stride):
    """Continuous geometry for every camera, subsampled by `stride`."""
    rows = trace[::stride]
    t = np.array([r["t"] for r in rows])
    alt = np.array([r["alt"] for r in rows])
    R = np.array([r["r"] for r in rows])
    D = np.array([r["d_act"] for r in rows])
    out = {}
    for c in cams:
        see = np.zeros(len(rows), bool); img = np.zeros(len(rows)); hit = np.zeros(len(rows), bool)
        ang = np.zeros(len(rows))
        for k in range(len(rows)):
            s, g, h, a = c.geometry(R[k], D[k])
            see[k], img[k], hit[k], ang[k] = s, g, h, a
        rgate = (img >= c.min_range) & (img <= c.max_range)
        out[c.name] = dict(see=see & rgate, img=img, hit=hit, ang=ang)
    return t, alt, R, D, out


def fixes(trace, cams, rng):
    """One position fix from each camera that can see the surface, at its own update rate."""
    nxt = {c.name: trace[0]["t"] for c in cams}
    rec = {c.name: dict(t=[], z=[], r=[], sig=[]) for c in cams}
    for row in trace:
        t, r, d = float(row["t"]), np.array(row["r"], float), np.array(row["d_act"], float)
        for c in cams:
            if t < nxt[c.name] or not c.active(r, None, None, d):
                continue
            nxt[c.name] = t + 1.0 / c.rate
            z, _, Rk = c.measure(r, None, None, rng, thrust_dir=d)
            rec[c.name]["t"].append(t); rec[c.name]["z"].append(z); rec[c.name]["r"].append(r)
            rec[c.name]["sig"].append(float(np.sqrt(Rk[0, 0])))
    for k in rec:
        for f in rec[k]:
            rec[k][f] = np.array(rec[k][f])
    return rec


def bands(t, mask):
    """Contiguous True runs of `mask` as (t_start, t_end) pairs."""
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
    ap.add_argument("--seed", type=int, default=1, help="camera noise seed, recorded in the title")
    ap.add_argument("--model", default=None, choices=(None, "twoterm", "flown", "floor"))
    ap.add_argument("--stride", type=int, default=4, help="sub-sampling for the continuous geometry")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    path = a.run if os.path.isabs(a.run) else os.path.join(DATA, a.run)
    run = json.load(open(path))
    tr = run["trace"]
    if "d_act" not in tr[0] or "r" not in tr[0]:
        raise SystemExit("trace lacks d_act / r: re-run closed_loop.py with the extended trace.")
    model = a.model or CN.CAM_SIGMA_MODEL
    cams = [PERC.Camera(sign=+1, sigma_model=model, name="fwd (+thrust)"),
            PERC.Camera(sign=-1, sigma_model=model, name="aft (-thrust)")]
    r_site, n_hat = site()

    t, alt, R, D, geo = sweep(tr, cams, a.stride)
    rec = fixes(tr, cams, np.random.default_rng(a.seed))
    col = {cams[0].name: "#1f77b4", cams[1].name: "#d62728"}
    axc = ("#1f77b4", "#d62728", "#2ca02c")
    lab = ("x", "y", "z")
    anyone = geo[cams[0].name]["see"] | geo[cams[1].name]["see"]

    fig = plt.figure(figsize=(13.5, 16.0))
    gs = fig.add_gridspec(5, 2, hspace=0.45, wspace=0.22, top=0.945)
    fig.suptitle(
        f"Camera block on the flown descent   -   {os.path.basename(path)}\n"
        f"FOV {CN.CAM_FOV_DEG} deg (ONC-W1), position fix, noise model '{model}'"
        + ({"twoterm": f" = sqrt(({CN.CAM_BEARING_SIGMA*1e6:.0f} urad x rho)^2 + "
                      f"({CN.CAM_SIGMA_FLOOR_M:.1f} m)^2)",
            "flown": f" = {CN.CAM_SIGMA_FRAC*100:.2f} % of imaging range",
            "floor": f" = {CN.CAM_BEARING_SIGMA*1e6:.0f} urad x rho"}[model])
        + f", {1/cams[0].rate:.0f} s cadence, seed {a.seed}", fontsize=11, y=0.985)

    # -- 1. who can see the surface, against altitude ---------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, alt / 1e3, "-", lw=1.6, color="k", label="altitude")
    for c in cams:
        for i, (b0, b1) in enumerate(bands(t, geo[c.name]["see"])):
            ax.axvspan(b0, b1, color=col[c.name], alpha=0.16, lw=0,
                       label=c.name if i == 0 else None)
    for i, (b0, b1) in enumerate(bands(t, ~anyone)):
        ax.axvspan(b0, b1, color="0.25", alpha=0.35, lw=0,
                   label="NEITHER" if i == 0 else None)
        ax.text(0.5 * (b0 + b1), alt.max() / 1e3 * 0.75, f"{b1-b0:.0f} s\nblind",
                ha="center", fontsize=8.5, color="0.15")
    ax.set_ylabel("altitude [km]"); ax.grid(alpha=0.3); ax.legend(fontsize=8.5, ncol=3)
    ax.set_title("which camera can see the surface, evaluated on the real 49 152-facet shape "
                 "(ray hit OR visible vertex inside the FOV cone)", fontsize=10)

    # -- 2. true position and the fixes taken -----------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    for i in range(3):
        ax.plot(t, R[:, i] / 1e3, "-", lw=1.4, color=axc[i], label=f"$r_{lab[i]}$ true")
    for c in cams:
        if len(rec[c.name]["t"]):
            for i in range(3):
                ax.plot(rec[c.name]["t"], rec[c.name]["z"][:, i] / 1e3, "o", ms=3.4,
                        mfc="none", mew=1.0, color=axc[i])
    ax.set_ylabel("position, ACAF [km]"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, ncol=3)
    ax.set_title("true position (lines) and the camera fixes (circles). "
                 "Landmarks are body-fixed, so the measurement is already in ACAF -- no rotation",
                 fontsize=10)

    # -- 3. the error, per axis, against the 1-sigma envelope --------------------------------------
    ax = fig.add_subplot(gs[2, :])
    sig_c = {c.name: c.sigma(np.clip(geo[c.name]["img"], 0, c.max_range)) for c in cams}
    for c in cams:
        m = geo[c.name]["see"]
        ss = np.where(m, sig_c[c.name], np.nan)
        ax.fill_between(t, -ss, ss, color=col[c.name], alpha=0.16, lw=0,
                        label=f"$\\pm1\\sigma$  {c.name}")
    for c in cams:
        if len(rec[c.name]["t"]):
            e = rec[c.name]["z"] - rec[c.name]["r"]
            for i in range(3):
                ax.plot(rec[c.name]["t"], e[:, i], "o", ms=4.0, color=axc[i],
                        mfc="none", mew=1.1, label=f"$e_{lab[i]}$" if c is cams[0] and i < 3 else None)
    ax.set_ylabel("z - r_true  [m]"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=5)
    ax.set_title("TEST: the fix errors must sit inside the envelope, and the envelope must follow "
                 "the IMAGING RANGE, not the altitude", fontsize=10)

    # -- 4. sigma against the alternatives ---------------------------------------------------------
    ax = fig.add_subplot(gs[3, 0])
    for c in cams:
        m = geo[c.name]["see"]
        ax.semilogy(t[m], sig_c[c.name][m], ".", ms=2.2, color=col[c.name], label=c.name)
    rng_all = np.where(anyone, np.minimum(geo[cams[0].name]["img"], geo[cams[1].name]["img"]), np.nan)
    ax.semilogy(t, CN.CAM_BEARING_SIGMA * rng_all, "--", lw=1.2, color="0.45",
                label=f"pixel term {CN.CAM_BEARING_SIGMA*1e6:.0f} urad x rho (autoNGC 0.1 px)")
    ax.axhline(CN.CAM_SIGMA_FLOOR_M, color="0.45", ls="-.", lw=1.2,
               label=f"floor {CN.CAM_SIGMA_FLOOR_M:.1f} m (OSIRIS-REx NFT, map+attitude)")
    ax.axhline(CN.OD_POS_SIGMA, color="k", ls=":", lw=1.3,
               label=f"OD handover {CN.OD_POS_SIGMA:.0f} m")
    ax.set_xlabel("t [s]"); ax.set_ylabel("1-sigma [m]"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5); ax.set_title("fix accuracy vs the OD it has to beat", fontsize=9.5)

    # -- 5. the geometry that drives all of it -----------------------------------------------------
    ax = fig.add_subplot(gs[3, 1])
    for c in cams:
        ax.plot(t, geo[c.name]["ang"], "-", lw=1.4, color=col[c.name], label=f"{c.name}")
    ax.axhline(CN.CAM_FOV_DEG / 2, color="k", ls="--", lw=1.3,
               label=f"half-FOV {CN.CAM_FOV_DEG/2:.2f} deg")
    ax.set_ylim(0, 180); ax.set_xlabel("t [s]"); ax.set_ylabel("angle to nearest surface [deg]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("boresight to the nearest visible surface point\n(0 where the ray is on the surface)",
                 fontsize=9.5)

    # -- 6. imaging range, and the normalised errors -----------------------------------------------
    ax = fig.add_subplot(gs[4, 0])
    for c in cams:
        m = geo[c.name]["see"]
        ax.semilogy(t[m], geo[c.name]["img"][m], ".", ms=2.2, color=col[c.name], label=c.name)
    ax.semilogy(t, np.maximum(alt, 1e-2), "-", lw=1.2, color="k", label="altitude")
    ax.set_xlabel("t [s]"); ax.set_ylabel("range [m]"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("imaging range vs altitude: they are NOT the same\n"
                 "(the frame looks at terrain off to the side)", fontsize=9.5)

    ax = fig.add_subplot(gs[4, 1])
    allz = []
    for c in cams:
        if len(rec[c.name]["t"]):
            e = (rec[c.name]["z"] - rec[c.name]["r"]) / rec[c.name]["sig"][:, None]
            allz.append(e.ravel())
            ax.plot(rec[c.name]["t"], e, "o", ms=3.4, mfc="none", mew=1.0, color=col[c.name])
    allz = np.concatenate(allz) if allz else np.zeros(0)
    for k, ls in ((1, "--"), (3, ":")):
        ax.axhline(k, color="k", ls=ls, lw=1.0); ax.axhline(-k, color="k", ls=ls, lw=1.0)
    inside = float(np.mean(np.abs(allz) <= 1.0)) if allz.size else float("nan")
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$(z-r)/\sigma$")
    ax.set_title(f"normalised error: {100*inside:.0f} % within $1\\sigma$ "
                 f"(expect 68 %), n = {allz.size}\nmean {allz.mean():+.3f}, std {allz.std():.3f} "
                 f"(expect 0, 1)", fontsize=9.5)
    ax.grid(alpha=0.3)

    os.makedirs(a.out, exist_ok=True)
    tag = f"_{model}_s{a.seed}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"cam_block{tag}.{ext}"), dpi=140, bbox_inches="tight")
    print(f"wrote {os.path.join(a.out, f'cam_block{tag}.png')}")

    print(f"\nrun         : {os.path.basename(path)}   tf {t[-1]:.1f} s")
    print(f"noise model : '{model}'   seed {a.seed}   cadence {1/cams[0].rate:.0f} s")
    for c in cams:
        g = geo[c.name]
        print(f"{c.name:15s}: sees surface {100*g['see'].mean():5.1f} % of the flight, "
              f"boresight ON surface {100*g['hit'].mean():5.1f} %, {len(rec[c.name]['t']):3d} fixes")
        for b0, b1 in bands(t, g["see"]):
            print(f"                 active {b0:7.1f} -> {b1:7.1f} s")
    print(f"BLIND (neither): {100*(~anyone).mean():.1f} % of the flight")
    for b0, b1 in bands(t, ~anyone):
        m = (t >= b0) & (t <= b1)
        print(f"                 {b0:7.1f} -> {b1:7.1f} s  ({b1-b0:5.1f} s)   "
              f"altitude {alt[m].max():8.0f} -> {alt[m].min():8.0f} m")
    if allz.size:
        print(f"normalised err : mean {allz.mean():+.4f}  std {allz.std():.4f}  "
              f"{100*inside:.1f} % inside 1 sigma (expect 68.3 %)")
    for c in cams:
        if len(rec[c.name]["t"]):
            s = rec[c.name]["sig"]
            print(f"{c.name:15s}: sigma {s.min():.3f} .. {s.max():.3f} m")


if __name__ == "__main__":
    main()

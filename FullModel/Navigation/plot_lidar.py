"""What the LIDAR block does on a real descent: the true slant range against the measured one.

Reads a closed-loop run, ray-casts the -thrust boresight against the real 49 152-facet shape at
every recorded sub-step, and takes a range measurement wherever all three gates pass. Nothing
re-runs the solver, so the seed and the gate can be swept for free.

WHAT IS PLOTTED AS "TRUE z". The measurement z IS the slant range, so the true value is the
ray-cast distance from the vehicle to the first surface the beam meets. It is deliberately shown
against the ALTITUDE as well, because the two are not the same quantity and the difference is the
whole point of ray-casting instead of using a tangent plane.

The panel that is a TEST rather than a picture is the normalised-error one: (z - h_true)/sigma must
be N(0,1), with sigma^2 including the q^2/12 the 0.5 m quantiser contributes.
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
    ap.add_argument("--seed", type=int, default=1, help="lidar noise seed, recorded in the title")
    ap.add_argument("--inc-max", type=float, default=None, help="incidence gate [deg]")
    ap.add_argument("--stride", type=int, default=2, help="sub-sampling for the continuous sweep")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    path = a.run if os.path.isabs(a.run) else os.path.join(DATA, a.run)
    tr = json.load(open(path))["trace"]
    if "d_act" not in tr[0] or "r" not in tr[0]:
        raise SystemExit("trace lacks d_act / r: re-run closed_loop.py with the extended trace.")
    dev = PERC.Lidar(inc_max=a.inc_max if a.inc_max is not None else CN.LIDAR_INCIDENCE_MAX_DEG)

    rows = tr[::max(1, a.stride)]
    t = np.array([r["t"] for r in rows]); alt = np.array([r["alt"] for r in rows])
    hit = np.zeros(len(rows), bool); rngm = np.full(len(rows), np.nan)
    inc = np.full(len(rows), np.nan)
    for k, row in enumerate(rows):
        h, d, i, _ = dev.geometry(np.array(row["r"], float), np.array(row["d_act"], float))
        hit[k], rngm[k], inc[k] = h, d, i
    g_inc = hit & (inc <= dev.inc_max)
    g_rng = hit & (rngm >= dev.min_range) & (rngm <= dev.max_range)
    on = hit & g_inc & g_rng
    sig = dev.sigma(np.where(np.isfinite(rngm), rngm, dev.max_range))

    # measurements at the instrument's own rate
    rng = np.random.default_rng(a.seed)
    nxt, mt, mz, mh, ms = tr[0]["t"], [], [], [], []
    for row in tr:
        tt = float(row["t"])
        if tt < nxt:
            continue
        r = np.array(row["r"], float); d = np.array(row["d_act"], float)
        if not dev.active(r, None, None, d):
            continue
        nxt = tt + 1.0 / dev.rate
        z, H, R = dev.measure(r, None, None, rng, thrust_dir=d)
        mt.append(tt); mz.append(float(z[0])); mh.append(dev.geometry(r, d)[1])
        ms.append(float(np.sqrt(R[0, 0])))
    mt, mz, mh, ms = map(np.array, (mt, mz, mh, ms))
    err = mz - mh
    nrm = err / ms

    fig = plt.figure(figsize=(13.5, 15.0))
    gs = fig.add_gridspec(5, 2, hspace=0.46, wspace=0.22, top=0.935)
    fig.suptitle(f"LIDAR block on the flown descent   -   {os.path.basename(path)}\n"
                 f"one beam along -thrust, Hayabusa2 LIDAR (Mizuno et al. 2017): "
                 f"{dev.min_range:.0f} m - {dev.max_range/1e3:.0f} km, "
                 f"{CN.LIDAR_SIGMA_NEAR_M:.0f}-{CN.LIDAR_SIGMA_FAR_M:.1f} m, "
                 f"{CN.LIDAR_QUANT_M} m resolution, {dev.rate:.0f} Hz   |   "
                 f"incidence gate {dev.inc_max:.0f} deg, seed {a.seed}", fontsize=11, y=0.985)

    # -- 1. true range vs measured, and why it is not the altitude -------------------------------
    ax = fig.add_subplot(gs[0, :])
    for i, (b0, b1) in enumerate(bands(t, on)):
        ax.axvspan(b0, b1, color="#2ca02c", alpha=0.13, lw=0, label="lidar ON" if i == 0 else None)
    for i, (b0, b1) in enumerate(bands(t, ~on)):
        ax.axvspan(b0, b1, color="0.3", alpha=0.28, lw=0, label="OFF" if i == 0 else None)
    ax.semilogy(t, rngm, "-", lw=1.7, color="k", label="true slant range (ray-cast)")
    ax.semilogy(t, np.maximum(alt, 1e-2), "--", lw=1.3, color="#1f77b4", label="altitude")
    ax.semilogy(mt, mz, ".", ms=4.5, color="#d62728", label=f"measured, {len(mt)} returns")
    ax.axhline(dev.min_range, color="0.4", ls=":", lw=1.2,
               label=f"instrument floor {dev.min_range:.0f} m")
    ax.set_ylabel("range [m]"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, ncol=3)
    ax.set_title("TRUE z (the slant range the beam actually travels) against the measurement, "
                 "and against the altitude it is NOT", fontsize=10)

    # -- 2. the same, zoomed to the last 250 s ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    m = t >= t[-1] - 250
    mm = mt >= t[-1] - 250
    ax.plot(t[m], rngm[m], "-", lw=1.8, color="k", label="true")
    ax.plot(t[m], alt[m], "--", lw=1.3, color="#1f77b4", label="altitude")
    ax.plot(mt[mm], mz[mm], "o", ms=4.5, mfc="none", mew=1.3, color="#d62728", label="measured")
    ax.axhline(dev.min_range, color="0.4", ls=":", lw=1.3)
    for b0, b1 in bands(t[m], ~on[m]):
        ax.axvspan(b0, b1, color="0.3", alpha=0.28, lw=0)
    ax.set_xlabel("t [s]"); ax.set_ylabel("range [m]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("last 250 s: the beam goes dark below the 30 m floor", fontsize=9.5)

    # -- 3. error against the envelope ------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ss = np.where(on, sig, np.nan)
    ax.fill_between(t, -ss, ss, color="#2ca02c", alpha=0.18, lw=0, label=r"$\pm1\sigma$ (Mizuno)")
    ax.plot(mt, err, "o", ms=4.0, mfc="none", mew=1.1, color="#d62728", label="z - true")
    ax.set_xlabel("t [s]"); ax.set_ylabel("range error [m]"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("measurement error inside the range-dependent envelope", fontsize=9.5)

    # -- 4. the three gates -----------------------------------------------------------------------
    ax = fig.add_subplot(gs[2, :])
    lv = {"beam hits a facet": hit, f"incidence <= {dev.inc_max:.0f} deg": g_inc,
          f"range in [{dev.min_range:.0f} m, {dev.max_range/1e3:.0f} km]": g_rng,
          "ALL THREE -> measurement": on}
    for i, (lab, mk) in enumerate(lv.items()):
        y = np.full(len(t), len(lv) - 1 - i, float)
        ax.plot(t, np.where(mk, y, np.nan), "-", lw=9,
                color=("#2ca02c" if lab.startswith("ALL") else "#1f77b4"), solid_capstyle="butt")
        ax.text(t[-1] * 1.005, len(lv) - 1 - i, f"  {100*mk.mean():.0f} %", va="center", fontsize=8)
    ax.set_yticks(range(len(lv))); ax.set_yticklabels(list(lv)[::-1], fontsize=8.5)
    ax.set_ylim(-0.6, len(lv) - 0.4); ax.grid(alpha=0.3, axis="x")
    # the bars only exist where a gate passes, so matplotlib would auto-scale x to the ON region
    # and make a half-flight window look like the whole descent
    ax.set_xlim(t[0], t[-1] * 1.05); ax.set_xlabel("t [s]")
    ax.set_title("which gate is binding, and when", fontsize=10)

    # -- 5. incidence angle and the Jacobian it drives --------------------------------------------
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, np.where(hit, inc, np.nan), "-", lw=1.5, color="#9467bd")
    ax.axhline(dev.inc_max, color="k", ls="--", lw=1.3, label=f"gate {dev.inc_max:.0f} deg")
    ax.set_xlabel("t [s]"); ax.set_ylabel("incidence [deg]"); ax.set_ylim(0, 90)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("incidence of the beam on the facet it hits", fontsize=9.5)

    ax = fig.add_subplot(gs[3, 1])
    Hn = 1.0 / np.cos(np.radians(np.clip(inc, 0, 89.9)))
    ax.semilogy(t, np.where(hit, Hn, np.nan), "-", lw=1.5, color="#9467bd")
    ax.axhline(1.0 / np.cos(np.radians(dev.inc_max)), color="k", ls="--", lw=1.3,
               label=f"at the gate: {1/np.cos(np.radians(dev.inc_max)):.2f}")
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$|H| = 1/\cos i$"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("range sensitivity to position: 1 m of position ->\n"
                 r"$|H|$ metres of range. This is why the gate exists", fontsize=9.5)

    # -- 6. sigma, and the statistical test -------------------------------------------------------
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(rngm[on] / 1e3, sig[on], ".", ms=2.5, color="#2ca02c", label="sigma(range)")
    ax.axhline(CN.LIDAR_SIGMA_NEAR_M, color="0.4", ls=":", lw=1.1,
               label=f"{CN.LIDAR_SIGMA_NEAR_M:.0f} m at {dev.min_range:.0f} m")
    ax.axhline(CN.LIDAR_SIGMA_FAR_M, color="0.4", ls="-.", lw=1.1,
               label=f"{CN.LIDAR_SIGMA_FAR_M:.1f} m at {dev.max_range/1e3:.0f} km")
    ax.set_xlabel("slant range [km]"); ax.set_ylabel("1-sigma [m]"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    ax.set_title("the two accuracies Mizuno quotes, interpolated", fontsize=9.5)

    ax = fig.add_subplot(gs[4, 1])
    ax.plot(mt, nrm, "o", ms=4.0, mfc="none", mew=1.1, color="#d62728")
    for k, ls in ((1, "--"), (3, ":")):
        ax.axhline(k, color="k", ls=ls, lw=1.0); ax.axhline(-k, color="k", ls=ls, lw=1.0)
    inside = float(np.mean(np.abs(nrm) <= 1.0)) if nrm.size else np.nan
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$(z - h)/\sigma$"); ax.grid(alpha=0.3)
    ax.set_title(f"TEST: {100*inside:.0f} % within $1\\sigma$ (expect 68 %), n = {nrm.size}\n"
                 f"mean {nrm.mean():+.3f}, std {nrm.std():.3f} (expect 0, 1)", fontsize=9.5)

    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"lidar_block_s{a.seed}.{ext}"), dpi=140,
                    bbox_inches="tight")
    print(f"wrote {os.path.join(a.out, f'lidar_block_s{a.seed}.png')}")

    print(f"\nrun            : {os.path.basename(path)}   tf {t[-1]:.1f} s   seed {a.seed}")
    print(f"gates          : hit {100*hit.mean():.1f} %   incidence<= {dev.inc_max:.0f} deg "
          f"{100*g_inc.mean():.1f} %   range window {100*g_rng.mean():.1f} %   "
          f"-> ON {100*on.mean():.1f} %")
    for b0, b1 in bands(t, ~on):
        m = (t >= b0) & (t <= b1)
        print(f"   OFF {b0:7.1f} -> {b1:7.1f} s ({b1-b0:6.1f} s)  alt {alt[m].max():8.1f} -> "
              f"{alt[m].min():7.1f} m   " +
              ("no return at all (the beam misses the body)" if not hit[m].any()
               else "below the instrument floor"
               if np.nanmin(rngm[m & hit]) < dev.min_range
               else f"a return on {100*hit[m].mean():.0f} % of it, incidence up to "
                    f"{np.nanmax(inc[m & hit]):.0f} deg"))
    print(f"returns        : {len(mt)}   sigma {ms.min():.3f} .. {ms.max():.3f} m")
    print(f"true range     : {np.nanmin(rngm):.2f} .. {np.nanmax(rngm):.1f} m   "
          f"altitude {alt.min():.2f} .. {alt.max():.1f} m")
    print(f"incidence      : {np.nanmin(inc):.1f} .. {np.nanmax(inc[hit]):.1f} deg   "
          f"|H| up to {np.nanmax(Hn[on]):.2f}")
    print(f"normalised err : mean {nrm.mean():+.4f}  std {nrm.std():.4f}  "
          f"{100*inside:.1f} % inside 1 sigma (expect 68.3 %)")


if __name__ == "__main__":
    main()

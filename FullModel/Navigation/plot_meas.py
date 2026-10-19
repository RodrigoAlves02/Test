#!/usr/bin/env python
"""What each sensor actually measured, against the truth and the filter's estimate.

The altimeter measures a scalar (height above the site's tangent plane); the cameras measure a full
position, so their altitude and lateral components are both plotted. The bottom row shows PRE-FIT
innovations -- measurement minus the filter's prediction BEFORE the update -- which is the quantity
that says whether the filter and the sensor agree. Post-fit residuals are shrunk by the update
itself and would flatter the filter.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "..", "SCvx", "Figures&data")
OUT = os.path.join(HERE, "Figures&data")
TRU, EST = "#2B2B2B", "#1B5E9C"
COL = {"alt": "#B8351F", "cam+": "#1B5E9C", "cam-": "#2E8B57"}


def main(tag="_navfull"):
    d = json.load(open(os.path.join(SC, f"closed_loop{tag}.json")))
    m = d.get("meas") or []
    if not m:
        raise SystemExit("no measurements recorded in that run")
    tr = [r for r in d["trace"] if "r_nav" in r]
    z = np.load(os.path.join(SC, "descent_K60.npz"))
    nhat = np.asarray(z["n_hat"], float); rsite = np.asarray(z["rf"], float)
    P = np.eye(3) - np.outer(nhat, nhat)

    t = np.array([r["t"] for r in tr])
    rt = np.array([r["r_true"] for r in tr]); rn = np.array([r["r_nav"] for r in tr])
    alt_t = (rt - rsite) @ nhat; alt_n = (rn - rsite) @ nhat
    lat_t = np.linalg.norm((rt - rsite) @ P.T, axis=1)
    lat_n = np.linalg.norm((rn - rsite) @ P.T, axis=1)

    fig, ax = plt.subplots(2, 2, figsize=(14.0, 8.0))
    for a_ in ax.ravel():
        a_.grid(alpha=0.22, lw=0.6); a_.set_xlabel("time [s]")

    # -- altitude ---------------------------------------------------------------------------------
    ax[0, 0].plot(t, alt_t, color=TRU, lw=2.0, zorder=4, label="truth")
    ax[0, 0].plot(t, alt_n, color=EST, lw=1.3, ls="--", zorder=5, label="EKF estimate")
    for key in ("alt", "cam+", "cam-"):
        mm = [r for r in m if r["s"] == key]
        if not mm:
            continue
        tt = np.array([r["t"] for r in mm])
        zz = np.array([r["z"] for r in mm])
        av = zz[:, 0] if key == "alt" else (zz - rsite) @ nhat
        ax[0, 0].plot(tt, av, ".", color=COL[key], ms=2.5, alpha=0.5, zorder=2,
                      label=f"{key} measured ({len(mm)})")
    ax[0, 0].set_ylabel("altitude above site [m]")
    ax[0, 0].set_title("Altitude: sensors, truth, estimate", fontsize=10)
    ax[0, 0].legend(fontsize=8)

    # -- lateral ----------------------------------------------------------------------------------
    ax[0, 1].plot(t, lat_t, color=TRU, lw=2.0, zorder=4, label="truth")
    ax[0, 1].plot(t, lat_n, color=EST, lw=1.3, ls="--", zorder=5, label="EKF estimate")
    for key in ("cam+", "cam-"):
        mm = [r for r in m if r["s"] == key]
        if not mm:
            continue
        tt = np.array([r["t"] for r in mm]); zz = np.array([r["z"] for r in mm])
        ax[0, 1].plot(tt, np.linalg.norm((zz - rsite) @ P.T, axis=1), ".", color=COL[key],
                      ms=2.5, alpha=0.5, zorder=2, label=f"{key} measured")
    ax[0, 1].set_ylabel("lateral distance from site [m]")
    ax[0, 1].set_title("Lateral: the altimeter cannot see this axis", fontsize=10)
    ax[0, 1].legend(fontsize=8)

    # -- innovations ------------------------------------------------------------------------------
    for key in ("alt", "cam+", "cam-"):
        mm = [r for r in m if r["s"] == key]
        if not mm:
            continue
        tt = np.array([r["t"] for r in mm])
        zz = np.array([r["z"] for r in mm]); hh = np.array([r["h"] for r in mm])
        inn = (zz - hh)[:, 0] if key == "alt" else np.linalg.norm(zz - hh, axis=1)
        sg = np.array([r["sig"] for r in mm])
        ax[1, 0].plot(tt, inn, ".", color=COL[key], ms=2.5, alpha=0.55, label=key)
        ax[1, 1].plot(tt, np.abs(inn) / np.maximum(sg, 1e-12), ".", color=COL[key],
                      ms=2.5, alpha=0.55, label=key)
    ax[1, 0].axhline(0, color=TRU, lw=0.8)
    ax[1, 0].set_ylabel("innovation, measured - predicted")
    ax[1, 0].set_title("Pre-fit innovations", fontsize=10); ax[1, 0].legend(fontsize=8)

    ax[1, 1].axhline(3.0, color="#B8351F", ls="--", lw=1.2)
    ax[1, 1].text(t[-1], 3.0, r" 3$\sigma$", color="#B8351F", fontsize=8, va="bottom", ha="right")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel(r"|innovation| / $\sigma$")
    ax[1, 1].set_title("Normalised innovation (filter consistency)", fontsize=10)
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"Sensor measurements vs estimate  —  {len(m)} updates, "
                 f"final nav error {np.linalg.norm(rn[-1]-rt[-1]):.3f} m, "
                 f"miss {d['miss']:.3f} m", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"nav_meas{tag}.{e}"), dpi=300, bbox_inches="tight")
    for key in ("alt", "cam+", "cam-"):
        mm = [r for r in m if r["s"] == key]
        if mm:
            zz = np.array([r["z"] for r in mm]); hh = np.array([r["h"] for r in mm])
            inn = (zz - hh)[:, 0] if key == "alt" else np.linalg.norm(zz - hh, axis=1)
            sg = np.array([r["sig"] for r in mm])
            print(f"  {key:>5}: {len(mm):5d} updates, |innov| mean {np.abs(inn).mean():8.3f}, "
                  f"normalised mean {np.mean(np.abs(inn)/np.maximum(sg,1e-12)):6.2f}")
    print("wrote " + os.path.join(OUT, f"nav_meas{tag}.{{pdf,png}}"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_navfull")

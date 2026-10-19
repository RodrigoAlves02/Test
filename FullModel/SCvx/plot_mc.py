#!/usr/bin/env python
"""Monte Carlo of the closed loop: where it lands, and what the miss is made of.

Reads Figures&data/mc/closed_loop_mc_p<pos>_v<vel>_s<seed>.json.  The grid is the one in
mc_cases.py: ten handover-dispersion configurations x 50 seeds, arranged as four FAMILIES

    nav only    (0, 0)                      truth exactly at the nominal handover; the only error
                                            is the filter's, so this is the landing-accuracy
                                            distribution of the navigation alone
    coupled     (10, 0.01) ... (1000, 1)    position and velocity dispersed together
    position    (10, 0) ... (1000, 0)       position only
    velocity    (0, 0.01) ... (0, 1)        velocity only

with the 10 m / 0.01 m/s OD error drawn on top of every case, from the dispersed truth state.
Family is identity (hue + marker), magnitude within a family is a three-step ramp, and the nav-only
median is drawn as a reference line on every response panel -- a configuration that does not clear
that line is not being limited by its dispersion.

Writes into Figures&data/:
  mc_landing     the footprint, the miss CDF, miss against navigation error, touchdown speed
  mc_response    every outcome against the dispersion decade, one series per family
  mc_traces      time histories, per-family median and spread
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter


def logticks(ax, axis="x", vals=(0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)):
    """Readable decade-and-a-half ticks. Matplotlib's default log labelling writes every minor
    tick over a narrow range and the labels collide into an unreadable smear."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    keep = [v for v in vals if lo <= v <= hi]
    a.set_major_locator(FixedLocator(keep)); a.set_minor_locator(NullLocator())
    a.set_major_formatter(ScalarFormatter())

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"closed_loop_mc_p([0-9.]+)_v([0-9.]+)_s(\d+)\.json$")
VTOUCH = 0.3

# family -> (marker, three-step hue ramp light->dark).  Hue carries identity, step carries
# magnitude, and the marker repeats the identity so the figure survives colour-blind readers.
FAM = {"nav only": ("P", ["#33333b"]),
       "coupled":  ("o", ["#9ecae1", "#4292c6", "#08519c"]),
       "position": ("s", ["#fdae6b", "#e6550d", "#8c2d04"]),
       "velocity": ("^", ["#a1d99b", "#41ab5d", "#005a32"])}
ORDER = ["nav only", "coupled", "position", "velocity"]


def classify(p, v):
    """(family, decade index 0..2, label)."""
    if p == 0 and v == 0:
        return "nav only", 0, "nav only"
    if p and v:
        return "coupled", int(round(np.log10(p))) - 1, f"{p:g} m + {v:g} m/s"
    if p:
        return "position", int(round(np.log10(p))) - 1, f"{p:g} m"
    return "velocity", int(round(np.log10(v))) + 2, f"{v:g} m/s"


def site_frame(dirpath):
    """r_site, the local vertical, and an in-SURFACE east/north pair. Cached: this loads the mesh.

    The footprint is drawn in the plane of the FACET the site sits on, not in the plane
    perpendicular to the local vertical. Every touchdown point lies in that facet -- it is one flat
    triangle 178 x 259 x 261 m and the misses are metres -- so facet coordinates are a true map of
    where the vehicle landed relative to the target. The two planes are 21.3 deg apart and separate
    at 0.39 m per metre of miss, which is why the datum plane was the wrong place to measure
    anything terminal.
    """
    f = os.path.join(dirpath, "site_frame.npz")
    if not os.path.exists(f):
        import sys
        sys.path.insert(0, HERE)
        import gravity as GR
        rs, n = GR.site_and_normal()
        m = GR._tri_mesh()
        loc, _, it = m.ray.intersects_location(ray_origins=np.array([rs + n * 500.0]),
                                               ray_directions=np.array([-n]))
        nf = np.asarray(m.face_normals[it[int(np.argmax(loc @ n))]], float)
        nf = nf * np.sign(nf @ rs)                       # outward
        e = np.cross([0.0, 0.0, 1.0], nf); e /= np.linalg.norm(e)
        np.savez(f, r_site=rs, n_hat=n, east=e, north=np.cross(nf, e), n_facet=nf)
    z = np.load(f)
    return z["r_site"], z["n_hat"], z["east"], z["north"]


def load(dirpath):
    r_site, n_hat, east, north = site_frame(dirpath)
    rows = []
    for f in sorted(glob.glob(os.path.join(dirpath, "closed_loop_mc_p*_v*_s*.json"))):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        cy, tr = d["cycles"], d["trace"]
        fin = lambda k: [c[k] for c in cy if c.get(k) is not None and np.isfinite(c[k])]
        rt = np.array([p["r_true"] for p in tr]); rn = np.array([p["r_nav"] for p in tr])
        vt = np.array([p["v_true"] for p in tr]); vn = np.array([p["v_nav"] for p in tr])
        # The miss is RECOMPUTED here from the touchdown point rather than read from the file, so
        # datasets written before the definition changed stay comparable with ones written after.
        dr = rt[-1] - r_site
        miss = float(np.linalg.norm(dr))
        fam, dec, lab = classify(float(m.group(1)), float(m.group(2)))
        rows.append(dict(
            pos=float(m.group(1)), vel=float(m.group(2)), seed=int(m.group(3)),
            fam=fam, dec=dec, label=lab,
            landed=bool(d["landed"]), miss=miss, miss_plane=float(np.linalg.norm(dr - n_hat * (dr @ n_hat))),
            speed=d["speed"], vert=d.get("vert"),
            fuel=d["fuel"], tf=d["t"], ang=d["ang"],
            de=float(dr @ east), dn=float(dr @ north),
            maxiter=sum(1 for c in cy if c["status"] == "maxiter"),
            n_fallback=d.get("n_fallback", 0), solve=sum(c["wall"] for c in cy),
            D_max=max(fin("D_m") or [0.0]),
            plan_miss=float(np.mean(fin("miss_plan_m") or [np.nan])),
            nav_pe_f=float(np.linalg.norm(rn[-1] - rt[-1])),
            nav_ve_f=float(np.linalg.norm(vn[-1] - vt[-1])),
            t=np.array([p["t"] for p in tr]), alt=np.array([p["alt"] for p in tr]),
            spd=np.linalg.norm(vt, axis=1), T=np.array([p["T_act"] for p in tr]),
            pe=np.linalg.norm(rn - rt, axis=1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "Figures&data", "mc"))
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()
    rows = load(a.dir)
    if not rows:
        raise SystemExit(f"no MC files in {a.dir}")
    cfgs = sorted({(r["pos"], r["vel"]) for r in rows},
                  key=lambda c: (ORDER.index(classify(*c)[0]), classify(*c)[1]))
    by = {c: [r for r in rows if (r["pos"], r["vel"]) == c] for c in cfgs}
    col = {c: FAM[classify(*c)[0]][1][min(classify(*c)[1], len(FAM[classify(*c)[0]][1]) - 1)]
           for c in cfgs}
    mkr = {c: FAM[classify(*c)[0]][0] for c in cfgs}
    base = by.get((0.0, 0.0))
    b_med = (lambda k: np.median([r[k] for r in base])) if base else (lambda k: None)

    # ---- table ---------------------------------------------------------------------------
    hdr = (f"{'configuration':>18} {'n':>4} {'land':>8} {'miss p50':>9} {'p95':>8} {'max':>8} "
           f"{'naverr p50':>11} {'v>0.3':>7} {'v_td p95':>9} {'fuel p50':>9} {'tf p50':>8} "
           f"{'maxit':>6}")
    print(f"{len(rows)} cases, {len(cfgs)} configurations\n{hdr}\n" + "-" * len(hdr))
    pc = lambda g, k, q: np.percentile([r[k] for r in g], q)
    for c in cfgs:
        g = by[c]
        print(f"{classify(*c)[2]:>18} {len(g):4d} {sum(r['landed'] for r in g):4d}/{len(g):<3d} "
              f"{pc(g,'miss',50):9.3f} {pc(g,'miss',95):8.3f} {max(r['miss'] for r in g):8.3f} "
              f"{pc(g,'nav_pe_f',50):11.3f} "
              f"{sum(1 for r in g if r['landed'] and r['speed'] > VTOUCH):4d}/{len(g):<2d} "
              f"{pc(g,'speed',95):9.4f} {pc(g,'fuel',50):9.3f} {pc(g,'tf',50):8.1f} "
              f"{sum(r['maxiter'] for r in g):6d}")

    hand = [Line2D([], [], color=col[c], marker=mkr[c], ls="none", mfc="none", mew=1.5, ms=7,
                   label=classify(*c)[2]) for c in cfgs]
    sub = (f"{len(rows)} cases.  Truth handover dispersed as labelled; on top of every case the "
           f"filter starts with a fresh 10 m / 0.01 m/s OD error drawn from the dispersed state.")

    # ================= figure 1: where it lands =============================================
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 10.2))
    fig.suptitle("Closed-loop Monte Carlo: the landing footprint\n" + sub, fontsize=11.5)

    ax = axs[0, 0]
    rmax = max(np.hypot(r["de"], r["dn"]) for r in rows) * 1.3 + 0.2
    for rr in (0.5, 1, 2, 5, 10, 20, 50, 100):
        if rr < rmax:
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, ec="0.78", lw=0.8, ls=":", zorder=0))
            ax.annotate(f"{rr:g} m", (rr * 0.707, rr * 0.707), fontsize=7, color="0.5",
                        ha="left", va="bottom", zorder=0)
    for c in cfgs:
        g = [r for r in by[c] if r["landed"]]
        ax.plot([r["de"] for r in g], [r["dn"] for r in g], mkr[c], ms=5.5, mfc="none", mew=1.2,
                color=col[c], alpha=0.85, zorder=3)
    # A run that terminated above the surface never landed there; showing it as a touchdown point
    # would be a lie about the footprint, so it is marked instead of hidden.
    nl = [r for r in rows if not r["landed"]]
    if nl:
        ax.plot([r["de"] for r in nl], [r["dn"] for r in nl], "x", ms=8, mew=1.8, color="crimson",
                alpha=0.9, zorder=5,
                label=f"did not reach the surface ({len(nl)})")
    ax.plot(0, 0, "x", ms=11, mew=2.2, color="crimson", zorder=4)
    ax.annotate("target", (0, 0), (7, -13), textcoords="offset points", fontsize=8.5,
                color="crimson")
    ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax); ax.set_aspect("equal")
    ax.set_xlabel("east of site [m]"); ax.set_ylabel("north of site [m]")
    ax.grid(alpha=0.22); ax.set_title("touchdown points in the site tangent plane", fontsize=10)
    h2 = list(hand)
    if nl:
        h2.append(Line2D([], [], color="crimson", marker="x", ls="none", mew=1.8, ms=8,
                         label=f"never reached the surface ({len(nl)})"))
    ax.legend(handles=h2, fontsize=7, title="handover dispersion", title_fontsize=7.5,
              loc="upper left", ncol=2, framealpha=0.9)

    ax = axs[0, 1]
    for c in cfgs:
        m = np.sort([r["miss"] for r in by[c]])
        ax.step(np.r_[0, m], np.r_[0, np.arange(1, len(m) + 1) / len(m)] * 100,
                where="post", color=col[c], lw=1.7,
                ls="-" if classify(*c)[0] != "nav only" else "--")
    ax.set_xlabel("landing miss [m]"); ax.set_ylabel("cases landing within [%]")
    ax.set_ylim(0, 103); ax.set_xscale("log"); ax.grid(alpha=0.3, which="both")
    logticks(ax)
    ax.set_title("miss distribution (empirical CDF)  -  the curves lie on top of each other",
                 fontsize=10)

    ax = axs[1, 0]
    for c in cfgs:
        g = by[c]
        ax.plot([r["nav_pe_f"] for r in g], [r["miss"] for r in g], mkr[c], ms=5.5, mfc="none",
                mew=1.2, color=col[c], alpha=0.85)
    lo = 0.05
    hi = max([r["miss"] for r in rows] + [r["nav_pe_f"] for r in rows]) * 1.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1)
    ax.annotate("miss = nav error", (0.25, 0.25), (6, 2), textcoords="offset points",
                rotation=45, fontsize=8.5, ha="left", va="bottom")
    pm = np.nanmedian([r["plan_miss"] for r in rows])
    ax.axhline(pm, color="seagreen", lw=1.4, ls="-.")
    ax.annotate(f"what the plan itself promises, {pm:.3f} m", (lo * 1.2, pm), (0, 4),
                textcoords="offset points", fontsize=8.5, color="seagreen")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    dec = (0.1, 0.3, 1, 3, 10, 30, 100, 300)
    logticks(ax, "x", dec); logticks(ax, "y", dec)
    ax.set_xlabel("navigation position error at touchdown [m]"); ax.set_ylabel("landing miss [m]")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("is the miss just the navigation error?", fontsize=10)

    ax = axs[1, 1]
    for c in cfgs:
        g = by[c]
        ax.plot([r["miss"] for r in g], [r["speed"] for r in g], mkr[c], ms=5.5, mfc="none",
                mew=1.2, color=col[c], alpha=0.85)
    ax.axhline(VTOUCH, color="crimson", ls="--", lw=1.4)
    ax.annotate(f"limit {VTOUCH} m/s", (0.985, VTOUCH), (-3, 4), xycoords=("axes fraction", "data"),
                textcoords="offset points", fontsize=8.5, color="crimson", ha="right")
    ax.set_ylim(0, max(VTOUCH * 1.12, max(r["speed"] for r in rows) * 1.08))
    ax.set_xscale("log")
    ax.set_xlabel("landing miss [m]"); ax.set_ylabel("touchdown speed [m/s]")
    logticks(ax)
    ax.grid(alpha=0.3, which="both")
    ld = [r for r in rows if r["landed"]]
    over = sum(1 for r in ld if r["speed"] > VTOUCH)
    ax.set_title(f"touchdown speed: {over}/{len(ld)} landings ({100*over/len(ld):.0f}%) break the "
                 f"{VTOUCH} m/s limit", fontsize=10)

    os.makedirs(a.out, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"mc_landing.{e}"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ================= figure 2: response to the dispersion decade ==========================
    fams = [f for f in ORDER[1:] if any(classify(*c)[0] == f for c in cfgs)]
    fig, axs = plt.subplots(2, 3, figsize=(13.8, 8.2))
    fig.suptitle("Closed-loop Monte Carlo: response to the dispersion magnitude\n" + sub,
                 fontsize=11.5)
    TICK = ["10 m / 0.01 m/s", "100 m / 0.1 m/s", "1000 m / 1 m/s"]

    def resp(ax, k, yl, title, log=False, hline=None, hlab=None):
        for f in fams:
            cs = sorted([c for c in cfgs if classify(*c)[0] == f], key=lambda c: classify(*c)[1])
            x = [classify(*c)[1] for c in cs]
            med = [np.median([r[k] for r in by[c]]) for c in cs]
            lo = [np.percentile([r[k] for r in by[c]], 5) for c in cs]
            hi = [np.percentile([r[k] for r in by[c]], 95) for c in cs]
            cc = FAM[f][1][1]
            ax.fill_between(x, lo, hi, color=cc, alpha=0.16, lw=0)
            ax.plot(x, med, "-", color=cc, lw=1.8, marker=FAM[f][0], ms=7, mfc="none", mew=1.6,
                    label=f)
        if base is not None:
            ax.axhline(b_med(k), color=FAM["nav only"][1][0], ls="--", lw=1.4)
            ax.annotate("nav only", (0.01, b_med(k)), (3, 4), xycoords=("axes fraction", "data"),
                        textcoords="offset points", fontsize=8, color=FAM["nav only"][1][0])
        if hline is not None:
            ax.axhline(hline, color="crimson", ls="--", lw=1.3)
            ax.annotate(hlab, (0.99, hline), (-3, -12), xycoords=("axes fraction", "data"),
                        textcoords="offset points", fontsize=8, color="crimson", ha="right")
        if log:
            ax.set_yscale("log")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(TICK, fontsize=8)
        ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.set_title(title, fontsize=10)

    resp(axs[0, 0], "miss", "landing miss [m]", "landing miss   (line = median, band = 5-95%)", True)
    axs[0, 0].legend(fontsize=8, title="dispersed", title_fontsize=8)
    resp(axs[0, 1], "nav_pe_f", "nav position error [m]", "navigation error at touchdown", True)
    resp(axs[0, 2], "speed", "touchdown speed [m/s]", "touchdown speed", False, VTOUCH,
         f"limit {VTOUCH} m/s")
    resp(axs[1, 0], "fuel", "propellant [kg]", "propellant")
    resp(axs[1, 1], "tf", "flight time [s]", "flight time")
    resp(axs[1, 2], "D_max", "max stage-1 D [m]", "stage-1 reachability slack", True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"mc_response.{e}"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ================= figure 3: time histories =============================================
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 7.4))
    fig.suptitle("Closed-loop Monte Carlo: time histories (line = median, band = full spread)\n"
                 + sub, fontsize=11.5)

    def band(ax, g, key, color, lo=None):
        tg = np.linspace(0.0, max(r["t"][-1] for r in g), 400)
        Y = np.array([np.interp(tg, r["t"], r[key], right=np.nan) for r in g])
        q0, q1 = np.nanmin(Y, 0), np.nanmax(Y, 0)
        ax.fill_between(tg, q0 if lo is None else np.maximum(q0, lo), q1, color=color,
                        alpha=0.16, lw=0)
        ax.plot(tg, np.nanmedian(Y, 0), color=color, lw=1.6)

    for c in cfgs:
        g = by[c]
        band(axs[0, 0], g, "alt", col[c]); band(axs[0, 1], g, "spd", col[c])
        band(axs[1, 0], g, "pe", col[c], lo=1e-3); band(axs[1, 1], g, "T", col[c])
    axs[0, 0].set_ylabel("altitude [m]"); axs[0, 0].set_title("descent", fontsize=10)
    axs[0, 0].legend(handles=hand, fontsize=7, ncol=2, title="handover dispersion",
                     title_fontsize=7.5)
    axs[0, 1].set_ylabel("speed [m/s]"); axs[0, 1].set_title("speed", fontsize=10)
    axs[1, 0].set_ylabel("nav position error [m]"); axs[1, 0].set_yscale("log")
    axs[1, 0].set_title("navigation position error", fontsize=10)
    axs[1, 1].set_ylabel("thrust [N]"); axs[1, 1].set_title("thrust delivered", fontsize=10)
    for ax in axs.ravel():
        ax.set_xlabel("time [s]"); ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"mc_traces.{e}"), dpi=140, bbox_inches="tight")
    print(f"\nwrote mc_landing, mc_response, mc_traces (png+pdf) to {a.out}")


if __name__ == "__main__":
    main()

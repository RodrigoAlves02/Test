"""See the cameras working: the asteroid inside the field of view, at ten instants of the descent.

TWO FIGURES, because "is the asteroid in the FOV" and "how much of the FOV does it cover" are
different questions and only the second one can define an activation rule.

  cam_fov_views  -- what each camera RECORDS. The FOV is sampled with an N x N grid of rays cast
                    against the real 49 152-facet shape; a ray that hits is a pixel on the body,
                    coloured by its range. This is a rendering and a measurement at once: the hit
                    fraction IS the FOV coverage, and the hit distances ARE the imaging range.
  cam_fov_3d     -- the same instants seen from outside, with both FOV pyramids drawn, so the
                    geometry behind the coverage number is visible.

WHY A RAY GRID RATHER THAN THE ANGLE TO THE NEAREST VERTEX. The vertex test answers "is any surface
point inside the cone", which is a yes/no about the limb and says nothing about how much of the
frame is filled. It also breaks below about 12 km range, where the 211 m mean vertex spacing leaves
no vertex near the boresight even though the surface fills the image. The ray grid is exact at every
scale and gives the coverage fraction and the range distribution for free.

IMAGE ORIENTATION. A boresight fixes only two of the three rotational degrees of freedom; the roll
about it is free. Here the image "up" axis is the local vertical projected into the image plane
(falling back to the body spin axis when the two align), which keeps the horizon roughly horizontal.
It rotates the picture and nothing else.
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
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import config_nav as CN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "SCvx", "Figures&data"))


def load_shape():
    sys.path.insert(0, CN.SCVX_DIR)
    import trimesh
    import config as C
    import gravity as GR
    v, f = GR._mesh(C.MESH_TAB)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    r_site, n_hat = GR.site_and_normal()
    return mesh, np.asarray(v, float), np.asarray(f), r_site, n_hat


def image_axes(u, n_hat):
    """Right-handed (e1 = image right, e2 = image up, u = boresight)."""
    ref = np.asarray(n_hat, float)
    if abs(ref @ u) > 0.98:
        ref = np.array([0.0, 0.0, 1.0])
        if abs(ref @ u) > 0.98:
            ref = np.array([1.0, 0.0, 0.0])
    e2 = ref - (ref @ u) * u
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, u)
    return e1 / np.linalg.norm(e1), e2


def fov_rays(u, e1, e2, half_deg, N):
    """Directions for an N x N grid across a SQUARE field of view, plus the tan-space coords."""
    tt = np.tan(np.radians(half_deg))
    ax = np.linspace(-tt, tt, N)
    U, V = np.meshgrid(ax, ax, indexing="xy")
    D = u[None, None, :] + U[..., None] * e1[None, None, :] + V[..., None] * e2[None, None, :]
    return D / np.linalg.norm(D, axis=-1, keepdims=True), U, V, tt


def render(mesh, r, u, n_hat, half_deg, N):
    """Cast the FOV grid. Returns (range image with NaN for sky, coverage, range stats)."""
    e1, e2 = image_axes(u, n_hat)
    D, U, V, tt = fov_rays(u, e1, e2, half_deg, N)
    O = np.repeat(np.asarray(r, float)[None, :], N * N, axis=0)
    loc, idx_ray, _ = mesh.ray.intersects_location(
        ray_origins=O, ray_directions=D.reshape(-1, 3), multiple_hits=False)
    img = np.full(N * N, np.nan)
    if len(loc):
        dist = np.linalg.norm(loc - np.asarray(r, float), axis=1)
        # keep the NEAREST hit per ray (multiple_hits=False is not a guarantee on all backends)
        order = np.argsort(dist)
        img[idx_ray[order][::-1]] = dist[order][::-1]
    img = img.reshape(N, N)
    hit = np.isfinite(img)
    cov = float(hit.mean())
    rng = img[hit]
    stats = dict(n=int(hit.sum()),
                 median=float(np.median(rng)) if rng.size else np.nan,
                 mean=float(rng.mean()) if rng.size else np.nan,
                 min=float(rng.min()) if rng.size else np.nan,
                 max=float(rng.max()) if rng.size else np.nan)
    return img, cov, stats, tt, e1, e2


def pick_times(tr, n, explicit):
    t = np.array([row["t"] for row in tr])
    if explicit:
        want = np.array(explicit, float)
    else:
        want = np.linspace(t[0], t[-1], n)
    return [int(np.argmin(np.abs(t - w))) for w in want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="closed_loop_imuplot.json")
    ap.add_argument("--n", type=int, default=10, help="number of instants")
    ap.add_argument("--times", type=float, nargs="*", default=None, help="explicit instants [s]")
    ap.add_argument("--grid", type=int, default=72, help="rays per FOV side")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    path = a.run if os.path.isabs(a.run) else os.path.join(DATA, a.run)
    tr = json.load(open(path))["trace"]
    mesh, V, F, r_site, n_hat = load_shape()
    half = CN.CAM_FOV_DEG / 2.0
    ks = pick_times(tr, a.n, a.times)
    cams = ((+1, "fwd (+thrust)", "#1f77b4"), (-1, "aft (-thrust)", "#d62728"))

    # ---------------------------------------------------------------- figure 1: the camera images
    rows = int(np.ceil(len(ks) / 5))
    fig = plt.figure(figsize=(19.0, 4.3 * rows + 1.0))
    gs = fig.add_gridspec(rows, 5, hspace=0.40, wspace=0.14, top=0.88)
    shots = []
    for j, k in enumerate(ks):
        row = tr[k]
        r = np.array(row["r"], float)
        d = np.array(row["d_act"], float); d /= np.linalg.norm(d)
        best = None
        for sign, name, colr in cams:
            u = sign * d
            img, cov, st, tt, e1, e2 = render(mesh, r, u, n_hat, half, a.grid)
            if best is None or cov > best[1]:
                best = (name, cov, img, st, tt, colr, u, sign)
        name, cov, img, st, tt, colr, u, sign = best
        other = [c for c in cams if c[1] != name][0]
        oimg, ocov, ost, *_ = render(mesh, r, other[0] * d, n_hat, half, a.grid)
        shots.append(dict(k=k, t=row["t"], alt=row["alt"], r=r, d=d, name=name, cov=cov,
                          st=st, sign=sign, ocov=ocov, oname=other[1]))

        ax = fig.add_subplot(gs[j // 5, j % 5])
        ext = [-half, half, -half, half]
        if np.isfinite(img).any():
            km = np.flipud(img) / 1e3
            lo, hi = float(np.nanmin(km)), float(np.nanmax(km))
            # at touchdown every ray returns ~0 km, which no log scale can hold
            norm = LogNorm(vmin=lo, vmax=hi) if lo > 1e-3 and hi > 1.05 * lo else None
            im = ax.imshow(km, extent=ext, cmap="viridis", norm=norm,
                           vmin=None if norm else lo, vmax=None if norm else max(hi, lo + 1e-6),
                           interpolation="nearest")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ticks = (np.geomspace(lo, hi, 4) if norm else np.linspace(lo, max(hi, lo + 1e-6), 3))
            cb.ax.minorticks_off()            # LogNorm draws its own minor labels otherwise
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{x:.3g}" for x in ticks])
            cb.ax.tick_params(labelsize=7, which="both")
            cb.set_label("range [km]", fontsize=7)
        ax.set_facecolor("#0b0b16")
        ax.add_patch(plt.Rectangle((-half, -half), 2 * half, 2 * half, fill=False,
                                   ec=colr, lw=2.2))
        ax.plot(0, 0, "+", color="w", ms=9, mew=1.4)
        ax.set_xlim(-half * 1.02, half * 1.02); ax.set_ylim(-half * 1.02, half * 1.02)
        ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30])
        ax.tick_params(labelsize=7)
        ax.set_xlabel("deg across FOV", fontsize=7)
        sig = CN.CAM_SIGMA_FRAC * st["median"] if np.isfinite(st["median"]) else np.nan
        ax.set_title(f"t = {row['t']:.0f} s    alt {row['alt']/1e3:.2f} km\n"
                     f"{name}   coverage {100*cov:.1f} %   (other {100*ocov:.1f} %)\n"
                     f"range med {st['median']/1e3:.2f} km   "
                     + (f"$\\sigma$ {sig:.1f} m" if np.isfinite(sig) else "no return"),
                     fontsize=8.6, color=("0.15" if cov > 0 else "#b00"))
    fig.suptitle(
        f"What the cameras record   -   {os.path.basename(path)}   -   square FOV "
        f"{CN.CAM_FOV_DEG} deg (ONC-W1), {a.grid}x{a.grid} rays against the 49 152-facet shape\n"
        f"pixel colour is the RANGE to the surface along that ray; dark blue-black is sky. "
        f"The camera shown is whichever of the two has more coverage.", fontsize=12.5, y=0.975)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"cam_fov_views.{ext}"), dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.join(a.out, 'cam_fov_views.png')}")
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2: external geometry
    fn = np.asarray(mesh.face_normals); fcen = np.asarray(mesh.triangles_center)
    lim = 2.0e4
    fig = plt.figure(figsize=(19.0, 3.4 * rows + 0.8))
    gs = fig.add_gridspec(rows, 5, hspace=-0.10, wspace=0.0, top=0.92, bottom=0.02)
    for j, s in enumerate(shots):
        ax = fig.add_subplot(gs[j // 5, j % 5], projection="3d")
        eye = np.array([1.0, -1.3, 0.75]); eye /= np.linalg.norm(eye)
        vis = np.einsum("ij,ij->i", fn, fcen - eye * 6e4) < 0
        shade = np.clip(-(fn[vis] @ eye), 0, 1)
        ax.add_collection3d(Poly3DCollection(V[F[vis]], facecolors=plt.cm.bone(0.22 + 0.62 * shade),
                                             edgecolors="none", zsort="average"))
        ax.scatter(*s["r"], s=26, color="k", depthshade=False, zorder=10)
        for sign, name, colr in cams:
            u = sign * s["d"]
            e1, e2 = image_axes(u, n_hat)
            tt = np.tan(np.radians(half))
            L = min(3.2e4, max(6e3, 1.35 * np.linalg.norm(s["r"])))
            corners = [s["r"] + L * (u + su * tt * e1 + sv * tt * e2) /
                       np.linalg.norm(u + su * tt * e1 + sv * tt * e2)
                       for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            segs = [[s["r"], c] for c in corners] + \
                   [[corners[i], corners[(i + 1) % 4]] for i in range(4)]
            on = (name == s["name"] and s["cov"] > 0) or (name != s["name"] and s["ocov"] > 0)
            ax.add_collection3d(Line3DCollection(segs, colors=colr, linewidths=1.5 if on else 0.7,
                                                 alpha=0.95 if on else 0.30))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-58)
        ax.set_axis_off()
        ax.set_title(f"t = {s['t']:.0f} s   alt {s['alt']/1e3:.2f} km\n"
                     f"fwd {100*(s['cov'] if s['name'].startswith('fwd') else s['ocov']):.0f} %   "
                     f"aft {100*(s['cov'] if s['name'].startswith('aft') else s['ocov']):.0f} %",
                     fontsize=9, y=0.96)
    fig.suptitle("The same instants from outside: asteroid, spacecraft (black dot) and both FOV "
                 "pyramids.   blue = fwd (+thrust), red = aft (-thrust); faded = no surface in "
                 "that frame", fontsize=12.5, y=0.965)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"cam_fov_3d.{ext}"), dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.join(a.out, 'cam_fov_3d.png')}")

    print(f"\n{'t [s]':>8} {'alt [km]':>9} {'camera':>14} {'cov %':>7} {'other %':>8} "
          f"{'rng med [km]':>13} {'rng min':>9} {'rng max':>9} {'sigma [m]':>10}")
    for s in shots:
        st = s["st"]
        sig = CN.CAM_SIGMA_FRAC * st["median"] if np.isfinite(st["median"]) else np.nan
        print(f"{s['t']:8.0f} {s['alt']/1e3:9.3f} {s['name']:>14} {100*s['cov']:7.1f} "
              f"{100*s['ocov']:8.1f} {st['median']/1e3:13.3f} {st['min']/1e3:9.3f} "
              f"{st['max']/1e3:9.3f} {sig:10.2f}")


if __name__ == "__main__":
    main()

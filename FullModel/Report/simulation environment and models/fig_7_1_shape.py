#!/usr/bin/env python
"""Figures for section 7.1 -- the shape model and the candidate landing sites.

    fig_eros_shape           the body as the simulation flies it, ver64q, region marked
    fig_region_slope         slope from the local vertical over the region, ON THE FLIGHT MESH

EVERYTHING IS MEASURED ON ver64q, THE FLIGHT MESH. An earlier version selected the site from the
3.1 M facet ver512q model, which resolves 19 m against the flight mesh's 152 m. That is a site the
simulation does not have: at the same latitude and longitude the two surfaces differ by 12.7 m in
radius and 8.7 deg in slope, so the guidance would have been targeting a facet whose character it
could not represent. The site must be chosen on the surface the trajectory is actually flown over.

THE QUANTITY IS SLOPE FROM THE LOCAL VERTICAL, not relief, with the vertical computed per facet as
-g_eff/||g_eff|| from the polyhedral field plus the centrifugal term. Flat with respect to the
direction a released object falls is what a lander needs; flat with respect to the radius is not a
physical statement on a body this elongated.
"""
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, ConnectionPatch
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation
import trimesh

TH = "/Users/rodrigoalves/Desktop/Thesis"
sys.path.insert(0, os.path.join(TH, "FullModel", "SCvx"))
import config as C
import gravity as GR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render as RD

LAT, LON = (-55.0, -25.0), (60.0, 100.0)      # the NEAR Shoemaker / Himeros quadrant
NEAR = (-35.7, 80.5)                          # where NEAR Shoemaker touched down
SITES = [(-36.975, 95.401, "A"), (-49.585, 78.014, "B"), (-43.167, 94.908, "C"),
         (-52.428, 74.489, "D"), (-53.717, 60.377, "E")]
CACHE = os.path.join(HERE, "data", "flight_mesh_slope.npz")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"{stem}.{ext}"), dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    print("wrote", stem + ".{pdf,png}", flush=True)


v, f = GR._mesh(C.MESH_TAB)
M = trimesh.Trimesh(vertices=v, faces=f, process=False)
cen, nf = M.triangles_center, M.face_normals
rr = np.linalg.norm(cen, axis=1)
clat = np.degrees(np.arcsin(cen[:, 2] / rr))
clon = np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) % 360.0
inreg = ((clat >= LAT[0]) & (clat <= LAT[1]) & (clon >= LON[0]) & (clon <= LON[1]))

if not os.path.exists(CACHE):
    fld, w = GR.Field(), C.W_BODY
    sel = np.where(inreg)[0]
    sl = np.empty(sel.size)
    for i, k in enumerate(sel):
        g = np.asarray(fld.g(cen[k]), float) - np.cross(w, np.cross(w, cen[k]))
        sl[i] = np.degrees(np.arccos(np.clip(abs(nf[k] @ (-g / np.linalg.norm(g))), -1, 1)))
    np.savez_compressed(CACHE, idx=sel, lat=clat[sel], lon=clon[sel], slope=sl, r=rr[sel],
                        centre=cen[sel])
z = np.load(CACHE)
rlat, rlon, slope = z["lat"], z["lon"], z["slope"]
props = dict(mesh=os.path.basename(C.MESH_TAB), vertices=int(v.shape[0]), facets=int(f.shape[0]),
             facet_m=float(np.sqrt(M.area / len(f))),
             extent_km=[round(float(x) / 1e3, 3) for x in (v.max(0) - v.min(0))],
             volume_km3=round(float(M.volume) / 1e9, 3), area_km2=round(float(M.area) / 1e6, 3),
             bulk_density=C.BULK_DENSITY, GM=float(6.67430e-11 * M.volume * C.BULK_DENSITY),
             rot_period_h=C.EROS_ROT_PERIOD / 3600.0, omega_rad_s=float(C.W_BODY[2]),
             r_min_km=round(float(np.linalg.norm(v, axis=1).min()) / 1e3, 4),
             r_max_km=round(float(np.linalg.norm(v, axis=1).max()) / 1e3, 4),
             region=dict(lat=LAT, lon=LON, facets=int(slope.size),
                         slope_median=float(np.median(slope)),
                         slope_p5=float(np.percentile(slope, 5)), slope_min=float(slope.min()),
                         frac_below_10deg=float((slope < 10).mean())),
             sites=[dict(name=n, lat=a, lon=o) for a, o, n in SITES])
json.dump(props, open(os.path.join(HERE, "data", "shape_properties.json"), "w"), indent=1)
print(f"region {slope.size} facets: median {np.median(slope):.2f}  p5 {np.percentile(slope,5):.2f}"
      f"  min {slope.min():.2f} deg", flush=True)

# ---------------------------------------------------------------- one figure, body + zoom
# THE TWO PANELS ARE ONE FIGURE so the zoom can be drawn as a zoom: a rectangle around the region
# on the body, and leader lines to the corners of the map. As separate subfigures the reader has to
# be told in the caption which part of the body the map covers; this way the figure says it.
W, H, SS = 1500, 1180, 3
eye, right, up = RD.camera(lon_deg=78.0, lat_deg=-30.0)
P = v - v.mean(0)
uu, vv = P @ right, P @ up
mpp = max(np.ptp(uu) / (W * 0.93), np.ptp(vv) / (H * 0.93))
centre = v.mean(0) + 0.5 * (uu.min() + uu.max()) * right + 0.5 * (vv.min() + vv.max()) * up
view = RD.View(eye, right, up, centre, mpp, W, H)
t0 = time.perf_counter()
nrm, fbuf, hit, zbuf = RD.rasterise(v, f, view, ss=SS, flat=True)
I = RD.shade(nrm, hit, view)
print(f"rasterised {f.shape[0]:,} facets  [{time.perf_counter()-t0:.0f}s]", flush=True)

reg_px = hit & inreg[np.clip(fbuf, 0, None)]
floor = np.array([0.10, 0.10, 0.105]); tint = np.array([0.85, 0.30, 0.07])
rgb = np.ones((*I.shape, 3))
rgb[hit] = (floor + np.stack([I] * 3, axis=-1) * (1 - floor))[hit]
rgb[reg_px] = np.clip(tint[None, :] * (0.38 + 0.78 * I[reg_px])[:, None], 0, 1)
rgb[RD.facet_edges(fbuf, hit)] *= 0.62
img = np.clip(RD.downsample(rgb, SS), 0, 1)
reg_ds = RD.downsample(reg_px.astype(float), SS) > 0.25
ys, xs = np.where(reg_ds)
x0, x1b, y0b, y1b = xs.min() - 12, xs.max() + 12, ys.min() - 12, ys.max() + 12

gx = np.linspace(LON[0], LON[1], 150)
gy = np.linspace(LAT[0], LAT[1], 150)
GX, GY = np.meshgrid(gx, gy)
S = griddata(np.column_stack([rlon, rlat]), slope, (GX, GY), method="linear")

fig = plt.figure(figsize=(7.1, 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)
axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])

axL.imshow(img, interpolation="bilinear", zorder=1)
# CROP TO THE BODY. The render is sized so the body fills it horizontally, which leaves a lot of
# empty frame above and below; the panel is half a figure wide here, so that space is expensive.
solid = RD.downsample(hit.astype(float), SS) > 0.5
bys, bxs = np.where(solid)
bx0, bx1 = bxs.min(), bxs.max()
by0, by1 = bys.min(), bys.max()
axL.set_xlim(bx0 - 30, bx1 + 30)
axL.set_ylim(by1 + 105, by0 - 185)
axL.set_axis_off()
axL.add_patch(Rectangle((x0, y0b), x1b - x0, y1b - y0b, fill=False, ec="#d94801", lw=1.3,
                        zorder=6))
halo = [pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()]
col = int(round(view.project(np.zeros((1, 3)))[0, 0]))
y_body = int(np.where(solid[:, col])[0].min())
y_tip, y_tail = max(50, y_body - 170), y_body - 8
hl, hw = 44.0, 14.0
axL.plot([col, col], [y_tail, y_tip + 0.55 * hl], color="#1d3557", lw=2.2,
         solid_capstyle="butt", zorder=4, path_effects=halo)
axL.fill([col, col + hw, col - hw], [y_tip, y_tip + hl, y_tip + hl], color="#1d3557",
         zorder=4, lw=0, path_effects=halo)
axL.text(col, y_tip - 12, r"$\boldsymbol{\omega}$", color="#1d3557", fontsize=13,
         ha="center", va="bottom", zorder=5, path_effects=halo)
bar = 10e3 / mpp
xb1, yb0 = bx1, by1 + 78.0
axL.plot([xb1 - bar, xb1], [yb0, yb0], color="0.12", lw=2.0, solid_capstyle="butt", zorder=5)
for xb in (xb1 - bar, xb1):
    axL.plot([xb, xb], [yb0 - 7, yb0 + 7], color="0.12", lw=2.0, zorder=5)
axL.text(xb1 - bar / 2, yb0 - 11, "10 km", ha="center", va="bottom", fontsize=9, color="0.12")

im = axR.pcolormesh(GX, GY, S, cmap="Blues", vmin=0, vmax=35, shading="gouraud", rasterized=True)
cs = axR.contour(GX, GY, S, levels=[10.0], colors="0.25", linewidths=0.6)
axR.clabel(cs, fmt=r"%d$^\circ$", fontsize=7, inline_spacing=3)
cb = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.02)
cb.set_label(r"slope from local vertical  [$^\circ$]", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
halo2 = [pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()]
# THE MARKERS ARE IDENTIFIED IN THE FIGURE, not in the caption.
axR.plot(NEAR[1], NEAR[0], "o", ms=6, mfc="none", mec="0.15", mew=1.4, zorder=6,
         label="NEAR touchdown")
for i, (la_, lo_, nm) in enumerate(SITES):
    axR.plot(lo_, la_, "*", ms=11, color="#d94801", mec="white", mew=0.8, zorder=7,
             label="candidate site" if i == 0 else None)
    axR.annotate(nm, (lo_, la_), xytext=(6, -10), textcoords="offset points", fontsize=9,
                 color="#8c2d04", fontweight="bold", zorder=7, path_effects=halo2)
leg = axR.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.88,
                 borderpad=0.4, handletextpad=0.4, labelspacing=0.3)
leg.get_frame().set_linewidth(0.5); leg.get_frame().set_edgecolor("0.6")
axR.set_xlabel(r"east longitude  [$^\circ$]", fontsize=9)
axR.set_ylabel(r"latitude  [$^\circ$]", fontsize=9)
axR.tick_params(labelsize=8)
axR.set_xlim(*LON); axR.set_ylim(*LAT)
for sp in axR.spines.values():
    sp.set_edgecolor("#d94801"); sp.set_linewidth(1.3)

# the leader lines that make it read as a zoom
for yr, ya in ((y0b, 1.0), (y1b, 0.0)):
    fig.add_artist(ConnectionPatch(xyA=(x1b, yr), coordsA=axL.transData,
                                   xyB=(0.0, ya), coordsB=axR.transAxes,
                                   color="#d94801", lw=0.9, ls=(0, (4, 3)), zorder=10))
save(fig, "fig_shape_and_region")
plt.close(fig)

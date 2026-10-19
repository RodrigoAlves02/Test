#!/usr/bin/env python
"""A z-buffered software renderer for the shape model.

WHY NOT Poly3DCollection: it has no z-buffer. It sorts whole facets by an average depth and paints
them back to front, so on a 49 152-facet concave body the sort is wrong along every limb and the
silhouette breaks up. It also shades per facet, which on a 152 m mesh turns the regolith texture
into noise rather than relief. This renders with a per-pixel depth test and interpolated vertex
normals, which is what makes the craters read.

ORTHOGRAPHIC, not perspective: the figure is a shape plot, so equal lengths must measure equal
across the frame for the scale bar to mean anything.

The projection is returned with the image so that annotations (the spin axis, the site marker, the
scale bar) are placed in the SAME camera, not eyeballed onto the picture afterwards.
"""
import numpy as np


def camera(lon_deg, lat_deg, roll_deg=0.0):
    """Orthonormal camera basis. eye points FROM the body TOWARD the camera."""
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    eye = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    right = np.cross([0.0, 0.0, 1.0], eye)
    n = np.linalg.norm(right)
    right = np.array([0.0, 1.0, 0.0]) if n < 1e-9 else right / n
    up = np.cross(eye, right)
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    right, up = c * right + s * up, -s * right + c * up
    return eye, right, up


class View:
    """Camera plus image geometry. project() maps world metres to pixel coordinates."""

    def __init__(self, eye, right, up, centre, mpp, w, h):
        self.eye, self.right, self.up = eye, right, up
        self.centre, self.mpp, self.w, self.h = centre, mpp, w, h

    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float)) - self.centre
        return np.column_stack([self.w / 2 + (P @ self.right) / self.mpp,
                                self.h / 2 - (P @ self.up) / self.mpp])

    def depth(self, P):
        return (np.atleast_2d(np.asarray(P, float)) - self.centre) @ self.eye


def vertex_normals(V, F):
    """Area-weighted vertex normals. Area weighting is what keeps the shading continuous where the
    quad patches of the Gaskell parameterisation meet and the triangle sizes jump."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    fn = np.cross(e1, e2)                       # magnitude = 2 * area, so weighting is implicit
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(n, 1e-300), fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)


def rasterise(V, F, view, ss=2, flat=False):
    """Depth test every facet at ss x resolution.

    Returns the unit normal per subpixel, the facet index per subpixel, and the coverage mask.
    Back faces are dropped first, which halves the loop and removes the far wall of the body from
    the depth test entirely.

    flat=True shades with the FACET normal instead of the interpolated vertex normal. Smooth
    shading is the prettier picture, but it hides the triangulation, and the triangulation is the
    object this figure is about: the polyhedral gravity model integrates over these facets, so the
    figure should show them.
    """
    vn, fn = vertex_normals(V, F)
    W, H = view.w * ss, view.h * ss
    mpp = view.mpp / ss
    P = V - view.centre
    X = W / 2 + (P @ view.right) / mpp
    Y = H / 2 - (P @ view.up) / mpp
    D = P @ view.eye

    keep = np.where(fn @ view.eye > 0.0)[0]
    zbuf = np.full((H, W), -np.inf)
    fbuf = np.full((H, W), -1, np.int64)
    nbuf = np.zeros((H, W, 3))

    x0, x1, x2 = X[F[keep, 0]], X[F[keep, 1]], X[F[keep, 2]]
    y0, y1, y2 = Y[F[keep, 0]], Y[F[keep, 1]], Y[F[keep, 2]]
    d0, d1, d2 = D[F[keep, 0]], D[F[keep, 1]], D[F[keep, 2]]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    lo_x = np.maximum(np.floor(np.minimum.reduce([x0, x1, x2])).astype(int), 0)
    hi_x = np.minimum(np.ceil(np.maximum.reduce([x0, x1, x2])).astype(int), W - 1)
    lo_y = np.maximum(np.floor(np.minimum.reduce([y0, y1, y2])).astype(int), 0)
    hi_y = np.minimum(np.ceil(np.maximum.reduce([y0, y1, y2])).astype(int), H - 1)

    for i in range(keep.size):
        if abs(den[i]) < 1e-12 or hi_x[i] < lo_x[i] or hi_y[i] < lo_y[i]:
            continue
        gx = np.arange(lo_x[i], hi_x[i] + 1)[None, :]
        gy = np.arange(lo_y[i], hi_y[i] + 1)[:, None]
        w0 = ((y1[i] - y2[i]) * (gx - x2[i]) + (x2[i] - x1[i]) * (gy - y2[i])) / den[i]
        w1 = ((y2[i] - y0[i]) * (gx - x2[i]) + (x0[i] - x2[i]) * (gy - y2[i])) / den[i]
        w2 = 1.0 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any():
            continue
        dd = w0 * d0[i] + w1 * d1[i] + w2 * d2[i]
        sl = (slice(lo_y[i], hi_y[i] + 1), slice(lo_x[i], hi_x[i] + 1))
        upd = m & (dd > zbuf[sl])
        if not upd.any():
            continue
        zbuf[sl][upd] = dd[upd]
        fbuf[sl][upd] = keep[i]
        if flat:
            nbuf[sl][upd] = fn[keep[i]]
        else:
            a, b, c = F[keep[i]]
            nn = (w0[..., None] * vn[a] + w1[..., None] * vn[b] + w2[..., None] * vn[c])
            nbuf[sl][upd] = nn[upd]
    nrm = np.linalg.norm(nbuf, axis=2, keepdims=True)
    return nbuf / np.maximum(nrm, 1e-300), fbuf, fbuf >= 0, zbuf


def shade(normals, hit, view, key=(-0.52, 0.40, 0.76), fill=(0.65, -0.15, 0.30),
          amb=0.12, k_key=0.74, k_fill=0.14, gamma=1.0):
    """Lambert with a key light and a weak opposing fill, in CAMERA coordinates.

    The key sits 40 deg off the view axis so there is a terminator and the relief has a direction;
    the fill stops the dark limb going to a flat black, which is what hides the shape. Both are
    given in (right, up, eye) so the lighting does not change when the view does.
    """
    def world(c):
        d = c[0] * view.right + c[1] * view.up + c[2] * view.eye
        return d / np.linalg.norm(d)

    I = (amb + k_key * np.clip(normals @ world(key), 0, None)
             + k_fill * np.clip(normals @ world(fill), 0, None))
    return np.clip(I, 0, 1) ** gamma * hit


def downsample(img, ss):
    """Box filter back to the requested size: the only antialiasing on the silhouette."""
    if ss == 1:
        return img
    h, w = img.shape[0] // ss, img.shape[1] // ss
    tail = img.shape[2:]
    return img[:h * ss, :w * ss].reshape(h, ss, w, ss, *tail).mean(axis=(1, 3))


def facet_edges(fbuf, hit):
    """Facet boundaries, taken as discontinuities of the id buffer.

    NOT drawn as line segments: projecting 73 728 edges and stroking them would z-fight with the
    surface they lie on and would need its own depth test. A change in the id buffer between
    neighbouring subpixels IS the visible boundary, already correctly occluded, and at ss > 1 it
    downsamples to an antialiased hairline of exactly the right weight.
    """
    e = np.zeros(fbuf.shape, bool)
    e[:, 1:] |= fbuf[:, 1:] != fbuf[:, :-1]
    e[1:, :] |= fbuf[1:, :] != fbuf[:-1, :]
    e[:, :-1] |= fbuf[:, :-1] != fbuf[:, 1:]
    e[:-1, :] |= fbuf[:-1, :] != fbuf[1:, :]
    return e & hit

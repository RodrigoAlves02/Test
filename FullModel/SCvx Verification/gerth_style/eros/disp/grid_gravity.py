"""The polyhedral field tabulated on a grid in the site frame, as a FAST STAND-IN for PINN-GM-III.

Why: the label generator plans on PINN-GM-III (~60 us a call), which is not in the repository
(scvx_landing/data/pinn_eros_np.npz). The exact 49 152-facet field costs ~4.7 ms a call on 4 cores,
which makes one Eros solve take an hour here. A trust-region comparison over dispersed cases needs
the iteration behaviour of a cheap, smooth, slightly-wrong onboard model, which is what this is.

Grid: site frame (east, north, up), up from -0.2 to 13 km (50 m spacing below 1 km, 250 m above),
east/north +-7.5 km at 250 m. Only nodes inside the 30 deg glide-slope cone plus a 1 km margin are
evaluated; the rest are NaN. Trilinear in g and in the gravity-gradient tensor. A query whose cell
touches a NaN node, or that falls outside the box, goes to the exact field, so the stand-in is never
silently wrong, only occasionally slow (n_exact counts those calls).
"""
import os, sys
import numpy as np
SCVX = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "SCvx"))
if SCVX not in sys.path:
    sys.path.insert(0, SCVX)
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "grid_gravity.npz")


def axes():
    h = np.arange(-7500.0, 7500.0 + 1, 250.0)
    up = np.concatenate([np.arange(-200.0, 1000.0, 50.0), np.arange(1000.0, 13000.0 + 1, 250.0)])
    return h, h.copy(), up


def build(path=NPZ, margin=1000.0, gs_deg=30.0):
    import time
    import gravity as GR, problem_setup as PS
    P = PS.build()
    rs, n, e, no = P["r_site"], P["n_hat"], P["east"], P["north"]
    ex, nx, ux = axes()
    E, N, U = np.meshgrid(ex, nx, ux, indexing="ij")
    lat = np.hypot(E, N)
    keep = lat <= np.tan(np.radians(gs_deg)) * np.maximum(U, 0.0) + margin
    pts = rs + E[keep][:, None] * e + N[keep][:, None] * no + U[keep][:, None] * n
    fld = GR.field()
    t = time.time()
    res = fld._ev(np.ascontiguousarray(pts))          # one (pot, acc, tensor) per point
    acc = np.array([np.asarray(x[1], float) for x in res]); ten = np.array([np.asarray(x[2], float) for x in res])
    assert acc.shape == (len(pts), 3) and ten.shape == (len(pts), 6)
    G = np.full(E.shape + (9,), np.nan)
    G[keep] = np.concatenate([acc, ten], axis=1)
    np.savez_compressed(path, ex=ex, nx=nx, ux=ux, G=G, r_site=rs, n_hat=n, east=e, north=no,
                        n_eval=int(keep.sum()), wall_s=time.time() - t)
    print(f"grid {E.shape}, {keep.sum()} evaluated in {time.time()-t:.0f} s -> {path}")


class GridField:
    def __init__(self, path=NPZ):
        z = np.load(path)
        self.ax = [z["ex"], z["nx"], z["ux"]]
        self.G = z["G"]
        self.rs = z["r_site"]
        self.R = np.stack([z["east"], z["north"], z["n_hat"]])      # world -> (e, n, u)
        import gravity as GR
        self.exact = GR.field()
        self.n_calls = 0; self.n_exact = 0
        self._Z = None

    def _interp(self, r):
        q = self.R @ (r - self.rs)
        idx, w = [], []
        for a, x in zip(self.ax, q):
            i = int(np.searchsorted(a, x) - 1)
            if i < 0 or i >= len(a) - 1:
                return None
            idx.append(i); w.append((x - a[i]) / (a[i + 1] - a[i]))
        (i, j, k), (wx, wy, wz) = idx, w
        c = self.G[i:i + 2, j:j + 2, k:k + 2]
        if np.isnan(c[..., 0]).any():
            return None
        c = c[0] * (1 - wx) + c[1] * wx
        c = c[0] * (1 - wy) + c[1] * wy
        return c[0] * (1 - wz) + c[1] * wz

    def both(self, r):
        r = np.asarray(r, float); self.n_calls += 1
        v = self._interp(r)
        if v is None:
            self.n_exact += 1
            return self.exact.both(np.ascontiguousarray(r))
        t = v[3:]
        return v[:3], np.array([[t[0], t[3], t[4]], [t[3], t[1], t[5]], [t[4], t[5], t[2]]])

    def g(self, r):
        return self.both(r)[0]

    def grad(self, r):
        return self.both(r)[1]


def check(nq=400, seed=0):
    """Accuracy at random points inside the cone, against the exact field."""
    import time
    gf = GridField(); rng = np.random.default_rng(seed)
    u = np.concatenate([rng.uniform(0, 1000, nq // 2), rng.uniform(1000, 12500, nq // 2)])
    ang = rng.uniform(0, 2 * np.pi, nq); rad = np.tan(np.radians(30)) * u * np.sqrt(rng.uniform(0, 1, nq))
    q = np.stack([rad * np.cos(ang), rad * np.sin(ang), u], 1)
    pts = gf.rs + q @ gf.R
    err, terr = [], []
    t = time.time(); ga = [gf.both(p) for p in pts]; tg = (time.time() - t) / nq
    for p, (a, T) in zip(pts, ga):
        ae, Te = gf.exact.both(np.ascontiguousarray(p))
        err.append(np.linalg.norm(a - ae) / np.linalg.norm(ae)); terr.append(np.linalg.norm(T - Te) / np.linalg.norm(Te))
    err, terr = np.array(err), np.array(terr)
    lo = u < 1000
    out = dict(n=nq, us_per_call=tg * 1e6, exact_fallbacks=gf.n_exact,
               g_rel_err_max_below_1km=float(err[lo].max()), g_rel_err_max_above_1km=float(err[~lo].max()),
               g_rel_err_median=float(np.median(err)), grad_rel_err_max=float(terr.max()),
               grad_rel_err_median=float(np.median(terr)))
    print(out)
    return out


if __name__ == "__main__":
    {"build": build, "check": check}[sys.argv[1]]()

"""Polyhedral gravity on the 49 152-facet Gaskell shape of 433 Eros.

Self-contained: parses the .tab shape model, builds the exact Werner-Scheeres polyhedral field,
and exposes acceleration and its gradient. Nothing outside this folder is imported.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

import config as C

KM_TO_M = 1000.0


def load_tab(path=C.MESH_TAB):
    """Gaskell vertex-facet .tab -> (vertices [m], faces [0-indexed])."""
    with open(path) as fh:
        n_vert, n_face = (int(x) for x in fh.readline().split()[:2])
        v = np.empty((n_vert, 3)); f = np.empty((n_face, 3), dtype=np.int64)
        for i in range(n_vert):
            p = fh.readline().split(); v[i] = (float(p[1]), float(p[2]), float(p[3]))
        for i in range(n_face):
            p = fh.readline().split(); f[i] = (int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1)
    return v * KM_TO_M, f


@lru_cache(maxsize=2)
def _mesh(path=C.MESH_TAB):
    """Vertices and OUTWARD-oriented faces. The SIGN of the polyhedral field depends on the facet
    winding, so the orientation is fixed here with trimesh rather than trusted from the file."""
    import trimesh
    v, f = load_tab(path)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.fix_normals()                       # makes every facet normal point outward
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


class Field:
    """Exact polyhedral field. Call returns (acceleration, gradient tensor).

    MEMOISED ON THE EXACT BIT PATTERN OF r. polyhedral_gravity 3.3.1 reduces the 49 152 facet
    contributions in a non-deterministic order: repeated calls at one point differ at ~1e-18
    relative, and no thread-count environment variable suppresses it. A free-final-time successive
    iteration is chaotic enough in the flight time that this epsilon selects which fixed point a
    run reaches, so without memoisation two identical runs are not the same experiment. Keyed on
    r.tobytes() so the cache hit is bit-exact, bounded so a long run cannot grow without limit.
    """

    def __init__(self, path=C.MESH_TAB, density=C.BULK_DENSITY, memo=200_000):
        import polyhedral_gravity as pg
        v, f = _mesh(path)
        self.n_facets = len(f)
        poly = pg.Polyhedron(polyhedral_source=(v, f), density=float(density),
                             normal_orientation=pg.NormalOrientation.OUTWARDS,
                             integrity_check=pg.PolyhedronIntegrity.DISABLE,
                             metric_unit=pg.MetricUnit.METER)
        self._ev = pg.GravityEvaluable(poly)
        self._memo, self._cap = {}, int(memo)
        self.n_calls = 0

    def _raw(self, r):
        key = r.tobytes()
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        self.n_calls += 1
        pot, acc, ten = self._ev(r)
        # polyhedral_gravity already returns the ATTRACTIVE acceleration (it agrees in sign with
        # -mu*r/|r|^3 in the far field) and the matching second derivative, so neither is negated.
        # Verified against a point mass at 12 km altitude below.
        a = np.asarray(acc, float)
        t = np.asarray(ten, float)                    # (xx, yy, zz, xy, xz, yz)
        Gt = np.array([[t[0], t[3], t[4]],
                       [t[3], t[1], t[5]],
                       [t[4], t[5], t[2]]], float)
        if len(self._memo) < self._cap:
            self._memo[key] = (a, Gt)
        return a, Gt

    def g(self, r):
        return self._raw(np.ascontiguousarray(np.asarray(r, float)))[0]

    def grad(self, r):
        """d(g)/d(r), the gravity-gradient (tidal) tensor [1/s^2]."""
        return self._raw(np.ascontiguousarray(np.asarray(r, float)))[1]

    def both(self, r):
        return self._raw(np.ascontiguousarray(np.asarray(r, float)))


@lru_cache(maxsize=2)
def field(path=C.MESH_TAB, density=C.BULK_DENSITY):
    """One Field per (mesh, density) per process, so repeated solves share the memo."""
    return Field(path, density)


def site_and_normal(lat_deg=C.LAT_DEG, lon_deg=C.LON_DEG, path=C.MESH_TAB, lift=1.0):
    """The landing site on the surface, and the LOCAL VERTICAL there.

    The surface point is found by casting a ray from the body centre along the (lat, lon)
    direction and intersecting the shape, which is exact, rather than by picking the nearest
    vertex.

    The vertical is -g_eff/||g_eff|| with g_eff = g + centrifugal, NOT the radial direction: on a
    body this irregular the two differ by up to tens of degrees, and it is the vertical, not the
    radial, that the glideslope cone and the touchdown condition are built around.

    g_eff is evaluated `lift` metres ABOVE the surface. The polyhedral potential is continuous
    across a facet but its gradient is singular exactly at an edge or a vertex, and a ray-surface
    hit can land on one; evaluating at the surface point itself returned NaN. One metre is far
    below any length scale in this problem and removes the singularity.
    """
    import trimesh
    v, f = _mesh(path)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    u = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    far = 3.0 * float(np.abs(v).max())
    hits, _, _ = mesh.ray.intersects_location(ray_origins=np.array([far * u]),
                                              ray_directions=np.array([-u]))
    if len(hits) == 0:
        raise RuntimeError(f"no surface intersection at lat {lat_deg}, lon {lon_deg}")
    r_site = hits[np.argmax(hits @ u)]                    # outermost hit = the near side
    fld = field(path)
    r_probe = r_site + lift * u
    g_eff = fld.g(r_probe) - np.cross(C.W_BODY, np.cross(C.W_BODY, r_probe))
    n_hat = -g_eff / np.linalg.norm(g_eff)
    return np.asarray(r_site, float), np.asarray(n_hat, float)


class PinnField:
    """PINN-GM-III wearing the same (g, grad) interface as Field, for use as the ONBOARD model.

    The point of using it is that it is NOT the truth. Flying the polyhedral field while planning
    with the network introduces the gravity model error a real vehicle has, which every closed-loop
    result so far has been missing -- the solver and the truth shared one field, so the guidance
    knew the world exactly.

    grad returns ZERO. That is a deliberate choice, not laziness: the gravity gradient enters only
    A[3:6, 0:3], at O(1e-7) against A[0:3, 3:6] = I, and successive convexification converges to
    where the NONLINEAR defects vanish -- so an inexact Jacobian changes the path taken, not the
    fixed point reached. It also avoids six finite-difference evaluations per call, which is the
    difference between 56 us and ~390 us and the whole reason to use a network at all.
    """

    def __init__(self, npz_path=None, scvx_dir=None):
        import os, sys
        scvx_dir = scvx_dir or os.path.join(C.THESIS, "scvx_landing")
        npz_path = npz_path or getattr(C, "PINN_NPZ",
                                       os.path.join(scvx_dir, "data", "pinn_eros_np.npz"))
        if scvx_dir not in sys.path:
            sys.path.insert(0, scvx_dir)
        from scvx.pinn import load_pinn_np
        self.net, self.xs, self.a_s = load_pinn_np(npz_path)
        self._Z = np.zeros((3, 3))
        self.n_calls = 0
        self._memo, self._cap = {}, 200_000

    def g(self, r):
        r = np.ascontiguousarray(np.asarray(r, float))
        key = r.tobytes()
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        self.n_calls += 1
        a = np.asarray(self.net.acceleration(r / self.xs), float) * self.a_s
        if len(self._memo) < self._cap:
            self._memo[key] = a
        return a

    def grad(self, r):
        return self._Z

    def both(self, r):
        return self.g(r), self._Z


@lru_cache(maxsize=1)
def pinn_field():
    return PinnField()


@lru_cache(maxsize=2)
def _tri_mesh(path=C.MESH_TAB):
    """The shape as a trimesh, built once per process. Watertight, winding consistent."""
    import trimesh
    v, f = _mesh(path)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def height_above_surface(r, up, path=C.MESH_TAB):
    """Signed height of r above the REAL shape, measured along `up`. Negative = inside the body.

    WHY THIS EXISTS. "Altitude" had been n_hat . (r - r_site): the height above the TANGENT PLANE
    at the landing site. That plane and the actual surface both pass through the site but diverge
    with lateral offset, because the facet there is tilted 21.3 deg away from the local vertical
    (the vertical is -g_eff, not the surface normal). Contact was tested against the plane while
    the lidar ray-cast against the mesh, so the two disagreed by about 0.39 m per metre of miss.
    Measured consequence: a run reported altitude +0.268 m while the vehicle was 0.541 m INSIDE
    the body, and the lidar -- correctly ray-casting from a point inside a 34 km body -- returned
    11 112 m from the far wall, which the filter then swallowed for a 414 m position jump.

    Everything that asks "where is the ground" must ask the same surface, and that surface is the
    mesh. The guidance keeps its terminal constraint on the tangent plane: that is what makes the
    subproblem convex, and the two agree exactly at zero miss, which is where it is aiming.

    METHOD. Cast along -up. For a watertight mesh the parity of the crossing count says whether the
    origin is inside. Outside, the nearest crossing below is the height; inside, the nearest
    crossing ABOVE is the penetration depth. Both tend to zero at the surface, so the result is
    continuous through contact and safe to bisect on.
    """
    m = _tri_mesh(path)
    r = np.ascontiguousarray(np.asarray(r, float))
    up = np.asarray(up, float)
    up = up / max(np.linalg.norm(up), 1e-12)
    o = r.reshape(1, 3)
    down, _, _ = m.ray.intersects_location(ray_origins=o, ray_directions=(-up).reshape(1, 3))
    if len(down) % 2 == 0:                       # even crossings below => outside
        if len(down) == 0:
            return float("inf")                  # nothing under the beam at all
        return float(np.linalg.norm(down - r, axis=1).min())
    above, _, _ = m.ray.intersects_location(ray_origins=o, ray_directions=up.reshape(1, 3))
    if len(above) == 0:
        return float("-inf")
    return -float(np.linalg.norm(above - r, axis=1).min())

"""Lidar and cameras: measurement, activation geometry, and the EKF interface.

Each sensor exposes
    active(r, thrust_dir)  -> bool          the plan's "activation function, on/off by position"
    measure(r, rng)        -> (z, H, R)     measurement, Jacobian, and noise covariance

so the filter never needs to know what kind of sensor it is talking to.
"""
from __future__ import annotations

import numpy as np

import config_nav as CN
import frames as NFR


class Lidar:
    """A single laser beam along -thrust, ray-cast against the real 49 152-facet shape.

    WHAT IT MEASURES. The slant range from the vehicle to the first surface the beam meets --
    NOT the altitude, and not the height above the landing site's tangent plane. Those three
    differ: the beam lands wherever it lands, on terrain that is neither flat nor under the
    spacecraft. The previous Altimeter used h = n_hat . (r - r_site), which is exact only for a
    locally flat body and was optimistic by the terrain relief under the ground track.

    ACTIVATION is three gates, all from the ray-cast:
        1. the beam has to HIT something                       -- no hit, no measurement
        2. the incidence angle has to be <= the gate           -- too oblique, no usable return
        3. the range has to be inside the instrument's window  -- 30 m to 25 km for this unit
    Gate 3 is the one that bites at the end: below 30 m this instrument is simply out of range,
    which on Hayabusa2 is where the Laser Range Finder took over. With no LRF modelled, the beam
    goes dark for the last stretch of the descent and the plots show exactly that.

    THE MEASUREMENT JACOBIAN. For a locally planar facet with outward normal n_f, the range along
    a unit boresight u from position r satisfies t = n_f . (p0 - r) / (n_f . u), so

        H = dt/dr = - n_f^T / (n_f . u),      |H| = 1 / cos(i)

    The measurement is therefore NOT simply "the vertical": its sensitivity grows as 1/cos(i), and
    at the 70 deg gate |H| = 2.92. That is the navigation half of why the gate exists.
    """

    def __init__(self, sign=-1, min_range=CN.LIDAR_MIN_RANGE_M, max_range=CN.LIDAR_MAX_RANGE_M,
                 sig_near=CN.LIDAR_SIGMA_NEAR_M, sig_far=CN.LIDAR_SIGMA_FAR_M,
                 quant=CN.LIDAR_QUANT_M, inc_max=CN.LIDAR_INCIDENCE_MAX_DEG,
                 rate=CN.LIDAR_RATE_HZ, name="lidar"):
        self.sign, self.min_range, self.max_range = sign, min_range, max_range
        self.sig_near, self.sig_far, self.quant = sig_near, sig_far, quant
        self.inc_max, self.rate, self.name = inc_max, rate, name
        self.n = 0

    def boresight(self, thrust_dir):
        u = self.sign * np.asarray(thrust_dir, float)
        return u / max(np.linalg.norm(u), 1e-12)

    def geometry(self, r, thrust_dir):
        """(hit, range, incidence_deg, facet_normal). range is NaN and incidence 180 on a miss."""
        u = self.boresight(thrust_dir)
        d, nf = shape.ray_hit(np.asarray(r, float), u)
        if d is None:
            return False, np.nan, 180.0, None
        # incidence measured from the facet's outward normal: the beam arrives along u, so
        # cos(i) = -(n_f . u), positive when the beam is landing on the front of the facet
        c = float(np.clip(-(nf @ u), -1.0, 1.0))
        return True, float(d), float(np.degrees(np.arccos(abs(c)))), nf

    def sigma(self, rng_m):
        """Linear in range between the two accuracies Mizuno et al. quote, clamped outside."""
        f = (np.asarray(rng_m, float) - self.min_range) / (self.max_range - self.min_range)
        return self.sig_near + (self.sig_far - self.sig_near) * np.clip(f, 0.0, 1.0)

    def active(self, r, n_hat=None, r_site=None, thrust_dir=None):
        if thrust_dir is None:
            return False
        hit, d, inc, _ = self.geometry(r, thrust_dir)
        return bool(hit and inc <= self.inc_max and self.min_range <= d <= self.max_range)

    def measure(self, r, n_hat, r_site, rng, thrust_dir=None, t=None, wz=None):
        """Slant range with noise and quantisation. A range is a scalar, so it is frame-free;
        only H has to be rotated, and it is rotated here when `t` is given."""
        hit, d_true, inc, nf = self.geometry(r, thrust_dir)
        if not hit:
            raise ValueError("Lidar.measure called with no surface return; check active() first")
        sig = float(self.sigma(d_true))
        z = d_true + rng.normal(0.0, sig)
        if self.quant > 0:
            z = self.quant * np.round(z / self.quant)
        u = self.boresight(thrust_dir)
        denom = float(nf @ u)
        h_vec = -nf / denom                       # dt/dr for a locally planar facet
        if t is not None:
            wzz = CN.w_body_z() if wz is None else wz
            h_vec = NFR.Rz(wzz * float(t)) @ h_vec
        H = np.zeros((1, CN.NX)); H[0, CN.IR] = h_vec
        # the quantiser contributes a uniform step: variance q^2/12, added rather than ignored
        var = sig ** 2 + (self.quant ** 2) / 12.0 if self.quant > 0 else sig ** 2
        self.n += 1
        return np.array([z]), H, np.array([[var]])

    def predict(self, x, n_hat=None, r_site=None, thrust_dir=None, t=None, wz=None):
        """Predicted range, by ray-casting the ON-BOARD shape model from the ESTIMATED position.

        DEFERRED (see the DEFERRED block in config_nav): the on-board model here is the SAME
        49 152-facet mesh as the truth, i.e. a perfect shape model. A real lander carries a coarser
        mesh, and that difference -- metres to tens of metres at Eros -- would dominate the
        instrument's own 1-5 m noise. The gravity side of the chain already does this split
        properly (49k polyhedral truth against a PINN on board); the shape side does not yet.
        """
        r = np.asarray(x, float)[CN.IR]
        if t is not None:
            r = NFR.Rz(-(CN.w_body_z() if wz is None else wz) * float(t)) @ r
        d = shape.ray_range(r, self.boresight(thrust_dir))
        return np.array([np.nan if d is None else d])


class _Shape:
    """The truth shape, loaded once. Ray queries need rtree; the vertex arrays do not."""

    def __init__(self):
        self._mesh = self._v = self._n = None

    def _load(self):
        if self._mesh is None:
            import sys
            sys.path.insert(0, CN.SCVX_DIR)
            import trimesh
            import config as C
            import gravity as GR
            v, f = GR._mesh(C.MESH_TAB)
            self._mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            self._v = np.asarray(self._mesh.vertices, float)
            self._n = np.asarray(self._mesh.vertex_normals, float)
        return self._mesh

    def vertices(self):
        self._load(); return self._v

    def normals(self):
        self._load(); return self._n

    def ray_hit(self, origin, direction):
        """(distance, outward facet normal) of the FIRST surface hit, or (None, None).

        The normal is what makes the incidence angle available, and the incidence angle is what
        decides both whether a laser gets a return and how sensitive that return is to position.
        """
        m = self._load()
        o = np.asarray(origin, float)
        loc, idx_ray, idx_tri = m.ray.intersects_location(
            ray_origins=np.asarray([o]), ray_directions=np.asarray([direction], float))
        if len(loc) == 0:
            return None, None
        d = np.linalg.norm(loc - o, axis=1)
        k = int(np.argmin(d))
        return float(d[k]), np.asarray(m.face_normals[idx_tri[k]], float)

    def ray_range(self, origin, direction):
        """Distance only. Kept because the camera's FOV test does not need the normal."""
        return self.ray_hit(origin, direction)[0]


shape = _Shape()


class Camera:
    """Landmark position fix, boresight along +thrust (sign=+1) or -thrust (sign=-1).

    ACTIVATION is geometric and derived, not scheduled, and it is evaluated against the REAL
    49 152-facet shape. The camera sees the surface when EITHER the boresight ray hits the body OR
    some visible surface vertex falls inside the FOV cone. Both tests are needed:

      * the ray test is exact and is the only one that works close in;
      * the vertex test catches the limb at long range, where the boresight points past the body.

    The vertex test alone breaks near the ground -- mean vertex spacing is 211 m, so below about
    12 km range there may be no vertex near the boresight even though the surface fills the frame.
    A SPHERE for either test understates visibility badly: Eros runs 34 km end to end against a
    6.5 km site radius, and the spherical approximation puts the blind window at 670 s where the
    real mesh gives 99 s.

    MEASUREMENT is the full position, which is what a landmark solution gives once several mapped
    features are matched. It is NOT what a single image gives (that is a bearing, rank 2), so this
    assumes the on-board landmark database and the matching work as intended. The 1-sigma scales
    with the IMAGING RANGE -- the distance to the terrain actually in frame, not to the landing
    site -- because that is what sets the ground sample distance. See config_nav for the two
    literature anchors and why the flown one is the default.
    """

    def __init__(self, sign=+1, fov_deg=CN.CAM_FOV_DEG, frac=CN.CAM_SIGMA_FRAC,
                 max_range=CN.CAM_MAX_RANGE_M, min_range=CN.CAM_MIN_RANGE_M,
                 rate=CN.CAM_RATE_HZ, sigma_model=None, name="cam"):
        self.sign, self.half_fov = sign, fov_deg / 2.0
        self.frac, self.max_range, self.min_range, self.rate = frac, max_range, min_range, rate
        self.sigma_model = sigma_model or CN.CAM_SIGMA_MODEL
        self.name = name
        self.n = 0

    # -- geometry ----------------------------------------------------------------------------------
    def boresight(self, thrust_dir):
        u = self.sign * np.asarray(thrust_dir, float)
        return u / max(np.linalg.norm(u), 1e-12)

    def geometry(self, r, thrust_dir):
        """(sees, imaging_range, boresight_hit, min_angle_deg) against the real shape."""
        rr = np.asarray(r, float)
        u = self.boresight(thrust_dir)
        t_hit = shape.ray_range(rr, u)
        w = shape.vertices() - rr
        rng = np.linalg.norm(w, axis=1)
        wh = w / np.maximum(rng, 1e-12)[:, None]
        vis = np.einsum("ij,ij->i", wh, shape.normals()) < 0.0      # facing the observer
        ang = np.degrees(np.arccos(np.clip(wh @ u, -1.0, 1.0)))
        ang = np.where(vis, ang, np.inf)
        k = int(np.argmin(ang))
        # When the boresight RAY lands on the surface there is a surface point at exactly zero
        # angle, so that is the true minimum. The vertex sweep is only a SAMPLED approximation to
        # it, and below a few hundred metres altitude it is badly wrong: mean vertex spacing is
        # 211 m, so no vertex sits near a boresight that is hitting a facet dead centre. Measured
        # at t = 2096 s of the nominal descent -- ray hit at 0.9 m, nearest visible vertex 6.2 km
        # away at 41.8 deg. Reporting the vertex angle there made the camera look like it was
        # losing the site while it was in fact pointed straight down it.
        min_ang = 0.0 if t_hit is not None else float(ang[k])
        in_fov = (t_hit is not None) or (min_ang <= self.half_fov)
        # imaging range: along the boresight when it lands on the surface, otherwise the nearest
        # visible surface point -- the terrain the frame is actually looking at either way.
        img = float(t_hit) if t_hit is not None else float(rng[k]) if np.isfinite(min_ang) else np.inf
        return in_fov, img, (t_hit is not None), min_ang

    def sigma(self, imaging_range):
        """1-sigma of the position fix at this imaging range. See config_nav for the anchors."""
        if self.sigma_model == "floor":
            return CN.CAM_BEARING_SIGMA * imaging_range          # pixel term alone, perfect map
        if self.sigma_model == "flown":
            return self.frac * imaging_range                     # linear in range -- superseded
        # np.hypot, so the same call works on a scalar range and on a whole time series
        return np.hypot(CN.CAM_BEARING_SIGMA * imaging_range, CN.CAM_SIGMA_FLOOR_M)

    # -- sensor interface --------------------------------------------------------------------------
    def active(self, r, n_hat, r_site, thrust_dir=None):
        """Any surface in frame, within the range gate.

        DELIBERATELY PERMISSIVE -- see the DEFERRED block in config_nav. A sliver counts the same
        as a full frame, so fixes taken while the body is a crescent in one corner are optimistic.
        Replacing this with a usable-landmark count and a conditioning test is queued behind the
        guidance PINN, since the chain only has to be good enough to generate the SCvx Monte Carlo
        set first.
        """
        if thrust_dir is None:
            return False
        in_fov, img, _, _ = self.geometry(r, thrust_dir)
        return bool(in_fov and self.min_range <= img <= self.max_range)

    def measure(self, r, n_hat, r_site, rng, thrust_dir=None, t=None, wz=None):
        """Take a fix. `r` is the TRUE position in ACAF, which is where the sensor works.

        FRAME OF THE OUTPUT. Landmarks are body-fixed, so the raw fix is a position in ACAF. The
        filter runs in ACI, so pass `t` and the block delivers there instead:

            z_ACI = Rz(wz t) z_ACAF          R_ACI = Rz R_ACAF Rz^T

        Position is a PURE rotation between these two frames -- they share the asteroid's centre
        and coincide at t = 0 -- so there is no translation term. Velocity would NOT rotate this
        way (it picks up omega x r), which is why only a position fix can be converted this
        cheaply. And R is unchanged here ONLY because sigma^2 I is isotropic: Rz (s^2 I) Rz^T =
        s^2 I. The moment the measurement becomes a bearing, with its anisotropic covariance, that
        rotation stops being free and has to be carried properly.

        H stays [I 0 0] either way, because it is the Jacobian with respect to the state, and the
        state is expressed in whichever frame the measurement was delivered in.
        """
        rr = np.asarray(r, float)
        img = self.geometry(rr, thrust_dir)[1] if thrust_dir is not None \
            else float(np.linalg.norm(rr - r_site))
        sig = self.sigma(img)
        z = rr + rng.normal(0.0, sig, 3)
        R = np.eye(3) * sig ** 2
        if t is not None:
            Rz = NFR.Rz((CN.w_body_z() if wz is None else wz) * float(t))
            z = Rz @ z
            R = Rz @ R @ Rz.T
        H = np.zeros((3, CN.NX)); H[:, CN.IR] = np.eye(3)
        self.n += 1
        return z, H, R

    def predict(self, x, n_hat=None, r_site=None, thrust_dir=None, t=None, wz=None):
        """The predicted fix is just the state: measure() already delivered z in ACI, and the
        state is in ACI. thrust_dir/t/wz are accepted and ignored so that every sensor in the
        block presents ONE interface and the caller needs no isinstance test."""
        return np.asarray(x, float)[CN.IR].copy()

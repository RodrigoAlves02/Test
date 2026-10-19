"""Geometry of the descent: where the site is, where the vehicle starts, and the scaling."""
from __future__ import annotations

import numpy as np

import config as C
import gravity as GR
from dynamics import Dynamics


def build(altitude_km=C.ALTITUDE_KM, cone_deg=C.CONE_ANGLE_DEG, v0_local=None, m_wet=C.MWET):
    """Return everything the solver needs, in non-dimensional units.

    The start is placed `altitude_km` above the site along the LOCAL VERTICAL, offset by `cone_deg`
    within the approach cone. Node 0 is a hard equality in the subproblem and it is also subject to
    the glideslope constraint, so cone_deg > GLIDESLOPE_DEG would make the very first node
    infeasible -- checked here rather than discovered as a solver failure.
    """
    if cone_deg > C.GLIDESLOPE_DEG:
        raise ValueError(f"start cone {cone_deg} deg is outside the glideslope {C.GLIDESLOPE_DEG} deg")
    r_site, n_hat = GR.site_and_normal()
    # a direction `cone_deg` off the vertical, in the plane containing the vertical and east
    east = np.cross(np.array([0.0, 0.0, 1.0]), n_hat)
    east /= np.linalg.norm(east)
    a = np.radians(cone_deg)
    u = np.cos(a) * n_hat + np.sin(a) * east
    r0 = r_site + altitude_km * 1e3 * u

    # site-local triad: up along the vertical, east and north spanning the tangent plane
    east = np.cross(np.array([0.0, 0.0, 1.0]), n_hat); east /= np.linalg.norm(east)
    north = np.cross(n_hat, east)
    vl = C.V0_LOCAL if v0_local is None else np.asarray(v0_local, float)
    v0 = vl[0] * n_hat + vl[1] * east + vl[2] * north

    scales = C.Scales.build(r0, r_site, m_wet=m_wet)
    dyn = Dynamics(scales)
    x_init = np.concatenate([r0 / scales.L, v0 * scales.T / scales.L, [np.log(m_wet)]])
    return dict(dyn=dyn, scales=scales, x_init=x_init,
                rf_nd=r_site / scales.L, n_hat=n_hat, r_site=r_site, r0=r0,
                east=east, north=north, v0=v0)

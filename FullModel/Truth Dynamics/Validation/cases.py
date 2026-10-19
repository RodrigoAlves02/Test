#!/usr/bin/env python
"""The three arcs the truth propagator has to handle, as a common interface.

Each builder returns a Case: an initial state, a time span, and a thrust(t). Nothing in a Case tells
the propagator where the thrust changes -- that is the point. The three arcs are deliberately very
different in character, and a single truth propagator setting has to cover all of them:

  parking  : long, unthrusted, many revolutions. Error growth is secular and the test is whether a
             setting survives the duration.
  braking  : one short burn, a long coast, a very short correction burn, another long coast. Two
             thrust discontinuities, and one segment far shorter than any sensible step.
  descent  : thirty zero-order-hold segments, thrust switching between its bounds throughout.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.join(HERE, "..", "..", "..", "scvx_landing")
sys.path.insert(0, SCVX)
sys.path.insert(0, HERE)

from truth_dynamics import EROS_GM, OMEGA, Vehicle, coast, segments_thrust, zoh_thrust  # noqa: E402


@dataclass
class Case:
    name: str
    y0: np.ndarray                       # [r(3), v(3), m(1)]
    t0: float
    tf: float
    thrust: Callable[[float], np.ndarray]
    veh: Vehicle
    note: str = ""


def circular_orbit(radius_km, retrograde=True, n_orbits=3.0, mass=600.0, gm=EROS_GM):
    """A circular orbit at a given radius, expressed in the body-fixed ROTATING frame.

    Retrograde by default. A retrograde orbit admits no C22 resonance at any radius, because
    4(n+w)^2 > 4n^2 > n^2 always, whereas a prograde orbit has resonant radii; that is the reason
    the parking orbit candidates are retrograde and it is worth the test being consistent with it.
    """
    R = radius_km * 1e3
    v_circ = np.sqrt(gm / R)
    r = np.array([R, 0.0, 0.0])
    v_in = np.array([0.0, -v_circ if retrograde else v_circ, 0.0])
    v_rot = v_in - np.cross(OMEGA, r)          # inertial -> rotating frame
    period = 2.0 * np.pi * np.sqrt(R ** 3 / gm)
    return Case(
        name=f"parking_{radius_km:g}km",
        y0=np.concatenate([r, v_rot, [mass]]),
        t0=0.0, tf=n_orbits * period,
        thrust=coast(), veh=Vehicle(),
        note=(f"{'retrograde' if retrograde else 'prograde'} circular, R = {radius_km:g} km, "
              f"period {period:.1f} s, {n_orbits:g} revolutions"))


def braking_case(mass=600.0):
    """The nominal braking + TCM chain already used elsewhere, taken as-is to exercise the model."""
    import transfer_sequence as T
    from deorbit import geometry, make_field
    _, rf, n_hat, r_h = geometry()
    _, g_raw = make_field("poly")

    class A:
        tb, tc, f_tcm = -7694.993879390684, 7768.421052631579, 0.70
        tcm_dv, tcm_dir, thrust, mass_ = 0.0, None, 20.0, mass
        dt_ref = 0.46875
    A.mass = mass
    r0, v0, m0, segs, dv1, info = T.build_mc(A, g_raw, rf, r_h, 0, disperse=False)
    tot = float(sum(s[1] for s in segs))
    return Case(
        name="braking",
        y0=np.concatenate([r0, v0, [m0]]),
        t0=0.0, tf=tot,
        thrust=segments_thrust(segs), veh=Vehicle(),
        note=" | ".join(f"{nm} {h:.3f}s" for nm, h, _ in segs))


def descent_case(npz=None):
    """An SCvx descent solution, flown as the zero-order-hold thrust it actually is."""
    npz = npz or os.path.join(SCVX, "descent_nominal.npz")
    z = np.load(npz)
    X, U, sig = z["X"], z["U"], float(z["sigma"])
    m_ref = np.exp(X[6, :-1])
    T_newton = m_ref * U[0:3, :-1]                 # optimiser control is T/m at the planned mass
    T_newton = np.concatenate([T_newton, T_newton[:, -1:]], axis=1)
    y0 = np.concatenate([X[0:3, 0], X[3:6, 0], [float(np.exp(X[6, 0]))]])
    return Case(
        name="descent",
        y0=y0, t0=0.0, tf=sig,
        thrust=zoh_thrust(T_newton, sig), veh=Vehicle(),
        note=(f"SCvx solution, t_f {sig:.2f} s, {X.shape[1]} nodes, "
              f"|T| {np.linalg.norm(T_newton, axis=0).min():.1f}-"
              f"{np.linalg.norm(T_newton, axis=0).max():.1f} N"))


BUILDERS = {"parking": circular_orbit, "braking": braking_case, "descent": descent_case}


def build(name, **kw):
    if name.startswith("parking"):
        return circular_orbit(kw.pop("radius_km", 35.0), **kw)
    if name == "braking":
        return braking_case(**kw)
    if name == "descent":
        return descent_case(**kw)
    raise ValueError(f"unknown case: {name}")

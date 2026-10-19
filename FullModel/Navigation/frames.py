"""Rotating <-> inertial conversions, in one place.

The two frames coincide at t = 0. The body spins about +z at W_BODY.

WHY THIS FILE EXISTS. Three things live in three frames and it is easy to mix them:
  * the TRUTH and the GUIDANCE work in the body-fixed ROTATING frame (the site is stationary there)
  * the IMU measures specific force in INERTIAL axes -- an accelerometer senses real forces, and
    Coriolis and centrifugal are not real forces, so the instrument never sees them
  * the gravity field is defined in BODY-FIXED coordinates
Every conversion in the navigation chain goes through here rather than being written inline.
"""
from __future__ import annotations

import numpy as np


def Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def vec_rot_to_inertial(a_rot, t, wz):
    """A FREE vector (a specific force, say): components only, no frame-rate term."""
    return Rz(wz * t) @ np.asarray(a_rot, float)


def vec_inertial_to_rot(a_I, t, wz):
    return Rz(-wz * t) @ np.asarray(a_I, float)

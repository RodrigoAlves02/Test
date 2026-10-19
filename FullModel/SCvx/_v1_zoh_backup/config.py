"""Vehicle, body, site and scaling constants for the Eros powered-descent problem.

Every number here is either a mission parameter or derived from one. Nothing is tuned.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# THREE LEVELS UP, NOT TWO. This file sits in SCvx/_v1_zoh_backup/, one deeper than the
# SCvx/config.py it was copied from, so the inherited "..", ".." resolved THESIS to FullModel/
# and MESH_TAB to a shape file that is not there. The constant was never changed; only the
# file moved.
THESIS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "..", ".."))
MESH_TAB = os.path.join(THESIS, "gaskell.ast-eros-2.shape-model_V1_1",
                        "data", "vertex", "ver64q.tab")      # 49 152 facets

# ---- body -------------------------------------------------------------------------------------
EROS_ROT_PERIOD = 5.270 * 3600.0          # s
W_BODY = np.array([0.0, 0.0, 2.0 * np.pi / EROS_ROT_PERIOD])
BULK_DENSITY = 2670.0                     # kg/m^3, Gaskell shape + NEAR mass

# ---- vehicle ----------------------------------------------------------------------------------
MWET = 600.0                              # kg
MDRY = 490.0                              # kg
TMAX = 20.0                               # N
TMIN = 5.0                                # N  -- hardware floor; the engine cannot throttle below
ISP = 225.0                               # s
G0 = 9.80665                              # m/s^2
ALPHA = 1.0 / (ISP * G0)                  # s/m, mass-flow coefficient: mdot = -ALPHA*||T||

# ---- landing site -----------------------------------------------------------------------------
LAT_DEG = -40.0                           # body-fixed latitude
LON_DEG = 80.7                            # body-fixed EAST longitude (279.3 W, the NEAR ponds)

# ---- mission constraints ----------------------------------------------------------------------
GLIDESLOPE_DEG = 30.0        # half-angle of the approach cone, measured from the LOCAL VERTICAL
# Max angle between the thrust direction and the LOCAL VERTICAL.
#
# 140 deg, not the 45 deg that "point the sensors down" suggests, and the reason is physical rather
# than a tuning choice. Tmin/m = 0.00833 m/s^2 exceeds gravity (0.0011 m/s^2 at 12 km to 0.0056 at
# the surface) EVERYWHERE on this descent, and the engine cannot be switched off. So at minimum
# throttle with the thrust anywhere within 45 deg of the vertical the net vertical acceleration is
# upward at every altitude (+4.8e-3 m/s^2 at 12 km, +3.3e-4 at the surface): the vehicle cannot
# descend at all. A feasible descent has to point the engine DOWN-and-retrograde for most of the
# flight and swing it up only for the terminal burn, which is exactly what the existing nominal
# does -- 132 deg from the vertical at handover, 19 deg at touchdown.
#
# For theta > 90 deg the original pointing constraint is nonconvex, but the relaxed half-space form
# used here, n.a >= cos(theta)*Gamma, stays convex and is a valid relaxation (Acikmese et al. 2013).
#
# OPEN ISSUE, flagged rather than assumed away: in 3DoF the thrust direction is the only attitude
# proxy available, so demanding the engine point retrograde and a body-fixed sensor point nadir are
# the same constraint pulling opposite ways. A nadir sensor requirement cannot be expressed here
# without either a gimbal, a sensor on another face, or 6DoF attitude states.
POINTING_DEG = 140.0
VTOUCH = 0.3                 # m/s, max speed at contact

# TERMINAL THRUST POINTING. The angle the thrust must lie within, of the local vertical, over the
# last TERMINAL_POINT_NODES hold intervals. 0 means exactly vertical at touchdown -- the vehicle
# sets down upright on its legs with the engine opposing gravity along the local vertical, which is
# what a landing wants. Mazouz gives the same constraint as Tc(tf) = ||Tc(tf)|| * n_tf.
#
# Written against the lossless-convexification slack as n.a == cos(theta)*Gamma it is LINEAR (an
# equality when theta = 0), and at theta = 0 it also forces ||a|| = Gamma there, so the relaxation
# is tight at the touchdown node by construction.
#
# Note n_hat is the LOCAL VERTICAL, i.e. -g_eff, so a vertical thrust exactly opposes effective
# gravity at the site -- which is only true because the vertical was built from g_eff rather than
# from the radial direction (see gravity.site_and_normal).
TERMINAL_POINT_DEG = 0.0
TERMINAL_POINT_NODES = 1

# ---- start ------------------------------------------------------------------------------------
ALTITUDE_KM = 12.0           # km above the landing site, along the local vertical
# Handover velocity delivered by the braking + TCM phase, in the site-local frame (up, east, north).
# This is mission data, not a solver choice: it is the state the upstream block hands over, and the
# descent problem starts from it. Handing over at rest is not a descent -- see POINTING_DEG.
V0_LOCAL = np.array([-3.2681, -7.5733, -2.4666])   # m/s
CONE_ANGLE_DEG = 0.0         # deg off the surface normal. 0 = straight overhead. Must be <=
                             # GLIDESLOPE_DEG or node 0 violates the cone it is constrained by.


@dataclass(frozen=True)
class Scales:
    """Non-dimensionalisation.

    The whole problem is solved in these units. Physical values span 1e-4 (gravity) to 1e4
    (range), and an SOCP solver's interior-point iteration is sensitive to that conditioning --
    a trust-region radius in metres and a mass in kilograms are not comparable quantities, so any
    penalty weight that mixes them is meaningless. Scaling makes every state O(1) and makes the
    PTR weights dimensionless numbers that mean the same thing on any problem instance.

    L is the range to fly. T is the time to cross it under maximum acceleration -- guess-free, and
    it makes the acceleration scale exactly the control authority, so a non-dimensional thrust of 1
    is full throttle.
    """
    L: float                      # m
    T: float                      # s
    A: float                      # m/s^2  (= L/T^2)

    @staticmethod
    def build(r0, rf, m_wet=MWET, t_max=TMAX):
        L = float(np.linalg.norm(np.asarray(r0, float) - np.asarray(rf, float)))
        A = t_max / m_wet
        T = float(np.sqrt(L / A))
        return Scales(L=L, T=T, A=A)

    # state x = [r(3), v(3), z=ln m]
    def x_nd(self, x):
        x = np.asarray(x, float)
        out = np.array(x, dtype=float)
        out[..., 0:3] = x[..., 0:3] / self.L
        out[..., 3:6] = x[..., 3:6] * self.T / self.L
        return out                                    # ln(mass) is already dimensionless

    def x_dim(self, x):
        x = np.asarray(x, float)
        out = np.array(x, dtype=float)
        out[..., 0:3] = x[..., 0:3] * self.L
        out[..., 3:6] = x[..., 3:6] * self.L / self.T
        return out

    def u_nd(self, u):    return np.asarray(u, float) / self.A      # thrust ACCELERATION
    def u_dim(self, u):   return np.asarray(u, float) * self.A
    def t_nd(self, t):    return float(t) / self.T
    def t_dim(self, t):   return float(t) * self.T
    @property
    def v(self):          return self.L / self.T                     # velocity scale
    @property
    def alpha_nd(self):   return ALPHA * self.A * self.T             # dz/dtau = -s_nd*alpha_nd*Gam
    @property
    def w_nd(self):       return W_BODY * self.T


def tf_bounds(v0_norm, range_m, m_wet=MWET, m_dry=MDRY, t_max=TMAX, t_min=TMIN, g_max=6e-3):
    """Physically meaningful bounds on the flight time. Both must be VALID, i.e. no feasible
    trajectory may lie outside them, so each is deliberately optimistic.

    t_lo is the larger of two necessary conditions:
      * the vehicle cannot null its initial velocity faster than max thrust allows,
        m_dry*||v0||/Tmax;
      * it cannot cross the range faster than maximum acceleration allows. Solving
        d = v0*t + 0.5*a*t^2 for t with a = Tmax/m_dry + g_max (every accelerating effect, all
        pointed helpfully -- optimistic, therefore a valid bound) gives
        t >= (sqrt(v0^2 + 2*a*d) - v0)/a.
    The second matters: with v0 = 0 the first is ZERO, and a flight time of zero was measured
    letting the optimiser drive t_f to 0.1 s and pay the virtual control to teleport the vehicle
    12 km instead of flying it.

    t_hi is the propellant limit: it cannot fly longer than m_fuel/(alpha*Tmin) at the throttle
    floor. That floor is why this bound is finite at all -- the engine cannot be shut off.
    """
    v0 = float(v0_norm); d = float(range_m)
    t_vel = m_dry * v0 / t_max
    a_opt = t_max / m_dry + g_max
    t_rng = (np.sqrt(v0 * v0 + 2.0 * a_opt * d) - v0) / a_opt
    t_lo = max(t_vel, t_rng)
    t_hi = (m_wet - m_dry) / (ALPHA * t_min)
    return t_lo, t_hi

"""Vehicle, body, site and scaling constants for the Eros powered-descent problem.

Every number here is either a mission parameter or derived from one. Nothing is tuned.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

THESIS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MESH_TAB = os.path.join(THESIS, "gaskell.ast-eros-2.shape-model_V1_1",
                        "data", "vertex", "ver64q.tab")      # 49 152 facets

# WHICH PINN-GM-III CHECKPOINT FLIES.
#
# pinn_eros_np.npz was trained on radii 0-10R. That spends 85.5% of its 50 000 samples above 25 km
# altitude and leaves 298 below 1 km, so the model interpolates across the shell a landing actually
# descends through, and its error there peaks at 2.8% (median over 200 sites at 300 m altitude)
# against 0.2% in the far field. The vertical component of that error is SIGNED -- the model pulls
# down harder than the truth -- which integrates into a -0.43 m altitude bias in the navigation
# solution by touchdown.
#
# pinn_eros_d3R_8x64_np.npz is the same model and the same loss (Eq 12 unmodified, no Hessian or
# Laplacian term) trained on radii 0-3R, which is the range Martin & Schaub use in Appendix B, at
# 33 729 parameters -- the paper's large-model scale. Measured median error over the same 200 sites:
#
#            altitude     10 m    100 m    300 m   1000 m   3000 m
#     0-10R (was)        0.376%   1.447%   2.808%   0.741%   0.414%
#     0-3R  (now)        0.183%   0.945%   0.345%   0.178%   0.081%
#
# It costs 75 us per call against 56 us, which buys an 8x reduction exactly where the landing is
# decided. Point this back at pinn_eros_np.npz to reproduce anything generated before the swap.
PINN_NPZ = os.path.join(THESIS, "scvx_landing", "data", "pinn_eros_d3R_8x64_np.npz")

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
# Written against the lossless-convexification slack as n.a >= cos(theta)*Gamma it is LINEAR.
#
# WHY NOT ZERO. At theta = 0 that line becomes n.a == Gamma, an EQUALITY, and together with
# ||a|| <= Gamma it pins a = Gamma*n_hat exactly. Two things follow and both were measured:
#
#   * NUMERICALLY, a zero-half-angle cone has no interior, which is the worst shape to hand an
#     interior-point solver. The endgame subproblems came back 'optimal_inaccurate' with the trust
#     region violated at 121% of its radius, and the next iteration then linearised about that bad
#     iterate and was reported infeasible.
#
#   * PHYSICALLY it is unachievable in the last instant. Re-planning can only satisfy it while
#     there is time to slew: alt/v_vert >= theta_err/SLEW_MAX_DEG_S. Measured at 1.58 cm altitude
#     and 0.296 m/s with the thrust 1.048 deg off vertical, that demanded 524 ms of flight against
#     53 ms of altitude remaining -- a factor of ten, and no flight-time bound can reconcile it.
#
# One degree costs nothing: the lateral acceleration it admits is sin(1 deg)*Tmax/m = 5.9e-4 m/s^2,
# against a touchdown speed limit of 0.3 m/s. What it gives up is that the LCvx relaxation is no
# longer tight at the touchdown node BY CONSTRUCTION -- it becomes something verify.py has to
# check, which it already does.
#
# Note n_hat is the LOCAL VERTICAL, i.e. -g_eff, so a vertical thrust exactly opposes effective
# gravity at the site -- which is only true because the vertical was built from g_eff rather than
# from the radial direction (see gravity.site_and_normal).
TERMINAL_POINT_DEG = 1.0
TERMINAL_POINT_NODES = 1

# Thrust-direction slew rate limit [deg/s]. This is the ACTUATOR's limit, now imposed inside the
# optimisation so the plan cannot ask for a turn the hardware refuses to deliver. Flown open loop
# through a real actuator the previous plans -- which had no slew constraint -- missed by 44.6 m
# against 0.013 m with an ideal actuator. None disables it.
SLEW_MAX_DEG_S = 2.0

# ---- start ------------------------------------------------------------------------------------
# THE REAL HANDOVER ALTITUDE, from the upstream braking/TCM phase, not a round number.
# scvx_landing/deorbit.py: HANDOVER_ALT = 11.771e3, 'gravity-centrifugal balance altitude
# above this site', and r_h = rf + HANDOVER_ALT * n_hat -- straight up the local vertical
# from the same site (LAT -40.0, LON 80.7), which is the geometry built here.
#
# IT MUST MATCH, because the handover dispersions in scvx_landing/mc_handover_poly.json are
# stored as offsets FROM that point: B @ (rE - r_h). Running the nominal at 12.0 km put a
# systematic 229 m bias into every Monte Carlo case that disperses around it. closed_loop's
# own rescale_plan docstring still quotes L = 11771 m at handover, from when this agreed.
ALTITUDE_KM = 11.771         # km above the landing site, along the local vertical
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


def tf_bounds(v0_norm, range_m, m_wet=MWET, m_dry=MDRY, t_max=TMAX, t_min=TMIN,
              g_max=6e-3, v_touch=VTOUCH):
    """Physically meaningful bounds on the flight time. Both must be VALID, i.e. no feasible
    trajectory may lie outside them, so each is deliberately optimistic.

    t_lo is the larger of two necessary conditions, both written with the same optimistic
    deceleration a_opt = Tmax/m_dry + g_max -- every accelerating effect, at the lightest mass, all
    pointed helpfully:
      * the vehicle cannot shed speed faster than a_opt, so t >= (||v0|| - v_touch)/a_opt;
      * it cannot cross the range faster than a_opt either. Solving d = v0*t + 0.5*a*t^2 gives
        t >= (sqrt(v0^2 + 2*a*d) - v0)/a.
    The second matters: with v0 = 0 the first is ZERO, and a flight time of zero was measured
    letting the optimiser drive t_f to 0.1 s and pay the virtual control to teleport the vehicle
    12 km instead of flying it.

    THE SPEED BRANCH NULLS TO v_touch, NOT TO ZERO, and that is not a refinement -- it is what
    makes the bound valid at all. A feasible trajectory has to arrive at or below VTOUCH, not at
    rest, so requiring ||v0|| to be driven to zero excludes trajectories that are perfectly legal.
    It bites exactly in the endgame, where ||v0|| approaches VTOUCH: measured on a navigation run
    at 0.32 m altitude and 0.2896 m/s, the old form demanded t_f >= 7.096 s while the vehicle could
    not stay above the surface for more than 1.10 s even at full throttle straight up. No
    trajectory anywhere on the body satisfies both, the virtual control never decayed, and stage 1
    reported infeasible -- which looked like an unreachable site and was not. Nulling to VTOUCH
    gives 0 s there, and t_rng's 1.231 s takes over.

    Both branches only ever got SMALLER with this change, which is the safe direction for a lower
    bound: a bound that is too low costs the optimiser a little search, a bound that is too high
    removes the answer.

    t_hi is the propellant limit: it cannot fly longer than m_fuel/(alpha*Tmin) at the throttle
    floor. That floor is why this bound is finite at all -- the engine cannot be shut off.
    """
    v0 = float(v0_norm); d = float(range_m)
    a_opt = t_max / m_dry + g_max
    t_vel = max(0.0, v0 - v_touch) / a_opt
    t_rng = (np.sqrt(v0 * v0 + 2.0 * a_opt * d) - v0) / a_opt
    t_lo = max(t_vel, t_rng)
    t_hi = (m_wet - m_dry) / (ALPHA * t_min)
    return t_lo, t_hi

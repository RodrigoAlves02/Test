#!/usr/bin/env python
"""VERIFICATION of the SCvx powered-descent solver against a published, cross-validated benchmark.

VERIFICATION, NOT VALIDATION. The question is "do we solve these equations correctly", which is
mathematics and is answered by reproducing a known solution. Validation -- "are these the right
equations for Eros" -- is a different question and is not what this folder does.

THE BENCHMARK. Johnston, Schiassi, Furfaro & Mortari, "Fuel-Efficient Powered Descent Guidance on
Large Planetary Bodies via the Theory of Functional Connections" (arXiv:2001.03572), flying the
Mars lander of Acikmese & Ploen, "Convex Programming Approach to Powered Descent Guidance for Mars
Landing", JGCD 30(5):1353-1366, 2007.

It is the right benchmark for us because the equations are ours:

    rdot = v
    vdot = g + T/m                 g CONSTANT here, polyhedral in the Eros problem
    zdot = -alpha ||T||            z = ln m, the same mass-depletion law
    Tmin <= ||T|| <= Tmax          the nonconvex bound LCvx removes with the slack Gamma
    free final time                the same free-tf scaling

and because the answer is not one author's number: the paper reports the Theory of Functional
Connections result beside a GPOPS-II pseudospectral solution and they agree to six significant
figures on propellant. An indirect method and a direct method converging on the same value is as
close to ground truth as this problem class gets.

WHAT IS UNDER TEST. ptr.run, subproblem.build and dynamics.Dynamics -- our actual solver, imported
unmodified. Only the CONFIGURATION changes: constant gravity, no rotating frame, Mars vehicle, no
glideslope, no pointing cone, no slew limit, and an exact soft landing (r(tf) = 0, v(tf) = 0)
instead of our |v| <= VTOUCH touchdown condition. The exact terminal state falls out of the
existing formulation with D_nd = 0 and VTOUCH = 0; no code change is needed to express it.

WHY IT MATTERS RIGHT NOW. The segment-level LCvx defect found on the Eros problem mis-charged
propellant by 3%. On 179.447 kg that is 5.4 kg -- far outside any tolerance -- so this benchmark
is a direct, external verdict on that fix rather than an internal argument about which bias is
smaller. Run it with SEGMENT_LCVX both ways.
"""
from __future__ import annotations

import numpy as np

# ---- the published cases -------------------------------------------------------------------
# Vehicle, Table II of the TFC paper (the Acikmese-Ploen lander).
G_MARS = np.array([0.0, 0.0, -3.7114])      # m/s^2
ISP = 225.0                                  # s
G0 = 9.807                                   # m/s^2, the value the paper uses
T_ENGINE = 3100.0                            # N per engine
N_THRUST = 6
PHI_CANT_DEG = 27.0

_c = np.cos(np.radians(PHI_CANT_DEG))
# The paper plots THROTTLE, not newtons: Tmin and Tmax are defined as 0.3 and 0.8 of the full
# cant-corrected stack thrust, so normalising by T_REF puts the bang-bang between 0.3 and 0.8 on a
# 0..1 axis, which is what its thrust panel shows.
T_REF = T_ENGINE * N_THRUST * _c             # 16572.70 N
TMIN = 0.3 * T_ENGINE * N_THRUST * _c        # 4971.81 N
TMAX = 0.8 * T_ENGINE * N_THRUST * _c        # 13258.18 N
ALPHA = 1.0 / (ISP * G0 * _c)                # 5.0863e-4 s/m  -- the cant angle enters here
M_WET = 1905.0                               # kg
M_DRY = 1505.0                               # kg, Acikmese-Ploen; slack in both cases

# Tables III/V and IV/VI. m_used, tf and the switch times are the published answers.
# Axis scales read off the paper's own figures, per case, so each of ours lays beside its
# counterpart without rescaling: Fig. 6 for min-max, Fig. 9 for max-min-max. The fourth panel in
# both is the THRUST DIRECTION PAIR, not the propellant -- case 2 settles it, since its angles peak
# at 214 deg (inside the 0..250 axis) while its propellant is 275 kg (outside it).
AXES = {
    "min-max": dict(t=(0, 30, 5), pos=(-1000, 1500, 500), vel=(-100, 50, 50),
                    acc=(-4, 4, 2), ang=(0, 200, 50), fuel=(0, 200, 50)),
    "max-min-max": dict(t=(0, 40, 10), pos=(-500, 1500, 500), vel=(-100, 100, 50),
                        acc=(-5, 5, 5), ang=(0, 250, 50), fuel=(0, 300, 50)),
}

# ---- THE MIN-MAX PROPELLANT DISCREPANCY, INVESTIGATED ----------------------------------------
#
# We reproduce the published FLIGHT TIMES (min-max +0.03% at K=120, max-min-max +0.28%) and the
# max-min-max PROPELLANT converges properly with the mesh (+0.161 / +0.102 / +0.078% at K=60/120/240
# with tf pinned, halving each refinement). The min-max propellant does NOT: it sits at +0.52% and
# does not move over an 8x refinement (+0.530% at K=30, +0.517% at K=240).
#
# WHAT IT IS NOT. Each of these was measured, not argued:
#
#   * not discretisation      -- flat over K = 30, 60, 120, 240.
#   * not the free-time search -- pinning tf to the published 31.2623 s still gives +0.538%.
#   * not our formulation     -- the 7-state ZOH LCvx version (../SCvx/.zoh7_backup, no direction
#                                split, no cross-product kinematics) gives +0.563% / +0.542% at
#                                K=60/120. Same place.
#   * not a transcription error -- Tables II, III and IV were extracted from the PDF and every
#                                constant above matches: r0, v0, m0, Tmin, Tmax, alpha.
#   * not a spurious intermediate-thrust arc -- the profile is clean bang-bang, 13 nodes at Tmin,
#                                one transition node, 45 at Tmax.
#   * NOT OUR FUEL COMPUTATION -- given the paper's OWN t1 = 7.4430 s and tf = 31.2623 s, our mass
#                                law returns 179.4469 kg, their 179.447 to 1e-4%. Our ZOH charging
#                                converges to it from BELOW as the mesh refines (-0.187% at K=30 to
#                                -0.006% at K=960), so the discretisation bias is negative and
#                                cannot manufacture a positive 0.5% excess.
#
# WHAT IT IS. The switch time. An independent PRIMER-VECTOR solve -- nothing in common with this
# code -- settles it. For constant gravity the costates give lambda_r-dot = 0, so lambda_v is linear
# in t and the optimal steering is d(t) = (a + b t)/||a + b t||. With tf held at the published value
# and t1 free that is 6 unknowns against 6 boundary conditions, square, and it solves EXACTLY
# (|r(tf)| = 2e-12 m):
#
#       source                        t1 [s]     propellant [kg]
#       paper (TFC and GPOPS-II)      7.4430     179.447
#       independent primer-vector     7.2473     180.2717
#       our SCvx, K=240, tf pinned    ~7.213     180.3739
#
# Our SCvx agrees with the independent indirect solution to +0.057%, which is the ZOH restriction
# penalty and is the right size. Both disagree with the paper by ~0.5%, and the whole difference is
# the 0.196 s shift: d(fuel)/d(t1) = alpha*(Tmin - Tmax) = -4.214 kg/s, and 0.196 * 4.214 = 0.826 kg,
# exactly the gap. Holding t1 at the paper's 7.4430 s, 60 random restarts of the shooting problem
# all stop 1.635 m / 0.583 m/s short of the target.
#
# HOW TO STATE IT. "Verified against an independent primer-vector solution, to which the solver
# agrees within 0.06%; the published min-max propellant was not reproducible by either method."
# Not "the paper is wrong" -- TFC and GPOPS-II agree with each other there and the authors verified
# through ode45 -- but two independent methods cannot recover it from the constants the paper states,
# and the discrepancy localises to a single number.

CASES = {
    "min-max": dict(
        r0=np.array([-900.0, 10.0, 1500.0]), v0=np.array([30.0, -10.0, -70.0]),
        m_used=179.447, tf=31.2623, switches=(7.4430,),
        note="TFC 179.447 / GPOPS-II 179.447 kg"),
    "max-min-max": dict(
        r0=np.array([-200.0, 100.0, 1500.0]), v0=np.array([85.0, 50.0, -65.0]),
        m_used=275.205, tf=44.823, switches=(32.418, 38.838),
        # Table V gives v0 = {85, 50, -65}; the caption of Fig. 8 prints {85, -50, -65}.
        # The table is taken as authoritative and the discrepancy is recorded, not silently chosen.
        note="TFC 275.205 / GPOPS-II 275.206 kg; paper's Fig.8 caption disagrees with Table V on v0_y"),
}


# ---- a THIRD case, with the constraints -------------------------------------------------------
# Malyuta, Reynolds, Szmuk, Lew, Bonalli, Pavone & Acikmese, "Convex Optimization for Trajectory
# Generation", IEEE CSM 2022 (arXiv:2106.09125), Problem (95) and parameters (108).
#
# Why this one: its Problem (95) is our formulation constraint for constraint, INCLUDING the
# rotating-frame terms -w x w x r - 2 w x v that the TFC cases did not exercise, plus the two
# constraints we had never verified -- a glideslope cone on the POSITION and a tilt cone on the
# THRUST. Both are written exactly as ours:
#     glideslope  e3' r  >= ||r||  cos(gamma_gs)       gamma_gs from the local vertical
#     pointing    Tc' e3 >= ||Tc|| cos(gamma_p)        gamma_p  from the local vertical
# It also discretizes the input ZOH, which is now our hold too.
#
# Two differences, recorded rather than hidden:
#   * they approximate the glideslope cone by four halfspaces, which CIRCUMSCRIBES it, so their
#     constraint is marginally looser than our exact second-order cone;
#   * they impose ||v|| <= vmax, which we do not implement. The paper states it is never activated
#     for these parameters, so omitting it should not move the optimum -- which the run can check.
MAL = dict(
    g=np.array([0.0, 0.0, -3.71]),
    isp=225.0, ge=9.807, m_wet=1905.0, m_dry=1505.0,
    rho_min=4971.0, rho_max=13258.0,
    omega_deg_s=np.array([3.5e-3, 0.0, 2.0e-3]),      # (3.5 ex + 2 ez) * 1e-3 deg/s
    gamma_gs=86.0, gamma_p=40.0, v_max=500.0 / 3.6,   # 500 km/h
    r0=np.array([2000.0, 0.0, 1500.0]),               # 2 ex + 1.5 ez km
    v0=np.array([80.0, 30.0, -75.0]),                 # 288 ex + 108 ey - 270 ez km/h
    tf=75.0,                                          # the published optimum
)
MAL["alpha"] = 1.0 / (MAL["isp"] * MAL["ge"])
MAL["omega"] = np.radians(MAL["omega_deg_s"])


class ConstantField:
    """A Dynamics-compatible gravity field that is uniform, as the benchmark assumes."""

    def __init__(self, g=G_MARS):
        self.g_vec = np.asarray(g, float)
        self._Z = np.zeros((3, 3))
        self.n_calls = 0

    def g(self, r):
        self.n_calls += 1
        return self.g_vec

    def grad(self, r):
        return self._Z

    def both(self, r):
        return self.g_vec, self._Z

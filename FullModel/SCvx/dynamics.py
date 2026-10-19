"""Non-dimensional rotating-frame dynamics, its Jacobians, and the ZOH discretisation.

STATE   x = [r(3), v(3), z, d(3)]   d is the thrust DIRECTION, a unit vector, and is a STATE.
CONTROL u = [omega(3), Gamma]       omega is the direction's ANGULAR RATE (d-dot = omega x d);
                                    Gamma is the thrust magnitude, zero-order held.

WHY THRUST IS A STATE. Three things follow from it, and they are the reason for the change:

  * THE SLEW RATE BECOMES A CONVEX CONSTRAINT. The thrust direction turns at |a x da|/||a||^2,
    which is not convex in a. But with the rate as an explicit control, the component of w
    perpendicular to a is exactly ||a|| times the angular rate, so ||P w|| <= G * omega_max bounds
    it -- a norm of a linear expression against a linear expression. P is built from the REFERENCE
    direction, which makes it a constant at each iteration and exact at convergence.

  * THE THRUST IS CONTINUOUS. With a zero-order hold on w, a is piecewise LINEAR in time rather
    than piecewise constant. That is first-order-hold behaviour on the thrust obtained
    structurally, without holding the thrust itself first-order -- which matters, because an FOH
    on the thrust VECTOR dips below the 5 N floor between nodes whenever consecutive thrusts
    differ in direction. Here the dip is bounded by the slew constraint instead.

  * a(0) IS A BOUNDARY CONDITION. Each re-plan starts from the thrust the engine is ACTUALLY
    producing, instead of commanding a step the actuator then has to chase. Open loop through a
    real actuator the previous formulation missed by 44.6 m against 0.013 m ideal.

The mass still enters only through the thrust bounds. The ONLY nonlinearity in the right-hand side
is still gravity: a enters v-dot linearly and w enters a-dot linearly.

All quantities are non-dimensional (config.Scales). Gravity is called with dimensional positions.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import config as C
import gravity as GR

# ZERO-ORDER HOLD ON THE THRUST ITSELF.
#
# The thrust used to be a STATE (NX = 10, x = [r, v, z, a]) with its rate w = a-dot as the control,
# because a rate is what a slew-rate limit bounds and you cannot bound the derivative of a
# discontinuous signal. That bought the slew constraint and cost three things:
#
#   1. ZOH on w makes a piecewise-LINEAR, while Gamma -- which LCvx drives equal to ||a|| -- stayed
#      zero-order held. Two quantities forced equal cannot be held to different orders: ||a|| <= G
#      was imposed only at the nodes, so the thrust ramped above the Gamma being charged in between
#      and dynamics.f() charges mass flow on Gamma. Measured on the Eros nominal: 11 of 39 segments
#      exceeded the charge, by up to 8.12 N, and the optimiser farmed it by parking a(0) exactly on
#      the 5 N floor. That is why every Monte Carlo case started at TMIN.
#   2. The magnitude was forced continuous even though it is never rate-constrained, so a bang-bang
#      switch could not sit at a node and had to smear across a segment.
#   3. Verified against a published benchmark (see ../SCvx Verification): FOH thrust converged
#      0.6% high on propellant with an intermediate-thrust arc where the optimum is bang-bang.
#
# With the thrust as a ZOH CONTROL all three go away: a and Gamma are held identically so ||a|| <= G
# at the node holds across the whole segment, the magnitude may jump at a node, and the slew limit
# is expressed on the DIFFERENCE between consecutive segment values, which needs no derivative at
# all -- a constraint on two decision variables rather than on a rate.
# THRUST SPLIT INTO DIRECTION AND MAGNITUDE, because the engine treats them differently.
#
# The throttle is a valve: it goes from minimum to maximum essentially instantly. The direction is
# the whole vehicle rotating at 2 deg/s. Gluing them into one vector -- which every previous
# formulation here did -- forces one rule on both, and whichever rule you pick is wrong for one of
# them. Carrying the vector as a STATE made the throttle ramp (measured: a 5 s ramp where the
# benchmark switches instantly). Carrying it as a ZOH CONTROL makes the direction jump, which no
# vehicle can follow.
#
# So: d is the DIRECTION, a state, continuous, driven by d-dot = omega x d with omega the control.
# ||omega|| IS the slew rate in rad per unit time -- not a proxy for it -- because the cross product
# cannot change ||d||, so every component of omega that does anything goes into rotation. Gamma is
# the MAGNITUDE, a ZOH control, free to jump at a node. Mass flow is charged on Gamma, and because
# ||d|| = 1 is conserved the DELIVERED thrust magnitude is Gamma too, so the charge matches the
# thrust identically -- the 3% mis-charge that came from holding Gamma and ramping ||a|| cannot
# occur, and neither can the 4.3% throttle-under-the-floor leak that ||d|| <= 1 allowed.
#
# ||d|| = 1 IS ENFORCED BY THE KINEMATICS, NOT BY A CONSTRAINT.
#
# d-dot = omega x d, so d.(d-dot) = 0 identically and therefore d/dt ||d||^2 = 0. The norm is a
# CONSERVED QUANTITY of the right-hand side. That matters because of where SCvx measures itself: the
# defect is the distance between each node and where the NONLINEAR propagation takes the previous
# one, and that propagation conserves the norm. Converged means the defect and the virtual control
# are spent, so at the solution ||d_k|| = ||d_0|| to integrator accuracy, for free.
#
# The earlier free rate, d-dot = w, had no such structure and both ways of patching it failed, each
# measured on this code:
#   * ||d|| = 1 linearised as dref.d = 1 pins only the projection onto the reference, so d grows
#     sideways. ||d|| reached 1.043 and the propellant came out 1.09% UNDER the published benchmark:
#     thrust nobody pays for, since the thrust is Gamma*d and the charge is Gamma.
#   * ||d|| <= 1 is convex and leaks the other way. It is tight only where the optimiser wants MORE
#     thrust; where the Tmin floor binds it wants LESS, so it shrinks d instead. ||d|| fell to 0.957
#     at the two nodes sitting on the floor and the plan delivered 4.787 N while charging 5 N --
#     throttling below the floor without ever violating a constraint.
# The cross product removes the choice rather than trading one leak for the other.
#
# WHAT IS LEFT. Two things, and both are handled in subproblem.py / ptr.py rather than here:
#   * omega has a NULL DIRECTION. omega parallel to d produces no motion, so B[d,omega] = -skew(d)
#     is rank 2 and omega is not unique. Fixed by the gauge row dref.omega = 0, which removes no
#     physical freedom -- only the choice of representative.
#   * ||d_0|| = 1 still has to be imposed ONCE, and the conservation then carries it everywhere.
#     Left as an inequality it would become a CONSTANT discount on every node instead of a local
#     one, which is worse, not better. Handled by a tangent plane plus a retraction in ptr.py.
#
# The thrust is still Gamma*d, a product of two decision variables, linearised about the reference
# the same way gravity is. The thrust lower bound is a box on Gamma rather than lossless
# convexification, so global optimality is not guaranteed the way it is for the LCvx formulation in
# ../SCvx Verification.
NX, NU = 10, 4          # [r, v, z, d(3)] and [w(3), Gamma]
ID = slice(7, 10)       # thrust DIRECTION in the state
IA = slice(0, 3)        # (kept for callers; the magnitude is u[3])


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


class Dynamics:
    """f(x,u) and its Jacobians, in non-dimensional units.

    The rotating-frame terms (Coriolis -2w x v and centrifugal -w x (w x r)) are LINEAR in the
    state, so they enter the Jacobian exactly and cost nothing in linearisation error. That matters
    here: Coriolis is 5.1x gravity at handover on a body that turns 34 deg during the descent.
    The ONLY nonlinearity in the whole right-hand side is g(r).
    """

    def __init__(self, scales: C.Scales, field=None):
        self.s = scales
        self.field = field if field is not None else GR.field()
        w = scales.w_nd
        self.wX = skew(w)
        self.Ccf = -self.wX @ self.wX          # d/dr of -w x (w x r)
        self.Cco = -2.0 * self.wX              # d/dv of -2 w x v
        self.alpha = scales.alpha_nd
        self._gscale = scales.L / scales.A     # d(g_nd)/d(r_nd) = (L/A) * dg/dr

    def g_nd(self, r_nd):
        return self.field.g(np.asarray(r_nd, float) * self.s.L) / self.s.A

    def G_nd(self, r_nd):
        return self.field.grad(np.asarray(r_nd, float) * self.s.L) * self._gscale

    def f(self, x, u):
        r, v, d = x[0:3], x[3:6], x[7:10]
        G = u[3]
        vdot = G * d + self.Cco @ v + self.Ccf @ r + self.g_nd(r)
        # d-dot = omega x d. The cross product is what keeps ||d|| = 1 exactly: it lies in the
        # tangent plane of the unit sphere at d, which is also exactly the two degrees of freedom
        # a pointing command has.
        return np.concatenate([v, vdot, [-self.alpha * G], np.cross(u[0:3], d)])

    def jac(self, x, u):
        """(df/dx, df/du). df/dx depends on the state only through the gravity gradient."""
        A = np.zeros((NX, NX)); B = np.zeros((NX, NU))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = self.G_nd(x[0:3]) + self.Ccf
        A[3:6, 3:6] = self.Cco
        # the bilinear thrust Gamma*d, linearised about the reference -- d(Gd)/dd = G,
        # d(Gd)/dG = d. Successive convexification closes the gap, as it does for gravity.
        A[3:6, 7:10] = float(u[3]) * np.eye(3)
        B[3:6, 3] = np.asarray(x[7:10], float)
        B[6, 3] = -self.alpha           # mass flow IS the thrust magnitude now: no mismatch
        # d(omega x d)/dd = skew(omega), and d(omega x d)/domega = -skew(d) because
        # omega x d = -(d x omega). Both exact and closed-form: the only nonlinearity in the whole
        # right-hand side that needs finite differences is still gravity, and not even that.
        A[7:10, 7:10] = skew(np.asarray(u[0:3], float))
        B[7:10, 0:3] = -skew(np.asarray(x[7:10], float))
        return A, B

    def rk4(self, x, u, h):
        """One nonlinear RK4 step of the TIME-scaled dynamics (h is in tau, caller folds in s)."""
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * h * k1, u)
        k3 = self.f(x + 0.5 * h * k2, u)
        k4 = self.f(x + h * k3, u)
        return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def propagate(self, x0, U, s, K, n_sub=20):
        """Nonlinear ZOH propagation across all segments. Returns the node states it reaches.

        This is the ground truth the linearisation is checked against every iteration.
        """
        dtau = 1.0 / (K - 1)
        x = np.asarray(x0, float).copy()
        out = [x.copy()]
        for k in range(K - 1):
            h = s * dtau / n_sub
            for _ in range(n_sub):
                x = self.rk4(x, U[:, k], h)
            out.append(x.copy())
        return np.array(out).T


def discretize(dyn: Dynamics, X, U, s, n_sub=10):
    """Linearise about (X, U, s) and discretise with a zero-order hold on u.

    Returns Ad, Bd, Sd, zd such that   x_{k+1} = Ad_k x_k + Bd_k u_k + Sd_k s + zd_k
    and Xprop, the NONLINEAR propagation of the same reference, whose disagreement with X is the
    defect that measures how much the linearisation lied.

    Time is normalised to tau in [0,1] and the flight time s enters as a dilation, x' = s f(x,u).
    Linearising that product about (xbar, ubar, sbar) gives
        A = sbar * df/dx,  B = sbar * df/du,  S = f(xbar, ubar),  z = -A xbar - B ubar
    (the sbar*f and -S*sbar terms cancel exactly, which is why z carries no constant from them).

    Each segment is split into n_sub substeps. Within a substep the matrices are frozen at the
    substep MIDPOINT and the transition is taken from a Van Loan block exponential, which is exact
    for frozen matrices and second-order accurate in the substep because of the midpoint rule. The
    reference is carried forward NONLINEARLY between substeps rather than interpolated, so the
    linearisation is taken about points the vehicle actually passes through -- this matters because
    the whole nonlinearity is gravity, and gravity is a function of where you really are.
    """
    K = X.shape[1]
    dtau = 1.0 / (K - 1)
    nA = NX + NU + 2                                   # [x | u | s | 1]
    Ad = np.zeros((K - 1, NX, NX)); Bd = np.zeros((K - 1, NX, NU))
    Sd = np.zeros((K - 1, NX)); zd = np.zeros((K - 1, NX))
    Xprop = np.zeros_like(X); Xprop[:, 0] = X[:, 0]

    for k in range(K - 1):
        Phi = np.eye(NX); Psi = np.zeros((NX, NU + 2))
        xr = X[:, k].copy(); u = U[:, k]
        h = dtau / n_sub
        for _ in range(n_sub):
            xm = dyn.rk4(xr, u, 0.5 * s * h)           # midpoint of the substep
            Ac, Bc = dyn.jac(xm, u)
            Ac = s * Ac; Bc = s * Bc
            Sc = dyn.f(xm, u)
            zc = -Ac @ xm - Bc @ u
            M = np.zeros((nA, nA))
            M[:NX, :NX] = Ac
            M[:NX, NX:NX + NU] = Bc
            M[:NX, NX + NU] = Sc
            M[:NX, NX + NU + 1] = zc
            E = expm(M * h)
            Phi_i, Psi_i = E[:NX, :NX], E[:NX, NX:]
            Phi = Phi_i @ Phi
            Psi = Phi_i @ Psi + Psi_i
            xr = dyn.rk4(xr, u, s * h)                 # advance the reference nonlinearly
        Ad[k] = Phi
        Bd[k] = Psi[:, :NU]
        Sd[k] = Psi[:, NU]
        zd[k] = Psi[:, NU + 1]
        Xprop[:, k + 1] = xr
    return Ad, Bd, Sd, zd, Xprop


def defect(dyn: Dynamics, X, U, s, n_sub=20):
    """Max per-segment defect in POSITION and VELOCITY: propagate each node forward nonlinearly with its own ZOH
    control and see how far it lands from the next node. Non-dimensional length units.

    This is the linearisation-quality signal. It replaces the trust-region ratio rho, which in the
    previous solver was measured to be flat in the trust-region radius (dL constant to three
    decimals while the radius fell 16x) and therefore carried no step-size information at all.
    This number does: it is a distance, it has units, and it goes to zero with the step.

    Also returns the L1 norm of the FULL ten-state defect summed over every segment. That is the
    quantity an SCvx merit function needs: the subproblem prices its virtual control as
    w_vc * sum|nu|, so the nonlinear counterpart must be the SAME norm over the SAME rows, or the
    predicted and actual reductions are not in the same units and the ratio rho is meaningless.
    """
    K = X.shape[1]; dtau = 1.0 / (K - 1)
    worst = worstv = 0.0; per = np.zeros(K - 1); perv = np.zeros(K - 1); l1 = 0.0
    # Velocity is reported separately and deliberately. A position-only defect was
    # measured reading 1.09e-3 m -- 'converged' -- on a trajectory whose per-node
    # velocity inconsistency was 2.7 m/s, because multiple shooting restarts each
    # segment from its own node and a velocity error never shows up in position.
    for k in range(K - 1):
        x = X[:, k].copy(); h = s * dtau / n_sub
        for _ in range(n_sub):
            x = dyn.rk4(x, U[:, k], h)
        per[k] = float(np.linalg.norm(x[0:3] - X[0:3, k + 1]))
        perv[k] = float(np.linalg.norm(x[3:6] - X[3:6, k + 1]))
        l1 += float(np.abs(x - X[:, k + 1]).sum())
        worst = max(worst, per[k]); worstv = max(worstv, perv[k])
    return worst, worstv, per, perv, l1

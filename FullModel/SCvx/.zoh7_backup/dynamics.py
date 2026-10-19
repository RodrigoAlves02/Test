"""Non-dimensional rotating-frame dynamics, its Jacobians, and the ZOH discretisation.

STATE   x = [r(3), v(3), z, a(3)]   a = T/m, the thrust acceleration, is a STATE.
CONTROL u = [w(3), G]               w = da/dtau is the thrust RATE; G is the lossless-
                                    convexification slack, ||a|| <= G.

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
NX, NU = 7, 4           # [r, v, z] and [a(3), Gamma]
IA = slice(0, 3)        # where the thrust acceleration lives in the CONTROL


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
        r, v, a = x[0:3], x[3:6], u[0:3]
        vdot = a + self.Cco @ v + self.Ccf @ r + self.g_nd(r)
        return np.concatenate([v, vdot, [-self.alpha * u[3]]])

    def jac(self, x, u):
        """(df/dx, df/du). df/dx depends on the state only through the gravity gradient."""
        A = np.zeros((NX, NX)); B = np.zeros((NX, NU))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = self.G_nd(x[0:3]) + self.Ccf
        A[3:6, 3:6] = self.Cco
        B[3:6, 0:3] = np.eye(3)         # thrust acceleration enters v-dot directly
        B[6, 3] = -self.alpha           # mass flow on the LCvx slack; ZOH makes it equal ||a||
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
    """
    K = X.shape[1]; dtau = 1.0 / (K - 1)
    worst = worstv = 0.0; per = np.zeros(K - 1); perv = np.zeros(K - 1)
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
        worst = max(worst, per[k]); worstv = max(worstv, perv[k])
    return worst, worstv, per, perv

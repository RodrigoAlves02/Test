"""Non-dimensional rotating-frame dynamics, its Jacobians, and the ZOH discretisation.

STATE   x = [r(3), v(3), z]      z = ln(m); mass enters the dynamics ONLY through the thrust
                                 bounds, because the control is an ACCELERATION (see below).
CONTROL u = [a(3), G]            a = T/m is thrust acceleration, G is the lossless-convexification
                                 slack with ||a|| <= G. Using acceleration rather than force is what
                                 removes the mass from the right-hand side and leaves the dynamics
                                 linear in the state apart from gravity.

All quantities are non-dimensional (config.Scales). Gravity is called with dimensional positions.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import config as C
import gravity as GR

NX, NU = 7, 4


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
        r, v = x[0:3], x[3:6]
        vdot = u[0:3] + self.Cco @ v + self.Ccf @ r + self.g_nd(r)
        return np.concatenate([v, vdot, [-self.alpha * u[3]]])

    def jac(self, x, u):
        """(df/dx, df/du). df/dx depends on the state only through the gravity gradient."""
        A = np.zeros((NX, NX)); B = np.zeros((NX, NU))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 0:3] = self.G_nd(x[0:3]) + self.Ccf
        A[3:6, 3:6] = self.Cco
        B[3:6, 0:3] = np.eye(3)
        B[6, 3] = -self.alpha
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
    # l1 IS FOR THE RATIO TEST, and it has to be the WHOLE state vector, not position alone. The
    # subproblem pays w_vc * sum|nu| over all NX rows for the defects its linear model admits; the
    # merit function must pay w_vc * (true L1 defect) over the same rows, or predicted and actual
    # reductions are in different units and rho is meaningless.
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

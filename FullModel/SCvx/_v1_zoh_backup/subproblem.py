"""The convex subproblem: one SOCP, two objectives.

Everything is non-dimensional. The only non-convexities in the original problem are
  (1) the thrust lower bound  ||T|| >= Tmin  -- removed losslessly (Acikmese & Ploen 2007) by the
      slack G with ||a|| <= G and the bounds placed on G instead of on ||a||;
  (2) the mass in  G_phys in [Tmin/m, Tmax/m]  -- handled by z = ln m and a Taylor expansion of
      e^-z about the reference;
  (3) the flight time multiplying the dynamics -- handled by successive linearisation outside.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

import config as C
from dynamics import NX, NU


def build(Ad, Bd, Sd, zd, Xref, Uref, sref, x_init, rf_nd, n_hat, scales,
          mode, D_nd=None, w_vc=1e4, w_tr=1.0, w_ts=1.0,
          s_lo=None, s_hi=None, R=None, Rs=None, gs_deg=C.GLIDESLOPE_DEG, point_deg=C.POINTING_DEG,
          vtouch_nd=None, term_point_deg=C.TERMINAL_POINT_DEG,
          term_point_nodes=C.TERMINAL_POINT_NODES):
    """Assemble and return (problem, variables). mode is 'error' or 'fuel'."""
    K = Xref.shape[1]
    X = cp.Variable((NX, K), name="X")
    U = cp.Variable((NU, K), name="U")
    s = cp.Variable(nonneg=True, name="s")
    nu = cp.Variable((NX, K - 1), name="nu")            # virtual control on the dynamics
    eta = cp.Variable(K, nonneg=True, name="eta")       # PTR trust epigraph, per node
    eta_s = cp.Variable(nonneg=True, name="eta_s")

    cons = []

    # -- dynamics, with virtual control so the subproblem is always feasible -------------------
    for k in range(K - 1):
        cons += [X[:, k + 1] == Ad[k] @ X[:, k] + Bd[k] @ U[:, k] + Sd[k] * s + zd[k] + nu[:, k]]

    # -- boundary conditions -------------------------------------------------------------------
    # HELD BY NAME, not by its index in cons. Its DUAL is the gradient of the optimal value with
    # respect to the initial state -- the discrete costate lambda(0) -- which is the only thing in
    # this problem that can be checked against a finite-difference sensitivity computed from an
    # entirely separate set of solves. Reaching it as cons[K-1] would break the moment a constraint
    # is added above it.
    ic = X[:, 0] == x_init
    cons += [ic]
    # land ON the surface: the component of the terminal miss along the local vertical is zero.
    cons += [n_hat @ (X[0:3, K - 1] - rf_nd) == 0]
    cons += [cp.norm(X[3:6, K - 1]) <= vtouch_nd]
    cons += [X[6, :] >= np.log(C.MDRY)]

    # -- lossless convexification of the thrust magnitude --------------------------------------
    rho1 = C.TMIN / scales.A
    rho2 = C.TMAX / scales.A
    zbar = Xref[6, :]
    # A ZERO-ORDER HOLD HAS K-1 CONTROL INTERVALS, NOT K. U[:, K-1] never enters the dynamics --
    # the hold on segment k uses U[:, k] for k = 0 .. K-2 -- so it drives nothing and costs nothing,
    # and the optimiser leaves it anywhere the constraints permit. Left free it reported a thrust of
    # 3.3 N against a 5 N floor and a lossless-convexification gap of 13.25 N at that one node,
    # which looked like a failed relaxation and was not: the gap over the real intervals is <= 0.007 N.
    # Tying it to the last real interval makes the reported control history mean what it says.
    cons += [U[:, K - 1] == U[:, K - 2]]
    for k in range(K):
        cons += [cp.norm(U[0:3, k]) <= U[3, k]]                     # ||a|| <= G  (LCvx)
        d = X[6, k] - zbar[k]
        mu1 = rho1 * np.exp(-zbar[k])
        mu2 = rho2 * np.exp(-zbar[k])
        # e^-z is convex, so its 2nd-order expansion bounds it from above for d>0 and the 1st-order
        # expansion bounds it from below everywhere. Lower bound -> SOC, upper bound -> affine.
        # Here the expansion is essentially exact: the mass changes by ~1% over a whole descent, so
        # |d| <= 0.01 and the discarded term is O(d^3/6) ~ 1.7e-7 relative.
        cons += [mu1 * (1.0 - d + 0.5 * cp.square(d)) <= U[3, k]]
        cons += [U[3, k] <= mu2 * (1.0 - d)]

    # -- glideslope: a cone about the LOCAL VERTICAL with its apex at the ACHIEVED LANDING POINT --
    # Not at the nominal site. When the site is out of reach the vehicle lands somewhere else, and
    # the cone that keeps it clear of the terrain belongs to where it actually touches down -- "the
    # apex of the cone concurs with the landing target position, a certain distance q from the
    # original target" (Blackmore et al. 2010). Anchoring it at the nominal site instead forces a
    # trajectory that diverts 100 m to stay inside a cone about a point it never reaches, which near
    # the ground is impossibly tight: measured making every dispersed minimum-landing-error solve
    # fail rather than return a reachable miss. r_K is a variable, so this stays a second-order cone.
    P = np.eye(3) - np.outer(n_hat, n_hat)                          # projector onto the tangent plane
    tg = np.tan(np.radians(gs_deg))
    for k in range(K - 1):
        dr = X[0:3, k] - X[0:3, K - 1]
        cons += [cp.norm(P @ dr) <= tg * (n_hat @ dr)]

    # -- pointing: thrust within point_deg of the local vertical -------------------------------
    # In 3DoF the thrust direction is the attitude proxy. Written against the LCvx slack rather
    # than ||a|| so it stays affine (Acikmese et al. 2013): n.a >= cos(theta) * G.
    cos_point = np.cos(np.radians(point_deg))
    for k in range(K):
        cons += [n_hat @ U[0:3, k] >= cos_point * U[3, k]]

    # -- terminal thrust pointing --------------------------------------------------------------
    # The touchdown burn is the hold on interval K-2; U[:, K-1] is the tied dummy and drives nothing.
    if term_point_deg is not None and term_point_nodes > 0:
        k_hi = K - 1                       # exclusive: the last REAL hold is K-2
        k_lo = max(0, k_hi - int(term_point_nodes))
        for k in range(k_lo, k_hi):
            if term_point_deg <= 0.0:
                cons += [U[0:3, k] == n_hat * U[3, k]]           # exactly along the local vertical
            else:
                cons += [n_hat @ U[0:3, k]
                         >= np.cos(np.radians(term_point_deg)) * U[3, k]]

    # -- trust region, penalised rather than imposed (PTR) --------------------------------------
    # The radius is a VARIABLE priced in the cost, not a hard bound adapted by a ratio test. The
    # previous solver adapted a hard radius from rho = dJ/dL, and dL was measured constant to three
    # decimals while the radius fell 16x, so the controller was steering on a signal that carried
    # no step-size information. Here the optimiser chooses its own step against a fixed price, and
    # the price is set from the measured linearisation error (see ptr.py).
    for k in range(K):
        cons += [cp.norm(cp.hstack([X[:, k] - Xref[:, k], U[:, k] - Uref[:, k]])) <= eta[k]]
    cons += [cp.abs(s - sref) <= eta_s]
    # A HARD CAP on top of the price. A priced radius alone cannot bound the step: the
    # virtual-control term dominated the objective by ~1000x, so the trust price was
    # invisible and the flight time swung 716 <-> 3155 s on successive iterations. The
    # cap makes the step bounded no matter how the objective happens to scale, and the
    # price still decides where inside the cap to go.
    if R is not None:
        cons += [eta <= R]
    if Rs is not None:
        cons += [eta_s <= Rs]
    if s_lo is not None:
        cons += [s >= s_lo]
    if s_hi is not None:
        cons += [s <= s_hi]

    # -- objective -------------------------------------------------------------------------------
    miss = cp.norm(P @ (X[0:3, K - 1] - rf_nd))
    pen = (w_vc * cp.sum(cp.abs(nu)) + w_tr * cp.sum(eta) + w_ts * eta_s)
    if mode == "error":
        obj = miss + pen
    elif mode == "fuel":
        # -z_K maximises final mass. Divided by alpha_nd so its gradient with respect to the
        # controls is O(1) instead of O(1e-3): dz/dtau = -alpha_nd*G, so this term is numerically
        # the integral of G, which is what the thrust variables actually control.
        obj = -X[6, K - 1] / scales.alpha_nd + pen
        cons += [miss <= D_nd]
    else:
        raise ValueError(mode)

    return cp.Problem(cp.Minimize(obj), cons), dict(X=X, U=U, s=s, nu=nu, eta=eta, eta_s=eta_s,
                                                    miss=miss, ic=ic)


SOLVER_OPTS = {
    # CLARABEL is the primary. SCS is the fallback and is BOUNDED: it is only reached when the
    # interior-point method has already failed, i.e. exactly when a first-order method is slowest.
    # Unbounded, it was measured taking tens of minutes on a single subproblem.
    "CLARABEL": dict(),
    "SCS": dict(max_iters=20000, time_limit_secs=15.0),
}


def solve(prob):
    """Solve, trying CLARABEL then SCS. Returns the solver name, or None if both failed."""
    for name in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=getattr(cp, name), **SOLVER_OPTS[name])
        except (cp.error.SolverError, Exception):
            continue
        if prob.status in ("optimal", "optimal_inaccurate"):
            return name, prob.status
    return None, prob.status

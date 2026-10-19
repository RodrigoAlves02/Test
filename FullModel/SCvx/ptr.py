"""The successive loop: penalised trust region, driven by the measured linearisation error.

No trust-region ratio. The step size is priced in the cost (subproblem.py) and the price is moved
by the one signal that actually measures whether the linear model told the truth: the distance
between where the nonlinear dynamics take each node and where the linear model said it would go.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import config as C
import subproblem as SP
from dynamics import NX, NU, Dynamics, defect, discretize


@dataclass
class Result:
    X: np.ndarray; U: np.ndarray; s: float
    status: str; iters: int; history: list = field(default_factory=list)
    miss_m: float = np.nan; fuel_kg: float = np.nan; tf_s: float = np.nan
    defect_m: float = np.nan; vc: float = np.nan; wall_s: float = 0.0
    D_nd: float = np.nan
    # ||d_k|| over all nodes. NOT a constraint residual -- nothing in the subproblem bounds it.
    # The kinematics d-dot = omega x d conserve it and node 0 pins it, so these two numbers are
    # the audit on that whole argument: they must read 1.0000 to within the defect. If they do
    # not, the thrust being delivered is not the Gamma being charged, which is the exact failure
    # that both earlier ||d|| treatments had (1.043 one way, 0.957 the other).
    dnorm_lo: float = np.nan; dnorm_hi: float = np.nan


def _gauge(X, U):
    """Put the reference in the SAME GAUGE the subproblem imposes: omega perpendicular to d.

    subproblem.build carries the row dref.omega = 0 to remove omega's null direction. That row is a
    hard equality with no slack, so if the REFERENCE violates it the solver must spend trust-region
    budget just to reach feasibility -- and the violation is not small. Uref satisfied the gauge
    against the PREVIOUS direction; between iterations d rotates by up to the trust radius, so the
    component of omega along the NEW d is about ||omega||*||delta d||. Measured on the min-max
    benchmark: ||omega|| up to 18.2, ||delta d|| about 0.1, so a required step of 1.8 against a
    trust cap of R = 0.3. CLARABEL correctly reported the subproblem PRIMAL INFEASIBLE on alternate
    iterations, SCS then returned 'optimal_inaccurate' junk violating the trust region by up to 75%,
    and that junk became the next reference. The iteration never settled.

    Projecting costs nothing physical, by exactly the argument that justifies the gauge row itself:
    the removed component produces no motion, since (omega + c*d) x d = omega x d. So f at the
    reference is unchanged, and the Jacobian loses only the spurious c*skew(d) term -- the
    linearisation gets BETTER, not worse.

    NODE K-1 IS NOT PROJECTED, and getting that wrong cost a benchmark. subproblem.build imposes the
    gauge only for k = 0..K-2, because U[:, K-1] is the ZOH dummy that drives nothing and is instead
    TIED to the last real interval by U[0:3, K-1] == U[0:3, K-2]. Projecting node K-1 against its own
    direction breaks that tie in the REFERENCE -- the two columns were equal, the projections are
    not, because d_{K-1} and d_{K-2} differ. The subproblem then has to pull both columns back to a
    common value with each one held inside its own trust radius, which is impossible once the
    references differ by more than 2R. Measured on the Malyuta case: ||omega|| 38.3 against R = 0.35,
    CLARABEL primal infeasible at iteration 1, and a constraint-group bisection that pointed at the
    tie row. Tying after projecting, exactly as the subproblem does, is the whole fix.
    """
    d = X[7:10, :]
    dn = np.linalg.norm(d, axis=0)
    dh = d / np.where(dn > 1e-12, dn, 1.0)
    U[0:3, :-1] -= dh[:, :-1] * np.sum(dh[:, :-1] * U[0:3, :-1], axis=0)
    U[0:3, -1] = U[0:3, -2]
    return U


# Per-segment turn the COLD GUESS is smoothed to, in degrees. A FIXED number on purpose: it must
# not depend on slew_max_deg_s, or the seed becomes a function of the constraint and the solver's
# answer follows the seed rather than the optimum. 3 deg/segment keeps the guess inside the cone
# of any limit down to ~0.16 deg/s at the guess's own flight time, which covers every case of
# interest, while leaving the guess smooth enough not to fight the first few iterations.
GUESS_MAX_TURN_DEG = 3.0

# Fraction of the flight over which the COLD GUESS is blended to the local vertical. Also a fixed
# number on purpose, and for the same reason as GUESS_MAX_TURN_DEG: it must not depend on
# term_point_deg / _nodes / _start, or the seed becomes a function of the constraint again.
#
# It is constraint-INDEPENDENT because every configuration requires the final node within
# TERMINAL_POINT_DEG of vertical -- a guess that ends vertical is closer to feasible for all of
# them and favours none. Without it the guess's terminal attitude is whatever the heuristic
# produces and it VARIES WITH K: measured 37.2 deg at K=60 against 11.3 deg at K=100. A 45->1 deg
# corridor over the last 5% then sat 36 deg outside the guess at K=60, which no trust radius can
# close in one step, and the first subproblem came back infeasible.
GUESS_TERMINAL_BLEND_FRAC = 0.10


def initial_guess(dyn, x_init, rf_nd, n_hat, scales, K, s_lo, s_hi,
                  term_point_deg=C.TERMINAL_POINT_DEG,
                  term_point_nodes=C.TERMINAL_POINT_NODES, slew_max_nd=None, s0_override=None,
                  point_deg=None):
    """A dynamically consistent guess for the AUGMENTED state x = [r, v, z, d].

    Position is a cubic Hermite carrying the boundary conditions (r0, v0 at the start, rf and rest
    at the end), so r-dot = v holds exactly at every node and the guess already satisfies the hard
    constraints it will be held to -- a guess that violates one by more than the trust radius makes
    the very first subproblem infeasible.

    The direction d is then wherever the remaining half of the dynamics points, normalised and walked
    into the pointing cone, and the magnitude Gamma is the length it needs, clipped into the
    achievable band. The rate control omega is the rotation that actually carries d_k to d_{k+1},
    not a finite difference -- see the loop at the bottom.

    EVERY d IN THE GUESS IS A UNIT VECTOR, and that is load-bearing, not tidiness. The kinematics
    conserve ||d||, so they conserve whatever the guess hands them: a guess at ||d|| = 0.8 would
    make the node-0 plane dref_0.d_0 = 1 demand ||d_0|| >= 1.25, and the first subproblem would be
    solving a different problem than the one asked for.

    s is seeded at twice the minimum-time bound: that bound assumes every accelerating effect points
    helpfully, so the true optimum always lies above it, and a factor of two needs no knowledge of
    the answer. The solver must be insensitive to this seed; the seed-ladder test checks it.
    """
    r0, v0, z0 = x_init[0:3], x_init[3:6], x_init[6]
    # s0_override SEEDS THE DURATION WITHOUT TOUCHING THE CONSISTENCY. 2*s_lo needs no knowledge of
    # the answer, which is what makes it a safe default, but it is only a bound-derived guess: on a
    # dispersed Eros descent it seeds 1098 s against a 2112 s optimum, and the trust region on s then
    # spends iterations walking the difference. A caller that already has a solved neighbour -- the
    # nominal, say -- knows the duration far better than the bound does. Note this is NOT a warm
    # start: the Hermite is still built through THIS case's r0 and v0, so the guess stays dynamically
    # consistent with its own initial condition. Handing the whole neighbouring SOLUTION over instead
    # was measured worse -- node 0 is anchored to the true IC while the rest keeps the neighbour's,
    # leaving a 405 m jump between the first two reference nodes, and stage 1 went from converging in
    # 11 iterations to maxiter at D = 105.8 m.
    s0 = float(np.clip(2.0 * s_lo if s0_override is None else s0_override, s_lo, s_hi))
    tau = np.linspace(0.0, 1.0, K); dtau = 1.0 / (K - 1)
    X = np.zeros((NX, K)); U = np.zeros((NU, K))

    t2, t3 = tau ** 2, tau ** 3
    h00 = 2 * t3 - 3 * t2 + 1; h10 = t3 - 2 * t2 + tau
    h01 = -2 * t3 + 3 * t2;    h11 = t3 - t2
    d00 = 6 * t2 - 6 * tau;    d10 = 3 * t2 - 4 * tau + 1
    d01 = -6 * t2 + 6 * tau;   d11 = 3 * t2 - 2 * tau
    m0 = s0 * v0
    for i in range(3):
        X[i] = h00 * r0[i] + h10 * m0[i] + h01 * rf_nd[i] + h11 * 0.0
        X[3 + i] = (d00 * r0[i] + d10 * m0[i] + d01 * rf_nd[i] + d11 * 0.0) / s0
    X[3:6, 0] = v0; X[3:6, K - 1] = 0.0
    X[6, 0] = z0

    rho1, rho2 = C.TMIN / scales.A, C.TMAX / scales.A
    # THE GUESS MUST USE THE CONE THE SUBPROBLEM USES. Reading C.POINTING_DEG here while the caller
    # passes point_deg to subproblem.build walks every guess into a 140 deg cone that the problem may
    # not impose -- and that is exactly the wrong starting point for the ~28 % of cases whose optimum
    # lies at 150-177 deg, which is where the cone was costing up to 10 % of the fuel.
    cos_pt = np.cos(np.radians(C.POINTING_DEG if point_deg is None else point_deg))
    for k in range(K):
        vdot = ((X[3:6, k + 1] - X[3:6, k]) / (s0 * dtau)) if k < K - 1 else np.zeros(3)
        a = vdot - dyn.Cco @ X[3:6, k] - dyn.Ccf @ X[0:3, k] - dyn.g_nd(X[0:3, k])
        an = np.linalg.norm(a)
        d = a / an if an > 1e-12 else n_hat.copy()
        c = float(d @ n_hat)
        if c < cos_pt:                                    # project into the pointing cone
            perp = d - c * n_hat; pn = np.linalg.norm(perp)
            d = (cos_pt * n_hat + np.sqrt(max(1.0 - cos_pt ** 2, 0.0)) * (perp / pn)) \
                if pn > 1e-12 else n_hat.copy()
        g = float(np.clip(an, rho1 * np.exp(-X[6, k]), rho2 * np.exp(-X[6, k])))
        X[7:10, k] = d                                    # DIRECTION is the state, unit norm
        U[3, k] = g                                       # MAGNITUDE is the ZOH control
        if k < K - 1:
            X[6, k + 1] = X[6, k] - dyn.alpha * g * s0 * dtau
    if term_point_deg is not None and term_point_nodes > 0 and term_point_deg <= 0:
        for k in range(max(0, K - int(term_point_nodes)), K):
            X[7:10, k] = n_hat
    # THE GUESS IS DELIBERATELY INDEPENDENT OF slew_max_nd. The parameter is accepted so callers
    # need not change, and is NOT USED.
    #
    # It used to slerp consecutive guess directions back onto the slew cone, which made the COLD
    # START a function of the constraint. That is fatal to the one property a constraint must have:
    # tightening it can only cost more. Measured, it did not --
    #
    #     tight tolerances, fuel [kg]:   none 5.49711   2.0 deg/s 5.50094
    #                                    5.0  5.50316   10.0      5.50331
    #
    # i.e. among the constrained solves the TIGHTEST limit was the CHEAPEST, exactly backwards,
    # and at default tolerances nine of eleven constrained solves beat the unconstrained one. A
    # tighter feasible set cannot contain a better optimum, so the solver was not finding optima --
    # it was finding whatever its seed led to, and the seed was being moved by the limit.
    #
    # The guess's own peak turn rate is 3.056 deg/s, so the slerp fired for EVERY limit below that,
    # reshaping up to 38 of 59 nodes and moving guess directions by up to 95 deg. 5 and 10 deg/s
    # left it untouched and came out 0.00015 kg apart -- the same run twice. 2 deg/s moved it and
    # came out anomalously cheap. That is the whole effect.
    #
    # An infeasible guess is not a problem: SCvx exists to start from one. The subproblem imposes
    # the slew bound on the VARIABLES, not on the reference, so iteration 1 already returns a
    # trajectory inside the cone; the trust region governs how fast it gets there.
    # THAT LAST CLAIM WAS WRONG, AND THIS IS THE CORRECTION. The DYNAMICS carry virtual control,
    # so an infeasible guess is survivable there. The PATH constraints do not: the slew bound is
    # hard on the variables and the trust radius is a hard cap, so when the reference sits far
    # outside the cone the subproblem has no solution at all. Measured with the slerp simply
    # deleted: every limit at or below 1.5 deg/s returned subproblem_infeasible at iteration 0.
    #
    # So the guess IS smoothed onto a cone -- but onto a FIXED one, GUESS_MAX_TURN_DEG per
    # segment, which does not depend on the requested limit. That keeps both properties at once:
    # every slew limit gets the identical cold start (so tightening a constraint can no longer
    # change the seed and come out cheaper), and the start is inside the cone of any limit down
    # to about 0.16 deg/s, so iteration 0 is feasible.
    GUESS_MAX_TURN = np.radians(GUESS_MAX_TURN_DEG)
    for k in range(K - 1):
        c = float(np.clip(X[7:10, k] @ X[7:10, k + 1], -1.0, 1.0))
        th = float(np.arccos(c))
        if th > GUESS_MAX_TURN:
            perp = X[7:10, k + 1] - c * X[7:10, k]
            npp = float(np.linalg.norm(perp))
            if npp > 1e-12:
                X[7:10, k + 1] = (np.cos(GUESS_MAX_TURN) * X[7:10, k]
                                  + np.sin(GUESS_MAX_TURN) * perp / npp)

    # AFTER the turn cap, not before. The cap walks FORWARD from node 0 limiting each
    # consecutive turn to GUESS_MAX_TURN_DEG, so a blend applied first is simply flattened
    # back out: measured, the guess still ended at 36.99 deg at K=60 (needing 7.7 deg/node,
    # above the 3 deg cap) while K=100 worked because it only needed 1.4 deg/node.
    # Applied after, the tail turns at ~0.22 deg/s on the guess's own flight time -- still
    # inside any slew cone that matters.
    _nb = max(2, int(round(GUESS_TERMINAL_BLEND_FRAC * (K - 1))) + 1)
    _k0 = K - _nb
    for k in range(_k0, K):
        wgt = (k - _k0) / max(1, (K - 1) - _k0)          # 0 at the first blended node, 1 at the last
        dk = X[7:10, k].copy()
        cth = float(np.clip(dk @ n_hat, -1.0, 1.0))
        th_now = float(np.arccos(cth))
        th_new = (1.0 - wgt) * th_now                    # linearly to zero at the final node
        al = th_now - th_new
        if al > 1e-12:
            perp = n_hat - cth * dk
            pn = float(np.linalg.norm(perp))
            if pn > 1e-12:
                X[7:10, k] = dk * np.cos(al) + (perp / pn) * np.sin(al)

    # omega, NOT a finite difference of d. The control is now the angular rate in d-dot = omega x d,
    # and (d_{k+1} - d_k)/dt is not that -- it is a chord, which has a component along neither axis
    # the rotation uses. The rotation that carries d_k to d_{k+1} turns by the angle between them
    # about the axis d_k x d_{k+1}, so that cross product, normalised and scaled by theta/dt, IS
    # omega. It is automatically perpendicular to d_k, so the guess satisfies the gauge row in
    # subproblem.build exactly rather than needing the trust region to walk it there.
    for k in range(K - 1):
        a_, b_ = X[7:10, k], X[7:10, k + 1]
        ax = np.cross(a_, b_); axn = float(np.linalg.norm(ax))
        th = float(np.arctan2(axn, float(np.clip(a_ @ b_, -1.0, 1.0))))
        U[0:3, k] = (ax / axn) * (th / (s0 * dtau)) if axn > 1e-12 else np.zeros(3)
    U[0:3, K - 1] = U[0:3, K - 2]
    return X, U, s0


def run(dyn, scales, x_init, rf_nd, n_hat, K=30, mode="fuel", D_nd=None,
        warm=None, iters=25, w_vc=1e4, w_tr=0.0, w_ts=0.0,
        R0=0.35, Rs0=0.50, R_bounds=(1e-4, 1.0),
        tol_obj=1e-4, tol_abs=None, tol_defect_m=5e-3, tol_defect_v=5e-4, tol_vc_m=1e-3,
        tol_vc_v=1e-4, tol_vc_kg=1e-4, tol_vc_rad=1e-4,
        d_hi_m=2.0, d_lo_m=0.05, w_growth=2.0, w_anneal_it=2.0, w_bounds=(1e-4, 1e3),
        n_sub_disc=10, n_sub_prop=20, verbose=True, label="", fix_tf_s=None, tf_seed_s=None,
        term_point_deg=C.TERMINAL_POINT_DEG, term_point_nodes=C.TERMINAL_POINT_NODES,
        slew_max_deg_s=C.SLEW_MAX_DEG_S, slew_form="chord", a_init=None, d_init=None,
        v_lat_max=None,
        term_point_start_deg=None,
        term_point_frac=None,
        align_theta0_deg=None, align_h_m=None,
        gs_deg=None, point_deg=None,
        ratio_test=False, rho0=0.0, rho1=0.1, rho2=0.5, rho_shrink=1.5, rho_grow=2.0,
        tr_conv_frac=0.9):
    """PTR loop. Returns a Result.

    THE TRUST REGION IS A HARD CAP, NOT A PRICE (w_tr = w_ts = 0 by default).

    Pricing the radius in the cost was tried and measured to fail on this problem, for a reason
    specific to it: 58% of the propellant is burned idling at the thrust floor, so fuel is dominated
    by flight duration, d(fuel)/d(tf) = 2.3e-3 kg/s. In the objective's units 100 s of flight time is
    worth 0.042, while the trust price for that same 100 s at w_ts = 1 is 0.167 -- FOUR TIMES the
    fuel it would save. The optimiser then refuses to move the flight time at all and reports
    convergence at whatever the seed handed it, which is the same guess-dependence the old solver
    had, reached by a different route. A hard cap bounds the step without putting a price on
    optimality, and the cap is what the defect adapts.

    Convergence needs all three of:
      * the objective has stopped moving        (relative change < tol_obj)
      * the linear model is telling the truth   (max nonlinear defect < tol_defect_m, in METRES)

    tol_defect_m is 5 mm, not the 0.5 m first tried. Half a metre per segment sounds small and is
    not: flown OPEN LOOP the per-segment defects accumulate, and a solve that stopped at 0.23 m/
    segment landed 0.196 m off the site with 0.247 m of terminal altitude left, against 0.7 mm for
    the previous solver. It also made the accuracy run-dependent -- two solves of the same problem
    finished at 0.0085 m and 0.229 m purely according to which side of the tolerance they landed.
      * the virtual control is spent            (every channel inside its own tolerance)

    THE VIRTUAL-CONTROL TEST IS PER CHANNEL, AND IN PHYSICAL UNITS. nu is a defect in the next
    node's STATE, and that state is not all lengths. Rows 0:3 are a position, 3:6 a velocity, row 6
    is ln(mass), and rows 7:10 are a unit DIRECTION, which is dimensionless -- its residual already
    IS radians. The old test summed all ten rows and multiplied by L, which is the right conversion
    for exactly the first three.

    That was tolerable while position dominated nu. Under the cross-product kinematics it does not:
    the ball ||d|| <= 1 caps a norm the LINEARISED rotation pushes above 1, nu absorbs the
    difference, and measured on the Eros nominal at the plateau 99.5% of |nu|_1 sits on the
    DIRECTION rows. Every channel there is negligible --

        position   5.0e-10 nd  x L   = 6.0e-06 m
        velocity   8.2e-10 nd  x L/T = 1.6e-08 m/s
        log-mass   4.7e-11     x m   = 2.8e-08 kg
        direction  2.6e-07 rad       = 0.05 arcsec of pointing

    -- but the aggregate reported 3.1e-03 "m" against a 1e-03 m bar, so a solve sitting at dJ/J
    4e-6 (bar 1e-4) with a 3.8e-05 m defect (bar 5e-03 m) was labelled maxiter. The benchmarks only
    reported converged because the verification runs pass a loosened tol_vc_m = 1e-2.

    NOT fixed by loosening the aggregate, which is why it is split instead: a bar loose enough to
    pass that pointing residual would also pass a genuine POSITION defect 500x larger.

    The four tolerances and where they come from:
      tol_vc_m   = 1e-3 m     position, five times under tol_defect_m, the original rationale
      tol_vc_v   = 1e-4 m/s   velocity, the same factor of five under tol_defect_v
      tol_vc_kg  = 1e-4 kg    mass, matching tol_abs for the fuel objective -- cheating by less
                              propellant than the objective can resolve is not cheating
      tol_vc_rad = 1e-4 rad   direction, 0.006 deg, far below any pointing that matters mechanically
    Each channel is divided by its own bar and the WORST ratio decides, so one number still says
    pass or fail and `vc_worst` in the history says which channel was binding.

    The first alone is what the previous solver used a six-way step-size conjunction for, and that
    conjunction was satisfiable by a collapsed trust region rather than by convergence. A defect in
    metres cannot be faked that way.
    """
    v0n = float(np.linalg.norm(x_init[3:6])) * scales.v
    rng_m = float(np.linalg.norm(x_init[0:3] - rf_nd)) * scales.L
    t_lo, t_hi = C.tf_bounds(v0n, rng_m)
    s_lo, s_hi = scales.t_nd(t_lo), scales.t_nd(t_hi)
    if fix_tf_s is not None:
        # Fixed-flight-time mode. With s pinned, the only remaining non-convexity is
        # gravity, whose linearisation is accurate to well under 1% of control
        # authority over a trust radius -- so each solve is essentially a convex
        # problem and its answer cannot depend on the guess. Sweeping tf this way
        # gives the ground-truth fuel-vs-time curve that the free-time solver's
        # answer has to sit on.
        s_lo = s_hi = scales.t_nd(float(fix_tf_s))

    vtouch_nd = C.VTOUCH / scales.v
    # TERMINAL CORRIDOR AS A FRACTION OF THE FLIGHT, NOT A NODE COUNT.
    # term_point_nodes is a count, so the same setting means a different physical requirement at
    # every K: at 2 deg/s and tf ~2040 s, N=5 covers the last 100 m at K=100 but only ~50 m at
    # K=200 and ~170 m at K=60. Monte Carlo makes it worse -- a dispersed initial condition
    # changes tf, so a fixed count slides around in altitude from case to case.
    #
    # A FRACTION is invariant to both. frac = 0.05 is the last 5% of the flight, which on the
    # nominal is the last 100 m (nodes 95-99 at K=100, entered at t/tf = 0.960 and 2.67 m/s).
    # The corridor spans frac*(K-1) segments, so the node count follows K automatically.
    if term_point_frac is not None:
        term_point_nodes = max(2, int(round(float(term_point_frac) * (K - 1))) + 1)
    Pn = np.eye(3) - np.outer(n_hat, n_hat)
    # rad per non-dimensional time unit
    slew_nd = None if slew_max_deg_s is None else np.radians(slew_max_deg_s) * scales.T

    if warm is not None:
        Xr, Ur, sr = warm
        Xr, Ur, sr = np.array(Xr, float), np.array(Ur, float), float(sr)
        # RESAMPLE THE WARM START ONTO THE REQUESTED MESH. Both discretize() and build() read K from
        # Xref.shape[1], so a warm start with a different node count silently REPLACED the K that was
        # asked for -- a call requesting K=40 with a 60-node guess solved K=60 and said nothing. That
        # makes any comparison across a mesh change meaningless without it being visible.
        if Xr.shape[1] != K:
            tw = np.linspace(0.0, 1.0, Xr.shape[1]); tq = np.linspace(0.0, 1.0, K)
            Xr = np.array([np.interp(tq, tw, Xr[i]) for i in range(Xr.shape[0])])
            Ur = np.array([np.interp(tq, tw, Ur[i]) for i in range(Ur.shape[0])])
        # ANCHOR ONLY r, v, z. The initial condition constrains those seven rows and nothing
        # else -- subproblem.build pins exactly X[0:7, 0] == x_init[0:7] -- while d_0 is a
        # DECISION VARIABLE, free unless a_init pins it to a measured thrust. Overwriting
        # Xr[7:10, 0] from x_init threw away the warm start's own handover direction and
        # replaced it with whatever seed the caller happened to carry, which is both wrong
        # and wasteful when the previous cycle already solved for it.
        #
        # It also means x_init may be SEVEN long. closed_loop passes [r, v, z] because in the
        # closed loop the thrust is measured and enters through a_init, not through the state;
        # run_descent passes ten because problem_setup seeds a direction too. Slicing to 0:7
        # accepts both instead of crashing on the seven (measured: ValueError broadcasting
        # (7,) into (10,) at cycle 0 of every closed-loop run).
        Xr[0:7, 0] = np.asarray(x_init, float)[0:7]
        # RENORMALISE THE DIRECTIONS. Interpolating unit vectors componentwise does not give unit
        # vectors -- the resample above shortens every one that turns -- and ||d|| is conserved by
        # the dynamics, so a short reference direction is not an error the solver recovers from,
        # it is the norm it will then hold. Cheap, and it makes the node-0 plane well posed.
        dn = np.linalg.norm(Xr[7:10, :], axis=0)
        Xr[7:10, :] /= np.where(dn > 1e-12, dn, 1.0)
        Ur = _gauge(Xr, Ur)
        if term_point_deg is not None and term_point_nodes > 0 and term_point_deg <= 0:
            for k in range(max(0, K - int(term_point_nodes)), K):
                Xr[7:10, k] = n_hat
        Ur[0:3, -1] = Ur[0:3, -2]               # and to the tied terminal rate
    else:
        Xr, Ur, sr = initial_guess(dyn, x_init, rf_nd, n_hat, scales, K, s_lo, s_hi,
                                   term_point_deg, term_point_nodes, slew_nd,
                                   s0_override=(None if tf_seed_s is None
                                                else scales.t_nd(float(tf_seed_s))),
                                   point_deg=point_deg)

    # ---- ANCHOR THE REFERENCE WHERE THE HANDOVER PINS IT --------------------------------------
    # d_init (and a_init) make X[7:10, 0] an EQUALITY in the subproblem, but neither the cold guess
    # nor the warm resample above knows that: the guess points d_0 wherever the Hermite's required
    # acceleration points, and the resample keeps whatever direction the warm start carried. The
    # reference then disagrees with the constraint at node 0 from the first iteration, and every
    # subproblem spends trust-region budget dragging it there instead of improving the trajectory.
    #
    # Only the reference is touched. The constraint itself lives in subproblem.build; this just
    # starts the linearisation at the point the constraint will hold it to.
    _d0 = a_init if a_init is not None else d_init
    if _d0 is not None:
        _d0 = np.asarray(_d0, float)
        _n0 = float(np.linalg.norm(_d0))
        if _n0 > 1e-12:
            Xr[7:10, 0] = _d0 / _n0
            # omega_0 is gauged against dref_0, which has just moved, so the gauge has to follow.
            Ur = _gauge(Xr, Ur)

    # ---- the SCvx merit function, for the ratio test ------------------------------------------
    # Same cost and same penalty as subproblem.build's objective, but evaluated on the NONLINEAR
    # dynamics: the subproblem pays w_vc * sum|nu| for the defects its LINEAR model admits, and
    # this pays w_vc * (true L1 defect) for the ones the real dynamics produce. Identical units,
    # so predicted and actual reductions are comparable and rho means something.
    def _merit(X_, U_, s_, l1_):
        cost = (-X_[6, -1] / scales.alpha_nd if mode == "fuel"
                else float(np.linalg.norm(Pn @ (X_[0:3, -1] - rf_nd))))
        return float(cost + w_vc * l1_)

    hist = []; t0 = time.perf_counter(); Jprev = np.inf; status = "maxiter"
    d_prev = None; R, Rs = R0, Rs0; merit_ref = None; n_rej = 0; n_floor_rej = 0
    if verbose:
        print(f"  {'it':>3}{'solver':>9}{'J':>12}{'dJ/J':>10}{'defect[m]':>11}{'defect[m/s]':>11}"
              f"{'vc/tol':>10}{'tf[s]':>9}{'R':>9}{'eta/R':>8}{'etas/Rs':>8}"
              f"{'|d|lo':>8}{'|d|hi':>8}")
    for it in range(iters):
        Ad, Bd, Sd, zd, _ = discretize(dyn, Xr, Ur, sr, n_sub=n_sub_disc)
        prob, V = SP.build(Ad, Bd, Sd, zd, Xr, Ur, sr, x_init, rf_nd, n_hat, scales,
                           mode=mode, D_nd=D_nd, w_vc=w_vc, w_tr=w_tr, w_ts=w_ts,
                           s_lo=s_lo, s_hi=s_hi, vtouch_nd=vtouch_nd, R=R, Rs=Rs,
                           term_point_deg=term_point_deg,
                           term_point_nodes=term_point_nodes,
                           term_point_start_deg=term_point_start_deg,
                           align_theta0_deg=align_theta0_deg,
                           align_h_nd=(None if align_h_m is None
                                       else float(align_h_m)/scales.L),
                           slew_max_nd=slew_nd, slew_form=slew_form, a_init=a_init, d_init=d_init,
                           v_lat_max_nd=(None if v_lat_max is None
                                         else float(v_lat_max)/scales.v),
                           **({} if gs_deg is None else dict(gs_deg=gs_deg)),
                           **({} if point_deg is None else dict(point_deg=point_deg)))
        name, st = SP.solve(prob)
        if name is None:
            status = f"subproblem_{st}"
            break
        Xn, Un, sn = np.asarray(V["X"].value), np.asarray(V["U"].value), float(V["s"].value)
        # is the trust region actually BINDING, or is it slack and therefore irrelevant?
        eta_v = float(np.max(np.asarray(V["eta"].value)))
        etas_v = float(np.asarray(V["eta_s"].value))
        nu_v = np.abs(np.asarray(V["nu"].value))
        vc = float(nu_v.sum())
        # WHERE the virtual control lives, by state block. A scalar |nu|_1 says the subproblem had to
        # cheat; it does not say on which equation, and the answer decides what to do about it. The
        # direction rows are the ones to watch: the ball ||d|| <= 1 caps a norm that the LINEARISED
        # rotation wants to drift above, so nu has a floor there set by the per-segment turn, and
        # that floor is a property of the formulation rather than a convergence failure.
        # EACH CHANNEL IN ITS OWN UNITS. nu is a defect in the next node's STATE, and that state is
        # not all lengths: rows 0:3 are a position (x L -> m), 3:6 a velocity (x L/T -> m/s), row 6
        # is ln(mass) (x m -> kg) and rows 7:10 are a unit DIRECTION, which is dimensionless, so the
        # residual there already IS radians. Converting the lot with L is only correct for the first
        # three.
        vc_r = float(nu_v[0:3].sum()) * scales.L
        vc_vel = float(nu_v[3:6].sum()) * scales.v
        vc_kg = float(nu_v[6].sum()) * float(np.exp(Xn[6, 0]))
        vc_rad = float(nu_v[7:10].sum())
        vc_blk = dict(r_m=vc_r, v_ms=vc_vel, m_kg=vc_kg, d_rad=vc_rad)
        # how close each channel is to its own bar; the worst one decides
        vc_ratio = max(vc_r / tol_vc_m, vc_vel / tol_vc_v,
                       vc_kg / tol_vc_kg, vc_rad / tol_vc_rad)
        vc_worst = max(((vc_r / tol_vc_m, "r"), (vc_vel / tol_vc_v, "v"),
                        (vc_kg / tol_vc_kg, "m"), (vc_rad / tol_vc_rad, "d")))[1]
        dpos_nd, dvel_nd, _, _, l1_c = defect(dyn, Xn, Un, sn, n_sub=n_sub_prop)
        dmax_m = dpos_nd * scales.L
        dvel_ms = dvel_nd * scales.v
        dnrm = np.linalg.norm(Xn[7:10, :], axis=0)
        dn_lo, dn_hi = float(dnrm.min()), float(dnrm.max())
        J = float(prob.value)
        # Converge on the PHYSICAL objective. The penalised value carries a large
        # constant (-z_K/alpha_nd ~ -705), so a relative change measured on it is
        # diluted by ~700x and reports convergence while the fuel is still moving.
        Jphys = (float(np.linalg.norm(Pn @ (Xn[0:3, -1] - rf_nd))) * scales.L if mode == "error"
                 else float(np.exp(Xn[6, 0]) - np.exp(Xn[6, -1])))
        # Absolute OR relative. A purely relative test never settles on an objective
        # that goes to zero: stage 1 drives the miss to ~3e-5 m and the relative change
        # between two such numbers stays at ~4e-3 forever, so it ran to maxiter with a
        # converged answer in hand. tol_abs carries the units of the stage (metres for
        # landing error, kilograms for fuel).
        #
        # tol_obj STAYS AT 1e-4. Three mid-descent cycles used to run to maxiter on this test
        # alone -- defect 1e-4 m, virtual control 5e-6 m, trajectories good to millimetres -- while
        # the iteration flip-flopped between two answers differing by about a gram of propellant on
        # a 1-2 kg burn. Loosening this tolerance to 1e-3 would have hidden that, but the cause was
        # the trust radius alternating 1.0 <-> 0.5, and annealing w_growth fixed it at the strict
        # tolerance instead (see the adaptation block). Keep the tight test: it is the only thing
        # that says successive convexification actually settled.
        dJ = abs(Jphys - Jprev)
        rel = dJ / max(1e-12, abs(Jphys))
        ta = tol_abs if tol_abs is not None else (1e-2 if mode == "error" else 1e-4)
        obj_ok = (dJ < ta) or (rel < tol_obj)
        # Jphys IS THE QUANTITY THE CONVERGENCE TEST JUDGES (kg of propellant in fuel mode, metres
        # of miss in error mode), so it belongs in the history. Without it you cannot tell a solve
        # that stopped because it ARRIVED from one that stopped because its per-iteration
        # improvement fell under tol_abs while still descending -- which looks identical in every
        # other column and is the difference between a converged answer and an abandoned one.
        # ---- STEP ACCEPTANCE (standard SCvx ratio test; ON by default) --------------------------
        # Without this every step is taken, including ones that make the merit WORSE, and the
        # monotone decrease the SCvx convergence proof relies on does not hold. The radius is then
        # steered off the measured defect instead, which says how good the LINEARISATION is but not
        # whether the STEP helped. Measured with it off, on a stalling pinned stage 1: J wandered
        # 0.01037, 0.01009, 0.01004, 0.01013, 0.01023, 0.00973 -- not descending -- with eta/R at
        # 0.2-4 %, so the steps were small because the subproblem wanted them small, not because
        # the cap was tight, and 200 iterations ended 39 m from the site.
        #
        # NOT ON BY DEFAULT YET, DELIBERATELY. Enabling it makes the rho-band branch below `continue`
        # past the defect-based radius rule AND its w_tr/w_ts annealing, which is tuned (see
        # w_anneal_it). That is swapping a tuned controller for an untuned one, and the only
        # evidence so far is two cases that were already failing. It needs a regression over normal
        # seeds -- status, iterations, fuel, tf -- before the default moves.
        #
        # ON THE w_tr/w_ts CAUTION THIS COMMENT USED TO CARRY. pred uses J, the subproblem optimum,
        # which carries w_tr*sum(eta) + w_ts*eta_s on top of cost + w_vc*l1, while _merit carries
        # only cost + w_vc*l1; with a priced trust region pred is inflated by a penalty act never
        # pays and rho comes out systematically small. That cannot happen here, because the two
        # branches are mutually exclusive: the weights are only ever moved by the defect-based rule
        # below, which the ratio-test branch skips with its own `continue`. w_tr and w_ts start at
        # 0.0 and with the ratio test on nothing raises them, so pred and act stay in the same
        # units. (With it OFF they do become nonzero, since max(0/w_eff, w_bounds[0]) = 1e-4.)
        rho = np.nan; accepted = True
        if ratio_test:
            if merit_ref is None:
                _, _, _, _, l1_r = defect(dyn, Xr, Ur, sr, n_sub=n_sub_prop)
                merit_ref = _merit(Xr, Ur, sr, l1_r)
            merit_c = _merit(Xn, Un, sn, l1_c)
            pred = merit_ref - J            # J is the subproblem optimum: the LINEAR model's merit
            act = merit_ref - merit_c       # what the NONLINEAR dynamics actually delivered
            # A non-positive predicted reduction means the linear model did not promise an
            # improvement; then the only sensible question is whether the truth improved.
            rho = (act / pred) if pred > 1e-14 else (1.0 if act > 0.0 else -1.0)
            accepted = rho >= rho0
        # THE SOLVER'S OWN VERDICT, not just which solver ran. subproblem.solve accepts
        # 'optimal_inaccurate' as a success, so a degraded subproblem becomes a flown plan
        # with nothing downstream aware of it -- and cvxpy prints its warning only once per
        # source line, so the log cannot be used to count them.
        hist.append(dict(it=it, solver=name, solver_status=st, J=J, jphys=Jphys, rel=rel,
                         defect_m=dmax_m,
                         defect_v=dvel_ms, vc=vc, tf_s=scales.t_dim(sn), w_tr=w_tr,
                         eta=eta_v, eta_s=etas_v, R=R, Rs=Rs, vc_m=vc_r,
                         vc_ratio=vc_ratio, vc_worst=vc_worst,
                         dnorm_lo=dn_lo, dnorm_hi=dn_hi, vc_blk=vc_blk,
                         rho=rho, accepted=accepted, n_rej=n_rej))
        if verbose:
            print(f"  {it:>3}{name:>9}{J:>12.5f}{rel:>10.2e}{dmax_m:>11.3e}{dvel_ms:>11.3e}"
                  f"{vc_ratio:>9.1e}{vc_worst:>1}{scales.t_dim(sn):>9.1f}{R:>9.1e}"
                  f"{eta_v/max(R,1e-30):>8.1%}{etas_v/max(Rs,1e-30):>8.1%}"
                  f"{dn_lo:>8.4f}{dn_hi:>8.4f}")
        # REJECT: keep the reference, shrink the radius, and do NOT advance Jprev or d_prev --
        # the next iteration must re-solve the SAME linearisation point inside a smaller region.
        if not accepted:
            n_rej += 1
            # A REJECTED STEP AT THE RADIUS FLOOR IS AN INFINITE LOOP, NOT SLOW PROGRESS. Rejecting
            # keeps the reference, and at R = R_bounds[0] the shrink is a no-op, so the next
            # iteration poses the IDENTICAL subproblem and gets the identical answer. Measured on
            # seed 852: iterations 100 to 199 reported J 0.055592, defect 4.151e-05, eta/R 100.0 %
            # and rho -0.104 every single time, to the last digit -- a hundred solves of one
            # problem. Stop and say so; the caller can then tell a stall from a case the physics
            # refuses, which a bare "maxiter" cannot.
            at_floor = (R <= R_bounds[0] * (1.0 + 1e-9)) and (Rs <= R_bounds[0] * (1.0 + 1e-9))
            R = max(R / rho_shrink, R_bounds[0]); Rs = max(Rs / rho_shrink, R_bounds[0])
            n_floor_rej = n_floor_rej + 1 if at_floor else 0
            if n_floor_rej >= 2:          # 2, not 1, so a one-off numerical blip cannot end a run
                status = "stalled"; break
            continue
        Xr, Ur, sr = Xn, Un, sn
        # Put omega back in the subproblem's gauge against the NEW directions. Without this the
        # hard row dref.omega = 0 is violated by the reference itself and costs trust-region budget
        # to reach -- measured sending CLARABEL primal infeasible on alternate iterations. See _gauge.
        Ur = _gauge(Xr, Ur)
        if ratio_test:
            merit_ref = merit_c
        # A STEP PINNED AGAINST THE TRUST CAP IS NOT CONVERGENCE. eta is the per-node step and R
        # its hard cap, so eta/R near 1 means the optimiser wanted to move further and was
        # stopped by the cap -- the iterate has not settled, the leash is just short. Without
        # this guard a collapsing radius reports success: measured on the 5 deg/s case with the
        # ratio test, R fell 0.35 -> 0.00195 (512x) while the fuel was still descending, and the
        # shrinking steps pushed dJ/J under tolerance at 5.6959 kg against the 5.5521 kg the
        # same problem reaches with a healthy radius. The defect and virtual control look fine
        # there precisely BECAUSE the steps are tiny, so they cannot catch it either.
        # BOTH trust regions, not just the state one. eta_s caps how far the FLIGHT TIME may
        # move per iteration, and tf is the variable that sets the fuel here: 58% of the
        # propellant is burned idling at the thrust floor, so d(fuel)/d(tf) = 2.3e-3 kg/s and
        # fuel is very nearly a function of tf alone. Checking eta/R but not eta_s/Rs let a
        # solve report success while s was still travelling at its cap, so tf -- and therefore
        # the fuel -- had not settled. That is how two solves of the SAME problem landed 6 s
        # apart in tf and 0.045 kg apart in fuel, and why tightening a slew limit could come
        # out CHEAPER than not having one, which a feasible-set argument forbids.
        tr_slack = (eta_v <= tr_conv_frac * R) and (etas_v <= tr_conv_frac * Rs)
        if obj_ok and dmax_m < tol_defect_m and dvel_ms < tol_defect_v and vc_ratio < 1.0 \
                and tr_slack:
            status = "converged"; break
        Jprev = Jphys
        # RHO-BAND RADIUS UPDATE, the standard SCvx schedule, used instead of the defect rule when
        # the ratio test is on: shrink when the model over-promised, grow when it under-promised.
        if ratio_test:
            if rho < rho1:
                R = max(R / rho_shrink, R_bounds[0]); Rs = max(Rs / rho_shrink, R_bounds[0])
            elif rho > rho2:
                R = min(R * rho_grow, R_bounds[1]); Rs = min(Rs * rho_grow, R_bounds[1])
            d_prev = dmax_m
            continue

        # Price the step from the measured linearisation error -- but from whether the step made
        # it BETTER, not from an absolute threshold. Early iterations always carry a large defect
        # because the straight-line reference is dynamically meaningless; raising the trust price
        # then pins the iterate to that bad reference, which was measured driving w_tr to its cap
        # by iteration 17 and freezing the solve. Reacting to the CHANGE lets the iteration escape a
        # bad start and still clamps down once it is close.
        # DO NOT REACT TO NOISE. The rule below triggers on the RELATIVE change in the defect, and
        # once the defect is far inside tolerance that ratio is meaningless: measured in the
        # endgame it alternated 7.0e-6 -> 1.02e-5 -> 7.0e-6 m, ratios of 1.46 and 0.68, which trips
        # shrink-then-grow-then-shrink forever. Both numbers are six orders below tol_defect_m, so
        # the iteration was chasing nothing and sat in a period-2 limit cycle to maxiter, at 30
        # iterations instead of about 6. Below this floor the step is already as good as the
        # tolerance asks for and the radius is left alone.
        #
        # THE FLOOR HAS TO BE LOW. At 0.1*tol_defect_m it also froze cycles whose defect was
        # 2.8e-4 to 3.3e-4 m and whose OBJECTIVE was still oscillating -- and freezing is exactly
        # the wrong response there, because the cure for an oscillating objective is to SHRINK the
        # radius, which the frozen rule can no longer do. Measured: two mid-descent cycles that had
        # converged in 15 and 5 iterations went to maxiter, with eta/R alternating 62.4/51.6%.
        # 0.01*tol_defect_m still catches the genuine noise case (defect 4e-7 m) and leaves those
        # alone.
        adapt_floor = 0.01 * tol_defect_m
        # ANNEAL THE STEP. A fixed factor cannot serve both ends of the iteration. Early on the
        # reference is a straight line and dynamically meaningless, so the radius has to move in
        # big jumps to escape it -- that is why w_growth is 2. Late on, a factor of 2 re-solves a
        # materially different problem every iteration, and when the defect ratio happens to
        # straddle the 1.2/0.8 triggers the radius alternates 1.0 <-> 0.5 forever, which keeps the
        # fuel objective oscillating by about a gram and never lets the iteration settle.
        #
        # Flattening w_growth to 1.3 was measured and made it WORSE: one cycle went from converged
        # to maxiter and another got slower, because the early phase lost the step size it needed.
        # Decaying it instead keeps the fast escape and damps the endgame:
        #     w_eff = 1 + (w_growth - 1) / (1 + it/w_anneal_it)
        # At w_anneal_it = 2 the decay is deliberately fast -- 2.00, 1.50, 1.33, 1.17 at iterations
        # 0, 2, 4, 10. Measured on the three offending cycles at the STRICT tol_obj = 1e-4:
        #     cycle 20   flat: converged 25it     annealed: converged 26it
        #     cycle 24   flat: MAXITER   30it     annealed: converged 18it
        #     cycle 29   flat: MAXITER   30it     annealed: converged 15it
        # Slower decay is worse, not better: w_anneal_it = 10 sent cycle 20 to maxiter, because the
        # early phase still needs the big step. The fix is to decay QUICKLY, not gently.
        w_eff = 1.0 + (w_growth - 1.0) / (1.0 + it / max(w_anneal_it, 1e-9))
        if d_prev is not None and max(dmax_m, d_prev) > adapt_floor:
            if dmax_m > d_prev * 1.2:                     # the step made the model worse -> shrink
                w_tr = min(w_tr * w_eff, w_bounds[1]); w_ts = min(w_ts * w_eff, w_bounds[1])
                R = max(R / w_eff, R_bounds[0]); Rs = max(Rs / w_eff, R_bounds[0])
            elif dmax_m < d_prev * 0.8 and dmax_m < d_hi_m:
                w_tr = max(w_tr / w_eff, w_bounds[0]); w_ts = max(w_ts / w_eff, w_bounds[0])
                R = min(R * w_eff, R_bounds[1]); Rs = min(Rs * w_eff, R_bounds[1])
        d_prev = dmax_m

    # A subproblem with virtual control is always solvable, so genuine infeasibility of the PROBLEM
    # shows up as virtual control that never goes away, not as a solver failure. Saying so beats
    # reporting "maxiter" with a meaningless objective.
    #
    # Judged on the POSITION channel in metres. The old test summed all ten state rows and
    # multiplied by L, a conversion right only for the first three, so under the cross-product
    # kinematics it was dominated by the dimensionless direction rows -- it would call a solve
    # infeasible over a pointing residual of a few microradians.
    #
    # AND IT DOES NOT OVERWRITE A SOLVER FAILURE. "subproblem_infeasible" / "subproblem_unbounded"
    # mean CLARABEL and then SCS both refused the convex subproblem, which is a different event with
    # a different fix than a converged-but-cheating iterate, and relabelling it "infeasible" threw
    # that away. Measured on Malyuta K=151: it broke at iteration 29 of 600 on a solver failure and
    # reported "infeasible", which reads as "no trajectory exists" rather than "the SOCP blew up".
    if status != "converged" and not status.startswith("subproblem_") \
            and hist and hist[-1]["vc_blk"]["r_m"] > 1e-1:
        status = "infeasible"
    wall = time.perf_counter() - t0
    P = np.eye(3) - np.outer(n_hat, n_hat)
    miss_m = float(np.linalg.norm(P @ (Xr[0:3, -1] - rf_nd))) * scales.L
    fuel = float(np.exp(Xr[6, 0]) - np.exp(Xr[6, -1]))
    dpos_nd, dvel_nd, _, _, _ = defect(dyn, Xr, Ur, sr, n_sub=n_sub_prop)
    dnrm_f = np.linalg.norm(Xr[7:10, :], axis=0)
    return Result(X=Xr, U=Ur, s=sr, status=status, iters=len(hist), history=hist,
                  miss_m=miss_m, fuel_kg=fuel, tf_s=scales.t_dim(sr),
                  defect_m=dpos_nd * scales.L,
                  vc=hist[-1]["vc"] if hist else np.nan, wall_s=wall,
                  D_nd=float(np.linalg.norm(P @ (Xr[0:3, -1] - rf_nd))),
                  dnorm_lo=float(dnrm_f.min()), dnorm_hi=float(dnrm_f.max()))

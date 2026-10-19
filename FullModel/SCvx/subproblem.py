"""The convex subproblem: one SOCP, two objectives.

Everything is non-dimensional. The non-convexities of the original problem, and where each goes:
  (1) the thrust lower bound  Tmin <= ||T||  -- a plain box on the magnitude Gamma, which is a
      scalar decision variable here, so the bound is convex as written. (The earlier formulation
      carried the thrust as one vector and needed lossless convexification, Acikmese & Ploen 2007,
      to get the same thing; splitting magnitude from direction makes it trivial instead.)
  (2) the mass in  Gamma_phys in [Tmin/m, Tmax/m]  -- handled by z = ln m and a Taylor expansion of
      e^-z about the reference;
  (3) the flight time multiplying the dynamics -- handled by successive linearisation outside;
  (4) the thrust Gamma*d, a product of two decision variables -- linearised about the reference each
      iteration, the same bargain already made for gravity, exact at convergence;
  (5) ||d|| = 1 -- relaxed to ||d|| <= 1, which is a cone and so convex. The relaxation is TIGHT on
      the upper side by construction and on the lower side because d-dot = omega x d gives the
      optimiser no way to shorten d without paying virtual control. See the long note at that row.
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
          term_point_nodes=C.TERMINAL_POINT_NODES, slew_max_nd=None, a_init=None, d_init=None,
          slew_form="chord", v_lat_max_nd=None,
          term_point_start_deg=None,
          align_theta0_deg=None, align_h_nd=None):
    """Assemble and return (problem, variables). mode is 'error' or 'fuel'."""
    K = Xref.shape[1]
    dtau = 1.0 / (K - 1)
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
    # Only the first seven states are the handed-over condition. The thrust state a(0) is
    # bound separately (a_init) because it may legitimately be free: at handover the solver
    # choosing the best thrust to start from is INFORMATION FOR THE UPSTREAM PHASE, whereas
    # in closed loop it is measured and must be matched.
    cons += [X[0:7, 0] == np.asarray(x_init, float)[0:7]]
    # land ON the surface: the component of the terminal miss along the local vertical is zero.
    cons += [n_hat @ (X[0:3, K - 1] - rf_nd) == 0]
    cons += [cp.norm(X[3:6, K - 1]) <= vtouch_nd]
    # -- LATERAL touchdown speed, optional -------------------------------------------------------
    # ||v_K|| <= vtouch is a DISC, not a cone: it admits 0.3 m/s purely HORIZONTAL with zero
    # descent rate, i.e. the vehicle skidding across the surface at contact. Bounding the
    # component in the local horizontal plane removes that. Linear in the state, so convex.
    #
    # P = I - n n^T projects onto the plane perpendicular to the local vertical, which is the
    # same projector the miss and the glideslope already use.
    if v_lat_max_nd is not None:
        _Pl = np.eye(3) - np.outer(n_hat, n_hat)
        cons += [cp.norm(_Pl @ X[3:6, K - 1]) <= v_lat_max_nd]
    cons += [X[6, :] >= np.log(C.MDRY)]

    # -- lossless convexification of the thrust magnitude --------------------------------------
    rho1 = C.TMIN / scales.A
    rho2 = C.TMAX / scales.A
    zbar = Xref[6, :]
    Dref = Xref[7:10, :]          # reference thrust DIRECTION
    # A ZERO-ORDER HOLD HAS K-1 CONTROL INTERVALS, NOT K. U[:, K-1] never enters the dynamics --
    # the hold on segment k uses U[:, k] for k = 0 .. K-2 -- so it drives nothing and costs nothing,
    # and the optimiser leaves it anywhere the constraints permit. Left free it reported a thrust of
    # 3.3 N against a 5 N floor and a lossless-convexification gap of 13.25 N at that one node,
    # which looked like a failed relaxation and was not: the gap over the real intervals is <= 0.007 N.
    # Tying it to the last real interval makes the reported control history mean what it says.
    # omega ONLY. Tying the full column -- including Gamma -- was tried and REVERTED: it makes
    # stage 1 fail at iteration 0 with subproblem_unbounded for every K >= 150, while the
    # omega-only form converges there in 4 iterations. Measured both ways at K=150 and K=200,
    # with and without a terminal corridor, so it is the tie and not the corridor.
    #
    # The consequence is that Gamma[K-1] stays undetermined: it drives nothing (no dynamics row
    # uses it), costs nothing, and the solver parks it at its reference. Measured across seven
    # solves it sat at ~0.61 nd (about 12 N) while the real Gamma[K-2] ranged 0.25 to 1.01 nd.
    # So the stored control history carries one meaningless value, at the final node.
    #
    # HANDLE THAT AT THE POINT OF USE: drop node K-1 from any exported control history, and from
    # thrust plots. It matters most for PINN training labels, where it would otherwise contribute
    # one fabricated sample per trajectory at the terminal state.
    cons += [U[0:3, K - 1] == U[0:3, K - 2]]
    # -- omega's NULL DIRECTION, removed ---------------------------------------------------------
    # omega x d is blind to the component of omega along d: B[d,omega] = -skew(d) has rank 2, with
    # d spanning its null space. So omega and omega + c*d drive IDENTICAL trajectories and the
    # solver picks c arbitrarily. Two reasons that is not acceptable: ||omega|| then overstates the
    # slew rate by whatever c happened to be, so a slew bound spends budget on a component that
    # rotates nothing; and the reported control history stops meaning what it says.
    #
    # The row below is GAUGE FIXING, not a restriction -- it removes no achievable trajectory, only
    # the freedom to describe the same one many ways. Written against the REFERENCE direction so it
    # stays linear (dref.omega, not d.omega, which would be bilinear), and exact at convergence
    # where dref = d. With it, ||omega_k|| is the angular rate of the thrust direction in rad per
    # unit tau, full stop.
    for k in range(K - 1):
        cons += [Dref[:, k] @ U[0:3, k] == 0.0]

    # -- the thrust box, and ||d_k|| <= 1 --------------------------------------------------------
    # ||d_k|| <= 1 is the same row as before the cross product, and it does a completely different
    # job now, so the reasoning is worth setting out rather than leaving it looking unchanged.
    #
    # WHY IT IS NEEDED (the UPPER side). The kinematics conserve ||d|| exactly, but only in
    # CONTINUOUS time -- a LINEAR model of a rotation leaves the sphere. The Bd block contributes
    # -skew(dref)*dt*(omega - omega_ref), which is perpendicular to dref, and adding a perpendicular
    # vector to a unit vector lengthens it by second order. With no slew limit the min-max benchmark
    # turns 34.7 deg per segment, and measured there WITHOUT this row ||d|| reached 1.146: the thrust
    # hit 13375 N against a 13258 N cap -- a real constraint violation, not bookkeeping -- and the
    # corrupted linear model put the iteration in a limit cycle that never converged. This row is
    # hard and exact and removes all of it. It is also the thing a tangent plane at node 0 could NOT
    # provide: a plane bounds ||d_0|| from BELOW and leaves the drift above unbounded.
    #
    # WHY IT NO LONGER LEAKS DOWN. This is the part the cross product changed. Under the old free
    # rate d-dot = w, w could point straight back along d and shorten it at zero cost, which is
    # exactly what the optimiser did at the Tmin floor: ||d|| 0.957, 4.787 N delivered against 5 N
    # charged -- throttling under the floor without violating a constraint. Under d-dot = omega x d
    # there is NO control direction that shortens d, because the rate is confined to the tangent
    # plane of the sphere, so at nodes 1..K-1 shrinking requires VIRTUAL CONTROL, priced at
    # w_vc = 1e4 and driven to zero by the convergence test. The exploit is not penalised, it is
    # unreachable.
    #
    # WHAT IS LEFT, STATED PLAINLY. d_0 is not produced by any dynamics row -- X[:,0] carries only
    # x_init[0:7] -- so ||d_0|| <= 1 is the one place the optimiser may still choose a short
    # direction for free, and the conservation would then carry it to every node. It is held up by an
    # INCENTIVE, not a theorem: delivered thrust is Gamma*||d||, so a direction shorter by a factor c
    # needs Gamma/c for the same acceleration and costs 1/c more propellant at every node where the
    # thrust bound is not binding -- 58 of 60 on the Eros nominal. That is an argument, not a proof,
    # so it is MEASURED rather than asserted: ptr.py reports min/max ||d_k|| every iteration and in
    # the Result, and the verification runs print it. One scalar, audited, against sixty
    # independently shrinkable ones before.
    #
    # NOT COMBINABLE with a tangent plane dref_0.d_0 = 1, and not for a subtle reason: Cauchy-Schwarz
    # gives dref_0.d_0 <= ||d_0|| <= 1 with equality only at d_0 = dref_0, so the two rows together
    # collapse to the single POINT dref_0 and re-clamp the handover direction to the guess. The
    # one-sided version dref_0.d_0 >= 1 - tol does bound the norm below, but the same tol also caps
    # how far d_0 may rotate per iteration (half-angle arccos(1-tol)): 1e-4 pins the norm to four
    # decimals and allows 0.8 deg per iteration, far too slow to find a handover direction. The two
    # requirements are one number pulled in opposite directions, so this formulation takes the hard
    # upper bound and measures the lower one.
    #
    # The Gamma box is unchanged: e^-z is convex, so its 2nd-order expansion bounds it from above for
    # d > 0 and the 1st-order expansion bounds it from below everywhere -- lower bound to an SOC,
    # upper bound affine. The expansion is essentially exact here: the mass changes by ~1% over a
    # whole descent, so |d| <= 0.01 and the discarded term is O(d^3/6) ~ 1.7e-7 relative.
    for k in range(K):
        cons += [cp.norm(X[7:10, k]) <= 1.0]
        d = X[6, k] - zbar[k]
        mu1 = rho1 * np.exp(-zbar[k])
        mu2 = rho2 * np.exp(-zbar[k])
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
        cons += [n_hat @ X[7:10, k] >= cos_point]    # linear in d: ||d|| = 1 already

    # -- terminal thrust pointing --------------------------------------------------------------
    # The touchdown burn is the hold on interval K-2; U[:, K-1] is the tied dummy and drives nothing.
    # With the thrust a STATE, this lands on the thrust at CONTACT rather than on the last hold
    # interval -- a is what the vehicle is actually producing at the terminal node.
    # A RAMP, NOT A DEADLINE. With a single angle applied over the last N nodes the constraint is
    # a deadline: fuel-optimal steering holds its preferred ~28 deg off vertical for as long as it
    # is allowed and then SNAPS inside one segment. Measured at K=100, raising N from 1 to 12 did
    # not spread that turn at all -- it only moved it earlier (t/tf 0.99 -> 0.89) and cost 292 g,
    # about 1.2 g for every second of coverage.
    #
    # term_point_start_deg makes the cone CLOSE progressively instead: theta_k interpolates from
    # term_point_start_deg at the first constrained node to term_point_deg at touchdown, so the
    # approach to vertical is prescribed rather than the endpoint. Leaving it None reproduces the
    # old single-angle behaviour exactly.
    if term_point_deg is not None and term_point_nodes > 0:
        k_lo = max(0, K - int(term_point_nodes))
        th0 = term_point_deg if term_point_start_deg is None else float(term_point_start_deg)
        span = max(1, (K - 1) - k_lo)
        for k in range(k_lo, K):
            frac = (k - k_lo) / span
            th_k = th0 + frac * (term_point_deg - th0)
            if th_k <= 0.0:
                cons += [X[7:10, k] == n_hat]                   # exactly along the local vertical
            else:
                cons += [n_hat @ X[7:10, k] >= np.cos(np.radians(th_k))]

    # -- ALIGNMENT CORRIDOR, KEYED TO ALTITUDE -----------------------------------------------
    # The thrust must lie within theta(h) of the local vertical, theta closing from
    # align_theta0_deg at altitude align_h_m down to term_point_deg at the surface.
    #
    # WHY ALTITUDE AND NOT A NODE COUNT OR A FLIGHT FRACTION. The dataset for a guidance network
    # maps (x, v, tau) -> (x, v, T), so every trajectory in it must solve THE SAME problem. A
    # corridor over "the last N nodes" or "the last 5% of the flight" is a property of the
    # DISCRETISATION, not of the physics: tf changes with the initial condition, so the corridor
    # opens at a different altitude, speed and dynamical regime in every sample. Two trajectories
    # would then obey different terminal laws with nothing in (x, v, tau) to explain the
    # difference -- an inconsistent label, not noise to average out. Keyed to altitude, a vehicle
    # passing 50 m obeys the same law whether it started at 11 km or 8 km.
    #
    # CONVEXITY. Bounding the HORIZONTAL COMPONENT of the direction rather than the angle keeps
    # it a second-order cone: ||P d|| is a norm of a linear function of d, and the right-hand
    # side is affine in r. No linearisation, no reference direction, exact at every iteration --
    # unlike the gauge row, whose dref dependence was what made the old slew bound inexact.
    #
    #     || P d_k ||  <=  a + b * ( n . (r_k - rf) ),     P = I - n n^T
    #     a = sin(term_point_deg),  b = (sin(theta0) - a)/h_ramp
    #
    # It also switches itself off: ||P d|| <= ||d|| <= 1 always, so once the affine bound passes
    # 1 the row is vacuous. At theta0 = 45 deg and h_ramp = 100 m that happens at about 142 m.
    if align_theta0_deg is not None and align_h_nd is not None and align_h_nd > 0:
        _Pa = np.eye(3) - np.outer(n_hat, n_hat)
        _a = float(np.sin(np.radians(term_point_deg if term_point_deg is not None else 1.0)))
        _b = (float(np.sin(np.radians(align_theta0_deg))) - _a) / float(align_h_nd)
        for k in range(K):
            h_k = n_hat @ (X[0:3, k] - rf_nd)              # altitude above the datum plane, nd
            cons += [cp.norm(_Pa @ X[7:10, k]) <= _a + _b * h_k]

    # -- SLEW RATE ------------------------------------------------------------------------------
    # EXACT, not linearised. When the thrust was one vector the turn rate was ||a x adot|| / ||a||^2,
    # non-convex in a, and had to be approximated by projecting adot perpendicular to a REFERENCE
    # direction and scaling by Gamma. Splitting direction from magnitude removes all of that, and the
    # cross-product kinematics finish the job: omega IS the angular rate in rad per unit tau -- not a
    # proxy for it, not a bound on something proportional to it -- because omega x d cannot change
    # ||d||, so nothing omega does is spent on anything but rotation. With the gauge row above
    # removing the null component, ||omega_k|| is exactly the physical slew rate and the limit is a
    # plain norm bound. No projector, no Gamma on the right-hand side, nothing to converge.
    #
    # It also holds BETWEEN nodes, not just at them: omega is zero-order held, so the rate is
    # CONSTANT across each segment and the node-wise bound is the whole-segment bound.
    #
    # NO FLIGHT-TIME FACTOR. The bound is a plain constant, and that is not an approximation -- s
    # genuinely cancels out of the physical rate. Work it through:
    #
    #     dynamics.f gives the direction row as          omega x d
    #     discretize applies the time dilation:          dd/dtau = s * (omega x d)
    #     tau in [0,1] maps to [0, tf] with tf = s*T:    dt = s*T*dtau
    #     so                                             dd/dt = s*(omega x d)/(s*T)
    #                                                          = (omega x d)/T
    #
    # and with the gauge row making omega perpendicular to d, and ||d|| = 1, the physical angular
    # rate is exactly ||omega||/T rad/s. The bound is therefore ||omega|| <= radians(deg_s)*T, which
    # is slew_max_nd as ptr.run already computes it. Nothing is bilinear and nothing needs a
    # reference value, so this constraint is EXACT at every iteration, not just at convergence.
    #
    # THIS CARRIED A SPURIOUS * sref AND WAS NEVER CAUGHT, because no benchmark limits the slew rate
    # and the Eros nominal ran under its own limit. Measured at a 1 deg/s request: sref = 3.4721, so
    # the solver was actually permitting 3.47 deg/s. It looked harmless -- tightening the limit DID
    # change the trajectory -- but only because initial_guess walks the directions into the correct
    # cone (ptr.initial_guess, lim = slew_max_nd * s0 * dtau, which is right). The guess was doing
    # the work and the constraint was inert: the achieved rate sat at 91.8% of the true bound and
    # 26.4% of the imposed one, so it never appeared to bind.
    #
    # Note this constraint was never exercised by any of the three published benchmarks in
    # ../SCvx Verification -- none of them limits how fast the thrust may rotate -- so it carries no
    # external reference. The asteroid descent, at 2 deg/s, is the first case that activates it.
    # ---- SUPERSEDED BY THE CHORD FORM BELOW -----------------------------------------------------
    # ||omega_k|| <= slew_max_nd measures the turn rate only if omega is perpendicular to d. The
    # gauge row above makes it perpendicular to DREF, not to d, so with theta = angle(d, dref),
    #
    #     ||omega|| <= ||omega_perp|| / cos(theta)
    #
    # and the bound actually enforced is ||omega_perp|| <= slew_max_nd * cos(theta): TIGHTER than
    # requested, by a factor that changes every iteration as dref chases d. A limit that is slack
    # at convergence can therefore bind mid-iteration, steer the solver onto a different path, and
    # change the converged trajectory. Measured: at default tolerances, slack limits moved the
    # trajectory 90-140 m, against a 5 m same-limit repeat floor.
    #
    # The claim in the note above -- "nothing needs a reference value, so this constraint is EXACT
    # at every iteration" -- is wrong for exactly this reason. The constraint ROW needs no
    # reference; its MEANING does, through the gauge.
    if slew_max_nd is not None and slew_form == "omega":
        for k in range(K - 1):
            cons += [cp.norm(U[0:3, k]) <= slew_max_nd]

    # ---- SLEW RATE, as a bound on the TURN ITSELF -----------------------------------------------
    # Bound the chord between consecutive directions instead of the angular rate that produces it.
    # omega is zero-order held, so d rotates about a fixed axis at a CONSTANT rate across a
    # segment, and the total turn per segment is therefore exactly equivalent to the rate:
    #
    #     Delta_theta_k = ||omega_phys|| * Delta_t = (||omega_nd||/T) * (s*T*dtau)
    #                   = ||omega_nd|| * s * dtau
    #
    # so the rate limit ||omega_phys|| <= Om_max, i.e. ||omega_nd|| <= Om_max*T = slew_max_nd,
    # is the angle limit
    #
    #     Delta_theta_k <= theta_max := slew_max_nd * s * dtau
    #
    # and for UNIT vectors the angle is exactly recoverable from the chord,
    #
    #     ||d_{k+1} - d_k||^2 = 2 - 2*cos(Delta_theta) = 4*sin^2(Delta_theta/2)
    #
    # Since 2*sin(x/2) is monotone on [0, pi], the angle bound is EQUIVALENT to
    #
    #     ||d_{k+1} - d_k|| <= 2*sin(theta_max/2)
    #
    # WHY THIS IS THE RIGHT FORM. It reads only the STATES d_k, d_{k+1}. No omega, so the gauge
    # cannot touch it; no dref, so no directional reference. Roll about the thrust axis cannot
    # inflate it, because roll does not move d and d is all it measures.
    #
    # THE s DEPENDENCE. theta_max contains the free final time, so the right-hand side is
    # g(s) = 2*sin(slew_max_nd*dtau*s/2), which is CONCAVE in s over the usable range (sin is
    # concave on [0, pi]). A norm bounded by a concave function is still a convex set, but CVXPY
    # has no concave sin atom, so g is linearised at sref -- its TANGENT, which is affine and
    # therefore an SOC:
    #
    #     ||d_{k+1} - d_k|| <= 2*sin(a*sref/2) + a*cos(a*sref/2)*(s - sref),   a = slew_max_nd*dtau
    #
    # This is exact at s = sref, i.e. at convergence, and second order in (s - sref) elsewhere.
    # The reference is a SCALAR flight time, not a direction -- a far milder dependence than the
    # dref the gauge needs, and one that converges early.
    #
    # WHY NOT THE SMALL-ANGLE FORM ||d_{k+1}-d_k|| <= theta_max. Because 2*sin(x/2) <= x, it
    # permits more turn than asked: 2*arcsin(theta_max/2) instead of theta_max, an excess of
    # ~theta_max^2/24. At 2 deg/s and tf 2038 s that is +7.3% at K=60 and +26.1% at K=40. Worse,
    # two unit vectors are at most 2 apart, so once theta_max >= 2 rad (114.6 deg) the constraint
    # is VACUOUS -- at 2 deg/s that is every K <= 36, which the closed loop's adaptive K reaches.
    # It would enforce nothing, silently.
    if slew_max_nd is not None and slew_form == "chord":
        a_coef = float(slew_max_nd * dtau)                 # theta_max = a_coef * s
        th_ref = float(np.clip(a_coef * sref, 0.0, np.pi * 0.999))   # keep the tangent slope > 0
        rhs = 2.0 * np.sin(0.5 * th_ref) + a_coef * np.cos(0.5 * th_ref) * (s - sref)
        for k in range(K - 1):
            cons += [cp.norm(X[7:10, k + 1] - X[7:10, k]) <= rhs]
        # NUMERICAL BACKSTOP, not the limit. With the turn bounded on the states, omega is
        # constrained only through the dynamics rows -- which virtual control can slacken
        # mid-iteration, letting omega run away before nu is driven out. 2x the request is loose
        # enough never to be the binding constraint at convergence and tight enough to keep the
        # subproblem well scaled.
        for k in range(K - 1):
            cons += [cp.norm(U[0:3, k]) <= 2.0 * slew_max_nd]

    # -- the thrust the engine is ALREADY producing ----------------------------------------------
    # In closed loop this is measured, so each re-plan starts from the actual thrust instead of
    # commanding a step the actuator then has to chase (a_init: direction AND magnitude). Left free
    # (None) the solver chooses the handover thrust, which is information for the upstream braking
    # phase rather than a constraint. d_init is the third case: the attitude is imposed, the
    # throttle is not.
    if a_init is not None and d_init is not None:
        raise ValueError("pass a_init or d_init, not both: they impose the same direction twice")
    if a_init is not None:
        # a_init is the thrust the engine is measured to be producing: split it the same way.
        _a = np.asarray(a_init, float); _n = float(np.linalg.norm(_a))
        cons += [X[7:10, 0] == (_a / _n if _n > 1e-12 else n_hat)]
        cons += [U[3, 0] == _n]
    elif d_init is not None:
        # DIRECTION ONLY. The braking phase delivers an ATTITUDE; the throttle at the first node is
        # the descent guidance's own decision. Used for PINN label generation, where pinning the
        # magnitude as well would make the dataset inconsistent: a handover node would carry a
        # forced Gamma while an early sub-trajectory node at the same state carries the freely
        # chosen one, and since the solution is bang-bang those differ by the whole throttle band.
        #
        # It also leaves the sub-trajectory labels exact. Thrust magnitude has no inter-temporal
        # coupling here -- nothing rate-limits it, it may jump between segments -- so the Gamma the
        # parent chose at node j IS the optimum of the problem restarted there, and not pinning it
        # changes no label. The DIRECTION is different: d is a state with a slew constraint, which
        # is why that one has to be imposed.
        _d = np.asarray(d_init, float); _n = float(np.linalg.norm(_d))
        cons += [X[7:10, 0] == (_d / _n if _n > 1e-12 else n_hat)]

    # -- trust region, penalised rather than imposed (PTR) --------------------------------------
    # The radius is a VARIABLE priced in the cost, not a hard bound adapted by a ratio test. The
    # previous solver adapted a hard radius from rho = dJ/dL, and dL was measured constant to three
    # decimals while the radius fell 16x, so the controller was steering on a signal that carried
    # no step-size information. Here the optimiser chooses its own step against a fixed price, and
    # the price is set from the measured linearisation error (see ptr.py).
    # THE omega BLOCK IS SCALED. Everything in this norm must be O(1) or the cap means a
    # different thing for different variables. The non-dimensionalisation was built to make the
    # STATES O(1) and it does -- r 1.55, v 0.44, d 1.00, Gamma <= 1.01 -- but omega was never
    # scaled: omega_nd = omega_phys*T, which is 10.372 PER deg/s at T = 594 s, and peaks at
    # 9.490 in the converged solution. So omega sat ~10x the largest state and ~40x Gamma inside
    # a shared cap of R = 0.35.
    #
    # What that cost, measured: R = 0.35 permitted the slew rate to move 0.0337 deg/s per
    # iteration, so reaching the solution's 0.92 deg/s from the guess's 0.16 needed >= 22
    # iterations of omega crawl alone, out of 47 -- while starving every other variable of trust
    # budget. Tiny steps then made the per-iteration objective change fall under the 1e-4 kg
    # absolute branch of obj_ok, so the solve reported convergence ~4 g short of the optimum.
    # That is why a SLACK slew limit changed the answer, and why tightening one could appear to
    # IMPROVE the fuel: smaller steps, earlier stop, different distance from the same optimum.
    #
    # omega is divided by radians(SLEW_MAX_DEG_S)*T, so one unit of trust means "one full slew
    # limit" and lands alongside Gamma and the states. This reweights the NORM only -- the
    # dynamics, the slew bound and the reported controls are untouched.
    w_om = 1.0 / max(np.radians(C.SLEW_MAX_DEG_S) * scales.T, 1e-12)
    Wu = np.array([w_om, w_om, w_om, 1.0])
    for k in range(K):
        cons += [cp.norm(cp.hstack([X[:, k] - Xref[:, k],
                                    cp.multiply(Wu, U[:, k] - Uref[:, k])])) <= eta[k]]
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
                                                    miss=miss)


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

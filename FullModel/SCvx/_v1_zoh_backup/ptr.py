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


def initial_guess(dyn, x_init, rf_nd, n_hat, scales, K, s_lo, s_hi,
                  term_point_deg=C.TERMINAL_POINT_DEG,
                  term_point_nodes=C.TERMINAL_POINT_NODES):
    """A DYNAMICALLY CONSISTENT straight-line guess.

    Position runs straight from the start to the site. Velocity is then the finite difference of
    that position, so the kinematic half of the dynamics (rdot = v) holds exactly at every node
    rather than being left for the virtual control to repair. Thrust is whatever the remaining half
    (vdot = a + Coriolis + centrifugal + g) needs, clipped into the achievable band, and the mass is
    integrated from it.

    The earlier guess interpolated velocity from v0 to zero independently of the position, which for
    a vehicle handed over at rest means v == 0 while r moves 12 km -- inconsistent at every node.
    That was measured leaving ||nu||_1 stuck near 0.7 with the virtual-control penalty dominating
    the whole objective, so the iteration only ever chased feasibility and never optimised.

    s is seeded at twice the minimum-time bound. That bound assumes every accelerating effect points
    helpfully, so the true optimum always lies above it, and a factor of two needs no knowledge of
    the answer. The solver must be insensitive to this seed; the seed-ladder test is what checks it.
    """
    r0, v0, z0 = x_init[0:3], x_init[3:6], x_init[6]
    s0 = float(np.clip(2.0 * s_lo, s_lo, s_hi))
    tau = np.linspace(0.0, 1.0, K); dtau = 1.0 / (K - 1)
    X = np.zeros((NX, K)); U = np.zeros((NU, K))

    # CUBIC HERMITE in position, with the boundary conditions built in:
    #   r(0)=r0, r(1)=rf, v(0)=v0, v(1)=0.
    # Velocity is then dr/dtau / s, so rdot = v holds exactly at every node AND the guess already
    # satisfies the initial condition and the terminal conditions it will be held to. A guess that
    # violates a HARD constraint by more than the trust radius makes the very first subproblem
    # infeasible -- measured with a straight-line guess whose node-0 velocity was 0.42 non-dim
    # against an initial condition of 0 and a radius of 0.10.
    t2, t3 = tau ** 2, tau ** 3
    h00 = 2 * t3 - 3 * t2 + 1; h10 = t3 - 2 * t2 + tau
    h01 = -2 * t3 + 3 * t2;    h11 = t3 - t2
    d00 = 6 * t2 - 6 * tau;    d10 = 3 * t2 - 4 * tau + 1
    d01 = -6 * t2 + 6 * tau;   d11 = 3 * t2 - 2 * tau
    m0 = s0 * v0                                     # dr/dtau at tau = 0
    for i in range(3):
        X[i] = h00 * r0[i] + h10 * m0[i] + h01 * rf_nd[i] + h11 * 0.0
        X[3 + i] = (d00 * r0[i] + d10 * m0[i] + d01 * rf_nd[i] + d11 * 0.0) / s0
    X[3:6, 0] = v0; X[3:6, K - 1] = 0.0
    X[6, 0] = z0

    rho1, rho2 = C.TMIN / scales.A, C.TMAX / scales.A
    cos_pt = np.cos(np.radians(C.POINTING_DEG))
    for k in range(K):
        if k < K - 1:
            vdot = (X[3:6, k + 1] - X[3:6, k]) / (s0 * dtau)
        else:
            vdot = np.zeros(3)
        a = vdot - dyn.Cco @ X[3:6, k] - dyn.Ccf @ X[0:3, k] - dyn.g_nd(X[0:3, k])
        an = np.linalg.norm(a)
        d = a / an if an > 1e-12 else n_hat.copy()
        # project into the pointing cone, so the guess does not violate a hard constraint either
        c = float(d @ n_hat)
        if c < cos_pt:
            perp = d - c * n_hat
            pn = np.linalg.norm(perp)
            d = (cos_pt * n_hat + np.sqrt(max(1.0 - cos_pt ** 2, 0.0)) * (perp / pn)) if pn > 1e-12 \
                else n_hat.copy()
        lo, hi = rho1 * np.exp(-X[6, k]), rho2 * np.exp(-X[6, k])
        g = float(np.clip(an, lo, hi))
        U[0:3, k] = d * g
        U[3, k] = g
        if k < K - 1:
            X[6, k + 1] = X[6, k] - dyn.alpha * g * s0 * dtau
    # the K-th control column is a tied duplicate of the last real hold interval
    # (see subproblem.build); the guess must already satisfy that or the first
    # subproblem is infeasible against the trust radius.
    # the terminal hold must already point where the terminal constraint demands, or the
    # first subproblem is infeasible against the trust radius
    if term_point_deg is not None and term_point_nodes > 0:
        for k in range(max(0, K - 1 - int(term_point_nodes)), K - 1):
            U[0:3, k] = n_hat * U[3, k]
    U[:, K - 1] = U[:, K - 2]
    return X, U, s0


def run(dyn, scales, x_init, rf_nd, n_hat, K=30, mode="fuel", D_nd=None,
        warm=None, iters=25, w_vc=1e4, w_tr=0.0, w_ts=0.0,
        R0=0.35, Rs0=0.50, R_bounds=(1e-4, 1.0),
        tol_obj=1e-4, tol_abs=None, tol_defect_m=5e-3, tol_defect_v=5e-4, tol_vc=1e-6,
        d_hi_m=2.0, d_lo_m=0.05, w_growth=2.0, w_bounds=(1e-4, 1e3),
        ratio_test=False, rho0=0.0, rho1=0.1, rho2=0.5, rho_shrink=1.5, rho_grow=2.0,
        tr_conv_frac=0.9, tr_conv_R=1e-5,
        n_sub_disc=10, n_sub_prop=20, verbose=True, label="", fix_tf_s=None,
        term_point_deg=C.TERMINAL_POINT_DEG, term_point_nodes=C.TERMINAL_POINT_NODES):
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
      * the virtual control is spent            (||nu||_1 < tol_vc)
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

    if warm is not None:
        Xr, Ur, sr = warm
        Xr, Ur, sr = np.array(Xr, float), np.array(Ur, float), float(sr)
        Xr[:, 0] = x_init                       # anchor the guess to the IC it is being reused on
        if term_point_deg is not None and term_point_nodes > 0 and term_point_deg <= 0:
            for k in range(max(0, K - 1 - int(term_point_nodes)), K - 1):
                Ur[0:3, k] = n_hat * Ur[3, k]
        Ur[:, -1] = Ur[:, -2]                   # and to the tied terminal control
    else:
        Xr, Ur, sr = initial_guess(dyn, x_init, rf_nd, n_hat, scales, K, s_lo, s_hi,
                                   term_point_deg, term_point_nodes)

    vtouch_nd = C.VTOUCH / scales.v
    Pn = np.eye(3) - np.outer(n_hat, n_hat)
    # ---- the SCvx merit function, for the ratio test ------------------------------------------
    # Same cost and same penalty as subproblem.build's objective, but evaluated on the NONLINEAR
    # dynamics: the subproblem pays w_vc * sum|nu| for the defects its LINEAR model admits, and
    # this pays w_vc * (true L1 defect) for the ones the real dynamics produce. Identical units, so
    # predicted and actual reductions are comparable and rho means something.
    def _merit(X_, U_, s_, l1_):
        cost = (-X_[6, -1] / scales.alpha_nd if mode == "fuel"
                else float(np.linalg.norm(Pn @ (X_[0:3, -1] - rf_nd))))
        return float(cost + w_vc * l1_)

    hist = []; t0 = time.perf_counter(); Jprev = np.inf; status = "maxiter"
    d_prev = None; R, Rs = R0, Rs0; merit_ref = None; n_rej = 0; n_floor_rej = 0
    ref_feas = False
    if verbose:
        print(f"  {'it':>3}{'solver':>9}{'J':>12}{'dJ/J':>10}{'defect[m]':>11}{'defect[m/s]':>11}"
              f"{'|nu|_1':>10}{'tf[s]':>9}{'R':>9}")
    for it in range(iters):
        Ad, Bd, Sd, zd, _ = discretize(dyn, Xr, Ur, sr, n_sub=n_sub_disc)
        prob, V = SP.build(Ad, Bd, Sd, zd, Xr, Ur, sr, x_init, rf_nd, n_hat, scales,
                           mode=mode, D_nd=D_nd, w_vc=w_vc, w_tr=w_tr, w_ts=w_ts,
                           s_lo=s_lo, s_hi=s_hi, vtouch_nd=vtouch_nd, R=R, Rs=Rs,
                           term_point_deg=term_point_deg,
                           term_point_nodes=term_point_nodes)
        name, st = SP.solve(prob)
        if name is None:
            status = f"subproblem_{st}"
            break
        Xn, Un, sn = np.asarray(V["X"].value), np.asarray(V["U"].value), float(V["s"].value)
        vc = float(np.abs(np.asarray(V["nu"].value)).sum())
        eta_v = float(np.max(np.asarray(V["eta"].value)))
        etas_v = float(np.asarray(V["eta_s"].value))
        dpos_nd, dvel_nd, _, _, l1_c = defect(dyn, Xn, Un, sn, n_sub=n_sub_prop)
        dmax_m = dpos_nd * scales.L
        dvel_ms = dvel_nd * scales.v
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
        dJ = abs(Jphys - Jprev)
        rel = dJ / max(1e-12, abs(Jphys))
        ta = tol_abs if tol_abs is not None else (1e-2 if mode == "error" else 1e-4)
        obj_ok = (dJ < ta) or (rel < tol_obj)
        # ---- THE RATIO TEST ------------------------------------------------------------------
        # rho = (what the NONLINEAR dynamics actually delivered) / (what the LINEAR model promised).
        # A step that over-promises is REJECTED: the reference does not move and the radius shrinks,
        # so the next iteration re-solves the same linearisation point in a smaller region. That
        # rejection is the whole reason a trust-region method converges, and it is what the
        # defect-based rule below cannot do -- it accepts every step and only tunes the radius
        # afterwards, which is how a period-2 cycle survives: a good step earns a bigger radius,
        # the bigger radius produces a bad step, the bad step shrinks it back, forever. Measured on
        # mass draw 304: J alternating -713.76038 / -713.76150 and R alternating 1.0 / 0.5 for 200
        # iterations, with every physical criterion already satisfied on both branches.
        rho = np.nan; accepted = True
        if ratio_test:
            if merit_ref is None:
                dp_r, dv_r, _, _, l1_r = defect(dyn, Xr, Ur, sr, n_sub=n_sub_prop)
                merit_ref = _merit(Xr, Ur, sr, l1_r)
                ref_feas = (dp_r * scales.L < tol_defect_m) and (dv_r * scales.v < tol_defect_v)
            merit_c = _merit(Xn, Un, sn, l1_c)
            pred = merit_ref - J            # J is the subproblem optimum: the LINEAR model's merit
            act = merit_ref - merit_c       # what the NONLINEAR dynamics actually delivered
            # A non-positive predicted reduction means the linear model did not promise an
            # improvement; then the only sensible question is whether the truth improved.
            rho = (act / pred) if pred > 1e-14 else (1.0 if act > 0.0 else -1.0)
            accepted = rho >= rho0
        hist.append(dict(it=it, solver=name, J=J, jphys=Jphys, rel=rel, defect_m=dmax_m, rho=rho,
                         accepted=accepted, n_rej=n_rej, eta=eta_v, eta_s=etas_v, R=R, Rs=Rs,
                         tf_ref=scales.t_dim(sr), defect_v=dvel_ms, vc=vc,
                         tf_s=scales.t_dim(sn), w_tr=w_tr))
        if verbose:
            print(f"  {it:>3}{name:>9}{J:>12.5f}{rel:>10.2e}{dmax_m:>11.3e}{dvel_ms:>11.3e}"
                  f"{vc:>10.2e}{scales.t_dim(sn):>9.1f}{R:>9.1e}"
                  + (f"{rho:>9.2f}{'' if accepted else ' REJ'}" if ratio_test else ""))
        # REJECT: keep the reference, shrink the radius, and do NOT advance Jprev or d_prev --
        # the next iteration must re-solve the SAME linearisation point inside a smaller region.
        #
        # THE REFERENCE UPDATE MUST COME AFTER THIS BLOCK, NOT BEFORE IT. With `Xr, Ur, sr = Xn,
        # Un, sn` placed above, rejection shrank the radius but the iterate advanced anyway, so
        # the ratio test was not a trust-region method at all -- it accepted every step and merely
        # tuned the radius, which is the behaviour it was brought in to replace. It also left
        # merit_ref (refreshed only on acceptance) describing a trajectory the solver had already
        # left, so both pred and act were measured against a stale baseline: that is why rho
        # converged to -1.8 instead of +1 as R -> 0.
        if not accepted:
            n_rej += 1
            # A REJECTED STEP AT THE RADIUS FLOOR IS AN INFINITE LOOP, NOT SLOW PROGRESS. Rejecting
            # keeps the reference, and at R = R_bounds[0] the shrink is a no-op, so the next
            # iteration poses the IDENTICAL subproblem and gets the identical answer. Stop and say
            # so, which lets a caller tell a stall from a case the physics refuses.
            # AN ITERATE THAT HAS ARRIVED REJECTS EVERYTHING, AND THAT IS NOT A FAILURE.
            # At the optimum the reference is a local minimum of the merit, so every candidate step
            # comes back with a worse true defect and rho < 0 -- correctly. The convergence test
            # below sits after this `continue`, so such an iterate could never be TESTED for having
            # arrived: measured from a reference with defect 9.85e-05 m, 58 of 61 steps rejected,
            # the radius collapsed to the floor, and the run reported `stalled` with the answer
            # already in hand and 1.0 g from the converged value.
            #
            # The standard trust-region termination covers exactly this: the radius has shrunk to
            # nothing and no improving step exists in that neighbourhood, with the reference itself
            # satisfying the dynamics -- which is what a local optimum is.
            if ref_feas and R <= tr_conv_R:
                status = "converged"; break
            at_floor = (R <= R_bounds[0] * (1.0 + 1e-9)) and (Rs <= R_bounds[0] * (1.0 + 1e-9))
            R = max(R / rho_shrink, R_bounds[0]); Rs = max(Rs / rho_shrink, R_bounds[0])
            n_floor_rej = n_floor_rej + 1 if at_floor else 0
            if n_floor_rej >= 2:          # 2, not 1, so a one-off numerical blip cannot end a run
                status = "stalled"; break
            continue
        # ACCEPTED: the iterate advances and the baseline moves with it.
        Xr, Ur, sr = Xn, Un, sn
        if ratio_test:
            merit_ref = merit_c
            ref_feas = (dmax_m < tol_defect_m) and (dvel_ms < tol_defect_v)
        # A STEP PINNED AGAINST THE TRUST CAP IS NOT CONVERGENCE. eta is the per-node step and R its
        # hard cap, so eta/R near 1 means the optimiser wanted to move further and was stopped by
        # the cap: the iterate has not settled, the leash is just short. With the ratio test on, a
        # collapsing radius would otherwise report success -- shrinking steps push dJ/J under
        # tolerance while the fuel is still descending, and the defect and virtual control look fine
        # precisely BECAUSE the steps are tiny. Both regions are checked: eta_s caps how far the
        # FLIGHT TIME may move, and tf is the variable that sets the fuel on this problem.
        tr_slack = (not ratio_test) or ((eta_v <= tr_conv_frac * R)
                                        and (etas_v <= tr_conv_frac * Rs))
        if obj_ok and dmax_m < tol_defect_m and dvel_ms < tol_defect_v and vc < tol_vc and tr_slack:
            status = "converged"; break
        Jprev = Jphys
        # RHO-BAND RADIUS UPDATE, the standard SCvx schedule, used INSTEAD of the defect rule when
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
        # once the defect is far inside tolerance that ratio is meaningless -- two numbers six
        # orders below tol_defect_m can give ratios of 1.46 and 0.68 and trip
        # shrink-then-grow-then-shrink forever. That is the period-2 cycle this solver was measured
        # sitting in. The floor is 0.01*tol_defect_m, not 0.1: at 0.1 it also froze cycles whose
        # defect was 2-3e-4 m and whose OBJECTIVE was still oscillating, and freezing is exactly
        # wrong there, because the cure for an oscillating objective is to SHRINK the radius, which
        # a frozen rule can no longer do.
        adapt_floor = 0.01 * tol_defect_m
        if d_prev is not None and max(dmax_m, d_prev) > adapt_floor:
            if dmax_m > d_prev * 1.2:                     # the step made the model worse -> shrink
                w_tr = min(w_tr * w_growth, w_bounds[1]); w_ts = min(w_ts * w_growth, w_bounds[1])
                R = max(R / w_growth, R_bounds[0]); Rs = max(Rs / w_growth, R_bounds[0])
            elif dmax_m < d_prev * 0.8 and dmax_m < d_hi_m:
                w_tr = max(w_tr / w_growth, w_bounds[0]); w_ts = max(w_ts / w_growth, w_bounds[0])
                R = min(R * w_growth, R_bounds[1]); Rs = min(Rs * w_growth, R_bounds[1])
        d_prev = dmax_m

    # A subproblem with virtual control is always solvable, so genuine infeasibility shows up as a
    # virtual control that never goes away rather than as a solver failure. Say so explicitly
    # instead of reporting "maxiter" with a meaningless objective.
    if status != "converged" and hist and hist[-1]["vc"] > 1e-3:
        status = "infeasible"
    wall = time.perf_counter() - t0
    P = np.eye(3) - np.outer(n_hat, n_hat)
    miss_m = float(np.linalg.norm(P @ (Xr[0:3, -1] - rf_nd))) * scales.L
    fuel = float(np.exp(Xr[6, 0]) - np.exp(Xr[6, -1]))
    dpos_nd, dvel_nd, _, _, _ = defect(dyn, Xr, Ur, sr, n_sub=n_sub_prop)
    return Result(X=Xr, U=Ur, s=sr, status=status, iters=len(hist), history=hist,
                  miss_m=miss_m, fuel_kg=fuel, tf_s=scales.t_dim(sr),
                  defect_m=dpos_nd * scales.L,
                  vc=hist[-1]["vc"] if hist else np.nan, wall_s=wall,
                  D_nd=float(np.linalg.norm(P @ (Xr[0:3, -1] - rf_nd))))

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
                  term_point_nodes=C.TERMINAL_POINT_NODES, slew_max_nd=None):
    """A dynamically consistent guess for the AUGMENTED state x = [r, v, z, a].

    Position is a cubic Hermite carrying the boundary conditions (r0, v0 at the start, rf and rest
    at the end), so r-dot = v holds exactly at every node and the guess already satisfies the hard
    constraints it will be held to -- a guess that violates one by more than the trust radius makes
    the very first subproblem infeasible.

    The thrust state a is then whatever the remaining half of the dynamics needs, clipped into the
    achievable band and into the pointing cone, and the rate control w is its finite difference, so
    a-dot = w holds exactly as well.

    s is seeded at twice the minimum-time bound: that bound assumes every accelerating effect points
    helpfully, so the true optimum always lies above it, and a factor of two needs no knowledge of
    the answer. The solver must be insensitive to this seed; the seed-ladder test checks it.
    """
    r0, v0, z0 = x_init[0:3], x_init[3:6], x_init[6]
    s0 = float(np.clip(2.0 * s_lo, s_lo, s_hi))
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
    cos_pt = np.cos(np.radians(C.POINTING_DEG))
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
        U[0:3, k] = d * g                                 # THRUST IS THE CONTROL (ZOH)
        U[3, k] = g
        if k < K - 1:
            X[6, k + 1] = X[6, k] - dyn.alpha * g * s0 * dtau
    if term_point_deg is not None and term_point_nodes > 0 and term_point_deg <= 0:
        for k in range(max(0, K - int(term_point_nodes)), K):
            U[0:3, k] = n_hat * U[3, k]
    # Walk the guess into the slew cone: with ZOH the limit is on the STEP between consecutive
    # holds, so clip each step rather than a rate. The last control drives nothing (K-1 intervals).
    if slew_max_nd is not None:
        for k in range(K - 1):
            step = U[0:3, k + 1] - U[0:3, k]
            aref = U[0:3, k]; na = float(np.linalg.norm(aref))
            if na < 1e-12:
                continue
            dhat = aref / na
            spar = (step @ dhat) * dhat; sperp = step - spar
            lim = slew_max_nd * s0 * dtau * U[3, k]
            npp = float(np.linalg.norm(sperp))
            if npp > lim > 0:
                U[0:3, k + 1] = U[0:3, k] + spar + sperp * (lim / npp)
    U[0:3, K - 1] = U[0:3, K - 2]
    return X, U, s0


def run(dyn, scales, x_init, rf_nd, n_hat, K=30, mode="fuel", D_nd=None,
        warm=None, iters=25, w_vc=1e4, w_tr=0.0, w_ts=0.0,
        R0=0.35, Rs0=0.50, R_bounds=(1e-4, 1.0),
        tol_obj=1e-4, tol_abs=None, tol_defect_m=5e-3, tol_defect_v=5e-4, tol_vc_m=1e-3,
        d_hi_m=2.0, d_lo_m=0.05, w_growth=2.0, w_anneal_it=2.0, w_bounds=(1e-4, 1e3),
        n_sub_disc=10, n_sub_prop=20, verbose=True, label="", fix_tf_s=None,
        term_point_deg=C.TERMINAL_POINT_DEG, term_point_nodes=C.TERMINAL_POINT_NODES,
        slew_max_deg_s=C.SLEW_MAX_DEG_S, a_init=None,
        gs_deg=None, point_deg=None):
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
      * the virtual control is spent            (||nu||_1 * L < tol_vc_m, in METRES)

    THE VIRTUAL-CONTROL TOLERANCE IS PHYSICAL, and that is not cosmetic. nu is non-dimensional, so
    testing it against a fixed number means testing against a bar that moves with the scale. With
    per-cycle rescaling L runs from 12 000 m at handover to 59 m in the endgame, so the old
    non-dimensional tol_vc = 1e-6 was a 1.2e-2 m bar at the first cycle and a 5.9e-5 m bar at the
    last -- a factor of 200 tighter -- while the solver's ACHIEVED virtual control sat at about
    5e-6 m throughout. Cycles that were identical in quality were graded differently depending
    only on how close the vehicle had come, and the late ones were being held to a standard they
    were never going to meet. Measured across a full descent: vc*L from 1.4e-6 to 1.6e-5 m on every
    converged cycle, against a bar that fell from 1.2e-2 to 5.9e-5 m.

    The default is 1e-3 m: five times under tol_defect_m, which is the real quality check. The two
    measure different things -- vc is how much the LINEARISED subproblem had to cheat, the defect
    is what the NONLINEAR propagation actually does -- and in the endgame the trajectory was good
    to 4e-7 m while the linearised solve wanted 1e-4 m of slack, so slack at that level is not
    hurting anything. A tighter 1e-4 m bar was measured sending two endgame cycles to maxiter on
    the virtual control alone, with the defect six orders inside tolerance.
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
        Xr[:, 0] = x_init                       # anchor the guess to the IC it is being reused on
        if term_point_deg is not None and term_point_nodes > 0 and term_point_deg <= 0:
            for k in range(max(0, K - int(term_point_nodes)), K):
                Ur[0:3, k] = n_hat * Ur[3, k]
        Ur[0:3, -1] = Ur[0:3, -2]               # and to the tied terminal rate
    else:
        Xr, Ur, sr = initial_guess(dyn, x_init, rf_nd, n_hat, scales, K, s_lo, s_hi,
                                   term_point_deg, term_point_nodes, slew_nd)

    hist = []; t0 = time.perf_counter(); Jprev = np.inf; status = "maxiter"
    d_prev = None; R, Rs = R0, Rs0
    if verbose:
        print(f"  {'it':>3}{'solver':>9}{'J':>12}{'dJ/J':>10}{'defect[m]':>11}{'defect[m/s]':>11}"
              f"{'|nu|_1':>10}{'tf[s]':>9}{'R':>9}{'eta/R':>8}{'etas/Rs':>8}")
    for it in range(iters):
        Ad, Bd, Sd, zd, _ = discretize(dyn, Xr, Ur, sr, n_sub=n_sub_disc)
        prob, V = SP.build(Ad, Bd, Sd, zd, Xr, Ur, sr, x_init, rf_nd, n_hat, scales,
                           mode=mode, D_nd=D_nd, w_vc=w_vc, w_tr=w_tr, w_ts=w_ts,
                           s_lo=s_lo, s_hi=s_hi, vtouch_nd=vtouch_nd, R=R, Rs=Rs,
                           term_point_deg=term_point_deg,
                           term_point_nodes=term_point_nodes,
                           slew_max_nd=slew_nd, a_init=a_init,
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
        vc = float(np.abs(np.asarray(V["nu"].value)).sum())
        dpos_nd, dvel_nd, _, _ = defect(dyn, Xn, Un, sn, n_sub=n_sub_prop)
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
        hist.append(dict(it=it, solver=name, J=J, rel=rel, defect_m=dmax_m,
                         defect_v=dvel_ms, vc=vc, tf_s=scales.t_dim(sn), w_tr=w_tr,
                         eta=eta_v, eta_s=etas_v, R=R, Rs=Rs, vc_m=vc * scales.L))
        if verbose:
            print(f"  {it:>3}{name:>9}{J:>12.5f}{rel:>10.2e}{dmax_m:>11.3e}{dvel_ms:>11.3e}"
                  f"{vc:>10.2e}{scales.t_dim(sn):>9.1f}{R:>9.1e}"
                  f"{eta_v/max(R,1e-30):>8.1%}{etas_v/max(Rs,1e-30):>8.1%}")
        Xr, Ur, sr = Xn, Un, sn
        vc_m = vc * scales.L                      # the virtual control in METRES of position
        if obj_ok and dmax_m < tol_defect_m and dvel_ms < tol_defect_v and vc_m < tol_vc_m:
            status = "converged"; break
        Jprev = Jphys
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

    # A subproblem with virtual control is always solvable, so genuine infeasibility shows up as a
    # virtual control that never goes away rather than as a solver failure. Say so explicitly
    # instead of reporting "maxiter" with a meaningless objective.
    if status != "converged" and hist and hist[-1]["vc"] * scales.L > 1e-1:
        status = "infeasible"
    wall = time.perf_counter() - t0
    P = np.eye(3) - np.outer(n_hat, n_hat)
    miss_m = float(np.linalg.norm(P @ (Xr[0:3, -1] - rf_nd))) * scales.L
    fuel = float(np.exp(Xr[6, 0]) - np.exp(Xr[6, -1]))
    dpos_nd, dvel_nd, _, _ = defect(dyn, Xr, Ur, sr, n_sub=n_sub_prop)
    return Result(X=Xr, U=Ur, s=sr, status=status, iters=len(hist), history=hist,
                  miss_m=miss_m, fuel_kg=fuel, tf_s=scales.t_dim(sr),
                  defect_m=dpos_nd * scales.L,
                  vc=hist[-1]["vc"] if hist else np.nan, wall_s=wall,
                  D_nd=float(np.linalg.norm(P @ (Xr[0:3, -1] - rf_nd))))

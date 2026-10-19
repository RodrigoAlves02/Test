#!/usr/bin/env python
"""Guidance: estimated state in, commanded thrust out.

This is the block the PINN will eventually occupy. For now it is SCvx, so that the rest of the full
model -- navigation, actuator, truth dynamics -- can be closed and verified against a guidance law
that is known to be correct, before the network is asked to imitate it.

    x_est (from navigation)  ->  [ SCvx ]  ->  T_cmd  ->  [ actuator model ]  ->  truth dynamics

Used as a receding horizon: called once per guidance cycle, it re-solves from wherever the vehicle
is believed to be, applies only the first control, and throws the rest of the plan away. That is
what makes the loop correct its own errors rather than integrate them.

THREE THINGS THIS BLOCK OWNS, and they are easy to get wrong if they leak elsewhere.

FRAME. The truth dynamics propagate in the INERTIAL frame, but SCvx is posed in the body-fixed
ROTATING frame, because that is where the landing site sits still and where the glideslope cone is
defined. So the estimated state is rotated in, the problem is solved, and the commanded thrust is
rotated back out. The conversion needs absolute time, since the two frames are only aligned at t=0.

UNITS. SCvx's control is an ACCELERATION, u = T/m, carried alongside ln(m). The actuator and the
truth dynamics want a force in newtons. The conversion is T = m*u at the mass the optimiser used,
and that mass is a state of the solution, not a constant.

WHICH CONTROL. The first node, u(tau=0), held until the next cycle. A real vehicle holds its
commanded thrust between guidance updates; sampling the plan more finely would be executing a
profile the guidance never actually commits to.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import contextlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.join(HERE, "..", "..", "scvx_landing")
sys.path.insert(0, SCVX)

@contextlib.contextmanager
def _in_scvx_dir():
    """Run inside scvx_landing.

    gen_dataset opens mc_handover_poly.json by a bare relative path, so Ctx construction and the
    solve both assume that directory is the working one. Handling it here keeps the assumption
    inside the block that depends on it, instead of making every caller chdir before using guidance.
    """
    prev = os.getcwd()
    os.chdir(os.path.abspath(SCVX))
    try:
        yield
    finally:
        os.chdir(prev)


EROS_ROT_PERIOD = 5.270 * 3600.0
WZ = 2.0 * np.pi / EROS_ROT_PERIOD
OMEGA = np.array([0.0, 0.0, WZ])


# --- frame conversion, shared convention with the truth dynamics -----------------------------------
def _Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def inertial_to_body(r_I, v_I, t):
    R = _Rz(-WZ * t)
    r_B = R @ r_I
    return r_B, R @ v_I - np.cross(OMEGA, r_B)


def body_to_inertial_vec(a_B, t):
    """A free vector -- a thrust, an acceleration -- carries no frame-rate term."""
    return _Rz(WZ * t) @ a_B


@dataclass
class Plan:
    """One SCvx solution, kept so the caller can inspect or plot what guidance intended."""
    t_query: float
    X: np.ndarray                 # (7, K) states in the ROTATING frame, x[6] = ln m
    U: np.ndarray                 # (4, K) controls, u[0:3] acceleration, u[3] the LCvx slack
    sigma: float                  # time of flight [s]
    solve_s: float
    iters: int = 0
    ok: bool = True
    error: str = ""
    reason: str = ""
    exit_via: str = ""

    @property
    def t_go(self):
        return self.sigma


# --- non-dimensional solver tuning ----------------------------------------------------------------
# Every PTR constant in tune.py carries units, so each one silently encodes the scale of the descent
# it was tuned on. eta0 = 1000 m is 8.5% of the 11 771 m nominal descent but 56% of the 1 775 m
# remaining in its last quarter -- 6.6 times oversized in relative terms. The step is then far too
# large for the linearisation to hold across, the defect explodes, the step is rejected, and the
# region has to halve its way back down, paying a full subproblem solve on each rejection.
#
# The fix is to carry the constants as DIMENSIONLESS numbers and restore them against the current
# problem's own scales. The values below are the tuned dimensional constants divided by the scales
# of the descent they were tuned on, so at full scale this reproduces the existing behaviour exactly
# and at any other scale it does the equivalent thing rather than the same-numbered thing.
# Retuned 2026-09-21 in OtherTests/solver_fix. The previous values were the shipped dimensional
# constants (1000 m / 500 s / 0.1) divided by their scales; those were measured to leave the solver
# 3.36% off the best flown fuel in the worst case, because rho here is a DEFECT-CONTRACTION ratio
# (rho ~ 1 - ||d_new||/||d_ref||, since dL is dominated by Wvc*||d_ref||_1 and the subproblem can
# always zero its virtual control inside the trust region). Demanding rho >= 0.90 to grow therefore
# demanded a tenfold defect cut per step, which essentially never happens, while r1t = 0.36 shrank
# on every other accepted step -- so the region ratcheted down and sigma's travel was choked off.
# See solver_fix/solver_fix.txt. Worst-case flown-fuel gap with these: 0.13%.
ETA0_ND = 0.025487        # state trust region, in units of |r0 - rf|          (300 m / 11771 m)
ETA_S0_ND = 0.222136      # time-dilation trust region, in units of t_go       (400 s / 1800.7 s)
ETA_U0_ND = 0.029975      # control trust region, in units of Tmax/m       (1e-3 / 0.033361)
TOL_SIG_ND = 5.5534e-04   # flight-time convergence tolerance, in units of t_go        (1 s)
TOL_TRAJ_ND = 8.495e-05   # trajectory convergence tolerance, in units of |r0 - rf|    (1 m)
RHO_SHRINK_FIX = 0.0      # shrink only when the step made the defect WORSE
RHO_GROW_FIX = 0.5        # grow when the step at least halved it


def nd_cfg(x_init, mass, rf, tune, sigma_ref=None):
    """The solver constants restored against this problem's own length, time and control scales.

    sigma_ref is the TIME SCALE. Pass the warm start's t_go whenever there is one: run_one takes the
    flight time from the guess when tf is free (driver.py, `sigma = float(sw)`), so that is the time
    the solve actually starts from, and the time-dilation trust region and tf tolerance have to be
    sized against it. With no guess there is nothing better than the ballistic estimate, which is
    what gen_dataset.solve computes -- but it is a poor stand-in: on the nominal descent it reads
    1502.8 s against a true t_go of 1800.7 s, so scaling eta_s0 by it undersizes the time region by
    17% on the very problem the constants were tuned for.
    """
    dist = float(np.linalg.norm(np.asarray(x_init, float)[0:3] - np.asarray(rf, float)))
    tf_min = 2.0 * np.sqrt(dist * tune.MDRY / tune.TMAX)
    tf_max = 2.0 * np.sqrt(dist * mass / tune.TMIN)
    if sigma_ref is not None and np.isfinite(sigma_ref) and sigma_ref > 0.0:
        sigma0 = float(np.clip(sigma_ref, tf_min, tf_max))
    else:
        sigma0 = float(np.clip(2.0 * np.sqrt(dist / (0.5 * (tune.TMIN + tune.TMAX) / mass)),
                               tf_min, tf_max))
    accel = tune.TMAX / mass
    return dict(eta0=ETA0_ND * dist,
                eta_s0=ETA_S0_ND * sigma0,
                eta_u0=ETA_U0_ND * accel,
                tol_sig=TOL_SIG_ND * sigma0,
                tol_traj=TOL_TRAJ_ND * dist), dist, sigma0


def load_nominal(path=None):
    """The nominal descent as an (X, U, sigma) warm start, t_go included."""
    path = path or os.path.join(SCVX, "descent_nominal.npz")
    z = np.load(path)
    return np.asarray(z["X"], float), np.asarray(z["U"], float), float(z["sigma"])


@dataclass
class SCvxGuidance:
    """SCvx as the guidance law.

    warm_start reuses the previous solution as the next guess. Successive cycles differ by one
    control interval, so the previous plan is an excellent guess -- but it has to be SHIFTED first.
    After dt has elapsed the vehicle is one interval along, so the useful guess is the remainder of
    the old plan re-sampled onto the node grid with t_go reduced by dt, not the original plan again.

    run_one overwrites node 0 of the guess with the new x_init itself, which is what stops a warm
    guess whose first node disagrees with the initial condition from being scored as feasible and
    returned unchanged. The shift here supplies the right SHAPE and the right flight time; the
    anchoring is already handled downstream.
    """

    warm_start: bool = True
    nondim: bool = True           # scale the PTR constants to the current problem
    # The trust region is no longer carried across cycles. Persisting it was a workaround for the
    # ratchet -- the region could only decay, so re-opening it each cycle cost iterations shrinking
    # it back. With r1t/r2t recalibrated the region recovers on its own, and carrying a radius that
    # one cycle collapsed into the next is then actively harmful. Left switchable for comparison.
    persist_tr: bool = False
    # The nominal descent, used to seed the FIRST cycle. Every trajectory in the dataset lies in a
    # tube around it, so it is a far better guess than the straight line: cold-started, the solver
    # lands in a different basin (sigma ~ 2060 s, 6.01 kg) than the one the nominal occupies
    # (1801 s, 5.78 kg) and stays there, because refinement is local. Its t_go seeds the flight time
    # too, which is the whole point -- the ballistic estimate is 300 s short.
    nominal: tuple = field(default=None, repr=False)   # (X, U, sigma) or None
    iters: int = 120
    # Per-cycle wall-clock budget [s]. The iteration count does not bound the time -- a hard
    # subproblem can spend seconds inside the conic solver -- and a guidance cycle that overruns its
    # own period is worse than one that returns its best iterate early, because the controller
    # re-plans next cycle regardless. 0 disables it.
    max_seconds: float = 20.0
    ctx: object = None
    last: Plan = field(default=None, repr=False)
    _tr: tuple = field(default=None, repr=False)
    n_calls: int = 0
    t_solve_total: float = 0.0

    def __post_init__(self):
        if self.ctx is None:
            import gen_dataset as G
            with _in_scvx_dir():
                self.ctx = G.Ctx()

    # -- the interface the full model uses ----------------------------------------------------------
    def command(self, t, r_I, v_I, mass):
        """Estimated inertial state -> commanded thrust vector in the INERTIAL frame [N].

        Returns (T_cmd_inertial, plan). T_cmd is None if the solve failed, which the caller must
        handle -- silently substituting a zero or a previous command would hide a guidance failure
        inside a Monte Carlo.
        """
        plan = self.solve(t, r_I, v_I, mass)
        if not plan.ok:
            return None, plan
        m0 = float(np.exp(plan.X[6, 0]))
        T_B = m0 * plan.U[0:3, 0]                 # acceleration -> force, at the optimiser's mass
        return body_to_inertial_vec(T_B, t), plan

    @staticmethod
    def _shift(plan, dt):
        """The previous plan advanced by dt, as an (X, U, sigma) guess.

        The old plan spans t in [0, sigma] on the node grid. One interval later the part still to be
        flown is t in [dt, sigma], so the new grid maps tau -> (dt + tau*(sigma-dt))/sigma in the old
        coordinates and both X and U are resampled there. sigma itself drops by dt, which matters
        because run_one takes the flight time from the guess when tf is free -- handing back the
        unshifted sigma would ask for a flight time one interval too long on every cycle.
        """
        K = plan.X.shape[1]
        sig_new = float(max(plan.sigma - dt, 1e-3))
        tau_old = np.linspace(0.0, 1.0, K)
        tau_q = np.clip((dt + np.linspace(0.0, 1.0, K) * sig_new) / plan.sigma, 0.0, 1.0)
        Xw = np.array([np.interp(tau_q, tau_old, plan.X[i]) for i in range(plan.X.shape[0])])
        Uw = np.array([np.interp(tau_q, tau_old, plan.U[i]) for i in range(plan.U.shape[0])])
        return Xw, Uw, sig_new

    def _nominal_tail(self, x_init):
        """The part of the nominal that still lies ahead of this state, as an (X, U, t_go) guess.

        Matched on range to the landing site, which is monotone along the descent, so the node whose
        remaining range is closest to the current one is the right place to cut.
        """
        Xn, Un, sn = self.nominal
        Xn = np.asarray(Xn, float); Un = np.asarray(Un, float); sn = float(sn)
        rf = np.asarray(self.ctx.rf, float)
        rng_now = float(np.linalg.norm(np.asarray(x_init, float)[0:3] - rf))
        rng_nom = np.linalg.norm(Xn[0:3] - rf[:, None], axis=0)
        j = int(np.argmin(np.abs(rng_nom - rng_now)))
        j = min(j, Xn.shape[1] - 2)                  # always leave at least one segment
        return Xn[:, j:], Un[:, j:], sn * (1.0 - j / (Xn.shape[1] - 1))

    def solve(self, t, r_I, v_I, mass):
        """Run SCvx from the estimated state. Returns a Plan; check .ok before using it."""
        import gen_dataset as G
        r_B, v_B = inertial_to_body(np.asarray(r_I, float), np.asarray(v_I, float), t)
        x_init = np.concatenate([r_B, v_B, [np.log(mass)]])

        warm = None
        if self.warm_start and self.last is not None and self.last.ok:
            dt = t - self.last.t_query
            if 0.0 <= dt < self.last.sigma:
                warm = self._shift(self.last, dt)
        if warm is None and self.nominal is not None:
            # First cycle, or recovery after a failed one. Seed from the nominal -- shape AND t_go --
            # but from the RIGHT PART of it. Handing over the whole nominal mid-flight hands over its
            # whole t_go as well, and sigma is a hard bound in the subproblem: 1800.7 s against a
            # tf_max of 691.8 s at a real MPC state 1005 m from the site made iteration 0 infeasible,
            # both solvers agreed, and the cycle was over before it began. Because the fallback fires
            # exactly when the previous cycle failed, that turned one bad cycle into a death spiral.
            # Match on remaining range and take the tail from there.
            warm = self._nominal_tail(x_init)

        # gen_dataset.solve builds cfg internally with no override path. solver_cfg passes eta0,
        # eta_u0 and eta_s0 as explicit keywords AND unpacks tune.EXTRA_CFG into the same update(),
        # so putting those keys in EXTRA_CFG raises "multiple values for keyword argument". The
        # radii therefore have to be set on the module attributes solver_cfg reads; only the two
        # tolerances, which it does not name explicitly, can go through EXTRA_CFG.
        import tune
        extra_prev = dict(tune.EXTRA_CFG)
        saved = (tune.ETA0, tune.ETA_U0, tune.ETA_S0)
        saved_rho = (tune.RHO_SHRINK, tune.RHO_GROW)
        if self.nondim:
            # The time scale is the warm start's t_go when there is one -- that is the flight time
            # run_one will actually begin from, not the ballistic estimate gen_dataset.solve builds.
            sig_ref = float(warm[2]) if warm is not None else None
            over, dist, sig0 = nd_cfg(x_init, mass, self.ctx.rf, tune, sigma_ref=sig_ref)
            tune.ETA0, tune.ETA_U0, tune.ETA_S0 = over["eta0"], over["eta_u0"], over["eta_s0"]
            tune.RHO_SHRINK, tune.RHO_GROW = RHO_SHRINK_FIX, RHO_GROW_FIX
            tune.EXTRA_CFG = {**extra_prev, "tol_sig": over["tol_sig"],
                              "tol_traj": over["tol_traj"], "max_seconds": self.max_seconds}
        # CARRY THE TRUST REGION. run_one resets the radii from cfg on every call, so each guidance
        # cycle re-opens eta_s to its cold-start value and then spends iterations shrinking it back
        # down -- measured at 11 of 14 iterations when warm-started from the exact solution, every
        # one of them a full rejected subproblem solve. The previous cycle already discovered the
        # radius the linearisation can support, so it is handed forward.
        #
        # That ratchet is now fixed at its cause (r2t lowered to 0.5, r1t to 0.0 -- see the constants
        # above), so this is off by default and kept only for A/B comparison.
        if self.persist_tr and self._tr is not None:
            tune.ETA0, tune.ETA_U0, tune.ETA_S0 = self._tr

        t0 = time.perf_counter()
        try:
            with _in_scvx_dir():
                cs, r, tr, cfg = G.solve(self.ctx, x_init, mass, warm=warm, iters=self.iters)
            dt = time.perf_counter() - t0
            plan = Plan(t_query=t, X=np.asarray(r["X"], float), U=np.asarray(r["U"], float),
                        sigma=float(r["sigma"]), solve_s=dt,
                        iters=int(len(tr)) if tr is not None else 0,
                        # "stalled" is NOT a failure here -- those runs were measured to carry a
                        # defect ~1000x smaller than the ones labelled converged, they just exhaust
                        # the trust region instead of tripping the trajectory criteria. An empty
                        # trace is a different thing: the subproblem never produced a step, so X and
                        # U are still the guess and the plan describes nothing.
                        ok=bool(tr is not None and len(tr) > 0
                                and str(r.get("reason", "")) != "infeasible"),
                        reason=str(r.get("reason", "")), exit_via=str(r.get("exit_via", "")))
            if self.persist_tr:
                self._tr = (float(r["eta_final"]), float(r["eta_u_final"]),
                            float(r["eta_s_final"]))
        except Exception as exc:
            # Keep the message. A guidance failure inside a Monte Carlo is worth diagnosing, and a
            # bare except that discards the reason makes that impossible.
            dt = time.perf_counter() - t0
            plan = Plan(t_query=t, X=np.zeros((7, 0)), U=np.zeros((4, 0)),
                        sigma=float("nan"), solve_s=dt, ok=False,
                        error=f"{type(exc).__name__}: {exc}")
        finally:
            tune.EXTRA_CFG = extra_prev
            tune.ETA0, tune.ETA_U0, tune.ETA_S0 = saved
            tune.RHO_SHRINK, tune.RHO_GROW = saved_rho
        self.n_calls += 1
        self.t_solve_total += dt
        self.last = plan
        return plan

    # -- diagnostics --------------------------------------------------------------------------------
    def thrust_profile(self, plan=None):
        """The whole planned thrust history in newtons, body frame -- for inspection, not for flight.

        The vehicle only ever executes the first entry. This exists so the planned profile can be
        plotted against what the actuator actually delivered.
        """
        p = plan or self.last
        if p is None or not p.ok:
            return None, None
        m = np.exp(p.X[6, :])
        T = m * p.U[0:3, :]
        tau = np.linspace(0.0, 1.0, p.X.shape[1])
        return tau * p.sigma, T

    @property
    def mean_solve_s(self):
        return self.t_solve_total / max(self.n_calls, 1)

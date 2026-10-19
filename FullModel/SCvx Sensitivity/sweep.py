#!/usr/bin/env python
"""How the optimal cost and the flight time move when ONE state component moves.

THE QUESTION. The guidance network has to learn a map from the state to the optimal trajectory. For
that map to be learnable it has to BE a map: single-valued, and smooth enough that a neighbouring
state gives a neighbouring answer. This sweep tests that directly on the only two scalars the whole
problem reduces to -- the propellant J and the flight time tf -- by perturbing one component of the
seven-state initial condition at a time and re-solving from scratch.

WHAT A RESULT LOOKS LIKE. Smooth, monotone curves mean the optimum is well-behaved and the label
disagreement measured in the network's direction channel is genuine sensitivity. Jumps -- two nearby
initial conditions returning propellant that differs by far more than the perturbation can explain
-- mean the solver is landing in different basins, and then the fix is in data generation and no
network architecture touches it. That distinction has been open all along and this is what settles
it.

SEVEN-STATE, NOT TEN. x = [r(3), v(3), ln m], control u = [a(3), Gamma]: the thrust vector IS the
control, so there is no direction state carrying history and no gauge freedom. Verified against
Malyuta et al. arXiv:2106.09125 Problem (95) at +0.83 / +0.79 / +0.49 % of the published flight time
on a K = 38 / 76 / 151 ladder -- see SCvx Verification/run_malyuta_v1.py.

ONE COMPONENT AT A TIME, 50 DRAWS EACH, UNIFORM:

    r_x, r_y, r_z    U(-500, +500) m
    v_x, v_y, v_z    U(-1, +1) m/s
    ln m    via m    U(-10, +10) kg

7 channels x 50 draws = 350 solves, plus the unperturbed nominal they are all measured against.
Components are those of the BODY-FIXED state, because that is what the seven-state vector holds; the
site-local decomposition (up / east / north) is recorded alongside every draw so the same data can
be read as altitude, downrange and crossrange sensitivity without re-solving.

THE CORE IS PINNED BEFORE GRAVITY IS IMPORTED, and this study is worthless without it.
polyhedral_gravity reduces its 49 152 facet contributions in a non-deterministic order, which moves
the fuel by 0.57 % between otherwise identical runs -- and a 10 kg mass perturbation on a 600 kg
vehicle is expected to move it by far less than that. Unpinned, the sweep measures the reduction
order and nothing else. The pin is sched_setaffinity, which exists on Linux only: this must run on
the cluster, not on a Mac.

Usage:
    sweep.py --nominal-only                  # solve and cache the unperturbed case
    sweep.py --index 0 --count 10            # draws 0..9 of the 350-run plan
    sweep.py --collate                       # gather the shards into one json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ---- the pin, before anything imports gravity ------------------------------------------------
# MC_SLOT picks WHICH core when several solves share one allocation. Indexing into the cpuset the
# process actually got is safer than a hard-coded id: Slurm does not promise 0..n-1.
if hasattr(os, "sched_getaffinity"):                 # Linux only; a no-op elsewhere
    _allowed = sorted(os.sched_getaffinity(0))
    _core = _allowed[int(os.environ.get("MC_SLOT", "0")) % len(_allowed)]
    os.sched_setaffinity(0, {_core})
# VECLIB_MAXIMUM_THREADS IS THE ONE THAT MATTERS ON A MAC, and leaving it out cost a day.
# numpy here is built against Accelerate, which reads NONE of the OMP / MKL / OPENBLAS / TBB
# variables -- so the BLAS stayed multithreaded, its reduction order moved with thread scheduling,
# and the free-final-time iteration amplified that into different CONVERGENCE OUTCOMES for
# bit-identical inputs: draw 77 hit maxiter at 200 iterations inside a ten-worker sweep and
# converged in 12 when re-run on its own. Any study that compares one solve against another needs
# every one of these set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.abspath(os.path.join(HERE, "..", "SCvx", "_v1_zoh_backup"))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
sys.path.insert(0, SCVX)                 # second: gravity
sys.path.insert(0, V1)                   # first: the seven-state solver wins every name

import config as C                                                           # noqa: E402

assert os.path.dirname(os.path.abspath(C.__file__)) == V1, \
    f"config resolved to {C.__file__}, not the seven-state copy"

# ---- site A, and the pointing cone off -------------------------------------------------------
# PATCHED BEFORE ptr IS IMPORTED. subproblem.build takes gs_deg and point_deg as DEFAULT arguments,
# and a default is evaluated once at definition time, so patching the module afterwards leaves the
# old values in place silently. v1's ptr.run does not forward either one, so this is the only route.
#
# POINTING_DEG GOES TO 180, i.e. OFF. v1 carries it at 140 deg, which is not a hardware limit --
# DATASETS.md records it binding at interior nodes on 27.5 % of trajectories, costing up to 10.4 %
# of the fuel where it binds, and causing 59 of 60 generation failures in the ten-state set. Left
# on, it would put a constraint artefact straight into the sensitivity curves this study exists to
# measure. The glideslope and the terminal pointing stay as they are: those are real.
SITE = dict(name="A", lat_deg=-36.975, lon_deg=95.401)
C.POINTING_DEG = 180.0
C.LAT_DEG, C.LON_DEG = SITE["lat_deg"], SITE["lon_deg"]

# WHICH PINN-GM-III CHECKPOINT. The seven-state config predates PINN_NPZ, so PinnField's own
# fallback would pick pinn_eros_np.npz -- the SUPERSEDED model. SCvx/config.py records why it was
# replaced: trained on radii 0-10R, it puts 85.5 % of its 50 000 samples above 25 km and leaves 298
# below 1 km, so its error peaks at 2.8 % at 300 m altitude with a SIGNED vertical bias, in exactly
# the shell a landing descends through. Taken from the ten-state config by path so there is one
# source of truth for it, and set BEFORE gravity is imported.
import importlib.util as _iu                                                 # noqa: E402
_spec = _iu.spec_from_file_location("config_10state", os.path.join(SCVX, "config.py"))
_c10 = _iu.module_from_spec(_spec); sys.modules["config_10state"] = _c10
_spec.loader.exec_module(_c10)
C.PINN_NPZ = _c10.PINN_NPZ
assert os.path.isfile(C.PINN_NPZ), f"PINN-GM-III checkpoint not found: {C.PINN_NPZ}"

import gravity as GR                                                         # noqa: E402
import ptr                                                                   # noqa: E402
import subproblem as SP                                                      # noqa: E402
from dynamics import NX, Dynamics                                            # noqa: E402

assert NX == 7, f"this is the seven-state sweep but dynamics reports NX = {NX}"
import inspect                                                               # noqa: E402
assert inspect.signature(SP.build).parameters["point_deg"].default == 180.0, \
    "the pointing cone default was not patched -- it is still active"

# PINN-GM-III, NOT THE POLYHEDRAL FIELD. This is the model the guidance carries, so the cost
# sensitivity measured through it is the sensitivity the guidance can actually see. Two properties
# matter here beyond that: it is a plain network forward pass, so it is DETERMINISTIC -- the
# polyhedral field reduces 49 152 facet contributions in a non-deterministic order, worth 0.57 % of
# the fuel between otherwise identical runs, which is the same order as a 500 m position draw is
# expected to move J and would have made that channel unreadable -- and it is roughly twenty times
# cheaper per call.
#
# THE PRICE IS AN INEXACT JACOBIAN. PinnField.grad returns zero, because an analytic gradient does
# not exist for it and six finite differences per call would cancel the speed it is used for. That
# is safe for the fixed point -- successive convexification drives the NONLINEAR defect to zero, so
# the Jacobian sets only the search direction -- but it is not neutral for THIS study: a different
# path can descend into a different basin, and basin-hopping is one of the things being looked for.
# If jumps appear, re-solve those draws with --gravity poly, whose gradient is exact and free, and
# see whether they survive.
FIELD = None                  # set in main(), once the checkpoint above is pinned
# The handover rung's draws, in non-dimensional units: +/-500 m on L = 11771 m and +/-1 m/s on
# v = L/T = 19.81 m/s. Every lower rung reuses these FRACTIONS, not the metres.
# THE TRUST-REGION FLOOR, lowered from v1's 1e-4. SCvx converges by SHRINKING the radius until
# the linear model is faithful enough that the step is accepted -- as R -> 0 the linearisation
# becomes exact to first order, so rho -> 1 and acceptance is guaranteed. A floor stops that
# shrinking while the step is still being rejected, and the next iteration then poses the
# IDENTICAL subproblem and gets the identical answer, forever. That is what the `stalled` status
# detects, and it is an implementation limit rather than anything the problem imposes.
R_FLOOR = 1e-10
DR_ND = 500.0 / 11771.0
DV_ND = 1.0 / 19.8098
K = 100                       # overridden by --k; the handover value, giving 20.8 s segments
SEG_TARGET_S = 20.8           # tf/(K-1) at the handover rung
ALTITUDE_KM = 11.771          # matches the ten-state dataset's handover, for comparability
TOL = dict(iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4)
BASE_SEED = 20261018

# channel, lo, hi, unit -- the plan, in order. Index i of the 350 is (channel i//50, draw i%50).
CHANNELS = [("r_x", -500.0, 500.0, "m"), ("r_y", -500.0, 500.0, "m"), ("r_z", -500.0, 500.0, "m"),
            ("v_x", -1.0, 1.0, "m/s"), ("v_y", -1.0, 1.0, "m/s"), ("v_z", -1.0, 1.0, "m/s"),
            ("m", -10.0, 10.0, "kg")]
N_DRAW = 50
N_TOTAL = len(CHANNELS) * N_DRAW
OUT = os.path.join(HERE, "Figures&data")


def geometry():
    """Site A, its local vertical, the site-local triad, and the nominal handover state.

    Reproduced here rather than taken from v1's problem_setup.build, which calls
    GR.site_and_normal() with no arguments and so would read whatever lat/lon the config defaults
    were when gravity.py was imported. Passing them explicitly removes that coupling and puts the
    site in the output where it can be checked.
    """
    r_site, n_hat = GR.site_and_normal(lat_deg=SITE["lat_deg"], lon_deg=SITE["lon_deg"])
    east = np.cross(np.array([0.0, 0.0, 1.0]), n_hat); east /= np.linalg.norm(east)
    north = np.cross(n_hat, east)
    r0 = r_site + ALTITUDE_KM * 1e3 * n_hat          # straight overhead: cone_deg = 0
    vl = np.asarray(C.V0_LOCAL, float)
    v0 = vl[0] * n_hat + vl[1] * east + vl[2] * north
    return dict(r_site=r_site, n_hat=n_hat, east=east, north=north, r0=r0, v0=v0, m0=C.MWET)


def base_state(alt_km):
    """The base initial condition at a given altitude, taken ON the nominal descent.

    NOT an arbitrary point at that altitude. The question is how the cost sensitivity behaves along
    the trajectory the vehicle actually flies, and the sub-trajectory expansion the network trains
    on takes exactly these points: the tail of an optimal trajectory is optimal for its own initial
    condition, because the plan's dynamics are autonomous in the rotating frame.

    Altitude is measured the way the handover is, along the LOCAL VERTICAL at the site, so the
    ladder and the 11.771 km handover are on one scale.

    Returns the node index too, because the warm start has to be the tail from that node.
    """
    z = np.load(os.path.join(OUT, "sweep_nominal_plan.npz"))
    r, v, m = z["r"].T, z["v"].T, z["m"]
    h = (r - G["r_site"]) @ G["n_hat"]
    j = int(np.argmin(np.abs(h - alt_km * 1e3)))
    return j, r[j].copy(), v[j].copy(), float(m[j]), float(h[j])


def warm_from_tail(sc, j):
    """The nominal's own tail from node j, resampled onto K nodes, in THIS draw's scales.

    The tail has fewer nodes than K, so it is resampled rather than truncated -- a warm start of
    the wrong length silently replaces the K that was asked for, which is the trap v1's ptr.run
    still has and run_malyuta_v1.resample exists to avoid.
    """
    z = np.load(os.path.join(OUT, "sweep_nominal_plan.npz"))
    r, v, m = z["r"][:, j:], z["v"][:, j:], z["m"][j:]
    a, gm = z["a"][:, j:], z["gamma"][j:]
    n = r.shape[1]
    tw, tq = np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, K)
    rs = lambda A: np.array([np.interp(tq, tw, A[i]) for i in range(A.shape[0])])
    X = np.vstack([rs(r) / sc.L, rs(v) / sc.v, np.log(np.interp(tq, tw, m))[None, :]])
    U = np.vstack([rs(a) / sc.A, (np.interp(tq, tw, gm) / sc.A)[None, :]])
    tf_tail = float(z["tf_s"]) * (1.0 - j / (len(z["m"]) - 1))
    return X, U, float(sc.t_nd(tf_tail))


def warm_from_rung(sc, tag):
    """THIS rung's own converged nominal, non-dimensionalised with this draw's scales.

    Every draw is a perturbation of the rung's unperturbed case, whose exact answer is already on
    disk -- so that is the warm start, not a resampling of the handover nominal's tail. The tail is
    a stretched interpolation of a different arc, and interpolating a bang-bang control across its
    switches turns them into ramps, which is a dynamically inconsistent guess.

    Measured with the tail: 17 % of draws needed a fallback schedule and 2.2 % failed outright,
    against 8 % retries and none failing at the handover rung, which warm-starts from its own
    nominal. The gap widens going down because the draws are a fixed +/-500 m while L shrinks from
    11.8 km to 0.9 km -- a draw is 4 % of the problem scale at handover and 22 % at 800 m -- so the
    guess has to be BETTER at low altitude, not worse.
    """
    z = np.load(os.path.join(OUT, f"sweep_nominal_plan{tag}.npz"))
    X = np.vstack([z["r"] / sc.L, z["v"] / sc.v, np.log(z["m"])[None, :]])
    U = np.vstack([z["a"] / sc.A, (z["gamma"] / sc.A)[None, :]])
    return X, U, float(sc.t_nd(float(z["tf_s"])))


def warm_from_nominal(sc, m0):
    """The cached nominal plan, non-dimensionalised with THIS draw's scales.

    WHY WARM-START AT ALL. Cold, from ptr.initial_guess's straight line, the free-final-time
    iteration has to climb from ~1350 s to ~2050 s with the trust region saturated, and it only
    just settles: the nominal's own trace reads 1354 -> 1576 -> 1725 -> 1948 -> 2073 -> 2043 ->
    2054 -> 2051 -> 2052. A perturbation can tip that decaying oscillation into one that never
    meets the defect tolerance, which is what produced maxiter on draws as small as 4 cm/s while
    500 m draws converged in 30 s. closed_loop.py has always done this -- it inherits the previous
    cycle's plan and converges in single digits -- and every draw here is a <= 4 % perturbation of
    one nominal, which is precisely the case warm-starting exists for.

    IT ALSO MAKES THE SWEEP A FUNCTION OF THE INITIAL CONDITION. Cold-started inside a worker, a
    draw's outcome depended on its POSITION IN THE SEQUENCE -- draw 212 converged alone and hit
    maxiter as the third of a batch -- so dJ was measuring the optimiser's history as well as the
    perturbation. Starting every draw from the same point removes that.
    """
    z = np.load(os.path.join(OUT, "sweep_nominal_plan.npz"))
    X = np.vstack([z["r"] / sc.L, z["v"] / sc.v, np.log(z["m"])[None, :]])
    U = np.vstack([z["a"] / sc.A, (z["gamma"] / sc.A)[None, :]])
    return X, U, float(sc.t_nd(float(z["tf_s"])))


def solve(r0, v0, m0, warm=None, verbose=False, ratio_test=False, d_margin=0.1,
          damped=False):
    """One free-final-time minimum-fuel descent. The two-stage chain, as the dataset uses it."""
    sc = C.Scales.build(r0, G["r_site"], m_wet=m0)
    if isinstance(warm, tuple) and warm and warm[0] == "rung":
        warm = warm_from_rung(sc, warm[1])
    elif isinstance(warm, tuple) and warm and warm[0] == "tail":
        warm = warm_from_tail(sc, warm[1])
    elif warm == "nominal":
        warm = warm_from_nominal(sc, m0)
    elif isinstance(warm, tuple) and warm and warm[0] == "tail":
        warm = warm_from_tail(sc, warm[1])
    dyn = Dynamics(sc, field=FIELD)
    x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(m0)]])
    common = dict(dyn=dyn, scales=sc, x_init=x_init, rf_nd=G["r_site"] / sc.L,
                  n_hat=G["n_hat"], K=K, verbose=verbose, **TOL)
    # THE TERMINAL SLACK IS A PARAMETER, AND A SENSITIVITY STUDY WANTS IT SMALL.
    # Stage 2 must land within D + d_margin, an ABSOLUTE length, so as a fraction of the problem it
    # grows as the range shrinks: the achieved miss spreads over about 75 mm at every rung, which is
    # 6.1e-06 of the range at the 11.8 km handover and 8.1e-05 at 800 m. The optimiser may land
    # anywhere in that ball and different landing points cost different propellant, so the slack
    # injects ambiguity into the very gradient being measured -- two draws 15 g apart in mass came
    # out 4.2 g apart in cost at the 800 m rung, 11 % of that rung's whole signal, both converged
    # first pass. It is also a flat direction in its own right, and flat directions are what this
    # free-final-time iteration cannot settle on.
    #
    # 0.1 m stays correct for GENERATING LABELS -- it is what pinn_case.py uses, and it keeps the
    # solver off a hard constraint -- but measuring dJ/dx0 wants the landing point pinned.
    # STAGE 1 IS ALWAYS COLD, AND SEEDING IT IS ACTIVELY HARMFUL. Its objective is the landing
    # error, which is DEGENERATE whenever the site is reachable -- every feasible landing scores
    # zero -- so it has no preference in tf and goes wherever its seed points it. Cold from
    # initial_guess's straight line it settles at ~1435 s in 4 iterations, every time. Seeded with
    # the nominal's 2056 s fuel plan it instead drifts upward by ~130 s per iteration with the
    # trust region pinned at its cap: measured 2184 -> 6730 s over 35 iterations and still
    # climbing, which is what turned a 15 s draw into a 500 s maxiter.
    r1 = ptr.run(mode="error", warm=None, **common)
    if not str(r1.status).startswith("converged"):
        return r1, None, sc
    # STAGE 2 IS THE ONE THAT WANTS THE WARM START. It carries a real objective in tf, it is the
    # solve that oscillates, and it is the answer this study records. Seeded with the nominal plan
    # it starts where it is going; seeded with stage 1's 1435 s answer it has to climb 600 s again.
    w2 = warm if warm is not None else (r1.X, r1.U, r1.s)
    # ONE SCHEDULE FOR THE WHOLE RUNG, when --damped is set. A draw that converges only on a
    # fallback was solved by a different algorithm from its neighbours, and the two schedules were
    # measured disagreeing by ~3 g -- so at a rung where half the draws need the fallback, the
    # fallback IS the dominant error term. Starting every draw on the damped radius costs
    # iterations and buys consistency.
    prim = dict(R0=0.1, R_bounds=(R_FLOOR, 0.15)) if damped else dict(R_bounds=(R_FLOOR, 1.0))
    r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + d_margin / sc.L, warm=w2,
                 ratio_test=ratio_test, **prim, **common)

    # A MAXITER HERE IS A LIMIT CYCLE, NOT A BAD TRAJECTORY, AND IT IS WORTH RETRYING.
    # Traced on draw 304: stage 2 alternates between two iterates forever --
    #
    #     odd   J -713.76038  tf 2059.3 s  defect 2.35e-04 m  R 1.0
    #     even  J -713.76150  tf 2060.1 s  defect 1.72e-04 m  R 0.5
    #
    # Every PHYSICAL criterion is met on both branches: the position defect is 2.3e-04 m against a
    # 1e-03 bar, the velocity defect 1.9e-05 against 1e-04, the virtual control 6e-10 against 1e-06.
    # The only test failing is the objective one -- the two branches differ by dJ/J = 6.2e-04, about
    # 3.4 g, and v1's acceptance bar for the fuel stage is tol_abs = 1e-4 kg = 0.1 g. It cannot
    # accept a cycle of that amplitude however long it runs.
    #
    # The trust radius alternating 1.0 <-> 0.5 says the growth rule is toggling, so the fix is to
    # damp the step rather than to loosen the bar: accepting the cycle would stamp 3.4 g of
    # ambiguity onto the data point, which is 3 % of the whole mass signal. Capping the radius costs
    # iterations and costs nothing in accuracy. Reported per draw so a retried point is never
    # mistaken for a first-pass one.
    # A LADDER OF FALLBACKS, EACH ONE A DIFFERENT FAILURE MODE.
    #   1. damped radius      -- for an oscillation the full radius keeps feeding
    #   2. the defect rule    -- for a STALL, where the ratio test rejects every step and the
    #                            radius collapses to its floor. Measured on the 2 km rung, where
    #                            the ratio test stalled twice and the defect rule converges: the
    #                            two schedules fail on disjoint sets, so trying both costs a few
    #                            minutes and loses no data point.
    # Whichever succeeds is recorded in the status, so a fallback result is never mistaken for a
    # first-pass one.
    for tag, kw in (("retry", dict(R0=0.1, R_bounds=(R_FLOOR, 0.15), ratio_test=ratio_test)),
                    ("defectrule", dict(ratio_test=False))):
        if str(r2.status).startswith("converged"):
            break
        rb = ptr.run(mode="fuel", D_nd=r1.D_nd + d_margin / sc.L, warm=w2, **kw, **common)
        if str(rb.status).startswith("converged"):
            rb.status = f"converged_{tag}"
            return r1, rb, sc
        r2.status = f"{r2.status}+{tag}_{rb.status}"
    return r1, r2, sc


def record(tag, idx, channel, delta, r0, v0, m0, r1, r2, sc, wall):
    """One row. The site-local decomposition of the perturbation goes in too, so the same data can
    be plotted against altitude, downrange and crossrange without re-solving anything."""
    dr = r0 - G["r0"]
    dv = v0 - G["v0"]
    ok = r2 is not None and str(r2.status).startswith("converged")
    return dict(
        tag=tag, index=idx, channel=channel, delta=float(delta),
        # what actually went in
        r0=[float(x) for x in r0], v0=[float(x) for x in v0], m0=float(m0),
        d_up_m=float(dr @ G["n_hat"]), d_east_m=float(dr @ G["east"]),
        d_north_m=float(dr @ G["north"]),
        dv_up_ms=float(dv @ G["n_hat"]), dv_east_ms=float(dv @ G["east"]),
        dv_north_ms=float(dv @ G["north"]),
        range0_m=float(np.linalg.norm(r0 - G["r_site"])),
        # what came out. J IS THE PROPELLANT, in kg -- the solver's own objective is scaled and
        # sign-flipped, and is not comparable across cases whose scales differ.
        status1=str(r1.status), iters1=int(r1.iters), D_m=float(r1.miss_m),
        status=(str(r2.status) if r2 is not None else "stage1_failed"),
        iters=(int(r2.iters) if r2 is not None else -1),
        J_kg=(float(r2.fuel_kg) if ok else float("nan")),
        tf_s=(float(r2.tf_s) if ok else float("nan")),
        miss_m=(float(r2.miss_m) if ok else float("nan")),
        defect_m=(float(r2.defect_m) if ok else float("nan")),
        converged=bool(ok), wall_s=float(wall))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nominal-only", action="store_true")
    ap.add_argument("--index", type=int, default=None, help="first draw of the 350-run plan")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--collate", action="store_true")
    ap.add_argument("--repeat-nominal", dest="repeat_nominal", type=int, default=0,
                    help="re-solve the UNPERTURBED case this many times and report the spread. "
                         "This is the noise floor every dJ in the sweep is read against: without "
                         "the core pin, polyhedral_gravity's reduction order moves the fuel between "
                         "otherwise identical runs, and a perturbation whose effect is smaller than "
                         "that spread is unmeasurable. Writes nothing; does not touch the cache.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--damped", action="store_true",
                    help="start every draw on the damped trust radius instead of reaching it only "
                         "as a fallback, so all draws in a rung share one schedule")
    ap.add_argument("--d-margin-m", dest="d_margin", type=float, default=0.1,
                    help="terminal slack: stage 2 lands within D + this. 0.1 m matches the label "
                         "convention; a sensitivity study wants it small so the landing point is "
                         "pinned and J is a clean function of the initial state")
    ap.add_argument("--k", dest="K", type=int, default=None,
                    help="node count. Default 100 at handover; with --alt-km it is chosen to hold "
                         "the SEGMENT LENGTH near the handover's 20.8 s, because K=100 over a 450 s "
                         "arc is 4.5 s segments -- seven times finer than the rung this is being "
                         "compared against, and fine enough that the ratio test is differencing "
                         "two near-zero numbers")
    ap.add_argument("--alt-km", dest="alt_km", type=float, default=None,
                    help="put the base state on the nominal at this altitude above the site "
                         "instead of at the 11.771 km handover")
    ap.add_argument("--tag", default="", help="suffix for the shard files, so rungs do not collide")
    ap.add_argument("--ratio-test", dest="ratio_test", action="store_true",
                    help="use the PTR ratio test (reject steps, rho-band radius) in stage 2 "
                         "instead of the defect-based radius rule")
    ap.add_argument("--cold", action="store_true",
                    help="cold-start every draw from ptr.initial_guess instead of warm-starting "
                         "from the cached nominal. Slower, and its outcome depends on the draw's "
                         "position in the worker sequence; kept only to reproduce that.")
    ap.add_argument("--gravity", default="pinn", choices=("pinn", "poly"),
                    help="pinn = PINN-GM-III, the onboard model, deterministic, zero Jacobian. "
                         "poly = the polyhedral truth, exact gradient, non-deterministic reduction")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    global G, FIELD, BASE_J
    FIELD = GR.pinn_field() if a.gravity == "pinn" else GR.field()
    G = geometry()
    BASE_J = None
    # THE LADDER, SET BEFORE ANYTHING ELSE so that --alt-km --nominal-only solves THIS rung's own
    # reference. Measuring every dJ against the handover nominal instead would report a 2 kg
    # "sensitivity" that is just the rest of the descent.
    if a.K is not None:
        globals()["K"] = int(a.K)
    if a.alt_km is not None:
        BASE_J, br, bv, bm, bh = base_state(a.alt_km)
        G["r0"], G["v0"], G["m0"] = br, bv, bm
        # THE DRAWS SCALE WITH THE PROBLEM, so every rung is the same problem for the solver.
        #
        # The non-dimensionalisation is already scale-free -- L = |r0 - r_site|, A = Tmax/m_wet
        # fixed, T = sqrt(L/A) -- and it works: |v0| stays 0.44-0.77 and s = tf/T stays 1.8-3.5
        # across the whole ladder. A FIXED +/-500 m draw is not scale-free, though: it is 4.2 % of
        # the range at handover and 21 % at 2 km, so against a trust radius of R0 = 0.35 that never
        # moves, one draw goes from eating 12 % of the trust region to 60 % of it. Measured, the
        # clean-convergence fraction tracks that almost exactly -- 69, 64, 51, 49 % against
        # dr_nd of 0.043, 0.059, 0.087, 0.212.
        #
        # Holding dr/L and dv/(L/T) at their handover values fixes it, and costs nothing: a slope
        # is a derivative, so a smaller span is if anything a better estimate of it (less curvature
        # contamination), and the result is reported per metre either way. Mass needs no such
        # treatment -- it does not scale with L.
        _sc = C.Scales.build(br, G["r_site"], m_wet=bm)
        span_r = DR_ND * _sc.L
        span_v = DV_ND * _sc.v
        for _i, (_c, _lo, _hi, _u) in enumerate(CHANNELS):
            if _c.startswith("r_"):
                CHANNELS[_i] = (_c, -span_r, span_r, _u)
            elif _c.startswith("v_"):
                CHANNELS[_i] = (_c, -span_v, span_v, _u)
        print(f"ladder rung: node {BASE_J}/{K-1}  altitude {bh/1e3:.3f} km  "
              f"range {np.linalg.norm(br - G['r_site'])/1e3:.3f} km  "
              f"|v| {np.linalg.norm(bv):.4f} m/s  m {bm:.3f} kg")
        print(f"draws: position +/-{span_r:.1f} m ({DR_ND:.4f} L)   "
              f"velocity +/-{span_v:.3f} m/s ({DV_ND:.4f} L/T)   mass +/-10 kg")
    nom_path = os.path.join(OUT, f"sweep_nominal{a.tag}.json")

    if a.collate:
        rows = []
        for f in sorted(os.listdir(OUT)):
            if f.startswith(f"sweep_shard{a.tag}_") and f.endswith(".json"):
                rows += json.load(open(os.path.join(OUT, f)))["rows"]
        nom = json.load(open(nom_path))
        out = os.path.join(OUT, f"sweep{a.tag}.json")
        json.dump(dict(site=SITE, altitude_km=ALTITUDE_KM, K=K, formulation="7-state",
                       gravity=a.gravity, pinn_npz=os.path.basename(C.PINN_NPZ),
                       alt_km=a.alt_km, base_node=BASE_J, d_margin_m=a.d_margin, damped=a.damped,
                       base_seed=BASE_SEED, channels=CHANNELS, n_draw=N_DRAW,
                       nominal=nom, rows=sorted(rows, key=lambda r: r["index"])),
                  open(out, "w"), indent=1)
        ok = sum(1 for r in rows if r["converged"])
        print(f"collated {len(rows)} of {N_TOTAL} draws, {ok} converged -> {out}")
        return

    print(f"site {SITE['name']}  lat {SITE['lat_deg']:+.3f}  lon {SITE['lon_deg']:+.3f}  "
          f"|r_site| {np.linalg.norm(G['r_site'])/1e3:.3f} km")
    print(f"handover {ALTITUDE_KM:g} km along the local vertical, range "
          f"{np.linalg.norm(G['r0']-G['r_site'])/1e3:.3f} km, |v0| {np.linalg.norm(G['v0']):.4f} m/s")
    if a.gravity == "pinn":
        print(f"K {K}   pointing cone OFF   gravity PINN-GM-III "
              f"{os.path.basename(C.PINN_NPZ)}  (deterministic, grad = 0)", flush=True)
    else:
        print(f"K {K}   pointing cone OFF   gravity polyhedral "
              f"{FIELD.n_facets:,} facets  (exact grad)", flush=True)

    if a.repeat_nominal:
        Js, tfs = [], []
        for j in range(a.repeat_nominal):
            t0 = time.perf_counter()
            r1, r2, sc = solve(G["r0"], G["v0"], G["m0"])
            w = time.perf_counter() - t0
            ok = r2 is not None and str(r2.status).startswith("converged")
            J = float(r2.fuel_kg) if ok else float("nan")
            tf = float(r2.tf_s) if ok else float("nan")
            Js.append(J); tfs.append(tf)
            print(f"  repeat {j:>2}  J {J:.9f} kg   tf {tf:.6f} s   "
                  f"{r2.status if r2 is not None else 'stage1_failed'}  [{w:.0f} s]", flush=True)
        Js, tfs = np.array(Js), np.array(tfs)
        print(f"\nnoise floor over {a.repeat_nominal} identical solves:")
        print(f"  J   spread {Js.max()-Js.min():.3e} kg = {100*(Js.max()-Js.min())/np.mean(Js):.4f} %"
              f"   std {Js.std():.3e} kg")
        print(f"  tf  spread {tfs.max()-tfs.min():.3e} s  = "
              f"{100*(tfs.max()-tfs.min())/np.mean(tfs):.4f} %   std {tfs.std():.3e} s")
        print(f"  pinned: {'YES' if hasattr(os,'sched_getaffinity') else 'NO (not Linux)'}")
        return

    if a.nominal_only:
        t0 = time.perf_counter()
        r1, r2, sc = solve(G["r0"], G["v0"], G["m0"], verbose=a.verbose,
                           warm=(None if BASE_J is None else ("tail", BASE_J)),
                           ratio_test=a.ratio_test, d_margin=a.d_margin, damped=a.damped)
        w = time.perf_counter() - t0
        row = record("nominal", -1, "none", 0.0, G["r0"], G["v0"], G["m0"], r1, r2, sc, w)
        json.dump(row, open(nom_path, "w"), indent=1)
        # THE PLAN ITSELF, IN SI, so every draw can warm-start from it. Stored dimensionally
        # because each draw has its OWN scales -- L = |r0 - r_site| moves with the position
        # perturbation and A = t_max/m_wet with the mass one -- so a non-dimensional plan would
        # have to be rescaled, which is the error-prone way round. In SI it just gets divided by
        # whichever scales the draw happens to have.
        np.savez(os.path.join(OUT, f"sweep_nominal_plan{a.tag}.npz"),
                 r=r2.X[0:3, :] * sc.L, v=r2.X[3:6, :] * sc.v, m=np.exp(r2.X[6, :]),
                 a=r2.U[0:3, :] * sc.A, gamma=r2.U[3, :] * sc.A, tf_s=float(r2.tf_s))
        print(f"nominal: {row['status']}  J {row['J_kg']:.5f} kg  tf {row['tf_s']:.2f} s  "
              f"miss {row['miss_m']:.4f} m  defect {row['defect_m']:.2e} m  "
              f"D {row['D_m']:.4f} m  [{w:.0f} s]")
        print(f"wrote {nom_path}")
        return

    if a.index is None:
        ap.error("pass --index, or --nominal-only, or --collate")
    nom = json.load(open(nom_path)) if os.path.exists(nom_path) else None
    warm = None if a.cold else ("nominal" if BASE_J is None else ("rung", a.tag))
    rows = []
    # THE LADDER. With --alt-km the base state moves onto the nominal at that altitude and every
    # draw perturbs THAT, warm-started from the nominal's own tail. The position span shrinks with
    # altitude: a +/-500 m draw at 800 m altitude is a quarter of the remaining range and would put
    # some draws through the surface, which is a different experiment, not a smaller one.
    for i in range(a.index, min(a.index + a.count, N_TOTAL)):
        ch, lo, hi, unit = CHANNELS[i // N_DRAW]
        draw = i % N_DRAW
        # ONE RNG PER DRAW, SEEDED ON (channel, draw). Any single index can be re-run on its own
        # and reproduce its own perturbation exactly, which a shared stream would not allow.
        rng = np.random.default_rng([BASE_SEED, i // N_DRAW, draw])
        delta = float(rng.uniform(lo, hi))
        r0, v0, m0 = G["r0"].copy(), G["v0"].copy(), float(G["m0"])
        if ch.startswith("r_"):
            r0["xyz".index(ch[2])] += delta
        elif ch.startswith("v_"):
            v0["xyz".index(ch[2])] += delta
        else:
            m0 += delta
        t0 = time.perf_counter()
        r1, r2, sc = solve(r0, v0, m0, warm=warm, verbose=a.verbose,
                           ratio_test=a.ratio_test, d_margin=a.d_margin, damped=a.damped)
        w = time.perf_counter() - t0
        row = record(f"{ch}[{draw}]", i, ch, delta, r0, v0, m0, r1, r2, sc, w)
        rows.append(row)
        dJ = (row["J_kg"] - nom["J_kg"]) if (nom and row["converged"]) else float("nan")
        dtf = (row["tf_s"] - nom["tf_s"]) if (nom and row["converged"]) else float("nan")
        print(f"  {i:>4} {ch:>4} {delta:>+9.3f} {unit:<4} {row['status']:<12} "
              f"J {row['J_kg']:>9.5f} kg  dJ {dJ:>+10.6f}  tf {row['tf_s']:>9.3f} s  "
              f"dtf {dtf:>+9.3f}  miss {row['miss_m']:>7.4f}  def {row['defect_m']:>8.1e}  "
              f"[{w:.0f} s]", flush=True)

    shard = os.path.join(OUT, f"sweep_shard{a.tag}_{a.index:04d}.json")
    json.dump(dict(index=a.index, count=len(rows), rows=rows), open(shard, "w"), indent=1)
    print(f"wrote {shard}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""One PINN training case: pin the CPU, disperse the handover, solve open loop, write the labels.

OPEN LOOP, BY DESIGN. This is not mc_run.py. There is no navigation, no actuator and no re-plan:
one dispersed handover state goes in, one converged SCvx solution comes out, and that solution is
the label. The closed loop is a separate experiment and must not be mixed into this dataset.

WHY THE PIN COMES FIRST. polyhedral_gravity fans every single-point evaluation out over the whole
node via TBB and honours none of OMP_NUM_THREADS, TBB_NUM_THREADS or OMP_THREAD_LIMIT. TBB sizes
its pool ONCE, from the affinity mask in force at the time of the first import, so the mask has to
be narrowed before `gravity` is imported and a later sched_setaffinity does nothing. Unpinned, the
solve also scatters: 0.57 % in fuel run to run, exactly zero when the core is pinned first.

THE HANDOVER ATTITUDE IS A BOUNDARY CONDITION; THE THROTTLE IS NOT. d_init pins only the thrust
DIRECTION at node 0 (subproblem.py). The braking phase delivers an attitude; what throttle to open
at the first node is the descent guidance's own decision.

Pinning the direction is what makes the sub-trajectory labels below legal. d is a state with a slew
constraint, so a tail starting at node j inherits d_j and a sub-problem with d(0) free could
re-point the thrust instantly where the parent could not -- the tail would not be its optimum.
Magnitude is the opposite case: nothing rate-limits Gamma, it may jump between segments, so the
Gamma the parent chose at node j already IS the optimum of the problem restarted there and pinning
it would change no label while making the dataset inconsistent at the handover node.

  direction  dispersed in a cone about the nominal optimal d(0) -- a mission parameter with a
             delivery accuracy, which is what the cone represents
  magnitude  free, a decision variable, an OUTPUT

WHAT IS WRITTEN. The PARENT trajectory only, once. The sub-trajectory expansion is a reindexing,
not new information -- storing all K tails explicitly would repeat the same arc K/2 times over -- so
the loader derives them:

    tail j:   tau'   = (tau_k - tau_j) / (1 - tau_j)      for k = j .. K-1
              tf_j   = tf * (1 - tau_j),   tau_j = j/(K-1)
              tgo    = tf_j * (1 - tau')
              inputs = (r_j, v_j, m_j, d_j)

How far up j may run is a LOADER choice, not a generation one -- the parent is stored whole, so the
cut-off can be changed without re-solving anything. It should be set by measurement: re-solve a
sample of tails from scratch and cut where the inherited tail's fuel departs from the fresh solve's.

That expansion is valid because the plan's dynamics are AUTONOMOUS IN THE ROTATING FRAME -- Coriolis
and centrifugal, no explicit t -- so the principle of optimality transfers. In the inertial frame
the gravity is time-varying and it would not.

THE THRUST LABEL STOPS AT K-2. U[0:3, K-1] is a dummy tied to K-2 and U[3, K-1] is not a real
control, so there is no thrust label at tau = 1.

EVERYTHING IS WRITTEN IN SI. The solver's scales are per-solve -- L = |r0 - r_site| -- so the same
physical state has different non-dimensional coordinates depending on which sample it came from,
and a tail has different scales again from its parent. Non-dimensionalise once, globally, at
training time; never with these.

Usage:  pinn_case.py --make-nominal --out DIR
        pinn_case.py --seed S --nominal DIR/nominal.npz --out DIR
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ---- the pin, before anything imports gravity ------------------------------------------------
# MC_SLOT picks WHICH core when several cases share one Slurm allocation. Indexing into the cpuset
# the process actually got is safer than a hard-coded id: Slurm does not promise 0..n-1.
if hasattr(os, "sched_getaffinity"):                 # Linux only; a no-op elsewhere
    _allowed = sorted(os.sched_getaffinity(0))
    _core = _allowed[int(os.environ.get("MC_SLOT", "0")) % len(_allowed)]
    os.sched_setaffinity(0, {_core})
    print(f"pinned to core {_core} of {len(_allowed)} allowed", flush=True)

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as C                                                      # noqa: E402
import gravity as GR                                                    # noqa: E402
import problem_setup as PS                                              # noqa: E402
import ptr                                                              # noqa: E402
from dynamics import Dynamics                                           # noqa: E402

# The solver settings the dataset is generated under. Fixed here rather than exposed, because every
# sample in a dataset must be solved under identical tolerances -- a case that stopped early is a
# different optimisation problem, and mixing them puts a systematic error in the labels that no
# amount of training can see. tol_vc_m = 1e-4 and tol_obj = 1e-7 are the tightened values; the
# defaults stop the fuel descent about 0.2 % early.
TOL = dict(iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4, tol_vc_m=1e-4)
ALIGN = dict(align_theta0_deg=45.0, align_h_m=100.0)
POINT_DEG = 180.0        # set from --point-deg before any solve; recorded in every case file


def solve(r0, v0, m0, K, d_init=None):
    """Solve from a handover state in SI. Returns (stage1, stage2, scales, P, seed_result).

    THREE SOLVES WHEN THE ATTITUDE IS PINNED, NOT TWO, AND THE THIRD IS THE CHEAP ONE.

    Run cold, the pinned fuel stage settles into a WORSE LOCAL OPTIMUM and never leaves it. Measured
    on seed 7: 6.63516 kg at 1636.63 s, maxiter at 200 iterations and 465 s, with the trust radius
    wide open (eta 5.5e-3 against R 0.90) -- so it was not creeping toward an answer, it was parked.
    The true optimum is 5.61817 kg at 2130.01 s. Every reading taken off the cold chain was wrong by
    18 % in fuel and 500 s in flight time.

    The free problem -- same initial condition, d(0) not pinned -- converges reliably and cheaply,
    and its solution is nearly FEASIBLE for the pinned one: on seed 7 the two handover directions
    differ by 10.56 deg, which the first 21.3 s segment slews out at a quarter of its 2 deg/s
    budget. So it is a seed in the right basin. Started there, the pinned fuel stage converges in
    THREE iterations and 7 s, and lands 0.1 % from the free optimum -- which is the real price of
    holding the handover attitude, against the 18 % the local optimum pretended it was.

    The whole chain is also FASTER than the broken one: 225 s against 465 s spent failing.

    D COMES FROM THE PINNED STAGE 1. It is the achievable landing accuracy of the problem that
    actually carries the attitude constraint; a free stage 1's D would hand the fuel stage a
    tolerance the pinned problem might not be able to meet. The warm start is only the reference the
    linearisation begins from and changes no constraint, so the returned solution is a solution of
    the pinned problem -- which is checked, not assumed: the caller verifies d(0) against d_init.
    """
    P = PS.build(r0_override=r0, v0_override=v0, m_wet=m0)
    sc = P["scales"]
    # PLAN GRAVITY IS PINN-GM-III, the same field the flight guidance plans on. The network is
    # being trained to reproduce what the solver produces in flight, so the labels have to carry
    # the solver's own gravity, not the polyhedral truth.
    dyn = Dynamics(sc, field=GR.pinn_field())
    cm = dict(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=P["n_hat"],
              K=K, verbose=False, slew_max_deg_s=C.SLEW_MAX_DEG_S,
              term_point_deg=C.TERMINAL_POINT_DEG, term_point_nodes=C.TERMINAL_POINT_NODES,
              point_deg=POINT_DEG, **ALIGN, **TOL)

    if d_init is None:                                                  # nothing to seed around
        r1 = ptr.run(mode="error", warm=None, d_init=None, **cm)
        r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L,
                     warm=(r1.X, r1.U, r1.s), d_init=None, **cm)
        return r1, r2, sc, P, None

    # FREE FIRST, THEN PINNED, AT BOTH STAGES. The free chain is the seed and nothing else; every
    # pinned solve starts from its free counterpart.
    #
    # Seeding the LANDING-ERROR stage matters for a different reason than seeding the fuel stage.
    # Stage 1 can never be infeasible -- it minimises the miss, so it always has an answer, and a
    # large D only means the site is out of reach. So a maxiter there and a genuinely unreachable
    # initial condition produce the SAME rejection, and they cannot be told apart: seed 3 reported
    # D = 2634 m on one cold run and D = 2255 m on another, neither converged, 380 m apart. If some
    # cases are dropped because the solve stalled rather than because the physics says no, the
    # surviving dataset is biased toward initial conditions that happen to be easy to solve, and
    # that bias is invisible in the data and would be learned. A converged D makes the reject
    # defensible either way.
    f1 = ptr.run(mode="error", warm=None, d_init=None, **cm)
    r1 = ptr.run(mode="error", warm=(f1.X, f1.U, f1.s), d_init=d_init, **cm)   # the D that governs
    f2 = ptr.run(mode="fuel", D_nd=f1.D_nd + 0.1 / sc.L,
                 warm=(f1.X, f1.U, f1.s), d_init=None, **cm)
    r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L,                # pinned, in the right basin
                 warm=(f2.X, f2.U, f2.s), d_init=d_init, **cm)
    return r1, r2, sc, P, f2


def si(X, U, sc, K):
    """The solution in SI: r [m], v [m/s], m [kg], d [-], T [N]. T has K-1 entries, not K."""
    r = (X[0:3, :] * sc.L).T
    v = (X[3:6, :] * sc.v).T
    m = np.exp(X[6, :])
    d = X[7:10, :].T
    T = U[3, 0:K - 1] * sc.A * m[0:K - 1]        # Gamma_nd * A = thrust acceleration; x m = N
    return r, v, m, d, T


def write_manifest(a, out, nom_info):
    """What this dataset IS, written beside it. A directory of npz files that cannot say how it was
    generated is unusable three weeks later, and we now have more than one of them."""
    import hashlib, json, datetime
    h = hashlib.sha1()
    for f in sorted(("pinn_case.py", "ptr.py", "subproblem.py", "dynamics.py",
                     "problem_setup.py", "gravity.py", "config.py")):
        fp = os.path.join(HERE, f)
        if os.path.exists(fp):
            h.update(open(fp, "rb").read())
    man = dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        code_sha1=h.hexdigest()[:12], K=a.K,
        dispersion=dict(pos_sigma_m=a.pos_sigma, vel_sigma_ms=a.vel_sigma,
                        mass_sigma_kg=a.mass_sigma, point_sigma_deg=a.point_sigma_deg),
        constraints=dict(point_deg=float(a.point_deg),
                         point_deg_note=("DISABLED" if a.point_deg >= 179.99 else "active"),
                         glideslope_deg=C.GLIDESLOPE_DEG,
                         terminal_point_deg=C.TERMINAL_POINT_DEG,
                         terminal_point_nodes=C.TERMINAL_POINT_NODES,
                         slew_max_deg_s=C.SLEW_MAX_DEG_S, vtouch=C.VTOUCH,
                         tmin=C.TMIN, tmax=C.TMAX, align=ALIGN),
        solver=dict(tol=TOL, chain="free s1 -> pinned s1 (D) -> free s2 -> pinned s2 (seeded)",
                    plan_gravity="pinn-gm-iii", handover="attitude pinned by d_init, throttle free"),
        gates=dict(d_max_m=a.d_max, dnorm=[0.999, 1.001], d0_err_deg=1e-3),
        nominal=nom_info)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(f"wrote {os.path.join(out, 'manifest.json')}")


def make_nominal(a):
    """The mission parameters: the handover attitude and throttle the braking phase must deliver.

    Solved with neither a_init nor d_init, which is exactly what that case is for -- subproblem.py
    says the free handover thrust is INFORMATION FOR THE UPSTREAM PHASE rather than a constraint.
    This is the one solve in the whole dataset that chooses its own handover direction; every
    dispersed case is then required to start from a perturbation of it.
    """
    P0 = PS.build(altitude_km=C.ALTITUDE_KM, cone_deg=0.0)
    t0 = time.perf_counter()
    r1, r2, sc, P, _ = solve(P0["r0"], P0["v0"], C.MWET, a.K, d_init=None)
    if r2.status != "converged":
        raise SystemExit(f"nominal solve did not converge: {r2.status}")
    r, v, m, d, T = si(r2.X, r2.U, sc, a.K)
    out = os.path.join(a.out, "nominal.npz")
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(out, X=r2.X, U=r2.U, s=r2.s, tf=r2.tf_s, fuel=r2.fuel_kg,
                        r0=P0["r0"], v0=P0["v0"], m0=C.MWET,
                        d0=d[0], T0=T[0], K=a.K,
                        r_site=P["r_site"], n_hat=P["n_hat"])
    print(f"nominal: {r2.status} in {r2.iters} it, fuel {r2.fuel_kg:.5f} kg, tf {r2.tf_s:.2f} s, "
          f"[{time.perf_counter()-t0:.0f}s]")
    print(f"  handover attitude d0 = [{d[0][0]:+.6f} {d[0][1]:+.6f} {d[0][2]:+.6f}]")
    print(f"  (its own chosen throttle was {T[0]:.4f} N; NOT imposed on the cases)")
    print(f"wrote {out}")
    write_manifest(a, a.out, dict(tf_s=float(r2.tf_s), fuel_kg=float(r2.fuel_kg),
                                  d0=[float(x) for x in d[0]], T0_N=float(T[0]),
                                  status=str(r2.status), iters=int(r2.iters)))


def draw(a, nom):
    """The dispersed handover state. One rng, seeded explicitly, recorded in the output."""
    rng = np.random.default_rng(a.seed)
    r0 = np.asarray(nom["r0"], float) + a.pos_sigma * rng.standard_normal(3)
    v0 = np.asarray(nom["v0"], float) + a.vel_sigma * rng.standard_normal(3)
    m0 = float(nom["m0"]) + a.mass_sigma * rng.standard_normal()

    # DIRECTION: a cone about the nominal attitude. Gaussian in the two directions tangent to it,
    # which is the small-angle form of a pointing error -- not uniform on the sphere, because the
    # braking phase delivers an attitude and this is its delivery accuracy.
    d_nom = np.asarray(nom["d0"], float); d_nom /= np.linalg.norm(d_nom)
    t1 = np.cross(d_nom, [0.0, 0.0, 1.0])
    if np.linalg.norm(t1) < 1e-8:
        t1 = np.cross(d_nom, [1.0, 0.0, 0.0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(d_nom, t1)
    s = np.radians(a.point_sigma_deg) * rng.standard_normal(2)
    d0 = d_nom + s[0] * t1 + s[1] * t2
    d0 /= np.linalg.norm(d0)
    return r0, v0, m0, d0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-nominal", action="store_true",
                    help="solve the undispersed case and write the mission parameters")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--seed", type=int, help="dispersion seed; recorded in the output")
    ap.add_argument("--nominal", help="path to nominal.npz from --make-nominal")
    # The handover dispersion. 1000 m is 4.86 deg off the local vertical at the 11.771 km handover,
    # well inside the 30 deg glideslope even at 3 sigma, so the draw itself never makes node 0
    # infeasible. Every value is recorded in each case file.
    ap.add_argument("--pos-sigma", dest="pos_sigma", type=float, default=1000.0,
                    help="handover position 1-sigma [m]")
    ap.add_argument("--vel-sigma", dest="vel_sigma", type=float, default=1.0,
                    help="handover velocity 1-sigma [m/s]")
    ap.add_argument("--mass-sigma", dest="mass_sigma", type=float, default=2.0,
                    help="handover mass 1-sigma [kg]")
    ap.add_argument("--point-sigma-deg", dest="point_sigma_deg", type=float, default=3.0,
                    help="handover pointing 1-sigma [deg], per tangent axis")
    ap.add_argument("--d-max", dest="d_max", type=float, default=1e-3,
                    help="reject a case whose stage-1 miss exceeds this [m]")
    # THE GLOBAL POINTING CONE, OFF BY DEFAULT. 180 deg makes n.d >= cos(180) = -1, satisfied
    # identically by any unit vector, so the constraint drops out rather than being special-cased.
    #
    # It was C.POINTING_DEG = 140 and it was the root of 59 of the first run's 60 failures. Measured
    # on 20 seeds, two levels: where the cone binds (27.5 % of trajectories press exactly on it) the
    # optimum wants 150-177 deg, and holding it at 140 cost 4.8 % of the fuel on median and 10.4 %
    # at worst, with 6-30x more iterations. Seven of eight failing seeds converge in 6-9 iterations
    # with it off, against 400 and "infeasible" with it on. Where it does not bind, removing it is a
    # no-op: fuel within 0.08 %, same iteration count.
    #
    # It was never a hardware limit. The landing attitude is set by the terminal pointing ramp and
    # the slew rate, not by this.
    ap.add_argument("--point-deg", dest="point_deg", type=float, default=180.0,
                    help="global thrust pointing cone [deg]; 180 disables it")
    a = ap.parse_args()

    global POINT_DEG
    POINT_DEG = float(a.point_deg)
    if a.make_nominal:
        return make_nominal(a)

    missing = [n for n in ("seed", "nominal") if getattr(a, n) is None]
    if missing:
        ap.error("missing required argument(s): " + ", ".join("--" + m.replace("_", "-")
                                                              for m in missing))

    nom = np.load(a.nominal)
    if int(nom["K"]) != a.K:
        ap.error(f"nominal was solved with K={int(nom['K'])}, this case asks for K={a.K}")
    os.makedirs(a.out, exist_ok=True)
    f_out = os.path.join(a.out, f"case_{a.seed:06d}.npz")
    if os.path.exists(f_out):
        print(f"already have {f_out}, nothing to do")
        return

    r0, v0, m0, d0 = draw(a, nom)
    print(f"seed {a.seed}: |dr| {np.linalg.norm(r0-nom['r0']):.3f} m  "
          f"|dv| {np.linalg.norm(v0-nom['v0']):.5f} m/s  dm {m0-float(nom['m0']):+.3f} kg  "
          f"dpoint {np.degrees(np.arccos(np.clip(d0@nom['d0'],-1,1))):.4f} deg", flush=True)

    t0 = time.perf_counter()
    r1, r2, sc, P, f2 = solve(r0, v0, m0, a.K, d_init=d0)   # attitude imposed, throttle free
    wall = time.perf_counter() - t0

    D_m = float(r1.D_nd) * sc.L
    print(f"  stage1 {r1.status} {r1.iters:>3}it  D {D_m:.4f} m   "
          f"seed(free) {f2.status} {f2.iters:>3}it {f2.fuel_kg:.5f} kg   "
          f"stage2 {r2.status} {r2.iters:>3}it  fuel {r2.fuel_kg:.5f} kg  tf {r2.tf_s:.2f} s  "
          f"[{wall:.0f}s]", flush=True)

    # QUALITY GATES. A case that failed any of these is not a label, and keeping it would put a
    # systematic error in the dataset that training cannot detect.
    reject = []
    if r1.status != "converged":
        reject.append(f"stage1 {r1.status}")
    if r2.status != "converged":
        reject.append(f"stage2 {r2.status}")
    _d_end = r2.X[7:10, 0] / max(float(np.linalg.norm(r2.X[7:10, 0])), 1e-12)
    _d_err = float(np.degrees(np.arccos(np.clip(float(_d_end @ d0), -1.0, 1.0))))
    if _d_err > 1e-3:
        # The fuel stage is seeded from a solution that does NOT satisfy the pin, so this is the
        # one gate that proves the seeding did not quietly carry the free answer through.
        reject.append(f"handover attitude not honoured, d(0) off by {_d_err:.6f} deg")
    if not (0.999 < r2.dnorm_lo and r2.dnorm_hi < 1.001):
        # ||d|| is conserved by the kinematics and pinned at node 0, so a drift away from 1 means
        # the thrust being delivered is not the Gamma being charged.
        reject.append(f"||d|| in [{r2.dnorm_lo:.4f}, {r2.dnorm_hi:.4f}]")
    if D_m > a.d_max:
        reject.append(f"cannot reach the site, D = {D_m:.6f} m > {a.d_max:g} m")
    if reject:
        print("  REJECTED: " + "; ".join(reject))
        np.savez_compressed(f_out.replace(".npz", ".rejected.npz"),
                            seed=a.seed, reason="; ".join(reject), r0=r0, v0=v0, m0=m0, d0=d0)
        return

    r, v, m, d, T = si(r2.X, r2.U, sc, a.K)
    tau = np.arange(a.K) / (a.K - 1)
    np.savez_compressed(
        f_out,
        # the labels, SI, parent trajectory only -- tails are a reindexing, see the module docstring
        tau=tau, r=r, v=v, m=m, d=d, T=T, tf=r2.tf_s,
        # the handover that produced them
        r0=r0, v0=v0, m0=m0, d0=d0,
        # provenance and quality
        seed=a.seed, K=a.K, D=D_m, d_max=a.d_max, fuel=r2.fuel_kg, miss_m=r2.miss_m,
        fuel_free=f2.fuel_kg, tf_free=f2.tf_s, d0_err_deg=_d_err,
        defect_m=r2.defect_m, dnorm_lo=r2.dnorm_lo, dnorm_hi=r2.dnorm_hi,
        iters1=r1.iters, iters2=r2.iters, wall_s=wall,
        pos_sigma=a.pos_sigma, vel_sigma=a.vel_sigma, mass_sigma=a.mass_sigma,
        point_sigma_deg=a.point_sigma_deg,
        r_site=P["r_site"], n_hat=P["n_hat"],
        plan_gravity="pinn", tol=repr(TOL), align=repr(ALIGN), point_deg=POINT_DEG)
    print(f"  wrote {f_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""One Monte Carlo case: pin the CPU, then run closed_loop with the given dispersion.

WHY THE PIN COMES FIRST. polyhedral_gravity fans every single-point evaluation out over the whole
node via TBB and honours none of OMP_NUM_THREADS, TBB_NUM_THREADS or OMP_THREAD_LIMIT -- measured
identical ~1030 s CPU under all three. TBB sizes its pool ONCE, from the affinity mask in force at
the time of the first import, so the mask has to be narrowed before `gravity` is imported and a
later sched_setaffinity does nothing. Pinned is also FASTER per solve (830 s vs 1030 s CPU) because
the per-call thread launch disappears.

Slurm with --cpus-per-task=1 usually sets the mask already; this makes it explicit and independent
of the site's task/affinity configuration, and it is a no-op where the mask is already one core.

TWO INDEPENDENT DRAWS PER CASE. `seed` sets WHERE THE VEHICLE ACTUALLY IS at handover (the truth
dispersion). `nav_seed` sets WHAT THE FILTER BELIEVES and what the sensors report: the OD error
vector, the IMU bias, and every measurement noise sample. Both must vary from case to case -- with
nav_seed frozen, a level at pos_sigma = 0 is the SAME RUN repeated and its spread is zero, which is
what the first 8 cases of job 860300 showed. nav_seed is offset into a disjoint range so the two
default_rng streams cannot line up and correlate the OD error with the truth dispersion.

BOTH THE GUIDANCE AND THE FILTER RUN PINN-GM-III, and the measurement that settled it. Planning on
the exact 49k field was tried: it costs 513 s per guidance cycle against 5.2 s, because packing
cases onto a node pins each to one core and polyhedral_gravity's speed comes from fanning out over
the whole node. What it buys is bounded by what the guidance's gravity error actually costs, which
is NOT the solver's reported miss_plan_m -- that is a constraint residual the solver drives to zero
whatever field it uses, and its terminal node sits on the datum plane, not the ground. Measured
properly, by flying the same state and thrust under each model against the polyhedral truth, the
PINN displaces the trajectory by 0.028-0.059 m over the 60 s a plan is flown before being re-solved.
And flown with PERFECT navigation the guidance lands within 0.090 m, against 2.69 m with the filter.
So exact-gravity planning would spend 100x the compute on a term already inside a 0.09 m error, in
a loop limited at 2.7 m. Set MC_OUT to keep a configuration change out of an existing dataset.

Usage: mc_run.py <pos_sigma_m> <vel_sigma_ms> <seed> [extra closed_loop args...]
"""
import os
import sys

NAV_SEED_BASE = 70000     # keeps the nav stream disjoint from the dispersion stream

# MC_SLOT picks WHICH core when several cases share one Slurm allocation (mc_array.slurm packs 48
# of them onto a node). Indexing into the cpuset the process actually got is safer than taskset -c
# with a hard-coded number: Slurm does not promise the allocated core ids are 0..n-1.
if hasattr(os, "sched_getaffinity"):                 # Linux only; a no-op elsewhere
    allowed = sorted(os.sched_getaffinity(0))
    core = allowed[int(os.environ.get("MC_SLOT", "0")) % len(allowed)]
    os.sched_setaffinity(0, {core})
    print(f"pinned to core {core} of {len(allowed)} allowed", flush=True)

pos, vel, seed = sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
nav_seed = str(NAV_SEED_BASE + int(seed))
tag = f"_mc_p{float(pos):g}_v{float(vel):g}_s{seed}"   # pos AND vel: (10, 0.01) and (10, 0) are different cases

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = ["closed_loop.py",
            "--nav", "--plan-gravity", "pinn", "--nav-gravity", "pinn",
            "--commit-alt", "5.0", "--tol-obj", "1e-3", "--trace-stride", "10",
            "--from-nominal",                         # disperse around the braking+TCM handover
            "--nav-seed", nav_seed,                   # independent navigation realisation
            "--seed", seed, "--pos-sigma", pos, "--vel-sigma", vel,
            "--out", os.path.join(HERE, "Figures&data", os.environ.get("MC_OUT", "mc")),
            "--tag", tag] + extra

import closed_loop                                    # noqa: E402  (after the pin)
closed_loop.main()

#!/usr/bin/env python
"""Step-size convergence for the truth propagator on a parking orbit, by step halving.

No reference trajectory is used. The arc is propagated at dt and again at dt/2, and the difference
between the two is taken as the error estimate at dt. For a scheme of order p the finer run carries
2^-p of the coarser run's error, so the difference is (1 - 2^-p) of it -- 94% for RK4 -- which makes
it a good estimate of the coarse run's error and, unlike a fixed reference, it costs nothing extra:
every propagation in the ladder serves as the fine partner of the one above it and the coarse
partner of the one below.

The ladder is a fixed set of round step sizes, 1024 s down to 2 s by halving, walked from COARSE to
FINE so the cheap runs come first. Every step is EXACTLY the size requested; tf is not a multiple of
these steps, so the propagator's termination condition trims only the final step, the one that would
overshoot the end of the arc.

Inertial frame, per the frame study. Polyhedral gravity, no thrust, no SRP, no third body.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VALID = os.path.join(HERE, "..", "..")
sys.path.insert(0, VALID)

import truth_dynamics as TD                                          # noqa: E402


def circular_inertial(radius_km, seed, gm=TD.EROS_GM, mass=600.0):
    """A randomly oriented circular orbit in the INERTIAL frame. Seeded; the seed is recorded."""
    rng = np.random.default_rng(seed)
    R = radius_km * 1e3
    u = rng.normal(size=3); u /= np.linalg.norm(u)
    w = rng.normal(size=3); w -= (w @ u) * u; w /= np.linalg.norm(w)
    v = np.sqrt(gm / R)
    period = 2.0 * np.pi * np.sqrt(R ** 3 / gm)
    return np.concatenate([R * u, v * w, [mass]]), period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--radius-km", type=float, default=35.0)
    ap.add_argument("--orbits", type=float, default=1.5)
    ap.add_argument("--scheme", default="rk4")
    ap.add_argument("--gravity", default="poly", choices=["poly", "pointmass", "pinn"])
    ap.add_argument("--mesh-q", type=int, default=64)
    ap.add_argument("--dt-max", type=float, default=1024.0, help="coarsest step")
    ap.add_argument("--dt-min", type=float, default=2.0, help="finest step allowed")
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    y0, period = circular_inertial(a.radius_km, a.seed)
    tf = a.orbits * period
    g = TD.Gravity(a.gravity, mesh_q=a.mesh_q)
    thrust = TD.coast()
    dts = []
    d = a.dt_max
    while d >= a.dt_min - 1e-12:
        dts.append(d); d /= 2.0

    print(f"seed {a.seed}, R = {a.radius_km:g} km, period {period:.1f} s, "
          f"{a.orbits:g} orbits -> tf = {tf:.1f} s")
    print(f"frame inertial, scheme {a.scheme.upper()}, gravity {a.gravity}"
          + (f" {getattr(g, 'faces', 0):,} facets" if getattr(g, "faces", None) else ""))
    print(f"ladder {a.dt_max:g} s down to {a.dt_min:g} s by halving; every step is exactly dt, "
          f"only the final step is trimmed\n")

    print(f"{'dt [s]':>10}{'last step':>11}{'steps':>9}{'evals':>10}{'cpu [s]':>9}"
          f"{'|dr| [m]':>13}{'order':>8}")
    Y, cpu, evals, steps, used, hact = [], [], [], [], [], []
    errs = []
    broke = None
    stop_reason = f"ran the full ladder down to dt_min = {a.dt_min:g} s"
    for i, dt in enumerate(dts):
        g.reset()
        t0 = time.perf_counter()
        y, n = TD.propagate_fixed(y0, 0.0, tf, dt, a.scheme, g, thrust, frame="inertial")
        c = time.perf_counter() - t0
        Y.append(y); cpu.append(c); evals.append(g.n); steps.append(n); used.append(dt)
        last = tf - dt * (n - 1)              # the trimmed final step
        hact.append(last)
        line = f"{dt:10.2f}{last:11.3f}{n:9d}{g.n:10d}{c:9.2f}"
        if i == 0:
            print(line + f"{'':13}{'':8}"); continue
        e = float(np.linalg.norm(Y[i - 1][0:3] - Y[i][0:3]))
        errs.append(e)
        o = float(np.log2(errs[-2] / e)) if len(errs) >= 2 and e > 0 else float("nan")
        print(line + f"{e:13.3e}" + ("" if np.isnan(o) else f"{o:8.2f}"))

    rows = []
    for i in range(len(errs)):
        dr = errs[i]
        dv = float(np.linalg.norm(Y[i][3:6] - Y[i + 1][3:6]))
        rows.append(dict(dt=used[i], last_step=hact[i], steps=steps[i], evals=evals[i],
                         cpu=cpu[i], dr=dr, dv=dv, order=float("nan")))
    for i in range(1, len(rows)):
        if rows[i]["dr"] > 0:
            rows[i - 1]["order"] = float(np.log2(rows[i - 1]["dr"] / rows[i]["dr"]))
    print(f"\nstopped: {stop_reason}")

    print(f"\n{'dt [s]':>12}{'|dr| [m]':>14}{'|dv| [m/s]':>14}{'order':>9}{'cpu [s]':>10}")
    for r in rows:
        o = "" if np.isnan(r["order"]) else f"{r['order']:9.2f}"
        print(f"{r['dt']:12.4f}{r['dr']:14.3e}{r['dv']:14.3e}{o:>9}{r['cpu']:10.2f}")

    fin = [r for r in rows if np.isfinite(r["order"])]
    if fin:
        floor = 10.0 * min(r["dr"] for r in rows if r["dr"] > 0)
        ok = [r for r in fin if r["order"] > 3.0 and r["dr"] > floor]
        if len(ok) >= 2:
            p = np.polyfit(np.log([r["dt"] for r in ok]), np.log([r["dr"] for r in ok]), 1)[0]
            print(f"\nfitted order {p:.2f} over dt {min(r['dt'] for r in ok):.2f}"
                  f"-{max(r['dt'] for r in ok):.2f} s ({len(ok)} points)")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"parking_benchmark_{a.scheme}_{a.radius_km:g}km.json")
    json.dump(dict(phase="parking", frame="inertial", scheme=a.scheme, seed=a.seed,
                   radius_km=a.radius_km, orbits=a.orbits, period=period, tf=tf,
                   gravity=a.gravity, facets=getattr(g, "faces", None),
                   method="step halving: error(dt) = |y(dt) - y(dt/2)|",
                   dt_max=a.dt_max, dt_min=a.dt_min, stop_reason=stop_reason,
                   step_note="every step is exactly dt; only the final step is trimmed to land on tf",
                   rows=rows), open(f, "w"), indent=1)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()

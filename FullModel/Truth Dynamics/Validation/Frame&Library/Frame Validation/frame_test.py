#!/usr/bin/env python
"""Inertial vs body-fixed integration: do they agree, and what does each one pay?

The polyhedral field is defined in the BODY frame, so the frame choice decides what extra algebra
each stage carries:

  inertial   the state is inertial, the field is not. Each stage rotates the position into the body
             frame, evaluates, and rotates the acceleration back out: 2 matrix-vector products per
             stage, plus one sine/cosine pair for the angle.

  rotating   the position is already in the body frame, so no rotation is needed, but the frame is
             non-inertial and the stage owes Coriolis and centrifugal terms:
             -2 w x v  and  -w x (w x r), which is 3 matrix-vector products per stage.

So the inertial frame trades 3 cheap products for 2 cheap products, and neither is obviously better
a priori. Both are measured here against the cost of the gravity call itself, which is the only term
that plausibly matters.

Two questions, and the correctness one comes first: a trajectory propagated in one frame and
converted must equal the trajectory propagated in the other. If that fails, no timing is meaningful.

Gravity defaults to a POINT MASS, for two reasons. It is cheap, so the frame algebra stops being
buried under a 355 us field evaluation and the difference being asked about becomes measurable. And
on a circular orbit it has a closed form, so both frames can be checked against analytic truth
rather than only against each other.

A point mass is spherically symmetric and would strictly need no rotation at all. The rotate-in /
rotate-out is kept in the inertial path regardless, because it represents what the body-fixed
polyhedral field demands -- that cost is the thing being priced, and removing it would measure a
no-op. Run with --gravity poly for what the ratio looks like in production.

Simple dynamics only: no SRP, no third body, no thrust.
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
SCVX = os.path.join(HERE, "..", "..", "..", "..", "scvx_landing")
sys.path.insert(0, SCVX)

EROS_ROT_PERIOD = 5.270 * 3600.0
EROS_GM = 4.4665e5
WZ = 2.0 * np.pi / EROS_ROT_PERIOD
OMEGA = np.array([0.0, 0.0, WZ])
WX = np.array([[0.0, -WZ, 0.0], [WZ, 0.0, 0.0], [0.0, 0.0, 0.0]])


# --- frame algebra --------------------------------------------------------------------------------
def Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def inertial_to_body(r_I, v_I, t):
    """Body frame is rotated by +wt from inertial at time t (aligned at t = 0)."""
    R = Rz(-WZ * t)
    r_B = R @ r_I
    v_B = R @ v_I - np.cross(OMEGA, r_B)
    return r_B, v_B


def body_to_inertial(r_B, v_B, t):
    R = Rz(WZ * t)
    return R @ r_B, R @ (v_B + np.cross(OMEGA, r_B))


# --- right-hand sides -----------------------------------------------------------------------------
class Timer:
    """Accumulates time inside the frame algebra and inside the gravity call separately.

    perf_counter costs order 100 ns per call against rotation work of order 1 us, so the frame
    figure carries a few percent of timer overhead. It is reported as an upper bound, and the
    microbenchmark at the end gives the same quantity without instrumentation as a cross-check.
    """

    def __init__(self):
        self.frame = 0.0
        self.grav = 0.0
        self.n = 0


def eom_inertial(t, y, gfun, tm: Timer):
    r_I, v_I = y[0:3], y[3:6]
    t0 = time.perf_counter()
    th = WZ * t
    c, s = np.cos(th), np.sin(th)
    r_B = np.array([c * r_I[0] + s * r_I[1], -s * r_I[0] + c * r_I[1], r_I[2]])
    tm.frame += time.perf_counter() - t0

    t0 = time.perf_counter()
    g_B = gfun(r_B)
    tm.grav += time.perf_counter() - t0
    tm.n += 1

    t0 = time.perf_counter()
    a_I = np.array([c * g_B[0] - s * g_B[1], s * g_B[0] + c * g_B[1], g_B[2]])
    tm.frame += time.perf_counter() - t0
    return np.concatenate([v_I, a_I])


def eom_rotating(t, y, gfun, tm: Timer):
    r_B, v_B = y[0:3], y[3:6]
    t0 = time.perf_counter()
    g_B = gfun(r_B)
    tm.grav += time.perf_counter() - t0
    tm.n += 1

    t0 = time.perf_counter()
    a_B = g_B - 2.0 * (WX @ v_B) - WX @ (WX @ r_B)
    tm.frame += time.perf_counter() - t0
    return np.concatenate([v_B, a_B])


def rk4(f, y0, t0, tf, dt):
    n = max(1, int(np.ceil((tf - t0) / dt)))
    h = (tf - t0) / n
    y, t = np.asarray(y0, float).copy(), t0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y, n


# --- setup ----------------------------------------------------------------------------------------
def circular_analytic(y0_I, t):
    """Exact inertial solution for a point mass on the circular orbit built by random_orbit().

    random_orbit places r0 = R u and v0 = sqrt(mu/R) w with u perpendicular to w, which is a closed
    circular orbit, so the motion is uniform rotation in the (u, w) plane at n = sqrt(mu/R^3). No
    Kepler solve is needed and the result carries no numerical error of its own.
    """
    r0, v0 = y0_I[0:3], y0_I[3:6]
    R = np.linalg.norm(r0)
    u = r0 / R
    w = v0 / np.linalg.norm(v0)
    n = np.sqrt(EROS_GM / R ** 3)
    c, s = np.cos(n * t), np.sin(n * t)
    return np.concatenate([R * (c * u + s * w), R * n * (-s * u + c * w)])


def random_orbit(seed, radius_km, rng=None):
    """A randomly oriented, near-circular INERTIAL orbit. Seeded, and the seed is reported."""
    rng = rng or np.random.default_rng(seed)
    R = radius_km * 1e3
    u = rng.normal(size=3); u /= np.linalg.norm(u)              # position direction
    w = rng.normal(size=3); w -= (w @ u) * u; w /= np.linalg.norm(w)   # in-plane, perpendicular
    v = np.sqrt(EROS_GM / R)
    return np.concatenate([R * u, v * w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--radius-km", type=float, default=35.0)
    ap.add_argument("--dt", type=float, default=60.0, help="deliberately coarse")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--mesh-q", type=int, default=64, help="64 -> the 49 152-facet model")
    ap.add_argument("--gravity", default="pointmass", choices=["pointmass", "poly"])
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()

    if a.gravity == "pointmass":
        nfaces = None
        def gfun(r):
            rn = np.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
            return -EROS_GM * r / (rn * rn * rn)
    else:
        from scvx.gravity import load_mesh, make_polyhedral
        from mesh_lod import density_for_gm
        mesh = load_mesh()
        nfaces = len(mesh.faces)
        ev = make_polyhedral(mesh, density=density_for_gm(mesh, gm=EROS_GM))
        gfun = lambda r: np.asarray(ev(r)[1], float)

    tf = a.hours * 3600.0
    y0_I = random_orbit(a.seed, a.radius_km)
    r_B0, v_B0 = inertial_to_body(y0_I[0:3], y0_I[3:6], 0.0)
    y0_B = np.concatenate([r_B0, v_B0])

    print(f"seed {a.seed}, R = {a.radius_km:g} km, dt = {a.dt:g} s, span = {tf:.0f} s "
          f"({a.hours:g} h), gravity " + (f"polyhedral {nfaces:,} facets" if nfaces
                                          else "point mass (exact circular solution available)"))
    print(f"round-trip of the initial state: "
          f"{np.linalg.norm(np.concatenate(body_to_inertial(r_B0, v_B0, 0.0)) - y0_I):.3e}\n")

    tm_I, tm_B = Timer(), Timer()
    t0 = time.perf_counter()
    yI, nI = rk4(lambda t, y: eom_inertial(t, y, gfun, tm_I), y0_I, 0.0, tf, a.dt)
    cpu_I = time.perf_counter() - t0
    t0 = time.perf_counter()
    yB, nB = rk4(lambda t, y: eom_rotating(t, y, gfun, tm_B), y0_B, 0.0, tf, a.dt)
    cpu_B = time.perf_counter() - t0

    yB_as_I = np.concatenate(body_to_inertial(yB[0:3], yB[3:6], tf))
    yI_as_B = np.concatenate(inertial_to_body(yI[0:3], yI[3:6], tf))

    # Averaged over every evaluation in the propagation, which is the figure that matters: a total
    # depends on the arc length, a per-evaluation average does not and can be multiplied by whatever
    # stage count a given integrator has.
    print(f"{'frame':>10}{'steps':>8}{'evals':>8}{'evals/step':>11}{'total [s]':>11}"
          f"{'frame us/eval':>15}{'grav us/eval':>14}{'frame %':>9}")
    for nm, n, tm, cpu in (("inertial", nI, tm_I, cpu_I), ("rotating", nB, tm_B, cpu_B)):
        print(f"{nm:>10}{n:8d}{tm.n:8d}{tm.n/n:11.1f}{cpu:11.4f}"
              f"{1e6*tm.frame/tm.n:15.3f}{1e6*tm.grav/tm.n:14.3f}{100*tm.frame/cpu:9.2f}")

    y_exact = circular_analytic(y0_I, tf) if a.gravity == "pointmass" else None
    dr = float(np.linalg.norm(yB_as_I[0:3] - yI[0:3]))
    dv = float(np.linalg.norm(yB_as_I[3:6] - yI[3:6]))
    dr2 = float(np.linalg.norm(yI_as_B[0:3] - yB[0:3]))
    print(f"\nagreement after {tf:.0f} s")
    print(f"  rotating -> inertial vs inertial : dr {dr:.4e} m   dv {dv:.4e} m/s")
    print(f"  inertial -> rotating vs rotating : dr {dr2:.4e} m")
    print(f"  final |r| inertial {np.linalg.norm(yI[0:3])/1e3:.6f} km, "
          f"rotating {np.linalg.norm(yB[0:3])/1e3:.6f} km")
    if y_exact is not None:
        eI = float(np.linalg.norm(yI[0:3] - y_exact[0:3]))
        eB = float(np.linalg.norm(yB_as_I[0:3] - y_exact[0:3]))
        print(f"\nagainst the EXACT circular solution (point mass, no numerical reference)")
        print(f"  inertial : {eI:.4e} m")
        print(f"  rotating : {eB:.4e} m   (converted to inertial first)")

    # cross-check the frame cost without instrumentation
    N = 200000
    r = y0_I[0:3].copy(); v = y0_I[3:6].copy()
    t0 = time.perf_counter()
    for i in range(N):
        th = WZ * i; c, s = np.cos(th), np.sin(th)
        _ = np.array([c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]])
        _ = np.array([c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]])
    us_I = 1e6 * (time.perf_counter() - t0) / N
    t0 = time.perf_counter()
    for _ in range(N):
        _ = v - 2.0 * (WX @ v) - WX @ (WX @ r)
    us_B = 1e6 * (time.perf_counter() - t0) / N
    g0 = time.perf_counter()
    for i in range(200):
        gfun(r * (1.0 + 1e-9 * i))
    us_g = 1e6 * (time.perf_counter() - g0) / 200
    print(f"\nper stage, uninstrumented microbenchmark")
    print(f"  inertial frame algebra : {us_I:8.3f} us")
    print(f"  rotating frame algebra : {us_B:8.3f} us")
    print(f"  one gravity evaluation : {us_g:8.3f} us   "
          f"({us_g/max(us_I,us_B):.0f}x the larger frame cost)")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "frame_test.json")
    json.dump(dict(seed=a.seed, radius_km=a.radius_km, dt=a.dt, span_s=tf,
                   gravity=a.gravity, facets=nfaces,
                   inertial=dict(steps=nI, evals=tm_I.n, evals_per_step=tm_I.n/nI, cpu=cpu_I,
                                 grav=tm_I.grav, frame=tm_I.frame,
                                 frame_us_per_eval=1e6*tm_I.frame/tm_I.n,
                                 grav_us_per_eval=1e6*tm_I.grav/tm_I.n),
                   rotating=dict(steps=nB, evals=tm_B.n, evals_per_step=tm_B.n/nB, cpu=cpu_B,
                                 grav=tm_B.grav, frame=tm_B.frame,
                                 frame_us_per_eval=1e6*tm_B.frame/tm_B.n,
                                 grav_us_per_eval=1e6*tm_B.grav/tm_B.n),
                   agree=dict(dr=dr, dv=dv, dr_reverse=dr2),
                   exact=(None if y_exact is None else dict(
                       inertial=float(np.linalg.norm(yI[0:3] - y_exact[0:3])),
                       rotating=float(np.linalg.norm(yB_as_I[0:3] - y_exact[0:3])))),
                   micro_us=dict(inertial=us_I, rotating=us_B, gravity=us_g)),
              open(f, "w"), indent=1)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()

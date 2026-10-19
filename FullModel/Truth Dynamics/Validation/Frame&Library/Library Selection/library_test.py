#!/usr/bin/env python
"""Which library should drive the truth propagator: locally coded, SciPy, or tudatpy.

All three back ends run in ONE process, share ONE mesh, ONE polyhedral evaluable and ONE
right-hand side. The only thing that differs is the code stepping the ODE, so the comparison is
about the integrator and not about the gravity model.

    rdot = v
    vdot = R(wt) g(R(-wt) r)

Inertial frame, six states, no thrust, no mass. The field is body-fixed, so every evaluation rotates
the position in and the acceleration back out. Same arc as the parking-orbit benchmark: the 35 km
circular orbit, seed 20260916, 1.5 revolutions.

TIMING is split three ways. Total wall clock is measured around the whole propagation; the time
spent inside the polyhedral call is accumulated separately; integration time is the difference. The
gravity term should be near-identical per evaluation across back ends -- it is literally the same
function -- so any difference in integration time is the library's own overhead, which for tudat
includes the C++ to Python boundary crossing on every stage.

Run with the tudat environment:
    ~/miniforge3/envs/tudat/bin/python library_test.py
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
SCVX = os.path.join(HERE, "..", "..", "..", "..", "..", "scvx_landing")
sys.path.insert(0, SCVX)

EROS_GM = 4.4665e5
EROS_ROT_PERIOD = 5.270 * 3600.0
WZ = 2.0 * np.pi / EROS_ROT_PERIOD


# --- the one right-hand side every back end uses --------------------------------------------------
G_CONST = 6.67430e-11
TAB = os.path.join(SCVX, "data", "eros_ver64q.tab")


def load_gaskell_tab(path):
    """Parse a Gaskell vertex-facet .tab shape model -> (vertices [m], faces [0-indexed])."""
    with open(path) as fh:
        nv, nf = (int(x) for x in fh.readline().split()[:2])
        V = np.empty((nv, 3))
        for i in range(nv):
            p = fh.readline().split()
            V[i] = (float(p[1]), float(p[2]), float(p[3]))
        F = np.empty((nf, 3), dtype=np.int64)
        for i in range(nf):
            p = fh.readline().split()
            F[i] = (int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1)
    return V * 1000.0, F


class Field:
    """Polyhedral gravity, counting its evaluations and timing itself.

    Built standalone rather than imported from the scvx package, whose __init__ pulls in cvxpy that
    the tudat environment has no reason to carry. Same mesh file, same Werner-Scheeres evaluable,
    same density rule.

    Not memoised. The scvx version caches on the bit pattern of r because polyhedral_gravity 3.3.1
    sums its facet contributions in a non-deterministic order, but that scatter is ~2e-18 relative,
    about 1e-20 m/s^2, which integrates to ~1e-11 m over this arc -- four orders below the 1e-6 m
    benchmark floor. Caching here would instead corrupt the timing, since a memo hit returns without
    touching the field.
    """

    def __init__(self, mesh_q=64):
        import trimesh
        import polyhedral_gravity as pg
        V, F = load_gaskell_tab(TAB)
        self.mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        self.mesh.fix_normals()
        self.faces = len(self.mesh.faces)
        density = EROS_GM / (G_CONST * float(self.mesh.volume))
        self.density = density
        poly = pg.Polyhedron(
            polyhedral_source=(np.asarray(self.mesh.vertices), np.asarray(self.mesh.faces)),
            density=density, normal_orientation=pg.NormalOrientation.OUTWARDS,
            integrity_check=pg.PolyhedronIntegrity.DISABLE, metric_unit=pg.MetricUnit.METER)
        self._ev = pg.GravityEvaluable(poly)
        self.n = 0
        self.t_grav = 0.0

    def reset(self):
        self.n, self.t_grav = 0, 0.0

    def accel_inertial(self, t, r_I):
        """Inertial acceleration at inertial position r_I and time t."""
        c, s = np.cos(WZ * t), np.sin(WZ * t)
        r_B = np.array([c * r_I[0] + s * r_I[1], -s * r_I[0] + c * r_I[1], r_I[2]])
        t0 = time.perf_counter()
        g_B = np.asarray(self._ev(r_B)[1], float)
        self.t_grav += time.perf_counter() - t0
        self.n += 1
        return np.array([c * g_B[0] - s * g_B[1], s * g_B[0] + c * g_B[1], g_B[2]])

    def rhs(self, t, y):
        return np.concatenate([y[3:6], self.accel_inertial(t, y[0:3])])


def circular_inertial(radius_km, seed):
    rng = np.random.default_rng(seed)
    R = radius_km * 1e3
    u = rng.normal(size=3); u /= np.linalg.norm(u)
    w = rng.normal(size=3); w -= (w @ u) * u; w /= np.linalg.norm(w)
    period = 2.0 * np.pi * np.sqrt(R ** 3 / EROS_GM)
    return np.concatenate([R * u, np.sqrt(EROS_GM / R) * w]), period


# --- back end 1: locally coded fixed step ----------------------------------------------------------
def local_rk4(field, y0, tf, dt):
    """Classical RK4, exact step, final step trimmed when it would overshoot."""
    f = field.rhs
    y, t, n = np.asarray(y0, float).copy(), 0.0, 0
    while True:
        rem = tf - t
        if rem <= 1e-9:
            break
        h = dt if dt < rem else rem
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        n += 1
    return y, n


# --- back end 2: SciPy variable step ---------------------------------------------------------------
from scipy.integrate import solve_ivp as _solve_ivp        # imported once, not inside the timer


def scipy_run(field, y0, tf, method, rtol, atol):
    s = _solve_ivp(field.rhs, (0.0, tf), np.asarray(y0, float), method=method,
                  rtol=rtol, atol=atol)
    if not s.success:
        raise RuntimeError(s.message)
    return s.y[:, -1], len(s.t) - 1


# --- back end 3: tudatpy ---------------------------------------------------------------------------
class Tudat:
    """tudat's integrators driving the SAME field, through a custom acceleration.

    tudat's own custom_state hook -- which takes an arbitrary dy/dt = f(t,x) -- raises a type
    inconsistency in the C++ layer whatever the state shape or time type, so the route used here is
    a translational propagator whose single acceleration is a Python callback. The callback reads
    the vehicle's inertial state back out of the environment, which tudat updates before each stage,
    so the dynamics integrated are identical to the other two back ends.
    """

    def __init__(self, field):
        from tudatpy.dynamics import environment_setup
        self.field = field
        bl = environment_setup.BodyListSettings("SSB", "J2000")
        bl.add_empty_settings("Eros")
        bl.get("Eros").ephemeris_settings = environment_setup.ephemeris.constant(
            np.zeros(6), "SSB", "J2000")
        self.bodies = environment_setup.create_system_of_bodies(bl)
        self.bodies.create_empty_body("Vehicle")

    def _models(self):
        from tudatpy.dynamics import propagation_setup as ps

        def acc(t):
            r = np.asarray(self.bodies.get("Vehicle").state, float).ravel()[0:3]
            return self.field.accel_inertial(float(t), r).reshape(3, 1)

        return ps.create_acceleration_models(
            self.bodies, {"Vehicle": {"Eros": [ps.acceleration.custom_acceleration(acc)]}},
            ["Vehicle"], ["Eros"])

    def run(self, y0, tf, integ):
        from tudatpy.dynamics import propagation_setup as ps, simulator
        from tudatpy.astro.time_representation import Time
        term = ps.propagator.time_termination(Time(tf), terminate_exactly_on_final_condition=True)
        prop = ps.propagator.translational(["Eros"], self._models(), ["Vehicle"],
                                           np.asarray(y0, float), Time(0.0), integ, term)
        sim = simulator.create_dynamics_simulator(self.bodies, prop)
        h = sim.propagation_results.state_history
        k = sorted(h.keys())
        return np.asarray(h[k[-1]], float).ravel(), len(k) - 1

    def fixed(self, y0, tf, dt, cset):
        from tudatpy.dynamics import propagation_setup as ps
        from tudatpy.astro.time_representation import Time
        return self.run(y0, tf, ps.integrator.runge_kutta_fixed_step(Time(dt), cset))

    def variable(self, y0, tf, cset, rtol, atol, dt0=32.0):
        from tudatpy.dynamics import propagation_setup as ps
        from tudatpy.astro.time_representation import Time
        ctl = ps.integrator.step_size_control_elementwise_scalar_tolerance(rtol, atol)
        val = ps.integrator.step_size_validation(1e-4, 1.0e5)
        return self.run(y0, tf, ps.integrator.runge_kutta_variable_step(
            Time(dt0), cset, ctl, val))


# --- driver ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--radius-km", type=float, default=35.0)
    ap.add_argument("--orbits", type=float, default=1.5)
    ap.add_argument("--dt-rk4", type=float, default=64.0)
    ap.add_argument("--dt-bench", type=float, default=32.0,
                    help="the benchmark step from parking_orbit_benchmark")
    ap.add_argument("--rtols", type=float, nargs="*",
                    default=[1e-6, 1e-8, 1e-10, 1e-12])
    ap.add_argument("--out", default=os.path.join(HERE, "Figures&data"))
    a = ap.parse_args()

    field = Field()
    y0, period = circular_inertial(a.radius_km, a.seed)
    # Warm up every back end on a throwaway arc. The first call into SciPy and into tudat pays
    # import, JIT and allocation costs that belong to neither integrator; timing them as if they
    # were per-evaluation work put the first SciPy run at 1877 us/eval against 9 us for the rest.
    _w = 200.0
    scipy_run(field, y0, _w, "RK45", 1e-6, 1e-9)
    scipy_run(field, y0, _w, "DOP853", 1e-6, 1e-9)
    local_rk4(field, y0, _w, 32.0)
    tf = a.orbits * period
    print(f"arc: 35 km circular, seed {a.seed}, {a.orbits:g} orbits, tf = {tf:.1f} s")
    print(f"gravity: polyhedral, {field.faces:,} facets, GM held at {EROS_GM:g}\n")

    rows = []

    def record(lib, scheme, setting, fn):
        field.reset()
        t0 = time.perf_counter()
        y, steps = fn()
        tot = time.perf_counter() - t0
        rows.append(dict(lib=lib, scheme=scheme, setting=setting, y=y[0:6].tolist(),
                         evals=field.n, steps=steps, total_s=tot, grav_s=field.t_grav,
                         integ_s=tot - field.t_grav))
        print(f"  {lib:6s} {scheme:8s} {setting:>14}{field.n:9d} ev {tot:8.2f} s tot"
              f"{field.t_grav:8.2f} s grav{tot - field.t_grav:8.2f} s integ")
        return rows[-1]

    from tudatpy.dynamics import propagation_setup as ps
    CS = ps.integrator.CoefficientSets
    td = Tudat(field)
    td.fixed(y0, _w, 32.0, CS.rk_4)
    td.variable(y0, _w, CS.rkf_45, 1e-6, 1e-9)
    td.variable(y0, _w, CS.rkdp_87, 1e-6, 1e-9)
    field.reset()

    print(f"{'lib':>8}{'scheme':>9}{'setting':>15}{'evals':>12}{'total':>11}{'gravity':>11}{'integr':>11}")
    print("  --- reference: the parking-orbit benchmark ---")
    ref = record("local", "rk4", f"dt={a.dt_bench:g}s", lambda: local_rk4(field, y0, tf, a.dt_bench))
    y_ref = np.array(ref["y"])

    print("  --- fixed step RK4 at the same dt (SciPy has no fixed-step method) ---")
    record("local", "rk4", f"dt={a.dt_rk4:g}s", lambda: local_rk4(field, y0, tf, a.dt_rk4))
    record("tudat", "rk4", f"dt={a.dt_rk4:g}s", lambda: td.fixed(y0, tf, a.dt_rk4, CS.rk_4))

    print("  --- variable step, order 5 ---")
    for rt in a.rtols:
        record("scipy", "RK45", f"rtol={rt:.0e}",
               lambda rt=rt: scipy_run(field, y0, tf, "RK45", rt, rt * 1e-3))
        record("tudat", "rkf_45", f"rtol={rt:.0e}",
               lambda rt=rt: td.variable(y0, tf, CS.rkf_45, rt, rt * 1e-3))

    print("  --- variable step, order 8 ---")
    for rt in a.rtols:
        record("scipy", "DOP853", f"rtol={rt:.0e}",
               lambda rt=rt: scipy_run(field, y0, tf, "DOP853", rt, rt * 1e-3))
        record("tudat", "rkdp_87", f"rtol={rt:.0e}",
               lambda rt=rt: td.variable(y0, tf, CS.rkdp_87, rt, rt * 1e-3))

    for r in rows:
        r["err"] = float(np.linalg.norm(np.array(r["y"])[0:3] - y_ref[0:3]))
        r["verr"] = float(np.linalg.norm(np.array(r["y"])[3:6] - y_ref[3:6]))
        r["us_per_eval_grav"] = 1e6 * r["grav_s"] / max(r["evals"], 1)
        r["us_per_eval_integ"] = 1e6 * r["integ_s"] / max(r["evals"], 1)
        r.pop("y")

    print(f"\n{'lib':>8}{'scheme':>9}{'setting':>15}{'error [m]':>13}{'evals':>9}"
          f"{'grav us/ev':>12}{'integ us/ev':>13}")
    for r in rows:
        print(f"{r['lib']:>8}{r['scheme']:>9}{r['setting']:>15}{r['err']:13.3e}{r['evals']:9d}"
              f"{r['us_per_eval_grav']:12.2f}{r['us_per_eval_integ']:13.2f}")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, "library_test.json")
    json.dump(dict(seed=a.seed, radius_km=a.radius_km, orbits=a.orbits, tf=tf,
                   facets=field.faces, dt_rk4=a.dt_rk4, dt_bench=a.dt_bench,
                   ref_err_estimate=1.004e-06,
                   note="error is measured against the local RK4 dt=32 s benchmark, whose own "
                        "error is about 1.0e-6 m; nothing below that is resolvable",
                   rows=rows), open(f, "w"), indent=1)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()

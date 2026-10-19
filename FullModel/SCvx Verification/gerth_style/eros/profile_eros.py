"""Where does one Eros PTR iteration go? Polyhedral field, K = 60, stage 1, 2 iterations.
Pinned to one core BEFORE gravity is imported (TBB sizes its pool from the affinity mask)."""
import os, sys, time, json
core = int(sys.argv[1]) if len(sys.argv) > 1 else 3
os.sched_setaffinity(0, {core})
SCVX = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "SCvx")); sys.path.insert(0, SCVX)
import numpy as np
import config as C, ptr, subproblem as SP, problem_setup as PS, dynamics as D
T = dict(disc=0.0, build=0.0, solve=0.0, defect=0.0); N = dict(disc=0, defect=0)
def timed(name, f):
    def w(*a, **k):
        fld = a[0].field if hasattr(a[0], "field") else None
        n0 = getattr(fld, "n_calls", 0); t = time.perf_counter(); out = f(*a, **k)
        T[name] += time.perf_counter() - t
        if name in N: N[name] += getattr(fld, "n_calls", 0) - n0
        return out
    return w
ptr.discretize = timed("disc", D.discretize); ptr.defect = timed("defect", D.defect)
SP.build = timed("build", SP.build); SP.solve = timed("solve", SP.solve)
P = PS.build()
fld = P["dyn"].field
t = time.perf_counter(); fld.g(np.array([20000.0, 0, 0])); fld.n_calls = 0
t0 = time.perf_counter(); r = ptr.run(dyn=P["dyn"], scales=P["scales"], x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=P["n_hat"],
                                      K=60, iters=2, verbose=False, mode="error")
wall = time.perf_counter() - t0
rng = np.random.default_rng(0); pts = P["r_site"] + 1000 + 8000 * rng.random((50, 3))
t = time.perf_counter()
for p in pts: fld._ev(p)
per = (time.perf_counter() - t) / 50
out = dict(core=core, wall_2it=wall, split=T, gravity_calls=N, total_calls=fld.n_calls, per_eval_ms=per * 1e3,
           gravity_share=fld.n_calls * per / wall, field=type(fld).__name__)
print(json.dumps(out, indent=1))
json.dump(out, open("profile_eros.json", "w"), indent=1)

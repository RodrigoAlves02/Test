"""Classic PTR (Reynolds/Szmuk/Malyuta): the trust region is a QUADRATIC price in the cost, no hard cap,
no ratio test, fixed weight. Implemented on a patched COPY of subproblem.py (sum of squares instead of
the norm in the epigraph rows); ptr.run is called with the cap removed (R = 1e3) and the weights frozen
(w_growth = 1). Also runs the other variants from a different flight-time seed.
usage: ptr_classic.py <case> <variant> [tf_seed_s]
  variant: ptr0.1 | ptr1 | ptr10 | default | ratio"""
import os, sys, json, time, tempfile
import numpy as np
case, variant = sys.argv[1], sys.argv[2]
seed = float(sys.argv[3]) if len(sys.argv) > 3 else None
VER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); SCVX = os.path.abspath(os.path.join(VER, "..", "SCvx"))
if variant.startswith("ptr"):
    src = open(os.path.join(SCVX, "subproblem.py")).read()
    a = """        cons += [cp.norm(cp.hstack([X[:, k] - Xref[:, k],
                                    cp.multiply(Wu, U[:, k] - Uref[:, k])])) <= eta[k]]
    cons += [cp.abs(s - sref) <= eta_s]"""
    assert a in src
    src = src.replace(a, """        cons += [cp.sum_squares(cp.hstack([X[:, k] - Xref[:, k],
                                           cp.multiply(Wu, U[:, k] - Uref[:, k])])) <= eta[k]]
    cons += [cp.square(s - sref) <= eta_s]""")
    tmp = tempfile.mkdtemp(); open(os.path.join(tmp, "subproblem.py"), "w").write(src)
    sys.path.insert(0, tmp)
sys.path.insert(1, VER); sys.argv = ["x"]
import run_verification as RV
import ptr, config as C, benchmark as B, subproblem as SP
from dynamics import Dynamics
cs = B.CASES[case]; r0, v0 = cs["r0"], cs["v0"]; rf = np.zeros(3); n = np.array([0, 0, 1.0])
sc = C.Scales.build(r0, rf, m_wet=B.M_WET, t_max=B.TMAX); dyn = Dynamics(sc, field=B.ConstantField())
x0 = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(B.M_WET)], n])
kw = dict(dyn=dyn, scales=sc, x_init=x0, rf_nd=rf / sc.L, n_hat=n, K=60, mode="fuel", D_nd=0.0, iters=150,
          verbose=False, slew_max_deg_s=None, a_init=None, term_point_deg=180.0, term_point_nodes=0,
          tol_obj=1e-6, tol_defect_m=1e-2, tf_seed_s=seed)
nostop = variant.endswith("_nostop")
if variant.startswith("ptr"):
    w = float(variant[3:].replace("_nostop", ""))
    kw.update(R0=1e3, Rs0=1e3, R_bounds=(1e-4, 1e3), w_tr=w, w_ts=w, w_growth=1.0)
    if nostop:                       # no objective-stagnation stopping: run all iterations, judge the J history
        kw.update(tol_obj=0.0, tol_abs=0.0)
elif variant == "ratio":
    kw.update(ratio_test=True)
t = time.time(); r = ptr.run(**kw); wall = time.time() - t
H = r.history; J = np.array([h["jphys"] for h in H]); fin = J[-1]
BEST = {"min-max": 180.3387, "max-min-max": 275.5594}[case]
hit = np.where(np.abs(J - BEST) / BEST < 1e-4)[0]
out = dict(case=case, variant=variant, seed=seed, best_ref=BEST, first_within_1e4_of_best=(int(hit[0]) if len(hit) else None),
           J_at=[round(float(J[i]), 3) for i in (4, 9, 19, 39, 79, len(J) - 1) if i < len(J)], status=r.status, it=r.iters, mp=r.fuel_kg, tf=r.tf_s, wall=wall,
           rejected=sum(1 for h in H if not h["accepted"]), within_1e4_at=int(np.argmax(np.abs(J - fin) / fin < 1e-4)),
           subproblem=os.path.dirname(SP.__file__) != SCVX)
print(json.dumps(out), flush=True)
tag = f"{case}_{variant}" + ("" if seed is None else f"_s{seed:g}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"C_{tag}.json"), "w"), indent=1)

"""Is carrying the trust region across cycles still worth anything now the ratchet is fixed?"""
import sys, os, json, numpy as np
GC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, GC)
from guidance import SCvxGuidance, load_nominal, _Rz, OMEGA, WZ

def b2i(r_B, v_B, t):
    R = _Rz(WZ * t); return R @ r_B, R @ (v_B + np.cross(OMEGA, r_B))

Xn, Un, sn = load_nominal()
res = {}
for persist in (False, True):
    for seed_nom in (True, False):
        tag = f"persist={persist} nominal_seed={seed_nom}"
        g = SCvxGuidance(nominal=(Xn, Un, sn) if seed_nom else None,
                         persist_tr=persist, iters=60)
        x = Xn[:, 0]; r_I, v_I = b2i(x[0:3], x[3:6], 0.0); m = float(np.exp(x[6])); t = 0.0
        its, oks, tot = [], 0, []
        for c in range(6):
            p = g.solve(t, r_I, v_I, m)
            if not p.ok:
                its.append(float("nan")); break
            its.append(p.iters); oks += (p.reason == "converged"); tot.append(t + p.sigma)
            K = p.X.shape[1]; j = min(max(int(round(30.0/(p.sigma/(K-1)))), 1), K-2)
            xj = p.X[:, j]; t = t + j*p.sigma/(K-1)
            r_I, v_I = b2i(xj[0:3], xj[3:6], t); m = float(np.exp(xj[6]))
        res[tag] = dict(iters=its, conv=oks, mean_s=g.mean_solve_s,
                        mission=tot, drift=float(max(tot)-min(tot)) if tot else float("nan"))
        print(f"{tag:<38} iters {its}  conv {oks}/6  mean {g.mean_solve_s:.1f}s  "
              f"t+t_go drift {res[tag]['drift']:.1f}s", flush=True)
b = min((k for k in res if res[k]["conv"] == 6), key=lambda k: np.nansum(res[k]["iters"]),
        default=None)
print(f"\nfewest total iterations with all 6 converged: {b}")
json.dump(res, open("step22_persist.json","w"), indent=1)

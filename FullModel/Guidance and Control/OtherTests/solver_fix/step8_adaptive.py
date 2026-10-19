"""Initial trust region matched to the warm start's own mismatch.

A trust region should start at roughly the size of the step it will need. When warm starting, that
size is knowable: it is how far the new problem sits from the guess. Too small and the run spends
its iterations growing; too large and it spends them shrinking. Both were observed.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import tune, gen_dataset as G
from scvx.core import nl_defects

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
dtau = np.diff(np.linspace(0, 1, X.shape[1]))
def defect(o):
    r=o["r"]; return float(np.abs(nl_defects(o["cs"]["dyn"], r["X"], r["U"], float(r["sigma"]), dtau)).sum())

rng = np.random.default_rng(31)
CASES = [("exact", x0.copy())]
for disp in (10.0, 50.0, 150.0, 270.0):
    d = rng.normal(size=3); d = d/np.linalg.norm(d)*disp
    xp = x0.copy(); xp[0:3] += d
    CASES.append((f"d{disp:.0f}m", xp))

def adaptive_cfg(xi, k, bex):
    mis = float(np.linalg.norm(xi[0:3] - X[0:3, 0]))       # how far the guess is from the problem
    eta0 = max(1.0, k * mis)
    return dict(eta0=eta0, eta_s0=max(0.5, eta0 / 20.0), eta_u0=max(1e-4, eta0 * 1e-4), bex=bex)

print(HDR + "   defect")
res = []
for k in (2.0, 10.0):
    for bex in (2.0,):
        its = []
        for lab, xi in CASES:
            ex = adaptive_cfg(xi, k, bex)
            o = run(xi, m, warm=(X, U, sig), extra=ex, ctx=ctx)
            s = summary(o, f"k{k:g} bex{bex:g} {lab}", sig_ref=sig); s["defect"] = defect(o)
            s["eta0"] = ex["eta0"]; res.append(s); its.append(s["iters"])
            print(line(s) + f"{s['defect']:10.2e}")
        print(f"   -> k={k:g} bex={bex:g}: mean {np.mean(its):.1f} iters, max {max(its)}\n")
json.dump(res, open("step8.json", "w"), indent=1)

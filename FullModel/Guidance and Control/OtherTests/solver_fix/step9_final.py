"""Final validation of the fixed solver against the recorded pre-fix baseline."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import tune, gen_dataset as G
import scvx.driver as D
from scvx.core import nl_defects

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
dtau = np.diff(np.linspace(0, 1, X.shape[1]))
def defect(o):
    r=o["r"]; return float(np.abs(nl_defects(o["cs"]["dyn"], r["X"], r["U"], float(r["sigma"]), dtau)).sum())

def adaptive(xi, k=2.0):
    """eta0 matched to how far the warm start sits from the problem it is being reused on."""
    mis = float(np.linalg.norm(np.asarray(xi)[0:3] - X[0:3, 0]))
    e0 = max(1.0, k * mis)
    return dict(eta0=e0, eta_s0=max(0.5, e0/20.0), eta_u0=max(1e-4, e0*1e-4))

print(f"driver caps: ETA_MAX {D.ETA_MAX:g} ES_MAX {D.ES_MAX:g} EU_MAX {D.EU_MAX:g}  "
      f"(cfg-overridable, never below the starting radius)")
print(f"tune       : ETA0 {tune.ETA0} ETA_U0 {tune.ETA_U0} ETA_S0 {tune.ETA_S0}\n")
print(HDR + "   defect")
res = []
rng = np.random.default_rng(20260921)
cases = [("exact", x0.copy())]
for disp in (10.0, 50.0, 150.0, 270.0):
    d = rng.normal(size=3); d = d/np.linalg.norm(d)*disp
    xp = x0.copy(); xp[0:3] += d
    cases.append((f"d{disp:.0f}m", xp))

for lab, xi in cases:
    o = run(xi, m, warm=(X, U, sig), extra=adaptive(xi), ctx=ctx)
    s = summary(o, f"FIXED warm {lab}", sig_ref=sig); s["defect"] = defect(o); res.append(s)
    print(line(s) + f"{s['defect']:10.2e}")
# cold, unchanged defaults, for reference
o = run(x0, m, warm=None, ctx=ctx)
s = summary(o, "FIXED cold (defaults)", sig_ref=sig); s["defect"] = defect(o); res.append(s)
print(line(s) + f"{s['defect']:10.2e}")

it = [s["iters"] for s in res[:-1]]
print(f"\nwarm-start iterations: exact {res[0]['iters']}, mean over dispersions "
      f"{np.mean(it[1:]):.1f}, max {max(it)}")
print(f"eta_s growth events across all runs: {sum(s['eta_s_grew'] for s in res)}")
print(f"all converged: {all(s['reason']=='converged' for s in res)}")
json.dump(res, open("step9_final.json","w"), indent=1)

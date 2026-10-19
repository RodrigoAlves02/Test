"""Final validation of the fixed solver -- NO monkeypatching, the real driver.py."""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import tune, gen_dataset as G
from scvx.core import nl_defects
import scvx.driver as D

ap = argparse.ArgumentParser()
ap.add_argument("--eta0", type=float, default=1.0)
ap.add_argument("--eta-s0", type=float, default=0.5)
ap.add_argument("--eta-u0", type=float, default=1e-4)
ap.add_argument("--bex", type=float, default=3.0)
ap.add_argument("--tag", default="fixed")
a = ap.parse_args()
CFG = dict(eta0=a.eta0, eta_s0=a.eta_s0, eta_u0=a.eta_u0, bex=a.bex)

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
dtau = np.diff(np.linspace(0, 1, X.shape[1]))
def defect(o):
    r=o["r"]; return float(np.abs(nl_defects(o["cs"]["dyn"], r["X"], r["U"], float(r["sigma"]), dtau)).sum())
print(f"driver caps (module): ETA_MAX {D.ETA_MAX:g} ES_MAX {D.ES_MAX:g} EU_MAX {D.EU_MAX:g}")
print(f"config: {CFG}\n")
print(HDR + "   defect")
res=[]
def go(lab, xi, warm, **ex):
    o = run(xi, m, warm=warm, extra={**CFG, **ex}, ctx=ctx)
    s = summary(o, lab, sig_ref=sig); s["defect"]=defect(o); res.append(s)
    print(line(s) + f"{s['defect']:10.2e}"); return s

go("warm=exact", x0, (X,U,sig))
rng = np.random.default_rng(20260921)
for disp in (10.0, 50.0, 150.0, 270.0):
    d = rng.normal(size=3); d = d/np.linalg.norm(d)*disp
    xp = x0.copy(); xp[0:3] += d
    go(f"warm nominal d{disp:.0f}m", xp, (X,U,sig))
go("cold", x0, None)
json.dump(dict(cfg=CFG, rows=res), open(f"step7_{a.tag}.json","w"), indent=1)
it = [s["iters"] for s in res]
gr = sum(s["eta_s_grew"] for s in res)
print(f"\nSUMMARY  mean iters {np.mean(it):.1f}  max {max(it)}  "
      f"eta_s growth events {gr}  all converged: {all(s['reason']=='converged' for s in res)}")

"""Wvc, rho thresholds, and the cold start."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import tune, gen_dataset as G
import scvx.driver as D
from scvx.core import nl_defects

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
D.ETA_MAX, D.ES_MAX, D.EU_MAX = 1e6, 4000.0, 1.0
dtau = np.diff(np.linspace(0, 1, X.shape[1]))
def defect(o):
    r=o["r"]; return float(np.abs(nl_defects(o["cs"]["dyn"], r["X"], r["U"], float(r["sigma"]), dtau)).sum())

res=[]; print(HDR + "   defect")
def go(lab, **ex):
    warm = ex.pop("warm", None)
    o = run(x0, m, warm=warm, extra=ex, ctx=ctx)
    s = summary(o, lab, sig_ref=sig); s["defect"]=defect(o); s["cfg"]=ex; res.append(s)
    print(line(s) + f"{s['defect']:10.2e}"); return s

W = (X, U, sig)
for w in (1.4e4, 1.4e3, 1.4e2):
    go(f"warm Wvc {w:.0f}", warm=W, eta0=1.0, eta_s0=0.5, eta_u0=1e-4, Wvc=w)
for r1, r2 in ((0.35,0.9),(0.25,0.7),(0.1,0.5)):
    go(f"warm r1{r1} r2{r2}", warm=W, eta0=1.0, eta_s0=0.5, eta_u0=1e-4, r1t=r1, r2t=r2)
print("  --- cold start: trust region size ---")
for f in (1.0, 0.3, 0.1, 0.03, 0.01):
    go(f"cold tr x{f:g}", eta0=1000*f, eta_s0=500*f, eta_u0=0.1*f)
json.dump([{k:v for k,v in s.items() if k!="cfg"} for s in res], open("step4.json","w"), indent=1)

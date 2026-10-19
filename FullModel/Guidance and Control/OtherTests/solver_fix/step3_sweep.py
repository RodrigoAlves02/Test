"""Sweep the knobs that set how big a step the merit will accept."""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import tune, gen_dataset as G
import scvx.driver as D
from scvx.core import nl_defects

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
D.ETA_MAX, D.ES_MAX, D.EU_MAX = 1e6, 4000.0, 1.0      # caps above any initial radius
dtau = np.diff(np.linspace(0, 1, X.shape[1]))

def defect(o):
    r = o["r"]
    return float(np.abs(nl_defects(o["cs"]["dyn"], r["X"], r["U"], float(r["sigma"]), dtau)).sum())

res = []
print(HDR + "   defect")
# 1) initial trust-region size, warm from exact
for f in (1.0, 0.1, 0.01, 1e-3):
    ex = dict(eta0=1000.0*f, eta_s0=500.0*f, eta_u0=0.1*f)
    o = run(x0, m, warm=(X,U,sig), extra=ex, ctx=ctx)
    s = summary(o, f"warm tr x{f:g}", sig_ref=sig); s["defect"]=defect(o); res.append(s)
    print(line(s) + f"{s['defect']:10.2e}")
# 2) virtual-control weight, warm from exact, small tr
for w in (1.4e4, 1.4e3, 1.4e2, 1.4e1):
    ex = dict(eta0=100.0, eta_s0=50.0, eta_u0=0.01, Wvc=w)
    o = run(x0, m, warm=(X,U,sig), extra=ex, ctx=ctx)
    s = summary(o, f"warm Wvc {w:.0f}", sig_ref=sig); s["defect"]=defect(o); res.append(s)
    print(line(s) + f"{s['defect']:10.2e}")
# 3) growth/shrink thresholds
for r1,r2 in ((0.35,0.9),(0.25,0.7),(0.1,0.5),(0.05,0.3)):
    ex = dict(eta0=100.0, eta_s0=50.0, eta_u0=0.01, r1t=r1, r2t=r2)
    o = run(x0, m, warm=(X,U,sig), extra=ex, ctx=ctx)
    s = summary(o, f"warm r1{r1} r2{r2}", sig_ref=sig); s["defect"]=defect(o); res.append(s)
    print(line(s) + f"{s['defect']:10.2e}")
json.dump(res, open("step3_sweep.json","w"), indent=1)

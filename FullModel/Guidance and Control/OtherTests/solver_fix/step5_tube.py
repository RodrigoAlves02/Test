"""The real use case: warm start from the nominal for a DISPERSED initial condition.

This is both the MPC cycle and the dataset 'tube' idea -- neighbouring problems, one good guess.
"""
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

rng = np.random.default_rng(20260921)
print(HDR + "   defect")
res = []
for disp in (10.0, 50.0, 150.0, 270.0):        # metres of handover dispersion
    for trial in range(2):
        dx = rng.normal(size=3); dx = dx/np.linalg.norm(dx)*disp
        dv = rng.normal(size=3)*0.0
        xp = x0.copy(); xp[0:3] += dx; xp[3:6] += dv
        for lab, ex in (("tr x1  ", dict(eta0=1000., eta_s0=500., eta_u0=0.1)),
                        ("tr x.01", dict(eta0=10., eta_s0=5., eta_u0=1e-3)),
                        ("tr x.001", dict(eta0=1., eta_s0=.5, eta_u0=1e-4))):
            o = run(xp, m, warm=(X, U, sig), extra=ex, ctx=ctx)
            s = summary(o, f"d{disp:.0f}m #{trial} {lab}", sig_ref=sig); s["defect"]=defect(o)
            s["disp"]=disp; s["cfg"]=lab; res.append(s)
            print(line(s) + f"{s['defect']:10.2e}")
json.dump([{k:v for k,v in s.items()} for s in res], open("step5.json","w"), indent=1)

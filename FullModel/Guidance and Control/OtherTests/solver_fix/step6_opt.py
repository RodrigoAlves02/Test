"""Optimise the config for the tube case: warm from nominal, dispersed IC."""
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

rng = np.random.default_rng(7)
CASES = []
for disp in (50.0, 150.0):
    for _ in range(2):
        d = rng.normal(size=3); d = d/np.linalg.norm(d)*disp
        xp = x0.copy(); xp[0:3] += d
        CASES.append((disp, xp))

def trial(lab, **ex):
    its, dss, fus, dfs = [], [], [], []
    for disp, xp in CASES:
        o = run(xp, m, warm=(X,U,sig), extra=dict(ex), ctx=ctx)
        s = summary(o, lab, sig_ref=sig)
        its.append(s["iters"]); dss.append(abs(s["sigma_err"])); fus.append(s["fuel"]); dfs.append(defect(o))
    print(f"{lab:<30} it {np.mean(its):5.1f}  |dsig| {np.mean(dss):7.2f}  fuel {np.mean(fus):7.4f}"
          f"  defect {np.mean(dfs):9.2e}  max_it {max(its)}")
    return dict(lab=lab, iters=float(np.mean(its)), dsig=float(np.mean(dss)),
                fuel=float(np.mean(fus)), defect=float(np.mean(dfs)), max_it=int(max(its)), cfg=ex)

res = []
base = dict(eta0=1.0, eta_s0=0.5, eta_u0=1e-4)
print("--- growth factor ---")
for bex in (2.0, 3.0, 5.0):
    res.append(trial(f"bex {bex}", **base, bex=bex))
print("--- rho thresholds ---")
for r1, r2 in ((0.35,0.9),(0.2,0.6),(0.1,0.4)):
    res.append(trial(f"r1 {r1} r2 {r2}", **base, r1t=r1, r2t=r2, bex=3.0))
print("--- initial radius ---")
for f in (1e-2, 1e-3, 1e-4):
    res.append(trial(f"tr x{f:g}", eta0=1000*f, eta_s0=500*f, eta_u0=0.1*f, bex=3.0))
json.dump(res, open("step6.json","w"), indent=1)

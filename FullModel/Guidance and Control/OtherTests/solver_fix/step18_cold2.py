"""Cold start in two stages. No single region size serves both jobs: a cold guess needs a control
region wide enough to reach the bang-bang structure, and the refinement then needs one narrow
enough that rho stays informative. Run them in sequence instead of compromising on one."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G
from scvx.core import nl_defects

X, U, sig, x0, rf = nominal(); Mw = float(np.exp(x0[6])); ctx = G.Ctx()
dtau = np.diff(np.linspace(0, 1, X.shape[1]))
REF = 5.6710                                         # best feasible fuel found for this x_init
S1 = dict(eta0=1000., eta_s0=500., eta_u0=0.1,  r1t=0.36, r2t=0.90)
S2 = dict(eta0=300.,  eta_s0=400., eta_u0=1e-3, r1t=0.0,  r2t=0.50)
def dfct(o):
    r = o["r"]; return float(np.abs(nl_defects(o["cs"]["dyn"], r["X"], r["U"],
                                               float(r["sigma"]), dtau)).sum())
print(f"reference fuel for this problem: {REF:.4f} kg\n")
print(f"{'recipe':<22}{'s1_it':>6}{'s2_it':>6}{'tot':>5}{'reason':>11}{'sigma':>9}{'fuel':>9}"
      f"{'gap':>8}{'miss':>9}{'defect':>9}")
rows = []
for n1 in (5, 8, 10, 12, 15, 20):
    o1 = run(x0, Mw, warm=None, extra=dict(S1), iters=n1, ctx=ctx)
    r1 = o1["r"]
    o2 = run(x0, Mw, warm=(r1["X"], r1["U"], float(r1["sigma"])), extra=dict(S2),
             iters=60, ctx=ctx)
    s1, s2 = summary(o1, ""), summary(o2, "")
    gap = (s2["fuel"] - REF) / REF * 100.0
    rows.append(dict(n1=n1, s1=s1, s2=s2, gap=gap, defect=dfct(o2)))
    print(f"{'2-stage n1=' + str(n1):<22}{s1['iters']:>6}{s2['iters']:>6}"
          f"{s1['iters']+s2['iters']:>5}{s2['reason']:>11}{s2['sigma']:>9.1f}{s2['fuel']:>9.4f}"
          f"{gap:>7.2f}%{s2['miss']:>9.1e}{rows[-1]['defect']:>9.1e}", flush=True)
for nm, cfg in [("1-stage baseline", S1), ("1-stage tuned", S2)]:
    o = run(x0, Mw, warm=None, extra=dict(cfg), iters=60, ctx=ctx); s = summary(o, "")
    g = (s["fuel"] - REF) / REF * 100.0
    rows.append(dict(n1=0, s1=None, s2=s, gap=g, defect=dfct(o), name=nm))
    print(f"{nm:<22}{'-':>6}{s['iters']:>6}{s['iters']:>5}{s['reason']:>11}{s['sigma']:>9.1f}"
          f"{s['fuel']:>9.4f}{g:>7.2f}%{s['miss']:>9.1e}{rows[-1]['defect']:>9.1e}")
ok = [r for r in rows if r["n1"] and r["s2"]["reason"] == "converged" and r["defect"] < 1e-2]
if ok:
    b = min(ok, key=lambda r: (round(r["gap"], 2), r["s1"]["iters"] + r["s2"]["iters"]))
    print(f"\nBEST COLD RECIPE: stage 1 = {b['n1']} iters of the wide region, then stage 2 tuned "
          f"-> {b['s1']['iters']+b['s2']['iters']} iters total, fuel {b['s2']['fuel']:.4f} "
          f"({b['gap']:+.2f}%)")
json.dump([{k: v for k, v in r.items() if k != "s1" or v} for r in rows],
          open("step18_cold2.json", "w"), indent=1, default=str)

"""Is (eta0=100, eta_s0=200) an interior optimum or an edge of the grid?"""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
SEEDS = (0.80, 0.90, 1.00, 1.10, 1.20)
def score(es0, e0, eu0):
    sgs, its, fus, nc = [], [], [], 0
    for f in SEEDS:
        o = run(x0, m, warm=(X, U, sig*f),
                extra=dict(eta0=e0, eta_s0=es0, eta_u0=eu0, r1t=0.0, r2t=0.30), iters=60, ctx=ctx)
        s = summary(o, "", sig_ref=sig)
        sgs.append(s["sigma"]); its.append(s["iters"]); fus.append(s["fuel"])
        nc += (s["reason"] == "converged")
    sgs, fus = np.array(sgs), np.array(fus)
    return dict(eta_s0=es0, eta0=e0, eta_u0=eu0, spread=float(sgs.max()-sgs.min()),
                mean_it=float(np.mean(its)), max_it=int(max(its)), nconv=nc,
                fuel_spread=float(fus.max()-fus.min()), sigmas=sgs.tolist())

print(f"{'eta_s0':>7}{'eta0':>7}{'eta_u0':>9} | {'spread':>8}{'mean_it':>9}{'max':>5}{'conv':>6}{'fuel_sp':>9}")
rows = []
for es0, e0 in itertools.product((150., 200., 300., 400.), (50., 100., 300., 1000.)):
    r = score(es0, e0, 1e-4); rows.append(r)
    print(f"{es0:>7.0f}{e0:>7.0f}{1e-4:>9.0e} | {r['spread']:>8.1f}{r['mean_it']:>9.1f}"
          f"{r['max_it']:>5}{r['nconv']:>4}/5{r['fuel_spread']:>9.4f}")
rows.sort(key=lambda r: (-r["nconv"], r["spread"], r["mean_it"]))
b = rows[0]; print(f"\n-> best radii eta_s0={b['eta_s0']:g} eta0={b['eta0']:g}\n")
for eu0 in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
    r = score(b["eta_s0"], b["eta0"], eu0); rows.append(r)
    print(f"{b['eta_s0']:>7.0f}{b['eta0']:>7.0f}{eu0:>9.0e} | {r['spread']:>8.1f}{r['mean_it']:>9.1f}"
          f"{r['max_it']:>5}{r['nconv']:>4}/5{r['fuel_spread']:>9.4f}")
rows.sort(key=lambda r: (-r["nconv"], r["spread"], r["mean_it"]))
b = rows[0]
print(f"\nBEST eta0={b['eta0']:g} eta_s0={b['eta_s0']:g} eta_u0={b['eta_u0']:g} r1t=0.0 r2t=0.30: "
      f"{b['nconv']}/5 converged, spread {b['spread']:.1f} s, mean {b['mean_it']:.1f} it")
json.dump(rows, open("step15_radii.json","w"), indent=1)

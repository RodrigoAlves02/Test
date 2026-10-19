"""Refine the time trust region and the grow threshold on the full seed ladder."""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
SEEDS = (0.80, 0.90, 1.00, 1.10, 1.20)
print(f"target sigma {sig:.1f} s; x_init identical in all {len(SEEDS)} runs per config\n")
print(f"{'r2t':>5}{'eta_s0':>8}{'eta0':>7} | {'spread':>8}{'mean_it':>9}{'max_it':>7}{'conv':>6}"
      f"{'fuel_sp':>9}   sigmas")
rows = []
for r2t, es0, e0 in itertools.product((0.30, 0.50), (25.0, 50.0, 100.0, 200.0), (100.0,)):
    sgs, its, fus, nc = [], [], [], 0
    for f in SEEDS:
        o = run(x0, m, warm=(X, U, sig*f),
                extra=dict(eta0=e0, eta_s0=es0, eta_u0=1e-4, r1t=0.0, r2t=r2t), iters=60, ctx=ctx)
        s = summary(o, "", sig_ref=sig)
        sgs.append(s["sigma"]); its.append(s["iters"]); fus.append(s["fuel"])
        nc += (s["reason"] == "converged")
    sgs, fus = np.array(sgs), np.array(fus)
    r = dict(r2t=r2t, eta_s0=es0, eta0=e0, spread=float(sgs.max()-sgs.min()),
             mean_it=float(np.mean(its)), max_it=int(max(its)), nconv=nc,
             fuel_spread=float(fus.max()-fus.min()), sigmas=sgs.tolist())
    rows.append(r)
    print(f"{r2t:>5.2f}{es0:>8.0f}{e0:>7.0f} | {r['spread']:>8.1f}{r['mean_it']:>9.1f}"
          f"{r['max_it']:>7}{nc:>4}/{len(SEEDS)}{r['fuel_spread']:>9.4f}   "
          + " ".join(f"{v:.0f}" for v in sgs))
rows.sort(key=lambda r: (-r["nconv"], r["spread"], r["mean_it"]))
b = rows[0]
print(f"\nBEST r2t={b['r2t']} eta_s0={b['eta_s0']:g} eta0={b['eta0']:g}: "
      f"{b['nconv']}/{len(SEEDS)} converged, spread {b['spread']:.1f} s, mean {b['mean_it']:.1f} it")
json.dump(rows, open("step14_refine.json","w"), indent=1)

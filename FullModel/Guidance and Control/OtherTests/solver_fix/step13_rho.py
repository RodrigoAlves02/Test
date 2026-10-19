"""rho here is a defect-contraction ratio, 1 - ||d_new||/||d_ref||, not a classic model-quality
ratio. Re-calibrate the thresholds to that meaning and see if sigma can travel."""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import gen_dataset as G

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
SEEDS = (0.90, 1.00, 1.10)
GRID = [("base   ", 0.36, 0.90), ("r.00/.50", 0.00, 0.50), ("r.00/.40", 0.00, 0.40),
        ("r.10/.50", 0.10, 0.50), ("r.00/.30", 0.00, 0.30), ("r.20/.60", 0.20, 0.60)]
print(f"target sigma {sig:.1f} s; same x_init every run\n" + HDR)
best, rows = None, []
for (tag, r1t, r2t), es0 in itertools.product(GRID, (0.5, 50.0)):
    sgs, its, ok = [], [], True
    for f in SEEDS:
        o = run(x0, m, warm=(X, U, sig*f),
                extra=dict(eta0=max(1.0, es0*2), eta_s0=es0, eta_u0=1e-4, r1t=r1t, r2t=r2t),
                iters=60, ctx=ctx)
        s = summary(o, f"{tag} es{es0:g} x{f:.2f}", sig_ref=sig); print(line(s))
        sgs.append(s["sigma"]); its.append(s["iters"]); ok &= (s["reason"] == "converged")
    sgs = np.array(sgs); spread = float(sgs.max() - sgs.min())
    row = dict(tag=tag, r1t=r1t, r2t=r2t, eta_s0=es0, spread=spread,
               mean_it=float(np.mean(its)), max_it=int(max(its)), all_conv=bool(ok),
               sigmas=sgs.tolist())
    rows.append(row)
    print(f"   -> spread {spread:7.1f} s  mean {np.mean(its):5.1f} it  all_conv {ok}\n")
    if best is None or (spread, np.mean(its)) < (best["spread"], best["mean_it"]): best = row
print(f"BEST: {best['tag']} eta_s0={best['eta_s0']:g}  spread {best['spread']:.1f} s  "
      f"mean {best['mean_it']:.1f} it")
json.dump(rows, open("step13_rho.json","w"), indent=1)

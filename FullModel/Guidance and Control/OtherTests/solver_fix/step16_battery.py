"""Final selection on a battery where the control actually has to move.

The sigma-seed ladder alone is a trap: it hands every run the exact optimal U, so a tiny eta_u0
scores well by freezing the control rather than by solving. These four families break that.
Optimality is scored against the best fuel any configuration found on that same problem.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G

X, U, sig, x0, rf = nominal(); Mw = float(np.exp(x0[6])); ctx = G.Ctx(); K = X.shape[1]
rng = np.random.default_rng(20260921)

PROBS = []                                                    # (name, x_init, mass, warm)
for f in (0.80, 1.00, 1.20):                                  # A: seed sigma wrong, U exact
    PROBS.append((f"A sig x{f:.2f}", x0.copy(), Mw, (X, U, sig*f)))
for d in (50.0, 150.0, 300.0):                                # B: x_init moved, U must adapt
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    xp = x0.copy(); xp[0:3] += v*d
    PROBS.append((f"B disp {d:.0f}m", xp, Mw, (X, U, sig)))
for frac in (0.25, 0.50, 0.75):                               # C: the real MPC case, mid-flight
    j = int(round(frac*(K-1)))
    xj = X[:, j].copy()
    PROBS.append((f"C at {frac:.0%}", xj, float(np.exp(xj[6])),
                  (X[:, j:], U[:, j:], sig*(1.0-j/(K-1)))))
PROBS.append(("D cold", x0.copy(), Mw, None))                 # D: no guess at all

CANDS = {
 "baseline":      dict(eta0=1000., eta_s0=500., eta_u0=0.1,  r1t=0.36, r2t=0.90),
 "eu1e-5":        dict(eta0=300.,  eta_s0=400., eta_u0=1e-5, r1t=0.0,  r2t=0.30),
 "eu1e-4":        dict(eta0=300.,  eta_s0=400., eta_u0=1e-4, r1t=0.0,  r2t=0.30),
 "eu1e-3":        dict(eta0=300.,  eta_s0=400., eta_u0=1e-3, r1t=0.0,  r2t=0.30),
 "small/eu1e-4":  dict(eta0=100.,  eta_s0=200., eta_u0=1e-4, r1t=0.0,  r2t=0.30),
 "eu1e-4/r2t.5":  dict(eta0=300.,  eta_s0=400., eta_u0=1e-4, r1t=0.0,  r2t=0.50),
}
R = {}
for cn, cfg in CANDS.items():
    R[cn] = {}
    for pn, xi, mm, w in PROBS:
        o = run(xi, mm, warm=w, extra=dict(cfg), iters=60, ctx=ctx)
        R[cn][pn] = summary(o, f"{cn} | {pn}", sig_ref=sig)
        s = R[cn][pn]
        print(f"{cn:<14}{pn:<14}{s['reason']:<10}{s['iters']:>4} it  sigma {s['sigma']:>7.1f}  "
              f"fuel {s['fuel']:.4f}  miss {s['miss']:.1e}", flush=True)
    print()

best = {pn: min(R[c][pn]["fuel"] for c in CANDS if R[c][pn]["reason"] == "converged")
        for pn, *_ in PROBS if any(R[c][pn]["reason"] == "converged" for c in CANDS)}
print(f"{'config':<15}{'conv':>6}{'mean_it':>9}{'max_it':>8}{'worst fuel gap':>16}{'mean gap':>10}")
rank = []
for cn in CANDS:
    ss = [R[cn][pn] for pn, *_ in PROBS]
    nc = sum(s["reason"] == "converged" for s in ss)
    gaps = [(R[cn][pn]["fuel"]-best[pn])/best[pn]*100 for pn, *_ in PROBS
            if pn in best and R[cn][pn]["reason"] == "converged"]
    g = dict(name=cn, nconv=nc, mean_it=float(np.mean([s["iters"] for s in ss])),
             max_it=int(max(s["iters"] for s in ss)),
             worst_gap=float(max(gaps)) if gaps else float("nan"),
             mean_gap=float(np.mean(gaps)) if gaps else float("nan"))
    rank.append(g)
    print(f"{cn:<15}{nc:>3}/{len(PROBS)}{g['mean_it']:>9.1f}{g['max_it']:>8}"
          f"{g['worst_gap']:>15.2f}%{g['mean_gap']:>9.2f}%")
rank.sort(key=lambda g: (-g["nconv"], g["worst_gap"], g["mean_it"]))
print(f"\nWINNER: {rank[0]['name']}  {CANDS[rank[0]['name']]}")
json.dump(dict(runs={c: {p: R[c][p] for p in R[c]} for c in R}, rank=rank, cands=CANDS),
          open("step16_battery.json", "w"), indent=1)

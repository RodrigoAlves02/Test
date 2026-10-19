"""Late-trajectory restarts. The closed loop stalls at cycle 6, ~87% along. If the subproblem gets
hard near the end, a restart there should reproduce it without running the whole loop."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G
import scvx.core as C

X, U, sig, x0, rf = nominal(); ctx = G.Ctx(); K = X.shape[1]
TUNED = dict(eta0=300., eta_s0=400., eta_u0=1e-3, r1t=0.0, r2t=0.50)
print(f"{'along':>7}{'t_go_seed':>11}{'it':>4}{'reason':>11}{'t_go':>9}{'wall':>8}"
      f"{'s/iter':>8}   solver mix")
rows = []
for frac in (0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975):
    j = int(round(frac*(K-1))); j = min(j, K-2)
    xj = X[:, j].copy(); mj = float(np.exp(xj[6])); sg = sig*(1.0 - j/(K-1))
    # scale the tuned radii to this shorter problem, the way guidance.nd_cfg does
    dist = float(np.linalg.norm(xj[0:3] - rf))
    cfg = dict(TUNED); cfg["eta0"] = 0.025487*dist; cfg["eta_s0"] = 0.222136*sg
    cfg["eta_u0"] = 0.029975*(20.0/mj)
    t0 = time.perf_counter()
    o = run(xj, mj, warm=(X[:, j:], U[:, j:], sg), extra=cfg, iters=40, ctx=ctx)
    w = time.perf_counter() - t0
    s = summary(o, "", sig_ref=sig)
    mix = {}
    for d in o["tr"]: mix[d["solver"]] = mix.get(d["solver"], 0) + 1
    rows.append(dict(frac=frac, **{k: v for k, v in s.items() if k not in ("label", "wall")},
                     wall=w, mix={str(k): v for k, v in mix.items()}))
    print(f"{frac:>7.3f}{sg:>11.1f}{s['iters']:>4}{s['reason']:>11}{s['sigma']:>9.1f}"
          f"{w:>8.1f}{w/max(s['iters'],1):>8.2f}   "
          + ", ".join(f"{k}x{v}" for k, v in mix.items()), flush=True)
json.dump(rows, open("step25_late.json","w"), indent=1)
bad = [r for r in rows if r["wall"]/max(r["iters"],1) > 2.0]
print("\nslow subproblems (>2 s/iteration): "
      + (", ".join(f"{r['frac']:.3f}" for r in bad) if bad else "none"))

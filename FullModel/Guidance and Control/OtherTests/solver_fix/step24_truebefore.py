"""Is the 'before' column the ORIGINAL solver, or the patched one running the old constants?
cap_clamp=False restores the original caps exactly, so the question can be answered rather than
assumed. Three columns: original driver + shipped constants, patched driver + shipped constants,
patched driver + chosen constants."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
SHIP = dict(eta0=1000., eta_s0=500., eta_u0=0.1, r1t=0.36, r2t=0.90)
CHOSEN = dict(eta0=300., eta_s0=400., eta_u0=1e-3, r1t=0.0, r2t=0.50)
COLS = [("original (cap_clamp off)", {**SHIP, "cap_clamp": False}),
        ("patched, shipped consts",  dict(SHIP)),
        ("patched, chosen consts",   dict(CHOSEN))]
FACT = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
print(f"{'seed':>8} | " + " | ".join(f"{n:>26}" for n, _ in COLS))
res = {n: [] for n, _ in COLS}
for f in FACT:
    cells = []
    for n, cfg in COLS:
        o = run(x0, m, warm=(X, U, sig*f), extra=dict(cfg), iters=60, ctx=ctx)
        s = summary(o, "", sig_ref=sig); res[n].append(s)
        cells.append(f"{s['sigma']:7.1f}s {s['fuel']:7.4f} g{s['eta_s_grew']:<2}{s['iters']:>3}it")
    print(f"{sig*f:>8.1f} | " + " | ".join(f"{c:>26}" for c in cells), flush=True)
print()
for n, _ in COLS:
    y = np.array([s["sigma"] for s in res[n]]); fu = np.array([s["fuel"] for s in res[n]])
    print(f"{n:<28} spread {y.max()-y.min():7.1f} s   fuel {fu.min():.4f}-{fu.max():.4f}   "
          f"eta_s growth events {sum(s['eta_s_grew'] for s in res[n]):>3}   "
          f"converged {sum(s['reason']=='converged' for s in res[n])}/{len(FACT)}")
json.dump(res, open("step24_truebefore.json","w"), indent=1)

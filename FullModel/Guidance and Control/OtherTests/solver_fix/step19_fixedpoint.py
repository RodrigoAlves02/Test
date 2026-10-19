"""Self-consistency: re-solve warm-started from the solver's OWN previous answer, repeatedly.
A converged solver is a fixed point of this map -- round 2 must reproduce round 1. Drift means
the run stopped somewhere that is not a solution. This also produces the true optimum to score
everything else against, and an improved nominal as a by-product."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G
from scvx.core import nl_defects

X, U, sig, x0, rf = nominal(); Mw = float(np.exp(x0[6])); ctx = G.Ctx()
dtau = np.diff(np.linspace(0, 1, X.shape[1]))
TUNED = dict(eta0=300., eta_s0=400., eta_u0=1e-3, r1t=0.0, r2t=0.50)
BASE  = dict(eta0=1000., eta_s0=500., eta_u0=0.1, r1t=0.36, r2t=0.90)

def chain(cfg, name, warm0, rounds=8):
    print(f"--- {name}")
    print(f"{'round':>6}{'reason':>11}{'it':>4}{'sigma':>9}{'dsigma':>9}{'fuel':>9}{'dfuel':>10}"
          f"{'miss':>9}{'defect':>9}")
    w, hist = warm0, []
    for i in range(rounds):
        o = run(x0, Mw, warm=w, extra=dict(cfg), iters=60, ctx=ctx)
        r = o["r"]; s = summary(o, "")
        d = float(np.abs(nl_defects(o["cs"]["dyn"], r["X"], r["U"], float(r["sigma"]), dtau)).sum())
        ds = s["sigma"] - hist[-1]["sigma"] if hist else float("nan")
        df = s["fuel"] - hist[-1]["fuel"] if hist else float("nan")
        s["defect"] = d; hist.append(s)
        print(f"{i+1:>6}{s['reason']:>11}{s['iters']:>4}{s['sigma']:>9.1f}{ds:>9.1f}"
              f"{s['fuel']:>9.4f}{df:>10.5f}{s['miss']:>9.1e}{d:>9.1e}", flush=True)
        w = (r["X"], r["U"], float(r["sigma"]))
        if hist and abs(df) < 1e-4 and abs(ds) < 0.5 and i: break
    print()
    return hist

out = {}
out["tuned_from_nominal"] = chain(TUNED, "TUNED, chained from the shipped nominal", (X, U, sig))
out["base_from_nominal"]  = chain(BASE,  "BASELINE, chained from the shipped nominal", (X, U, sig))
c = run(x0, Mw, warm=None, extra=dict(BASE), iters=60, ctx=ctx)["r"]
out["tuned_from_cold"] = chain(TUNED, "TUNED, chained from a cold baseline solve",
                               (c["X"], c["U"], float(c["sigma"])))

for k, h in out.items():
    f = [s["fuel"] for s in h]; g = [s["sigma"] for s in h]
    print(f"{k:<22} fuel {f[0]:.4f} -> {f[-1]:.4f} ({(f[-1]-f[0])/f[0]*100:+.2f}%)   "
          f"sigma {g[0]:.1f} -> {g[-1]:.1f}   last move dfuel "
          f"{f[-1]-f[-2] if len(f)>1 else float('nan'):+.5f}")
json.dump(out, open("step19_fixedpoint.json","w"), indent=1)

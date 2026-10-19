"""SAME problem, different initial guesses. A converged optimiser must return one answer.
Nothing physical varies here: x_init is identical in every run, only the seed sigma moves."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import gen_dataset as G

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
print(f"x_init fixed; nominal sigma {sig:.1f} s\n" + HDR)
res = []
for f in (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20):
    o = run(x0, m, warm=(X, U, sig*f), extra=dict(eta0=1.0, eta_s0=0.5, eta_u0=1e-4),
            iters=60, ctx=ctx)
    s = summary(o, f"seed sigma x{f:.2f} ({sig*f:6.1f}s)", sig_ref=sig); print(line(s)); res.append(s)
sg = np.array([s["sigma"] for s in res]); fu = np.array([s["fuel"] for s in res])
print(f"\nsigma: {sg.min():.1f} .. {sg.max():.1f}  spread {sg.ptp():.1f} s ({sg.ptp()/sg.mean()*100:.1f}%)")
print(f"fuel : {fu.min():.4f} .. {fu.max():.4f}  spread {fu.ptp():.4f} kg ({fu.ptp()/fu.mean()*100:.1f}%)")
print("-> the answer is a function of the guess; it tracks the seed" if sg.ptp() > 5
      else "-> guess-invariant")
json.dump(res, open("step12_invariance.json","w"), indent=1)

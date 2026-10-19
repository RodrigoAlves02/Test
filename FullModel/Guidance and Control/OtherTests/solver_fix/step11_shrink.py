"""Does sizing the trust region by step quality (not feasibility) reach the same optimum
from every starting guess?  That invariance is the test an optimiser has to pass."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import gen_dataset as G

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
def adaptive(xi, k=2.0):
    mis = float(np.linalg.norm(np.asarray(xi)[0:3] - X[0:3, 0]))
    e0 = max(1.0, k * mis)
    return dict(eta0=e0, eta_s0=max(0.5, e0/20.0), eta_u0=max(1e-4, e0*1e-4))

rng = np.random.default_rng(20260921)
cases = [("exact", x0.copy())]
for disp in (10.0, 50.0, 150.0, 270.0):
    d = rng.normal(size=3); d /= np.linalg.norm(d); xp = x0.copy(); xp[0:3] += d*disp
    cases.append((f"d{disp:.0f}m", xp))

print(f"true optimum sigma {sig:.1f} s\n" + HDR)
out = {}
for soa in (True, False):
    tag = "SoA" if soa else "quality"
    sigs = []
    for lab, xi in cases:
        e = adaptive(xi); e["shrink_on_accept"] = soa
        o = run(xi, m, warm=(X, U, sig), extra=e, iters=60, ctx=ctx)
        s = summary(o, f"{tag} warm {lab}", sig_ref=sig); print(line(s)); sigs.append(s["sigma"])
        out.setdefault(tag, []).append(s)
    # cold, same switch, stock trust region
    o = run(x0, m, warm=None, extra={"shrink_on_accept": soa}, iters=60, ctx=ctx)
    s = summary(o, f"{tag} cold", sig_ref=sig); print(line(s)); out[tag].append(s)
    print(f"   -> warm sigma spread {max(sigs)-min(sigs):8.1f} s   cold-vs-warm "
          f"{s['sigma']-sigs[0]:+.1f} s\n")
json.dump(out, open("step11_shrink.json","w"), indent=1)

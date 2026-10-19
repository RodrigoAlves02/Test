"""Baseline: what the solver does today, and the cap bug in isolation."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import tune, gen_dataset as G

X, U, sig, x0, rf = nominal()
m = float(np.exp(x0[6]))
ctx = G.Ctx()
print(f"nominal: sigma {sig:.2f} s, {X.shape[1]} nodes, m {m:.2f} kg")
print(f"tune   : ETA0 {tune.ETA0} ETA_U0 {tune.ETA_U0} ETA_S0 {tune.ETA_S0}")
from scvx.driver import ETA_MIN, ETA_MAX, ES_MIN, ES_MAX, EU_MIN, EU_MAX
print(f"caps   : ETA_MAX {ETA_MAX:g}  ES_MAX {ES_MAX:g}  EU_MAX {EU_MAX:g}")
print(f"  -> ETA_S0 {tune.ETA_S0} {'>' if tune.ETA_S0>ES_MAX else '<='} ES_MAX {ES_MAX}")
print(f"  -> ETA_U0 {tune.ETA_U0} {'>' if tune.ETA_U0>EU_MAX else '<='} EU_MAX {EU_MAX}\n")
print(HDR)
rows = []
for lab, warm in (("cold", None), ("warm=exact", (X, U, sig))):
    o = run(x0, m, warm=warm, ctx=ctx)
    s = summary(o, lab, sig_ref=sig); rows.append((s, o)); print(line(s))
print()
# does eta_s actually move? dump it for the warm case
tr = rows[1][1]["tr"]
print("warm=exact, per-iteration eta_s / rho:")
for d in tr[:16]:
    print(f"   it{d['it']:3d}  eta_s {d['eta_s']:8.2f}  eta_u {d['eta_u']:7.4f}  eta {d['eta']:9.2f}"
          f"  rho {d['rho']:12.3f}  acc {str(d['accepted'])[0]}")
np.save("baseline_rows.npy", np.array([r[0] for r in rows], dtype=object), allow_pickle=True)

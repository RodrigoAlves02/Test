"""Fix 1: the growth caps sit below the initial radii, so 'grow' clamps downward."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary, HDR, line
import tune, gen_dataset as G
import scvx.driver as D

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
orig = (D.ETA_MAX, D.ES_MAX, D.EU_MAX)
print(f"original caps: ETA_MAX {D.ETA_MAX:g} ES_MAX {D.ES_MAX:g} EU_MAX {D.EU_MAX:g}")
print(HDR)
for lab, caps in (("baseline caps", orig),
                  ("caps >= eta0", (1e6, 4000.0, 1.0))):
    D.ETA_MAX, D.ES_MAX, D.EU_MAX = caps
    for tag, warm in (("cold", None), ("warm=exact", (X, U, sig))):
        o = run(x0, m, warm=warm, ctx=ctx)
        print(line(summary(o, f"{lab} / {tag}", sig_ref=sig)))
D.ETA_MAX, D.ES_MAX, D.EU_MAX = orig

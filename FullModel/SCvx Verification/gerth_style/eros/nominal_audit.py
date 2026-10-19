"""Regenerate the Eros nominal from the current code: run_descent.solve (two-stage, K = 60, config
defaults) + verify.audit (DOP853 rtol 1e-12, constraints at and between nodes). verify.HERE is
redirected so nothing is written into the repository's Figures&data."""
import os, sys, json, time
import numpy as np
SCVX = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "SCvx")); sys.path.insert(0, SCVX)
OUT = os.path.dirname(os.path.abspath(__file__))
import config as C, verify as V, run_descent as RD
store = {}
_orig = RD.solve
def _wrapped(*a, **k):
    out = _orig(*a, **k); store["out"] = out; return out
V.RD.solve = _wrapped
V.HERE = OUT
print("config: K=60  POINTING_DEG", C.POINTING_DEG, " TERMINAL_POINT_DEG", C.TERMINAL_POINT_DEG,
      " SLEW", C.SLEW_MAX_DEG_S, " GLIDESLOPE", C.GLIDESLOPE_DEG, " ALT_KM", C.ALTITUDE_KM, " V0", C.V0_LOCAL, flush=True)
t = time.time()
res = V.audit(K=60, iters=35, tag="_K60_rerun")
o = store["out"]; r1, r2 = o["stage1"], o["stage2"]
summ = dict(wall_s=time.time() - t, audit=res,
            stage1=dict(status=r1.status, it=r1.iters, miss_m=r1.miss_m, tf=r1.tf_s),
            stage2=dict(status=r2.status, it=r2.iters, fuel=r2.fuel_kg, tf=r2.tf_s, dnorm=[r2.dnorm_lo, r2.dnorm_hi]),
            hist=[[{k: (v if not isinstance(v, dict) else v) for k, v in h.items()} for h in H] for H in (r1.history, r2.history)])
json.dump(summ, open(os.path.join(OUT, "nominal_audit.json"), "w"), indent=1, default=float)
print(f"\nDONE in {summ['wall_s']/60:.1f} min", flush=True)

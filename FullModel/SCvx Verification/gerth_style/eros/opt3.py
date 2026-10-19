"""OPT-3 on Eros: the 10-state solver with the slew limit OFF vs the 7-state LCvx solver, same handover
(11.771 km, V0_LOCAL), same site, glide slope 30 deg, pointing 140 deg, terminal pointing 1 deg on the
last node, K = 60, two-stage, then verify.audit (DOP853 fly-out, constraints at and between nodes).
Config is patched BEFORE run_descent/ptr/problem_setup are imported (their defaults bind at import).
usage: opt3.py 10 | 7"""
import os, sys, json, time
import numpy as np
which = sys.argv[1]
SCVX = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "SCvx")); V1 = os.path.join(SCVX, "_v1_zoh_backup")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"opt3_{which}"); os.makedirs(os.path.join(OUT, "Figures&data"), exist_ok=True)
sys.path.insert(0, SCVX)
if which == "7":
    sys.path.insert(0, V1)
import config as C
if which == "7":
    assert os.path.dirname(os.path.abspath(C.__file__)) == V1
    C.TERMINAL_POINT_DEG = 1.0; C.TERMINAL_POINT_NODES = 1      # 10-state default
    C.ALTITUDE_KM = 11.771                                      # 10-state default
import run_descent as RD, verify as V
if which == "10":
    # slew limit OFF the way ptr.run supports it (slew_nd = None). NOT via config: subproblem.py:410
    # reads C.SLEW_MAX_DEG_S for the omega trust weight and crashes on None. The config stays at 2 deg/s,
    # so the trust-region weights are the nominal's.
    _run = RD.ptr.run
    def _noslew(*a, **k):
        k["slew_max_deg_s"] = None; return _run(*a, **k)
    RD.ptr.run = _noslew
from dynamics import NX
store = {}; _orig = RD.solve
def _wrapped(*a, **k):
    out = _orig(*a, **k); store["out"] = out; return out
V.RD.solve = _wrapped; V.HERE = OUT
print("config:", NX, "states  POINTING", C.POINTING_DEG, " TERM", C.TERMINAL_POINT_DEG, C.TERMINAL_POINT_NODES,
      " SLEW", getattr(C, "SLEW_MAX_DEG_S", "n/a"), " GS", C.GLIDESLOPE_DEG, " ALT", C.ALTITUDE_KM, " V0", C.V0_LOCAL, flush=True)
t = time.time()
res = V.audit(K=60, iters=60, tag=f"_opt3_{which}")
o = store["out"]; r1, r2 = o["stage1"], o["stage2"]; sc = o["setup"]["scales"]
row = dict(formulation=f"{NX}-state", wall_min=(time.time() - t) / 60, audit=res,
           stage1=dict(status=r1.status, it=r1.iters, miss=r1.miss_m, tf=r1.tf_s),
           stage2=dict(status=r2.status, it=r2.iters, fuel=r2.fuel_kg, tf=r2.tf_s, miss=r2.miss_m, defect_m=r2.defect_m))
if NX == 10:
    row["stage2"]["dnorm"] = [r2.dnorm_lo, r2.dnorm_hi]
else:
    m = np.exp(r2.X[6]); row["stage2"]["lcvx_gap_N"] = float(np.max((r2.U[3, :-1] - np.linalg.norm(r2.U[0:3, :-1], axis=0)) * m[:-1] * sc.A))
J = lambda z: z.tolist() if hasattr(z, "tolist") else str(z)
print("ROW " + json.dumps(row, default=J), flush=True)
json.dump(dict(row, hist=[r1.history, r2.history]), open(os.path.join(OUT, "opt3.json"), "w"), indent=1, default=J)

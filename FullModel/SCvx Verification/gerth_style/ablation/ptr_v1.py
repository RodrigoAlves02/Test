"""Same question on the 7-state LCvx solver (SCvx/_v1_zoh_backup): repo default loop vs ratio test vs
classic PTR (quadratic price, no cap, fixed weight, no objective stopping). TFC min-max / max-min-max, K = 60.
usage: ptr_v1.py <case> <variant>   variant: default | ratio | ptr<w>"""
import os, sys, json, time, tempfile
import numpy as np
case, variant = sys.argv[1], sys.argv[2]
FM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
V1 = os.path.join(FM, "SCvx", "_v1_zoh_backup"); SCVX = os.path.join(FM, "SCvx"); VER = os.path.join(FM, "SCvx Verification")
if variant.startswith("ptr"):
    src = open(os.path.join(V1, "subproblem.py")).read()
    a = "        cons += [cp.norm(cp.hstack([X[:, k] - Xref[:, k], U[:, k] - Uref[:, k]])) <= eta[k]]\n    cons += [cp.abs(s - sref) <= eta_s]"
    assert a in src
    src = src.replace(a, "        cons += [cp.sum_squares(cp.hstack([X[:, k] - Xref[:, k], U[:, k] - Uref[:, k]])) <= eta[k]]\n    cons += [cp.square(s - sref) <= eta_s]")
    tmp = tempfile.mkdtemp(); open(os.path.join(tmp, "subproblem.py"), "w").write(src)
sys.path.insert(0, SCVX); sys.path.insert(0, VER); sys.path.insert(0, V1)
if variant.startswith("ptr"):
    sys.path.insert(0, tmp)
import config as C
assert os.path.dirname(os.path.abspath(C.__file__)) == V1
import benchmark as B
C.TMIN, C.TMAX = B.TMIN, B.TMAX; C.MWET, C.MDRY = B.M_WET, B.M_DRY; C.ISP, C.G0, C.ALPHA = B.ISP, B.G0, B.ALPHA
C.W_BODY = np.zeros(3); C.VTOUCH = 0.0; C.GLIDESLOPE_DEG = 89.9; C.POINTING_DEG = 180.0
C.TERMINAL_POINT_DEG = 180.0; C.TERMINAL_POINT_NODES = 0
_t = C.tf_bounds
C.tf_bounds = lambda v0n, rng, **kw: _t(v0n, rng, m_wet=B.M_WET, m_dry=B.M_DRY, t_max=B.TMAX, t_min=B.TMIN,
                                        g_max=abs(B.G_MARS[2]))
import ptr, subproblem as SP
from dynamics import Dynamics, NX
assert NX == 7
cs = B.CASES[case]; r0, v0 = cs["r0"], cs["v0"]; rf = np.zeros(3); n = np.array([0, 0, 1.0])
sc = C.Scales.build(r0, rf, m_wet=B.M_WET, t_max=B.TMAX); dyn = Dynamics(sc, field=B.ConstantField())
x0 = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(B.M_WET)]])
kw = dict(K=60, mode="fuel", D_nd=0.0, iters=150, verbose=False, term_point_deg=180.0, term_point_nodes=0,
          tol_obj=1e-6, tol_defect_m=1e-2)
if variant == "ratio":
    kw.update(ratio_test=True)
elif variant.startswith("ptr"):
    w = float(variant[3:])
    kw.update(R0=1e3, Rs0=1e3, R_bounds=(1e-4, 1e3), w_tr=w, w_ts=w, w_growth=1.0, tol_obj=0.0, tol_abs=0.0)
t = time.time(); r = ptr.run(dyn, sc, x0, rf / sc.L, n, **kw); wall = time.time() - t
H = r.history; J = np.array([h.get("jphys", np.nan) for h in H])
BEST = {"min-max": 180.3387, "max-min-max": 275.5594}[case]
hit = np.where(np.abs(J - BEST) / BEST < 1e-3)[0]
out = dict(formulation="7-state", case=case, variant=variant, status=r.status, it=r.iters, mp=r.fuel_kg, tf=r.tf_s,
           wall=wall, first_within_0p1pct_of_best=(int(hit[0]) if len(hit) else None),
           J_at=[round(float(J[i]), 3) for i in (4, 9, 19, 39, 79, len(J) - 1) if i < len(J)],
           patched=os.path.dirname(SP.__file__) != V1)
print(json.dumps(out), flush=True)

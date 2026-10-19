"""Follow-up to gerth_cases.py. Case A: FT-1 with the glide-slope rows REMOVED (patched copy of
subproblem.py, the only change), so the trajectory may go underground as in Gerth/A&P.
Cases B, C: Sec 7.2 / 7.3 with the dry mass lowered to 1305 kg, to see whether the 1505 kg dry
mass (400 kg propellant) was binding."""
import os, sys, time, json
import numpy as np
import tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "..", "SCvx"))
case = sys.argv[1]
if case == "A":
    # Patched COPY of subproblem.py in a temp dir: gs_deg=None removes the glide-slope rows. The
    # repository file is not touched.
    src = open(os.path.join(SCVX, "subproblem.py")).read()
    old = """    for k in range(K - 1):
        dr = X[0:3, k] - X[0:3, K - 1]
        cons += [cp.norm(P @ dr) <= tg * (n_hat @ dr)]"""
    assert old in src, "subproblem.py changed; update the patch"
    src = src.replace(old, old.replace("for k in range(K - 1):", "for k in (range(K - 1) if gs_deg is not None else []):"))
    src = src.replace("tg = np.tan(np.radians(gs_deg))", "tg = np.tan(np.radians(gs_deg)) if gs_deg is not None else 0.0")
    tmp = tempfile.mkdtemp(); open(os.path.join(tmp, "subproblem.py"), "w").write(src)
    sys.path.insert(0, tmp)
sys.path.insert(1, SCVX); sys.path.insert(2, os.path.abspath(os.path.join(HERE, "..")))
import config as C
g = 3.7114; g0 = 9.807; m0 = 1905.0; Isp_eff = 225.0 * np.cos(np.radians(27.0))
mdry = 1505.0 if case == "A" else 1305.0
Tl, Th = 4971.8, 13258.2
C.TMIN, C.TMAX, C.MWET, C.MDRY = Tl, Th, m0, mdry
C.ISP, C.G0 = Isp_eff, g0; C.ALPHA = 1.0 / (Isp_eff * g0)
C.W_BODY = np.zeros(3); C.VTOUCH = 0.0
C.GLIDESLOPE_DEG = 89.9; C.POINTING_DEG = 180.0; C.TERMINAL_POINT_DEG = 180.0; C.TERMINAL_POINT_NODES = 0
_tfb = C.tf_bounds
C.tf_bounds = lambda v0n, rng, **kw: _tfb(v0n, rng, m_wet=m0, m_dry=mdry, t_max=Th, t_min=Tl, g_max=g, v_touch=0.0)
import ptr, benchmark as B, subproblem as SP
print("subproblem from", SP.__file__, flush=True)
from dynamics import Dynamics
r0 = np.array([2000.0, 0.0, 1500.0]); v0 = np.array([100.0, 0.0, -75.0]); rf = np.zeros(3); n = np.array([0.0, 0.0, 1.0])
sc = C.Scales.build(r0, rf, m_wet=m0, t_max=Th)
dyn = Dynamics(sc, field=B.ConstantField(np.array([0.0, 0.0, -g])))
x0 = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(m0)], n])
tf, gs, pt = {"A": (72.0, None, 180.0), "B": (81.0, 86.0, 180.0), "C": (95.0, 86.0, 35.0)}[case]
t = time.time()
r = ptr.run(dyn=dyn, scales=sc, x_init=x0, rf_nd=rf / sc.L, n_hat=n, K=int(tf) + 1, mode="fuel", D_nd=0.0,
            iters=300, verbose=False, slew_max_deg_s=None, a_init=None, term_point_deg=180.0, term_point_nodes=0,
            gs_deg=gs, point_deg=pt, fix_tf_s=tf, tol_obj=1e-6, tol_defect_m=1e-2)
z = r.X[2] * sc.L
m = np.exp(r.X[6]); T = r.U[3] * sc.A * m
h = r.history[-1]
row = dict(case=case, tf=tf, gs=gs, pt=pt, mdry=mdry, status=r.status, it=r.iters, mp=r.fuel_kg,
           zmin=float(z.min()), t_under=float(np.sum(z[:-1] < -1e-6) * tf / (len(z) - 1)),
           mass_floor_hit=bool(m.min() <= mdry + 1e-3), vc_ratio=h["vc_ratio"], vc_worst=h["vc_worst"],
           defect_m=r.defect_m, rel=h["rel"], wall=time.time() - t, throttle=(T / 16572.7).tolist())
print(json.dumps({k: v for k, v in row.items() if k != "throttle"}), flush=True)
json.dump(row, open(os.path.join(HERE, f"gerth_cases2_{case}.json"), "w"), indent=1, default=float)

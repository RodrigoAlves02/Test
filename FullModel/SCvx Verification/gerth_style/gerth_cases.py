"""Reproduce Gerth's FT-1 / Sec. 7.1-7.3 numbers (Acikmese & Ploen 2007 Mars case 1) with the user's
UNMODIFIED 10-state solver at Gerth's own tf and dt = 1 s. Gerth: Table 6.3, Figs 7.2, 7.6, 7.9."""
import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "..", "SCvx"))
sys.path.insert(0, SCVX); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import config as C
g = 3.7114; g0 = 9.807; m0 = 1905.0; mdry = 1505.0; Isp_eff = 225.0 * np.cos(np.radians(27.0))
Tl, Th = 4971.8, 13258.2
C.TMIN, C.TMAX, C.MWET, C.MDRY = Tl, Th, m0, mdry
C.ISP, C.G0 = Isp_eff, g0; C.ALPHA = 1.0 / (Isp_eff * g0)
C.W_BODY = np.zeros(3); C.VTOUCH = 0.0
C.GLIDESLOPE_DEG = 89.9; C.POINTING_DEG = 180.0; C.TERMINAL_POINT_DEG = 180.0; C.TERMINAL_POINT_NODES = 0
_tfb = C.tf_bounds
C.tf_bounds = lambda v0n, rng, **kw: _tfb(v0n, rng, m_wet=m0, m_dry=mdry, t_max=Th, t_min=Tl,
                                          g_max=g, v_touch=0.0)
import ptr, benchmark as B
from dynamics import Dynamics
r0 = np.array([2000.0, 0.0, 1500.0]); v0 = np.array([100.0, 0.0, -75.0]); rf = np.zeros(3)
n = np.array([0.0, 0.0, 1.0])
sc = C.Scales.build(r0, rf, m_wet=m0, t_max=Th)
dyn = Dynamics(sc, field=B.ConstantField(np.array([0.0, 0.0, -g])))
x0 = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(m0)], n])

def solve(tf, gs=89.9, pt=180.0, K=None, iters=300, warm=None):
    K = K or int(round(tf)) + 1
    t = time.time()
    r = ptr.run(dyn=dyn, scales=sc, x_init=x0, rf_nd=rf / sc.L, n_hat=n, K=K, mode="fuel", D_nd=0.0,
                iters=iters, verbose=False, slew_max_deg_s=None, a_init=None, term_point_deg=180.0,
                term_point_nodes=0, gs_deg=gs, point_deg=pt, fix_tf_s=tf, warm=warm,
                tol_obj=1e-6, tol_defect_m=1e-2)
    z = r.X[2] * sc.L
    return r, dict(tf=tf, K=K, gs=gs, pt=pt, status=r.status, it=r.iters, mp=r.fuel_kg,
                   zmin=float(z.min()), wall=time.time() - t)

cases = [("FT-1 (A&P case 1, unconstrained)", 72.0, 89.9, 180.0, "A&P 387.9 kg / Gerth 389.9 kg; Gerth z_min < 0 (30 s subsurface)"),
         ("Sec 7.2 glide slope 4 deg (=86 deg from vertical)", 81.0, 86.0, 180.0, "Gerth 401.6 kg at 81 s"),
         ("Sec 7.3 + thrust pointing 35 deg", 95.0, 86.0, 35.0, "Gerth tof 95 s (mass not tabulated)")]
res = []
for name, tf, gs, pt, ref in cases:
    r, row = solve(tf, gs, pt)
    row["name"] = name; row["ref"] = ref; res.append(row)
    print(f"{name:52s} tf {tf:5.1f} s K {row['K']:3d}  {row['status']:9s} it {row['it']:3d}  "
          f"m_p {row['mp']:8.3f} kg  z_min {row['zmin']:9.2f} m  [{row['wall']:.0f} s]   ref: {ref}", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "gerth_cases.json"), "w"), indent=1, default=float)

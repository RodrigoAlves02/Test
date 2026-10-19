"""Gerth CVX-1 ported to the user's SCvx: free-fall then full-thrust vertical landing (Meditch 1964).
Analytic reference solved to machine precision (2 equations, 2 unknowns), then the user's
UNMODIFIED solver is run on the same problem at several K."""
import os, sys, time, json
import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "..", "SCvx"))
sys.path.insert(0, SCVX)
import config as C

# ---- Gerth Table 6.6 inputs --------------------------------------------------------------
g, g0, m0, Isp, Tl, Th, h0 = 1.6249, 9.807, 100.0, 300.0, 0.0, 243.735, 1000.0
ve = g0 * Isp; mdot = Th / ve

def state_after(ts, tb):
    v1 = -g * ts; h1 = h0 - 0.5 * g * ts**2
    u = 1.0 - mdot * tb / m0
    v = v1 - g * tb - ve * np.log(u)
    h = h1 + v1 * tb - 0.5 * g * tb**2 + ve * (m0 / mdot) * (u * np.log(u) - u + 1.0)
    return h, v

ts, tb = fsolve(lambda p: state_after(*p), [20.0, 39.0], xtol=1e-14)
ana = dict(ts=ts, tgo=ts + tb, mp=mdot * tb)
print("ANALYTIC  t_s %.6f s  t_go %.6f s  m_p %.6f kg   (residual %s)" % (ts, ts + tb, mdot * tb,
      np.abs(state_after(ts, tb))))

# ---- configure the user's solver exactly as run_verification.py does -----------------------
C.TMIN, C.TMAX = Tl, Th
C.MWET, C.MDRY = m0, 50.0
C.ISP, C.G0 = Isp, g0
C.ALPHA = 1.0 / (Isp * g0)
C.W_BODY = np.zeros(3)
C.VTOUCH = 0.0
C.GLIDESLOPE_DEG = 89.9
C.POINTING_DEG = 180.0
C.TERMINAL_POINT_DEG = 180.0
C.TERMINAL_POINT_NODES = 0
_tfb = C.tf_bounds
# NB config.tf_bounds divides by t_min, so Tmin = 0 (Gerth's case) breaks the upper bound
# (same weakness as Gerth eq. 6.99). The lower bound is kept; the upper bound is capped at 300 s.
C.tf_bounds = lambda v0n, rng, **kw: (_tfb(v0n, rng, m_wet=m0, m_dry=50.0, t_max=Th, t_min=1e-9,
                                           g_max=g, v_touch=0.0)[0], 300.0)
import ptr
from dynamics import Dynamics
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import benchmark as B

def run(K, warm=None, iters=150):
    r0 = np.array([0.0, 0.0, h0]); rf = np.zeros(3); n = np.array([0.0, 0.0, 1.0])
    sc = C.Scales.build(r0, rf, m_wet=m0, t_max=Th)
    dyn = Dynamics(sc, field=B.ConstantField(np.array([0.0, 0.0, -g])))
    x0 = np.concatenate([r0 / sc.L, np.zeros(3), [np.log(m0)], n])
    t = time.time()
    r = ptr.run(dyn=dyn, scales=sc, x_init=x0, rf_nd=rf / sc.L, n_hat=n, K=K, mode="fuel",
                D_nd=0.0, iters=iters, verbose=False, slew_max_deg_s=None, a_init=None,
                term_point_deg=180.0, term_point_nodes=0, warm=warm,
                tol_obj=1e-6, tol_defect_m=1e-2)
    m = np.exp(r.X[6]); T = r.U[3] * sc.A * m          # thrust magnitude, N
    tk = np.linspace(0, r.tf_s, K)
    on = np.where(T[:-1] > 0.9 * Th)[0]                 # Gerth's switching definition
    ts_num = tk[on[0]] if len(on) else np.nan
    # impulse-equivalent switch: the burn delivers sum(T_k dt) at full thrust, so it started at
    ts_eq = r.tf_s - float(np.sum(T[:-1]) * (r.tf_s / (K - 1))) / Th
    return dict(K=K, status=r.status, it=r.iters, tgo=r.tf_s, mp=r.fuel_kg, ts=ts_num, ts_eq=ts_eq,
                rf=float(np.linalg.norm(r.X[0:3, -1]) * sc.L), vf=float(np.linalg.norm(r.X[3:6, -1]) * sc.v),
                dt=r.tf_s / (K - 1), wall=time.time() - t, defect=r.defect_m,
                Tmid=int(np.sum((T[:-1] > 0.05 * Th) & (T[:-1] < 0.95 * Th)))), (r.X, r.U, r.s)

rows = []; warm = None
for K in (31, 61, 121, 241):
    row, warm = run(K, warm=warm)
    rows.append(row)
    print("K %4d dt %.3f s  %-9s it %3d  t_go %.4f (%+.3f%%)  m_p %.4f (%+.3f%%)  t_s(0.9Th) %.3f (%+.3f s)  t_s(impulse) %.4f (%+.4f s)  "
          "|r_f| %.1e |v_f| %.1e  intermediate-thrust nodes %d  [%.0f s]" % (
          row["K"], row["dt"], row["status"], row["it"], row["tgo"], 100 * (row["tgo"] / ana["tgo"] - 1),
          row["mp"], 100 * (row["mp"] / ana["mp"] - 1), row["ts"], row["ts"] - ana["ts"], row["ts_eq"], row["ts_eq"] - ana["ts"],
          row["rf"], row["vf"], row["Tmid"], row["wall"]), flush=True)
json.dump(dict(analytic=ana, rows=rows), open(os.path.join(os.path.dirname(__file__), "vertical_drop.json"), "w"),
          indent=1, default=float)

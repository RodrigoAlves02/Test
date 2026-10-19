"""Is the 10-state FT-1 result (392.28 kg) a local optimum, or a parameterisation effect?
Warm-start the 10-state solver (glide-slope rows removed, same patch as gerth_cases2 A) from the
convex LCvx solution, which flies underground at 388.28 kg, and see where it converges."""
import os, sys, json, numpy as np, cvxpy as cp
HERE = os.path.dirname(os.path.abspath(__file__))
import tempfile
SCVX = os.path.abspath(os.path.join(HERE, "..", "..", "SCvx"))
src = open(os.path.join(SCVX, "subproblem.py")).read()
old = """    for k in range(K - 1):
        dr = X[0:3, k] - X[0:3, K - 1]
        cons += [cp.norm(P @ dr) <= tg * (n_hat @ dr)]"""
assert old in src, "subproblem.py changed; update the patch"
src = src.replace(old, old.replace("for k in range(K - 1):", "for k in (range(K - 1) if gs_deg is not None else []):"))
src = src.replace("tg = np.tan(np.radians(gs_deg))", "tg = np.tan(np.radians(gs_deg)) if gs_deg is not None else 0.0")
_tmp = tempfile.mkdtemp(); open(os.path.join(_tmp, "subproblem.py"), "w").write(src)
sys.path.insert(0, _tmp); sys.path.insert(1, SCVX); sys.path.insert(2, os.path.abspath(os.path.join(HERE, "..")))
import config as C
g = 3.7114; g0 = 9.807; m0 = 1905.0; mdry = 1505.0; Isp = 225.0 * np.cos(np.radians(27.0)); a = 1 / (Isp * g0)
Tl, Th = 4971.8, 13258.2
C.TMIN, C.TMAX, C.MWET, C.MDRY = Tl, Th, m0, mdry; C.ISP, C.G0 = Isp, g0; C.ALPHA = a
C.W_BODY = np.zeros(3); C.VTOUCH = 0.0; C.GLIDESLOPE_DEG = None; C.POINTING_DEG = 180.0; C.TERMINAL_POINT_DEG = 180.0; C.TERMINAL_POINT_NODES = 0
_tfb = C.tf_bounds
C.tf_bounds = lambda v0n, rng, **kw: _tfb(v0n, rng, m_wet=m0, m_dry=mdry, t_max=Th, t_min=Tl, g_max=g, v_touch=0.0)
import ptr, benchmark as B
from dynamics import Dynamics
r0 = np.array([2000.0, 0, 1500.0]); v0 = np.array([100.0, 0, -75.0]); n = np.array([0, 0, 1.0]); rf = np.zeros(3)
tf = 72.0; N = 72; dt = tf / N; K = N + 1
# --- convex LCvx solution (same as lcvx_ft1.py) ---
gv = np.array([0, 0, -g])
r = cp.Variable((3, K)); v = cp.Variable((3, K)); z = cp.Variable(K); u = cp.Variable((3, N)); s = cp.Variable(N)
t = np.arange(N) * dt; z0 = np.log(m0 - a * Th * t); mu1 = Tl * np.exp(-z0); mu2 = Th * np.exp(-z0)
c = [r[:, 0] == r0, v[:, 0] == v0, z[0] == np.log(m0), r[:, N] == 0, v[:, N] == 0, z >= np.log(mdry)]
for k in range(N):
    c += [v[:, k + 1] == v[:, k] + (u[:, k] + gv) * dt, r[:, k + 1] == r[:, k] + v[:, k] * dt + 0.5 * (u[:, k] + gv) * dt ** 2,
          z[k + 1] == z[k] - a * s[k] * dt, cp.norm(u[:, k]) <= s[k],
          mu1[k] * (1 - (z[k] - z0[k]) + 0.5 * cp.square(z[k] - z0[k])) <= s[k], s[k] <= mu2[k] * (1 - (z[k] - z0[k]))]
cp.Problem(cp.Maximize(z[N]), c).solve(solver="CLARABEL")
print(f"LCvx: {m0 - np.exp(z.value[N]):.3f} kg, z_min {r.value[2].min():.1f} m")
# --- map it to the 10-state variables as a warm start ---
sc = C.Scales.build(r0, rf, m_wet=m0, t_max=Th)
dyn = Dynamics(sc, field=B.ConstantField(gv))
X = np.zeros((10, K)); U = np.zeros((4, K))
X[0:3] = r.value / sc.L; X[3:6] = v.value * sc.T / sc.L; X[6] = z.value
uu = np.hstack([u.value, u.value[:, -1:]])
d = uu / np.linalg.norm(uu, axis=0); X[7:10] = d
U[3, :N] = s.value / sc.A; U[3, N] = U[3, N - 1]
snd = sc.t_nd(tf); dtau = 1.0 / (K - 1)
for k in range(K - 1):
    ax = np.cross(d[:, k], d[:, k + 1]); axn = np.linalg.norm(ax)
    th = np.arctan2(axn, np.clip(d[:, k] @ d[:, k + 1], -1, 1))
    U[0:3, k] = ax / axn * th / (snd * dtau) if axn > 1e-12 else 0.0
U[0:3, K - 1] = U[0:3, K - 2]
x0 = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(m0)], d[:, 0]])
import subproblem as SP, inspect
assert SP.build.__defaults__ is not None and inspect.signature(SP.build).parameters["gs_deg"].default is None, "cone still on"
print("subproblem:", SP.__file__, "| default gs_deg:", inspect.signature(SP.build).parameters["gs_deg"].default)
kw = dict(dyn=dyn, scales=sc, x_init=x0, rf_nd=rf / sc.L, n_hat=n, K=K, mode="fuel", D_nd=0.0, iters=300,
          verbose=False, slew_max_deg_s=None, a_init=None, term_point_deg=180.0, term_point_nodes=0,
          point_deg=180.0, fix_tf_s=tf, tol_obj=1e-6, tol_defect_m=1e-2)
cold = ptr.run(**kw)
zc = cold.X[2] * sc.L
print(f"COLD: {cold.status} it {cold.iters} m_p {cold.fuel_kg:.3f} z_min {zc.min():.1f} t_under {np.sum(zc[:-1] < -1e-6)*dt:.0f} s", flush=True)
res = ptr.run(warm=(X, U, snd), **kw)
_unused = dict(dyn=dyn, scales=sc, x_init=x0, rf_nd=rf / sc.L, n_hat=n, K=K, mode="fuel", D_nd=0.0, iters=300,
              verbose=False, slew_max_deg_s=None, a_init=None, term_point_deg=180.0, term_point_nodes=0,
              gs_deg=None, point_deg=180.0, fix_tf_s=tf, warm=(X, U, snd), tol_obj=1e-6, tol_defect_m=1e-2)
zz = res.X[2] * sc.L
out = dict(lcvx_mp=float(m0 - np.exp(z.value[N])), warm_status=res.status, warm_it=res.iters, warm_mp=res.fuel_kg,
           warm_zmin=float(zz.min()), warm_t_under=float(np.sum(zz[:-1] < -1e-6) * dt), cold_mp=cold.fuel_kg, cold_status=cold.status, cold_zmin=float(zc.min()))
print(json.dumps(out))
json.dump(out, open(os.path.join(HERE, "ft1_warm.json"), "w"), indent=1)

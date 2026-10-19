"""Unit tests of the user's dynamics.py / discretize(), in the spirit of Gerth UNIT TEST 6.1 (DYN-1, DYN-2),
plus Jacobian, Taylor-remainder and sub-step convergence tests that Gerth did not have."""
import sys, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "..", "SCvx"))
sys.path.insert(0, SCVX)
import config as C
C.W_BODY = np.zeros(3)
import dynamics as DY
from dynamics import Dynamics, discretize, NX, NU

class ConstField:
    def __init__(self, g): self.gv = np.asarray(g, float)
    def g(self, r): return self.gv
    def grad(self, r): return np.zeros((3, 3))

class PointMass:                     # nonlinear field with analytic gradient
    def __init__(self, mu): self.mu = mu
    def g(self, r):
        r = np.asarray(r, float); n = np.linalg.norm(r); return -self.mu * r / n**3
    def grad(self, r):
        r = np.asarray(r, float); n = np.linalg.norm(r)
        return -self.mu * (np.eye(3) / n**3 - 3.0 * np.outer(r, r) / n**5)

out = {}
# ---------------- DYN-1: free fall, constant gravity, Gamma = 0 ---------------------------------
g = 1.6249; h0 = 100.0; K = 11; tf = 10.0
sc = C.Scales.build(np.array([0, 0, h0]), np.zeros(3), m_wet=100.0, t_max=243.735)
dyn = Dynamics(sc, field=ConstField([0, 0, -g]))
s = sc.t_nd(tf)
X = np.zeros((NX, K)); U = np.zeros((NU, K))
X[0:3, 0] = [0, 0, h0 / sc.L]; X[6, :] = np.log(100.0); X[9, :] = 1.0     # d = up, Gamma = 0
Ad, Bd, Sd, zd, Xp = discretize(dyn, np.repeat(X[:, :1], K, 1), U, s)
x = X[:, 0].copy(); lin = [x.copy()]
for k in range(K - 1):
    x = Ad[k] @ x + Bd[k] @ U[:, k] + Sd[k] * s + zd[k]; lin.append(x.copy())
lin = np.array(lin).T
t = np.linspace(0, tf, K)
h_exact = h0 - 0.5 * g * t**2; v_exact = -g * t
e_h = np.max(np.abs(lin[2] * sc.L - h_exact)); e_v = np.max(np.abs(lin[5] * sc.v - v_exact))
print(f"DYN-1 free fall   N=10 dt=1 s : final h {lin[2,-1]*sc.L:.6f} m (exact {h_exact[-1]:.6f}), "
      f"v {lin[5,-1]*sc.v:.6f} m/s (exact {v_exact[-1]:.6f});  max node error h {e_h:.2e} m, v {e_v:.2e} m/s")
out["DYN1"] = dict(h_f=lin[2, -1] * sc.L, h_exact=h_exact[-1], v_f=lin[5, -1] * sc.v, v_exact=v_exact[-1],
                   err_h=e_h, err_v=e_v)

# ---------------- DYN-2: hover, Gamma = g (thrust ACCELERATION), d = up ---------------------------
Isp, g0 = 300.0, 9.80665
C.ISP = Isp; C.G0 = g0; C.ALPHA = 1.0 / (Isp * g0)
K = 101; tf = 100.0
dyn = Dynamics(sc, field=ConstField([0, 0, -g])); s = sc.t_nd(tf)
X = np.zeros((NX, K)); U = np.zeros((NU, K))
X[0:3, :] = np.array([[0], [0], [h0 / sc.L]]); X[6, :] = np.log(100.0); X[9, :] = 1.0
U[3, :] = g / sc.A
Ad, Bd, Sd, zd, Xp = discretize(dyn, X, U, s)
x = X[:, 0].copy()
for k in range(K - 1):
    x = Ad[k] @ x + Bd[k] @ U[:, k] + Sd[k] * s + zd[k]
m_f = np.exp(x[6]); m_exact = 100.0 * np.exp(-g * tf / (Isp * g0))
print(f"DYN-2 hover       N=100 dt=1 s: h_f {x[2]*sc.L:.9f} m, v_f {x[5]*sc.v:.2e} m/s, "
      f"m_f {m_f:.9f} kg (exact {m_exact:.9f}), rel. mass error {abs(m_f/m_exact-1):.2e}")
out["DYN2"] = dict(h_f=x[2] * sc.L, v_f=x[5] * sc.v, m_f=m_f, m_exact=m_exact)

# ---------------- JAC: analytic Jacobians vs central finite differences ---------------------------
C.W_BODY = np.array([0.0, 0.0, 2 * np.pi / (5.27 * 3600)])
mu = 6.687e-11 * 6.687e15                         # ~ Eros GM [m^3/s^2]
r0 = np.array([9000.0, 4000.0, 6000.0])
sc = C.Scales.build(r0, r0 * 0.6, m_wet=600.0, t_max=20.0)
dyn = Dynamics(sc, field=PointMass(mu))
rng = np.random.default_rng(1)
dvec = rng.normal(size=3); dvec /= np.linalg.norm(dvec)
xb = np.concatenate([r0 / sc.L, rng.normal(size=3) * 0.3, [np.log(590.0)], dvec])
ub = np.concatenate([rng.normal(size=3) * 0.5, [0.6]])
A, B = dyn.jac(xb, ub)
Afd = np.zeros_like(A); Bfd = np.zeros_like(B); h = 1e-6
for i in range(NX):
    e = np.zeros(NX); e[i] = h; Afd[:, i] = (dyn.f(xb + e, ub) - dyn.f(xb - e, ub)) / (2 * h)
for i in range(NU):
    e = np.zeros(NU); e[i] = h; Bfd[:, i] = (dyn.f(xb, ub + e) - dyn.f(xb, ub - e)) / (2 * h)
eA = np.max(np.abs(A - Afd)) / np.max(np.abs(A)); eB = np.max(np.abs(B - Bfd)) / np.max(np.abs(B))
print(f"JAC  analytic vs FD (point-mass gravity + rotating frame): max rel err A {eA:.1e}, B {eB:.1e}")
out["JAC"] = dict(eA=eA, eB=eB)

# ---------------- TAYLOR: one-segment linear model error vs perturbation size (expect slope 2) -----
K = 21; s = sc.t_nd(300.0)
Xr = np.repeat(xb[:, None], K, 1); Ur = np.repeat(ub[:, None], K, 1)
Xr = dyn.propagate(xb, Ur, s, K)                  # dynamically consistent reference
Ad, Bd, Sd, zd, Xp = discretize(dyn, Xr, Ur, s, n_sub=10)
k = 7
dx = rng.normal(size=NX); du = rng.normal(size=NU); ds = 1.0
def prop1(x, u, s_, nsub=200):
    hh = s_ * (1.0 / (K - 1)) / nsub
    for _ in range(nsub): x = dyn.rk4(x, u, hh)
    return x
eps_list = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]; errs = []
xn_ref = prop1(Xr[:, k], Ur[:, k], s); xl_ref = Ad[k] @ Xr[:, k] + Bd[k] @ Ur[:, k] + Sd[k] * s + zd[k]
for eps in eps_list:
    x0 = Xr[:, k] + eps * dx * 1e-2; u0 = Ur[:, k] + eps * du * 1e-1; s0 = s + eps * ds * 1e-2 * s
    xn = prop1(x0, u0, s0)
    xl = Ad[k] @ x0 + Bd[k] @ u0 + Sd[k] * s0 + zd[k]
    errs.append(np.linalg.norm((xn - xn_ref) - (xl - xl_ref)))   # remove the O(h^2) discretisation floor
sl = np.polyfit(np.log(eps_list), np.log(errs), 1)[0]
cons = np.linalg.norm(prop1(Xr[:, k], Ur[:, k], s) - (Ad[k] @ Xr[:, k] + Bd[k] @ Ur[:, k] + Sd[k] * s + zd[k]))
print(f"TAYLOR linearisation remainder: errors {['%.1e' % e for e in errs]} -> log-log slope {sl:.2f} "
      f"(2 expected); zero-perturbation consistency {cons:.1e}")
out["TAYLOR"] = dict(eps=eps_list, err=errs, slope=sl, consistency=cons)

# ---------------- SUBSTEP: Van Loan midpoint sub-stepping convergence (expect order 2) ------------
ref = discretize(dyn, Xr, Ur, s, n_sub=256)
ns = [1, 2, 4, 8, 16]; eAd = []
for n_ in ns:
    D = discretize(dyn, Xr, Ur, s, n_sub=n_)
    eAd.append(max(np.max(np.abs(D[0] - ref[0])), np.max(np.abs(D[1] - ref[1])), np.max(np.abs(D[3] - ref[3]))))
sl2 = np.polyfit(np.log(ns), np.log(eAd), 1)[0]
print(f"SUBSTEP Ad/Bd/zd error vs n_sub {ns}: {['%.1e' % e for e in eAd]} -> order {-sl2:.2f} (2 expected)")
out["SUBSTEP"] = dict(n_sub=ns, err=eAd, order=-sl2)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "unit_dynamics.json"), "w"), indent=1, default=float)

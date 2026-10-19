"""Is the open-loop divergence of the nominal caused by the direction-state defects?
Fly the stored control (descent_K60_rerun.npz) through DOP853 twice on the grid field:
  free  : x carried node to node (what verify.audit does)
  reset : same, but d is reset to the plan's d_k at every node (r, v, m still carried)"""
import os, sys
os.sched_setaffinity(0, {1})
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "SCvx"))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "disp"))
from scipy.integrate import solve_ivp
import config as C
from dynamics import Dynamics
from grid_gravity import GridField
z = np.load("Figures&data/descent_K60_rerun.npz"); X, U, s, K = z["X"], z["U"], float(z["s"]), int(z["K"])
sc = C.Scales(L=float(z["L"]), T=float(z["T"]), A=float(z["A"]))
dyn = Dynamics(sc, field=GridField()); n = z["n_hat"]; rf = z["rf"]; Pl = np.eye(3) - np.outer(n, n)
h = s / (K - 1)
# u-const : (omega_k, Gamma_k) held, Gamma = T/m, so thrust in N falls as mass burns (the solver's own ZOH)
# T-const : thrust in N held at Gamma_k * m_k (a real constant-thrust engine), so Gamma(t) = Gamma_k m_k / m(t)
def fT(y, uk, zk):
    u = uk.copy(); u[3] = uk[3] * np.exp(zk - y[6]); return dyn.f(y, u)
for mode in ("free u-const", "reset u-const", "free T-const", "reset T-const"):
    x = X[:, 0].copy()
    for k in range(K - 1):
        if mode.startswith("reset"): x[7:10] = X[7:10, k]
        uk = U[:, k]; zk = X[6, k]
        rhs = (lambda t, y: dyn.f(y, uk)) if "u-const" in mode else (lambda t, y: fT(y, uk, zk))
        x = solve_ivp(rhs, (0.0, h), x, method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1]
    r = x[0:3] * sc.L - rf
    print(f"{mode:13s}: lateral miss {np.linalg.norm(Pl @ r):8.3f} m   altitude {n @ r:8.3f} m   speed {np.linalg.norm(x[3:6]) * sc.L / sc.T:.4f} m/s"
          f"   |dr| vs plan {np.linalg.norm(x[0:3] - X[0:3, -1]) * sc.L:8.3f} m   grid fallbacks {dyn.field.n_exact}")

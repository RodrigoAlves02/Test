"""Fixed-step RK8 against adaptive DOP853 on an IDENTICAL control, so the integrator is the only
thing that differs. Reference: DOP853 at rtol 1e-13."""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, OT); sys.path.insert(0, os.path.abspath(os.path.join(OT, "..")))
from scipy.integrate import solve_ivp
import closed_loop as CL
import guidance as GD, gen_dataset as G, tune

with GD._in_scvx_dir(): ctx = G.Ctx()
z = np.load(os.path.join(HERE, "stall_state.npz"))
t0 = float(z["t"]); y10 = np.asarray(z["y"], float)
y0 = y10[0:7]                                   # the rewritten rhs is 7-state
alpha = 1.0 / (tune.ISP * 9.80665)
n = [0]; graw = ctx.truth.g
def g(r):
    n[0] += 1; return graw(r)
T_vec = 8.5 * np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])
ARC = 30.0

n[0] = 0; t1 = time.perf_counter()
ref = solve_ivp(CL.rhs, (t0, t0+ARC), y0, method="DOP853", rtol=1e-13, atol=1e-15,
                args=(T_vec, g, alpha))
print(f"reference DOP853 rtol 1e-13: {n[0]} gravity calls, {time.perf_counter()-t1:.2f} s")
yref = ref.y[:, -1]

print(f"\n{'method':>22}{'calls':>8}{'wall':>8}{'pos err':>12}{'vel err':>12}{'mass err':>11}")
for rtol in (1e-12, 1e-10, 1e-8):
    n[0] = 0; t1 = time.perf_counter()
    s = solve_ivp(CL.rhs, (t0, t0+ARC), y0, method="DOP853", rtol=rtol, atol=rtol*1e-2,
                  args=(T_vec, g, alpha))
    w = time.perf_counter()-t1; ye = s.y[:, -1]
    print(f"{'DOP853 rtol %.0e' % rtol:>22}{n[0]:>8}{w:>8.2f}"
          f"{np.linalg.norm(ye[0:3]-yref[0:3]):>12.2e}{np.linalg.norm(ye[3:6]-yref[3:6]):>12.2e}"
          f"{abs(ye[6]-yref[6]):>11.2e}")
for dt in (1.0, 0.5, 0.25, 0.125):
    n[0] = 0; t1 = time.perf_counter()
    _t, ye = CL.rk8_fixed(CL.rhs, t0, t0+ARC, y0, dt, args=(T_vec, g, alpha))
    w = time.perf_counter()-t1
    print(f"{'RK8 dt %.3f s' % dt:>22}{n[0]:>8}{w:>8.2f}"
          f"{np.linalg.norm(ye[0:3]-yref[0:3]):>12.2e}{np.linalg.norm(ye[3:6]-yref[3:6]):>12.2e}"
          f"{abs(ye[6]-yref[6]):>11.2e}")

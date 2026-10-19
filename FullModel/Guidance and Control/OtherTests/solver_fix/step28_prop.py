"""How expensive is one closed-loop propagation arc, and what drives it?"""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, OT); sys.path.insert(0, os.path.abspath(os.path.join(OT, "..")))
from scipy.integrate import solve_ivp
import closed_loop as CL
import guidance as GD, gen_dataset as G, tune
from actuator_model import Actuator

with GD._in_scvx_dir(): ctx = G.Ctx()
z = np.load(os.path.join(HERE, "stall_state.npz"))
t0 = float(z["t"]); y0 = np.asarray(z["y"], float)
act = Actuator(tau=0.5, slew_max_deg_s=2.0, t_min=tune.TMIN, t_max=tune.TMAX)
alpha = 1.0 / (tune.ISP * 9.80665)
n = [0]
gb = ctx.truth.g
def counted(r):
    n[0] += 1; return gb(r)
ctx.truth.g = counted
# a command that genuinely asks for a turn, as at the cycle where it stalls
T_cmd = 8.5 * np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])
print(f"{'rtol':>8}{'max_step':>10}{'rhs calls':>11}{'wall':>9}{'us/call':>9}")
for rtol, ms in ((1e-12, 7.5), (1e-12, 30.0), (1e-10, 7.5), (1e-8, 7.5), (1e-6, 7.5)):
    n[0] = 0; t1 = time.perf_counter()
    s = solve_ivp(CL.rhs, (t0, t0+30.0), y0, method="DOP853", rtol=rtol, atol=rtol*1e-2,
                  args=(T_cmd, ctx.truth.g, act, alpha), dense_output=True, max_step=ms)
    w = time.perf_counter() - t1
    print(f"{rtol:>8.0e}{ms:>10.1f}{n[0]:>11}{w:>9.2f}{w/max(n[0],1)*1e6:>9.0f}", flush=True)

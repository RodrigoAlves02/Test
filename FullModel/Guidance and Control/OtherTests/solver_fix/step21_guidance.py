"""End-to-end check of guidance.py: receding-horizon cycles seeded from the nominal, t_go included.
Cycle 1 has no previous plan, so it must fall back to the nominal rather than to a straight line."""
import sys, os, numpy as np
GC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, GC)
import guidance as GD
from guidance import SCvxGuidance, load_nominal, _Rz, OMEGA, WZ


def body_to_inertial(r_B, v_B, t):
    R = _Rz(WZ * t)
    return R @ r_B, R @ (v_B + np.cross(OMEGA, r_B))

Xn, Un, sn = load_nominal()
print(f"nominal warm start: K={Xn.shape[1]} nodes, t_go {sn:.1f} s")
g = SCvxGuidance(nominal=(Xn, Un, sn), iters=60)
x0 = Xn[:, 0]
r_I, v_I = body_to_inertial(x0[0:3], x0[3:6], 0.0)
mass = float(np.exp(x0[6]))

print(f"\n{'cycle':>6}{'t':>8}{'it':>4}{'reason':>11}{'exit':>7}{'t_go':>9}{'solve_s':>9}")
t, dt = 0.0, 30.0
for c in range(6):
    p = g.solve(t, r_I, v_I, mass)
    if not p.ok:
        print(f"{c+1:>6}{t:>8.1f}  FAILED: {p.error}"); break
    print(f"{c+1:>6}{t:>8.1f}{p.iters:>4}{p.reason:>11}{p.exit_via:>7}{p.sigma:>9.1f}"
          f"{p.solve_s:>9.2f}", flush=True)
    # fly one control interval along the plan itself, so the next cycle is a genuine shift
    j = int(round(dt / (p.sigma / (p.X.shape[1] - 1))))
    j = min(max(j, 1), p.X.shape[1] - 2)
    xj = p.X[:, j]
    t = t + j * p.sigma / (p.X.shape[1] - 1)
    r_I, v_I = body_to_inertial(xj[0:3], xj[3:6], t)
    mass = float(np.exp(xj[6]))
print(f"\nmean solve {g.mean_solve_s:.2f} s over {g.n_calls} cycles")

"""Replay the exact guidance solve that stalls in the closed loop, with the iteration exposed."""
import sys, os, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
import guidance as GD
from guidance import nd_cfg, inertial_to_body, load_nominal
import gen_dataset as G, tune
from scvx.core import LAST_FAIL

z = np.load(os.path.join(HERE, "stall_state.npz"))
t = float(z["t"]); y = np.asarray(z["y"], float)
print(f"replaying cycle k={int(z['k'])} at t={t:.1f}s")
r_B, v_B = inertial_to_body(y[0:3], y[3:6], t)
x_init = np.concatenate([r_B, v_B, [np.log(float(y[6]))]])
with GD._in_scvx_dir():
    ctx = G.Ctx()
dist = float(np.linalg.norm(x_init[0:3] - ctx.rf))
print(f"x_init: alt-ish range to site {dist:.1f} m, mass {float(y[6]):.2f} kg, "
      f"|v| {np.linalg.norm(v_B):.4f} m/s")

Xn, Un, sn = load_nominal()
g = GD.SCvxGuidance(nominal=(Xn, Un, sn), iters=120, ctx=ctx)
over, dist, sig0 = nd_cfg(x_init, float(y[6]), ctx.rf, tune, sigma_ref=sn)
print(f"nd_cfg -> eta0 {over['eta0']:.3f}  eta_s0 {over['eta_s0']:.3f}  "
      f"eta_u0 {over['eta_u0']:.3e}  tol_sig {over['tol_sig']:.3f}  tol_traj {over['tol_traj']:.3f}")

saved = (tune.ETA0, tune.ETA_U0, tune.ETA_S0, tune.RHO_SHRINK, tune.RHO_GROW, dict(tune.EXTRA_CFG))
tune.ETA0, tune.ETA_U0, tune.ETA_S0 = over["eta0"], over["eta_u0"], over["eta_s0"]
tune.RHO_SHRINK, tune.RHO_GROW = GD.RHO_SHRINK_FIX, GD.RHO_GROW_FIX
tune.EXTRA_CFG = {**saved[5], "tol_sig": over["tol_sig"], "tol_traj": over["tol_traj"]}
tr = []
t0 = time.perf_counter()
try:
    with GD._in_scvx_dir():
        cs, r, tr, cfg = G.solve(ctx, x_init, float(y[6]), warm=(Xn, Un, sn), iters=120)
finally:
    (tune.ETA0, tune.ETA_U0, tune.ETA_S0, tune.RHO_SHRINK, tune.RHO_GROW,
     tune.EXTRA_CFG) = saved
w = time.perf_counter() - t0
print(f"\nwall {w:.1f}s, {len(tr)} traced iterations, reason {r['reason']}")
print(f"{'it':>3}{'solver':>10}{'eta':>10}{'eta_s':>9}{'eta_u':>10}{'rho':>11}{'dL':>12}{'A':>2}")
for d in tr[:40]:
    print(f"{d['it']:>3}{str(d['solver']):>10}{d['eta']:>10.3f}{d['eta_s']:>9.3f}"
          f"{d['eta_u']:>10.2e}{d['rho']:>11.3f}{d['dL']:>12.3f}{'Y' if d['accepted'] else '.':>2}")
print(f"\nuntraced iterations (subproblem returned None or a bad status): {120-len(tr) if r['reason']!='converged' else 0}")
from scvx.core import LAST_FAIL as LF
print(f"last subproblem attempt statuses: {LF}")

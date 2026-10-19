"""Is the stalled state genuinely infeasible, or just badly guessed?"""
import sys, os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", ""))); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import guidance as GD
from guidance import nd_cfg, inertial_to_body, load_nominal
import gen_dataset as G, tune
import scvx.core as C

z = np.load(os.path.join(HERE, "stall_state.npz"))
t = float(z["t"]); y = np.asarray(z["y"], float)
r_B, v_B = inertial_to_body(y[0:3], y[3:6], t)
mass = float(y[6]); x_init = np.concatenate([r_B, v_B, [np.log(mass)]])
with GD._in_scvx_dir(): ctx = G.Ctx()
rf = np.asarray(ctx.rf, float); n_hat = np.asarray(ctx.n_hat, float)
dist = float(np.linalg.norm(x_init[0:3] - rf))
alt = float(n_hat @ (x_init[0:3] - rf))
Xn, Un, sn = load_nominal(); K = Xn.shape[1]
# where along the nominal does this range sit?
dn = np.linalg.norm(Xn[0:3] - rf[:, None], axis=0)
j = int(np.argmin(np.abs(dn - dist)))
print(f"state: range {dist:.1f} m, altitude {alt:.1f} m, |v| {np.linalg.norm(v_B):.4f} m/s, "
      f"mass {mass:.2f} kg")
print(f"nominal reaches this range at node {j}/{K-1}, its t_go there {sn*(1-j/(K-1)):.1f} s")
print(f"glideslope: cone half-angle {tune.GS if hasattr(tune,'GS') else G.GS} deg, "
      f"state angle from vertical "
      f"{np.degrees(np.arccos(np.clip(n_hat@(x_init[0:3]-rf)/max(dist,1e-9),-1,1))):.2f} deg")

def attempt(label, warm, sig_ref):
    over, _, _ = nd_cfg(x_init, mass, rf, tune, sigma_ref=sig_ref)
    sv = (tune.ETA0, tune.ETA_U0, tune.ETA_S0, tune.RHO_SHRINK, tune.RHO_GROW, dict(tune.EXTRA_CFG))
    tune.ETA0, tune.ETA_U0, tune.ETA_S0 = over["eta0"], over["eta_u0"], over["eta_s0"]
    tune.RHO_SHRINK, tune.RHO_GROW = GD.RHO_SHRINK_FIX, GD.RHO_GROW_FIX
    tune.EXTRA_CFG = {**sv[5], "tol_sig": over["tol_sig"], "tol_traj": over["tol_traj"]}
    t0 = time.perf_counter()
    try:
        with GD._in_scvx_dir():
            cs, r, tr, cfg = G.solve(ctx, x_init, mass, warm=warm, iters=60)
        w = time.perf_counter() - t0
        print(f"{label:<34}{r['reason']:>11}{len(tr):>4}it  t_go {r['sigma']:>7.1f}  "
              f"{w:>6.1f}s  last={C.LAST_FAIL[-2:] if len(tr)==0 else 'n/a'}")
    finally:
        (tune.ETA0, tune.ETA_U0, tune.ETA_S0, tune.RHO_SHRINK, tune.RHO_GROW,
         tune.EXTRA_CFG) = sv

print()
attempt("full nominal (wrong t_go)", (Xn, Un, sn), sn)
sgj = sn*(1.0 - j/(K-1))
attempt(f"nominal tail from node {j}", (Xn[:, j:], Un[:, j:], sgj), sgj)
attempt("cold (straight line)", None, None)
for f in (0.6, 0.8, 1.0, 1.3, 1.8):
    attempt(f"nominal tail, t_go x{f:.1f}", (Xn[:, j:], Un[:, j:], sgj*f), sgj*f)

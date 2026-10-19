"""7-state: make the direction steps between holds flyable. Two affine rows, same form as the
pointing row, on a PATCHED COPY of _v1_zoh_backup/subproblem.py (the repository file is untouched):

  slew budget between holds   ah_ref[k] . a[k+1] >= cos(w_max*beta*h_ref) * G[k+1]   and the mirror row
  reachable first hold        c_hat . a[0]        >= cos(w_max*beta*h_ref) * G[0]
                              (c_hat = engine direction at handover = the nominal 7-state first hold)

with an L1-priced slack (weight w_vc, like the virtual control) so a reference that violates the
budget cannot make the subproblem infeasible. h_ref is the hold length at the reference s.

usage: disp7_slew.py <seed> <beta> [core]       beta = 0 runs the unmodified rows (baseline)
Also flies every plan through a slew-limited direction tracker (geodesic turn at w_max, Gamma held)."""
import os, sys, json, time, tempfile, shutil
seed, beta = int(sys.argv[1]), float(sys.argv[2]); core = int(sys.argv[3]) if len(sys.argv) > 3 else 2
os.sched_setaffinity(0, {core})
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "SCvx"))
V1 = os.path.join(SCVX, "_v1_zoh_backup")
W_MAX_DEG_S = 2.0

src = open(os.path.join(V1, "subproblem.py")).read()
anchor = "    # -- trust region, penalised rather than imposed (PTR)"
assert src.count(anchor) == 1 and src.count("    pen = (w_vc * cp.sum(cp.abs(nu)) + w_tr * cp.sum(eta) + w_ts * eta_s)") == 1
rows = '''    # -- [PATCH] flyable direction steps -------------------------------------------------------
    _xi = None
    if SLEW_ROWS is not None and SLEW_ROWS["beta"] > 0:
        _h = float(sref) * scales.T / (K - 1)
        _c = float(np.cos(min(np.radians(SLEW_ROWS["w_deg_s"]) * SLEW_ROWS["beta"] * _h, np.pi)))
        _A = np.asarray(Uref[0:3], float); _ah = _A / np.maximum(np.linalg.norm(_A, axis=0), 1e-12)
        _xi = cp.Variable(2 * (K - 2) + 1, nonneg=True, name="xi")
        for k in range(K - 2):
            cons += [_ah[:, k] @ U[0:3, k + 1] >= _c * U[3, k + 1] - _xi[2 * k],
                     _ah[:, k + 1] @ U[0:3, k] >= _c * U[3, k] - _xi[2 * k + 1]]
        if SLEW_ROWS.get("c_hat") is not None:
            _ch = np.asarray(SLEW_ROWS["c_hat"], float); _ch = _ch / np.linalg.norm(_ch)
            cons += [_ch @ U[0:3, 0] >= _c * U[3, 0] - _xi[-1]]
        else:
            cons += [_xi[-1] == 0]

'''
src = src.replace(anchor, rows + anchor)
src = src.replace("    pen = (w_vc * cp.sum(cp.abs(nu)) + w_tr * cp.sum(eta) + w_ts * eta_s)",
                  "    pen = (w_vc * cp.sum(cp.abs(nu)) + w_tr * cp.sum(eta) + w_ts * eta_s\n"
                  "           + (w_vc * cp.sum(_xi) if _xi is not None else 0))")
src = src.replace("from dynamics import NX, NU", "from dynamics import NX, NU\nSLEW_ROWS = None", 1)
tmp = tempfile.mkdtemp(prefix="sp7_"); open(os.path.join(tmp, "subproblem.py"), "w").write(src)
for p in (HERE, SCVX, V1, tmp): sys.path.insert(0, p)          # tmp first: ptr imports the patched copy
import config as C
assert os.path.dirname(os.path.abspath(C.__file__)) == V1
C.TERMINAL_POINT_DEG = 1.0; C.TERMINAL_POINT_NODES = 1; C.ALTITUDE_KM = 11.771; C.POINTING_DEG = 180.0
import subproblem as SP
assert os.path.dirname(os.path.abspath(SP.__file__)) == tmp
import problem_setup as PS, ptr
from dynamics import Dynamics
from grid_gravity import GridField
from scipy.integrate import solve_ivp

P0 = PS.build()
r0, v0, m0 = P0["r0"], P0["v0"], C.MWET
if seed > 0:
    rng = np.random.default_rng(seed)
    r0 = r0 + 1000.0 * rng.standard_normal(3); v0 = v0 + 1.0 * rng.standard_normal(3); m0 = m0 + 2.0 * rng.standard_normal()
P = PS.build(m_wet=m0); sc = C.Scales.build(r0, P["r_site"], m_wet=m0)
x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(m0)]]); rf_nd = P["r_site"] / sc.L; n = P["n_hat"]
dyn = Dynamics(sc, field=GridField())
nom = np.load(os.path.join(HERE, "res", "7_0_default.npz")); c_hat = nom["U"][0:3, 0]
SP.SLEW_ROWS = dict(beta=beta, w_deg_s=W_MAX_DEG_S, c_hat=c_hat)
cm = dict(dyn=dyn, scales=sc, x_init=x_init, rf_nd=rf_nd, n_hat=n, K=60, verbose=False,
          iters=200, tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4)
t = time.time()
r1 = ptr.run(mode="error", warm=None, **cm)
r2 = ptr.run(mode="fuel", D_nd=r1.D_nd + 0.1 / sc.L, warm=(r1.X, r1.U, r1.s), **cm)
wall = time.time() - t
X, U, s = r2.X, r2.U, r2.s; K = 60; h = s / (K - 1); h_s = h * sc.T
a = U[0:3, :K - 1]; G = U[3, :K - 1]; an = np.linalg.norm(a, axis=0); ah = a / an
step = np.degrees(np.arccos(np.clip(np.sum(ah[:, 1:] * ah[:, :-1], 0), -1, 1)))
step0 = float(np.degrees(np.arccos(np.clip(ah[:, 0] @ (c_hat / np.linalg.norm(c_hat)), -1, 1))))
Pl = np.eye(3) - np.outer(n, n)


def fly(actuator):
    """Hold Gamma; direction = commanded (ideal) or turned geodesically at W_MAX from where it was."""
    x = X[:, 0].copy(); d = c_hat / np.linalg.norm(c_hat); w = np.radians(W_MAX_DEG_S) * sc.T   # rad per nd time
    for k in range(K - 1):
        tgt = ah[:, k]; Gk = G[k]
        if not actuator:
            rhs = lambda t, y: dyn.f(y, np.concatenate([tgt * Gk, [Gk]]))
            x = solve_ivp(rhs, (0, h), x, method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]; continue
        ang = float(np.arccos(np.clip(d @ tgt, -1, 1)))
        e = tgt - (d @ tgt) * d; e = e / np.linalg.norm(e) if np.linalg.norm(e) > 1e-12 else d
        def dir_at(t, d=d, e=e, ang=ang):
            ph = min(w * t, ang); return np.cos(ph) * d + np.sin(ph) * e
        rhs = lambda t, y: dyn.f(y, np.concatenate([dir_at(t) * Gk, [Gk]]))
        t_turn = min(ang / w, h)
        if t_turn > 0:
            x = solve_ivp(rhs, (0, t_turn), x, method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
        if t_turn < h:
            x = solve_ivp(lambda t, y: dyn.f(y, np.concatenate([tgt * Gk, [Gk]])), (t_turn, h), x,
                          method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
        d = dir_at(t_turn)
    rr = x[0:3] * sc.L - P["r_site"]
    return dict(miss_m=float(np.linalg.norm(Pl @ rr)), alt_m=float(n @ rr), speed=float(np.linalg.norm(x[3:6]) * sc.v))


out = dict(seed=seed, beta=beta, wall_s=wall, s1=dict(status=r1.status, it=r1.iters), s2=dict(status=r2.status, it=r2.iters),
           fuel=r2.fuel_kg, tf=r2.tf_s, hold_s=h_s, budget_deg=W_MAX_DEG_S * beta * h_s if beta > 0 else None,
           step_max_deg=float(step.max()), step0_deg=step0, n_steps_gt10=int((step > 10).sum()),
           turn_frac_worst=float((np.r_[step0, step] / W_MAX_DEG_S / h_s).max()),
           lcvx_gap_rel=float(((G - an) / G).max()), ideal=fly(False), actuator=fly(True))
print(json.dumps(out), flush=True)
os.makedirs(os.path.join(HERE, "res7"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "res7", f"7_{seed}_b{beta:g}.json"), "w"), indent=1)
np.savez(os.path.join(HERE, "res7", f"7_{seed}_b{beta:g}.npz"), X=X, U=U, s=s, L=sc.L, T=sc.T, A=sc.A)
shutil.rmtree(tmp, ignore_errors=True)

"""Fly stored 7-state plans through a slew-limited direction tracker, three command strategies.
  ideal : direction = the plan's hold direction, instantly (what the plan assumes)
  lag   : command = the plan's hold direction; the tracker turns geodesically at W_MAX (2 deg/s)
  lead  : command = the NEXT hold's direction from Dtheta/(2 W_MAX) before the node, so each turn is
          centred on the node and its direction error averages out to first order
Gamma held per hold; RK4 at 0.5 s sub-steps on the grid field; engine starts at c_hat.
usage: fly7.py <npz> [<npz> ...]"""
import os, sys, json
os.sched_setaffinity(0, {int(os.environ.get("CORE", "2"))})
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "SCvx")); V1 = os.path.join(SCVX, "_v1_zoh_backup")
for p in (HERE, SCVX, V1): sys.path.insert(0, p)
import config as C
C.ALTITUDE_KM = 11.771
import problem_setup as PS
from dynamics import Dynamics
from grid_gravity import GridField
P = PS.build(); n = P["n_hat"]; Pl = np.eye(3) - np.outer(n, n); fld = GridField()
c_hat = np.load(os.path.join(HERE, "res", "7_0_default.npz"))["U"][0:3, 0]; c_hat = c_hat / np.linalg.norm(c_hat)
W = np.radians(2.0)


def rot_toward(d, c, maxang):
    ang = float(np.arccos(np.clip(d @ c, -1, 1)))
    if ang <= maxang or ang < 1e-12:
        return c.copy()
    e = c - (d @ c) * d; e /= np.linalg.norm(e)
    return np.cos(maxang) * d + np.sin(maxang) * e


def fly(path, mode, dt=0.5):
    z = np.load(path); X, U, s = z["X"], z["U"], float(z["s"])
    sc = C.Scales(L=float(z["L"]), T=float(z["T"]), A=float(z["A"])); dyn = Dynamics(sc, field=fld)
    K = X.shape[1]; h_s = s * sc.T / (K - 1); tf = s * sc.T
    a = U[0:3, :K - 1]; G = U[3, :K - 1]; ah = a / np.linalg.norm(a, axis=0)
    step = np.r_[np.arccos(np.clip(ah[:, 0] @ c_hat, -1, 1)), np.arccos(np.clip(np.sum(ah[:, 1:] * ah[:, :-1], 0), -1, 1))]
    # sub-steps ALIGNED with the holds: a sub-step straddling a node would fly part of the new hold
    # with the old control (measured: 22.9 m of spurious miss on the ideal flight at dt = 0.5 s).
    x = X[:, 0].copy(); d = c_hat.copy(); err = []
    nsub = int(np.ceil(h_s / dt)); hs = h_s / nsub
    for k in range(K - 1):
        for j in range(nsub):
            t = k * h_s + j * hs
            if mode == "ideal":
                d = ah[:, k]
            else:
                c = ah[:, k]
                if mode == "lead" and k + 1 <= K - 2 and ((k + 1) * h_s - t) <= step[k + 1] / (2 * W) + 1e-9:
                    c = ah[:, k + 1]
                d = rot_toward(d, c, W * hs)
            err.append(np.degrees(np.arccos(np.clip(d @ ah[:, k], -1, 1))))
            x = dyn.rk4(x, np.concatenate([d * G[k], [G[k]]]), hs / sc.T)
    rr = x[0:3] * sc.L - P["r_site"]
    return dict(miss_m=round(float(np.linalg.norm(Pl @ rr)), 3), alt_m=round(float(n @ rr), 3),
                speed=round(float(np.linalg.norm(x[3:6]) * sc.v), 4), mean_dir_err_deg=round(float(np.mean(err)), 3))


for p in sys.argv[1:]:
    row = {"plan": os.path.basename(p)}
    for mode in ("ideal", "lag", "lead"):
        row[mode] = fly(p, mode)
    print(json.dumps(row), flush=True)

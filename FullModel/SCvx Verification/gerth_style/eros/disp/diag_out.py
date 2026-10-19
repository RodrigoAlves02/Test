"""Where do the out-of-grid gravity queries land? Seed 0, 10-state, stage 1, default variant.
Fallbacks here return a point-mass field (fast) and log the query in the site frame (e, n, u)."""
import os, sys, json
os.sched_setaffinity(0, {2})
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "SCvx"))); sys.path.insert(0, HERE)
import config as C, problem_setup as PS, ptr
from dynamics import Dynamics
import grid_gravity as G
class Diag(G.GridField):
    def __init__(self):
        super().__init__(); self.out = []
        self.mu = 6.6743e-11 * 6.692e15
    def both(self, r):
        r = np.asarray(r, float); self.n_calls += 1
        v = self._interp(r)
        if v is None:
            self.n_exact += 1; self.out.append(self.R @ (r - self.rs))
            return -self.mu * r / np.linalg.norm(r) ** 3, np.zeros((3, 3))
        t = v[3:]
        return v[:3], np.array([[t[0], t[3], t[4]], [t[3], t[1], t[5]], [t[4], t[5], t[2]]])
P = PS.build(); sc = P["scales"]; f = Diag()
dyn = Dynamics(sc, field=f)
r1 = ptr.run(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"], n_hat=P["n_hat"], K=60, verbose=True, iters=200,
             tol_obj=1e-7, tol_defect_m=1e-3, tol_defect_v=1e-4, tol_vc_m=1e-4, point_deg=180.0,
             align_theta0_deg=45.0, align_h_m=100.0, mode="error", warm=None)
Q = np.array(f.out)
print("calls", f.n_calls, "out", len(Q))
if len(Q):
    lat = np.hypot(Q[:, 0], Q[:, 1])
    print("e", Q[:, 0].min(), Q[:, 0].max(), " n", Q[:, 1].min(), Q[:, 1].max(), " u", Q[:, 2].min(), Q[:, 2].max())
    print("u<-200:", int((Q[:, 2] < -200).sum()), " u>13000:", int((Q[:, 2] > 13000).sum()), " |lat|>7500:", int((np.abs(Q[:, :2]) > 7500).any(1).sum()),
          " cone+1km NaN cell:", int(((Q[:, 2] >= -200) & (Q[:, 2] <= 13000) & (np.abs(Q[:, :2]) <= 7500).all(1)).sum()))
    for p in (50, 90, 99, 100):
        print(p, np.percentile(Q[:, 2], p), np.percentile(lat, p))
    np.save(os.path.join(HERE, "diag_out.npy"), Q)

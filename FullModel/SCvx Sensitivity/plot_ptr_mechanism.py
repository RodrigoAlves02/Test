#!/usr/bin/env python
"""What one PTR iteration actually does, drawn from a real solve rather than sketched.

The four panels answer the four things that are hard to see from the code alone:

  (a) the trajectory is 40 NODES, not a curve. Each is a decision variable.
  (b) the dynamics are linearised ONCE PER SEGMENT, about the reference there, and the nodes are
      tied together by that linear map alone -- so the leg the REAL dynamics fly from a node does
      not land on the next node. That gap is the defect.
  (c) the defect across all segments, iteration by iteration, as the reference improves. It is not
      monotone, and that non-monotonicity is what step rejection exists to handle.
  (d) how hard each node presses against the trust cap. Pinned at 1.0 nearly everywhere means the
      step size is set by R, not chosen by the optimiser -- which is why the trust-region floor
      matters so much.

Run at K = 40 rather than the flight K = 100 purely so the nodes are far enough apart to see.
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Figures&data")
V1 = os.path.abspath(os.path.join(HERE, "..", "SCvx", "_v1_zoh_backup"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "SCvx")))
sys.path.insert(0, V1)
import config as C
C.POINTING_DEG = 180.0
C.LAT_DEG, C.LON_DEG = -36.975, 95.401
import gravity as GR, ptr, subproblem as SP
from dynamics import Dynamics, discretize

K, N_IT = 40, 8
r_site, n_hat = GR.site_and_normal(lat_deg=C.LAT_DEG, lon_deg=C.LON_DEG)
east = np.cross([0, 0, 1.0], n_hat); east /= np.linalg.norm(east)
north = np.cross(n_hat, east)
r0 = r_site + 11771.0 * n_hat
vl = np.asarray(C.V0_LOCAL, float)
v0 = vl[0] * n_hat + vl[1] * east + vl[2] * north
sc = C.Scales.build(r0, r_site, m_wet=C.MWET)
dyn = Dynamics(sc, field=GR.pinn_field())
x_init = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(C.MWET)]])

r1 = ptr.run(dyn=dyn, scales=sc, x_init=x_init, rf_nd=r_site / sc.L, n_hat=n_hat, K=K,
             mode="error", warm=None, iters=60, verbose=False)
print(f"stage 1: {r1.status}, D = {r1.miss_m:.4f} m")
Xr, Ur, sr = r1.X.copy(), r1.U.copy(), float(r1.s)
D_nd = r1.D_nd + 0.1 / sc.L
R, Rs = 0.35, 0.50
snap = []
for it in range(N_IT):
    Ad, Bd, Sd, zd, _ = discretize(dyn, Xr, Ur, sr, n_sub=10)
    prob, V = SP.build(Ad, Bd, Sd, zd, Xr, Ur, sr, x_init, r_site / sc.L, n_hat, sc,
                       mode="fuel", D_nd=D_nd, w_vc=1e4, w_tr=0.0, w_ts=0.0,
                       vtouch_nd=C.VTOUCH / sc.v, R=R, Rs=Rs,
                       term_point_deg=C.TERMINAL_POINT_DEG,
                       term_point_nodes=C.TERMINAL_POINT_NODES)
    SP.solve(prob)
    Xn, Un, sn = np.asarray(V["X"].value), np.asarray(V["U"].value), float(V["s"].value)
    eta = np.asarray(V["eta"].value)
    legs, dgap = [], np.zeros(K - 1)
    for k in range(K - 1):
        x = Xn[:, k].copy(); h = sn / (K - 1) / 10
        path = [x.copy()]
        for _ in range(10):
            x = dyn.rk4(x, Un[:, k], h); path.append(x.copy())
        legs.append(np.array(path))
        dgap[k] = np.linalg.norm(x[0:3] - Xn[0:3, k + 1]) * sc.L
    snap.append(dict(it=it, X=Xn.copy(), legs=legs, dgap=dgap.copy(), eta=eta.copy(), R=R))
    print(f"  it {it}  max defect {dgap.max():9.3f} m   max eta/R {eta.max()/R:6.1%}")
    Xr, Ur, sr = Xn, Un, sn
    R = min(R * 2, 1.0) if (dgap.max() < 1.0 and it > 2) else max(R / 2, 1e-4)

loc = lambda r: np.array([(r - r_site) @ east, (r - r_site) @ n_hat])
fig, ax = plt.subplots(2, 2, figsize=(7.6, 5.6))
s = snap[0]
P = np.array([loc(s["X"][0:3, k] * sc.L) for k in range(K)])
ax[0][0].plot(P[:, 0] / 1e3, P[:, 1] / 1e3, "o", ms=3.0, color="#1f5fa8", zorder=3,
              label=r"nodes $x_k$, each a decision variable")
for k in range(K - 1):
    Q = np.array([loc(p[0:3] * sc.L) for p in s["legs"][k]])
    ax[0][0].plot(Q[:, 0] / 1e3, Q[:, 1] / 1e3, "-", lw=0.8, color="#c2541b", zorder=2,
                  label=r"leg the REAL dynamics fly from $x_k$" if k == 0 else None)
ax[0][0].plot(0, 0, "*", ms=11, color="#2e7d32", zorder=4, label="site")
ax[0][0].set_xlabel("downrange [km]", fontsize=8)
ax[0][0].set_ylabel("height [km]", fontsize=8)
ax[0][0].legend(fontsize=6.3, frameon=False, loc="upper left")

kw = int(np.argmax(s["dgap"]))
Q = np.array([loc(p[0:3] * sc.L) for p in s["legs"][kw]])
a, b = loc(s["X"][0:3, kw] * sc.L), loc(s["X"][0:3, kw + 1] * sc.L)
ax[0][1].plot(Q[:, 0], Q[:, 1], "-", lw=1.6, color="#c2541b", zorder=2,
              label="where the real dynamics arrive")
ax[0][1].plot([a[0]], [a[1]], "o", ms=7, color="#1f5fa8", zorder=3)
ax[0][1].plot([b[0]], [b[1]], "o", ms=7, color="#1f5fa8", zorder=3,
              label=r"next node $x_{k+1}$, where the LINEAR model says")
ax[0][1].plot(Q[-1, 0], Q[-1, 1], "x", ms=9, mew=2.2, color="#c2541b", zorder=4)
ax[0][1].annotate(rf"$x_{{{kw}}}$", (a[0], a[1]), textcoords="offset points",
                  xytext=(8, -12), fontsize=8)
ax[0][1].annotate(rf"$x_{{{kw+1}}}$", (b[0], b[1]), textcoords="offset points",
                  xytext=(-30, 10), fontsize=8)
ax[0][1].annotate("", xy=(b[0], b[1]), xytext=(Q[-1, 0], Q[-1, 1]),
                  arrowprops=dict(arrowstyle="<->", color="#b00020", lw=1.3))
ax[0][1].text(0.5 * (b[0] + Q[-1, 0]) - 2, 0.5 * (b[1] + Q[-1, 1]) - 55,
              f"defect  {s['dgap'][kw]:.1f} m", color="#b00020", fontsize=7.6, ha="center")
ax[0][1].margins(0.22, 0.30)
ax[0][1].set_xlabel("downrange [m]", fontsize=8)
ax[0][1].set_ylabel("height [m]", fontsize=8)
ax[0][1].legend(fontsize=6.3, frameon=False, loc="upper right")

for s in snap:
    c = plt.cm.viridis(s["it"] / (N_IT - 1))
    ax[1][0].semilogy(np.arange(K - 1), np.maximum(s["dgap"], 1e-6), "-", lw=1.0, color=c,
                      label=f"iteration {s['it']}" if s["it"] in (0, N_IT - 1) else None)
    ax[1][1].plot(np.arange(K), s["eta"] / s["R"], "-", lw=1.0, color=c)
ax[1][0].set_xlabel("segment $k$", fontsize=8)
ax[1][0].set_ylabel("defect [m]: nonlinear leg vs node", fontsize=8)
ax[1][0].legend(fontsize=6.6, frameon=False)
ax[1][1].axhline(1.0, color="#b00020", lw=1.0, ls="--")
ax[1][1].text(1, 1.03, "trust cap $R$", color="#b00020", fontsize=7)
ax[1][1].set_xlabel("node $k$", fontsize=8)
ax[1][1].set_ylabel(r"$\eta_k / R$", fontsize=8)
ax[1][1].set_ylim(0, 1.15)
for a_ in ax.ravel():
    a_.tick_params(labelsize=7)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.085, top=0.985, wspace=0.28, hspace=0.33)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"ptr_mechanism.{ext}"), dpi=200)
print(f"wrote {OUT}/ptr_mechanism.{{pdf,png}}")

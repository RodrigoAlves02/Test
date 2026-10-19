"""Independent single-SOCP lossless-convexification solve of A&P (2007) case 1 at fixed tf (Gerth FT-1),
ZOH with controls on intervals k = 0..N-1 (A&P / user convention). Convex => global optimum of the
relaxed problem, which LCvx theory says is the optimum of the original."""
import numpy as np, cvxpy as cp, sys, json
g = np.array([0, 0, -3.7114]); m0 = 1905.0; mdry = float(sys.argv[2]) if len(sys.argv) > 2 else 1505.0
Isp = 225.0 * np.cos(np.radians(27.0)); g0 = 9.807; a = 1.0 / (Isp * g0); Tl, Th = 4971.8, 13258.2
r0 = np.array([2000.0, 0, 1500.0]); v0 = np.array([100.0, 0, -75.0])
def solve(tf, nosub=False, gs_deg=None, pt_deg=None, N=None):
    N = N or int(round(tf)); dt = tf / N
    r = cp.Variable((3, N + 1)); v = cp.Variable((3, N + 1)); z = cp.Variable(N + 1)
    u = cp.Variable((3, N)); s = cp.Variable(N)
    t = np.arange(N) * dt
    z0 = np.log(m0 - a * Th * t); mu1 = Tl * np.exp(-z0); mu2 = Th * np.exp(-z0)
    c = [r[:, 0] == r0, v[:, 0] == v0, z[0] == np.log(m0), r[:, N] == 0, v[:, N] == 0, z >= np.log(mdry)]
    for k in range(N):
        c += [v[:, k + 1] == v[:, k] + (u[:, k] + g) * dt,
              r[:, k + 1] == r[:, k] + v[:, k] * dt + 0.5 * (u[:, k] + g) * dt ** 2,
              z[k + 1] == z[k] - a * s[k] * dt,
              cp.norm(u[:, k]) <= s[k],
              mu1[k] * (1 - (z[k] - z0[k]) + 0.5 * cp.square(z[k] - z0[k])) <= s[k],
              s[k] <= mu2[k] * (1 - (z[k] - z0[k]))]
        if pt_deg is not None: c += [u[2, k] >= np.cos(np.radians(pt_deg)) * s[k]]
    if nosub: c += [r[2, :] >= 0]
    if gs_deg is not None:
        c += [cp.norm(r[0:2, k]) <= np.tan(np.radians(gs_deg)) * r[2, k] for k in range(N)]
    p = cp.Problem(cp.Maximize(z[N]), c); p.solve(solver="CLARABEL")
    if z.value is None: return dict(tf=tf, nosub=nosub, gs=gs_deg, pt=pt_deg, status=p.status)
    mp = m0 - np.exp(z.value[N]); gap = float(np.max(s.value - np.linalg.norm(u.value, axis=0)))
    return dict(tf=tf, nosub=nosub, gs=gs_deg, pt=pt_deg, status=p.status, mp=mp, zmin=float(r.value[2].min()),
                t_under=float(np.sum(r.value[2] < -1e-6) * dt), lcvx_gap=gap)
if __name__ == "__main__":
    # usage: lcvx_ft1.py [cases|sweep] [m_dry]
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":      # pointing case over t_f
        for tf in [80, 85, 86, 87, 88, 89, 90, 95, 100, 110, 120, 130]:
            r = solve(float(tf), gs_deg=86, pt_deg=35); print(tf, r["status"], round(r.get("mp", float("nan")), 2))
    else:
        for rw in [solve(72), solve(72, nosub=True), solve(81, gs_deg=86), solve(95, gs_deg=86, pt_deg=35)]:
            print(rw)

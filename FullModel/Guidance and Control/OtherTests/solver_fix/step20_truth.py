"""Score every candidate on TRUTH, not on what the solver reports.

The reported `fuel` and `miss` are read off the solver's own discrete solution, which carries a
nonlinear defect. A run can post a low fuel by leaving that defect large. So take the control the
solver returns, fly it through DOP853 at rtol 1e-12 restarted at every ZOH node, and measure where
the vehicle actually ends up and what it actually burned. That number cannot be gamed.
"""
import sys, os, json, numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G

X, U, sig, x0, rf = nominal(); Mw = float(np.exp(x0[6])); ctx = G.Ctx(); K = X.shape[1]
rng = np.random.default_rng(20260921)

def fly(dyn, xi, Uc, sigma, rtol=1e-12, atol=1e-14):
    """Segment-wise DOP853: the propagator is restarted at every thrust switch."""
    h = sigma / (Uc.shape[1] - 1); x = np.asarray(xi, float).copy()
    for k in range(Uc.shape[1] - 1):
        uk = Uc[:, k]
        s = solve_ivp(lambda t, y: dyn.f(y, uk), (0.0, h), x, method="DOP853", rtol=rtol, atol=atol)
        if not s.success: raise RuntimeError(s.message)
        x = s.y[:, -1]
    return x

PROB = {"x0": [(f"sig x{f:.2f}", x0.copy(), Mw, (X, U, sig*f)) for f in (0.80, 1.00, 1.20)]}
PROB["x0"].append(("cold", x0.copy(), Mw, None))
for d in (150.0, 300.0):
    v = rng.normal(size=3); v /= np.linalg.norm(v); xp = x0.copy(); xp[0:3] += v*d
    PROB[f"disp{d:.0f}"] = [("warm", xp, Mw, (X, U, sig))]
for frac in (0.25, 0.50, 0.75):
    j = int(round(frac*(K-1))); xj = X[:, j].copy()
    PROB[f"mpc{frac:.0%}"] = [("shift", xj, float(np.exp(xj[6])),
                              (X[:, j:], U[:, j:], sig*(1.0-j/(K-1))))]

CANDS = {
 "baseline":     dict(eta0=1000., eta_s0=500., eta_u0=0.1,  r1t=0.36, r2t=0.90),
 "eu1e-3":       dict(eta0=300.,  eta_s0=400., eta_u0=1e-3, r1t=0.0,  r2t=0.30),
 "eu1e-3 r2t.5": dict(eta0=300.,  eta_s0=400., eta_u0=1e-3, r1t=0.0,  r2t=0.50),
 "eu3e-3":       dict(eta0=300.,  eta_s0=400., eta_u0=3e-3, r1t=0.0,  r2t=0.30),
}
print(f"{'config':<14}{'problem':<9}{'seed':<9}{'it':>4}{'reason':>11}{'sigma':>8}"
      f"{'fuel_rep':>10}{'fuel_true':>10}{'d_fuel':>9}{'miss_true':>11}")
R = {}
for cn, cfg in CANDS.items():
    R[cn] = {}
    for pn, seeds in PROB.items():
        for sn, xi, mm, w in seeds:
            o = run(xi, mm, warm=w, extra=dict(cfg), iters=60, ctx=ctx)
            r = o["r"]; s = summary(o, "")
            try:
                xe = fly(o["cs"]["dyn"], xi, r["U"], float(r["sigma"]))
                s["fuel_true"] = float(mm - np.exp(xe[6]))
                s["miss_true"] = float(np.linalg.norm(xe[0:3] - rf))
                s["vel_true"] = float(np.linalg.norm(xe[3:6]))
            except Exception as e:
                s["fuel_true"] = s["miss_true"] = s["vel_true"] = float("nan")
                s["fly_err"] = f"{type(e).__name__}: {e}"
            R[cn][f"{pn}|{sn}"] = s
            print(f"{cn:<14}{pn:<9}{sn:<9}{s['iters']:>4}{s['reason']:>11}{s['sigma']:>8.1f}"
                  f"{s['fuel']:>10.4f}{s['fuel_true']:>10.4f}"
                  f"{s['fuel_true']-s['fuel']:>9.4f}{s['miss_true']:>11.2e}", flush=True)
    print()
json.dump(R, open("step20_truth.json","w"), indent=1)

VALID = 1.0     # m: a solution whose flown terminal miss exceeds this is not a solution
REF = {}
for pn, seeds in PROB.items():
    v = [R[c][f"{pn}|{sn}"]["fuel_true"] for c in CANDS for sn, *_ in seeds
         if R[c][f"{pn}|{sn}"]["reason"] == "converged" and R[c][f"{pn}|{sn}"]["miss_true"] < VALID]
    REF[pn] = min(v) if v else float("nan")
print("best FLOWN fuel per problem: " + "  ".join(f"{k} {v:.4f}" for k, v in REF.items()))
print(f"\n{'config':<15}{'valid':>7}{'mean_it':>9}{'worst gap':>11}{'mean gap':>10}{'worst miss':>12}")
rank = []
for cn in CANDS:
    ss = list(R[cn].values())
    val = [s for s in ss if s["reason"] == "converged" and s["miss_true"] < VALID]
    gaps = []
    for pn, seeds in PROB.items():
        f = [R[cn][f"{pn}|{sn}"]["fuel_true"] for sn, *_ in seeds
             if R[cn][f"{pn}|{sn}"]["reason"] == "converged"
             and R[cn][f"{pn}|{sn}"]["miss_true"] < VALID]
        if f and REF[pn] == REF[pn]: gaps.append((min(f)-REF[pn])/REF[pn]*100.0)
    g = dict(name=cn, nvalid=len(val), ntot=len(ss),
             mean_it=float(np.mean([s["iters"] for s in ss])),
             worst_gap=float(max(gaps)) if gaps else float("nan"),
             mean_gap=float(np.mean(gaps)) if gaps else float("nan"),
             worst_miss=float(max(s["miss_true"] for s in ss)))
    rank.append(g)
    print(f"{cn:<15}{len(val):>4}/{len(ss)}{g['mean_it']:>9.1f}{g['worst_gap']:>10.2f}%"
          f"{g['mean_gap']:>9.2f}%{g['worst_miss']:>12.2e}")
rank.sort(key=lambda g: (-g["nvalid"], g["worst_gap"], g["mean_it"]))
print(f"\nWINNER ON TRUTH: {rank[0]['name']}  {CANDS[rank[0]['name']]}")
json.dump(dict(runs=R, ref=REF, rank=rank, cands=CANDS), open("step20_truth.json","w"), indent=1)

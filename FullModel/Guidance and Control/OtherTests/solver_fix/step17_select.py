"""Final configuration selection, scored per PROBLEM (not per run).

The three sigma-seeds and the cold start all share one x_init, so they are ONE problem with four
initialisations; scoring them separately let a configuration look good by reproducing its own guess.
A configuration is judged on: does it converge, and does it find that problem's best fuel from any
initialisation.  Feasibility is checked independently via the nonlinear defect.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run, summary
import gen_dataset as G
from scvx.core import nl_defects

X, U, sig, x0, rf = nominal(); Mw = float(np.exp(x0[6])); ctx = G.Ctx(); K = X.shape[1]
dtau = np.diff(np.linspace(0, 1, K))
rng = np.random.default_rng(20260921)

PROB = {}
PROB["x0"] = [(f"sig x{f:.2f}", x0.copy(), Mw, (X, U, sig*f)) for f in (0.80, 1.00, 1.20)]
PROB["x0"].append(("cold", x0.copy(), Mw, None))
for d in (50.0, 150.0, 300.0):
    v = rng.normal(size=3); v /= np.linalg.norm(v); xp = x0.copy(); xp[0:3] += v*d
    PROB[f"disp{d:.0f}"] = [("warm", xp, Mw, (X, U, sig))]
for frac in (0.25, 0.50, 0.75):
    j = int(round(frac*(K-1))); xj = X[:, j].copy()
    PROB[f"mpc{frac:.0%}"] = [("shift", xj, float(np.exp(xj[6])),
                              (X[:, j:], U[:, j:], sig*(1.0-j/(K-1))))]

CANDS = {
 "baseline":     dict(eta0=1000., eta_s0=500., eta_u0=0.1,  r1t=0.36, r2t=0.90),
 "eu1e-3":       dict(eta0=300.,  eta_s0=400., eta_u0=1e-3, r1t=0.0,  r2t=0.30),
 "eu3e-3":       dict(eta0=300.,  eta_s0=400., eta_u0=3e-3, r1t=0.0,  r2t=0.30),
 "eu1e-2":       dict(eta0=300.,  eta_s0=400., eta_u0=1e-2, r1t=0.0,  r2t=0.30),
 "eu1e-1":       dict(eta0=300.,  eta_s0=400., eta_u0=0.1,  r1t=0.0,  r2t=0.30),
 "eu1e-3 r2t.5": dict(eta0=300.,  eta_s0=400., eta_u0=1e-3, r1t=0.0,  r2t=0.50),
}
R = {}
for cn, cfg in CANDS.items():
    R[cn] = {}
    for pn, seeds in PROB.items():
        for sn, xi, mm, w in seeds:
            o = run(xi, mm, warm=w, extra=dict(cfg), iters=60, ctx=ctx)
            s = summary(o, f"{cn}|{pn}|{sn}", sig_ref=sig)
            r_ = o["r"]
            s["defect"] = float(np.abs(nl_defects(o["cs"]["dyn"], r_["X"], r_["U"],
                                                  float(r_["sigma"]), dtau)).sum())
            R[cn][f"{pn}|{sn}"] = s
            print(f"{cn:<14}{pn:<9}{sn:<9}{s['reason']:<10}{s['iters']:>3}it "
                  f"sig {s['sigma']:>7.1f} fuel {s['fuel']:8.4f} miss {s['miss']:.0e} "
                  f"def {s['defect']:.1e}", flush=True)
    print()

REF = {}
for pn, seeds in PROB.items():
    v = [R[c][f"{pn}|{sn}"]["fuel"] for c in CANDS for sn, *_ in seeds
         if R[c][f"{pn}|{sn}"]["reason"] == "converged"
         and R[c][f"{pn}|{sn}"]["defect"] < 1e-2]
    REF[pn] = min(v) if v else float("nan")
print("best feasible fuel per problem: " + "  ".join(f"{k} {v:.4f}" for k, v in REF.items()) + "\n")

print(f"{'config':<15}{'conv':>7}{'mean_it':>9}{'max':>5}{'worst gap':>11}{'mean gap':>10}{'maxdef':>9}")
rank = []
for cn in CANDS:
    ss = list(R[cn].values())
    nc = sum(s["reason"] == "converged" for s in ss)
    gaps = []
    for pn, seeds in PROB.items():
        f = [R[cn][f"{pn}|{sn}"]["fuel"] for sn, *_ in seeds
             if R[cn][f"{pn}|{sn}"]["reason"] == "converged"]
        if f and REF[pn] == REF[pn]:
            gaps.append((min(f) - REF[pn]) / REF[pn] * 100.0)
    g = dict(name=cn, nconv=nc, ntot=len(ss),
             mean_it=float(np.mean([s["iters"] for s in ss])),
             max_it=int(max(s["iters"] for s in ss)),
             worst_gap=float(max(gaps)), mean_gap=float(np.mean(gaps)),
             max_defect=float(max(s["defect"] for s in ss if s["reason"] == "converged")))
    rank.append(g)
    print(f"{cn:<15}{nc:>4}/{len(ss)}{g['mean_it']:>9.1f}{g['max_it']:>5}"
          f"{g['worst_gap']:>10.2f}%{g['mean_gap']:>9.2f}%{g['max_defect']:>9.1e}")
rank.sort(key=lambda g: (round(g["worst_gap"], 2), -g["nconv"], g["mean_it"]))
print(f"\nWINNER (best worst-case optimality, then robustness): {rank[0]['name']}"
      f"  {CANDS[rank[0]['name']]}")
json.dump(dict(runs=R, ref=REF, rank=rank, cands=CANDS), open("step17_select.json","w"), indent=1)

"""Does the trust-region machinery in ptr.run do anything? Four variants on a TFC benchmark (free tf):
  default  : accept every step, hard cap R adapted from the measured defect (repo default)
  ratio    : ratio_test=True (SCvx accept/reject + rho-band radius update)
  fixedR   : no radius adaptation (w_growth = 1), R = 0.35
  noTR     : cap effectively removed (R0 = Rs0 = R_max = 1e3)
usage: ptr_ablation.py <case> <variant>"""
import os, sys, json, time
import numpy as np
case, variant = sys.argv[1], sys.argv[2]
VER = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, VER); sys.argv = ["x"]
import run_verification as RV
import ptr, config as C, benchmark as B
from dynamics import Dynamics
cs = B.CASES[case]; r0, v0 = cs["r0"], cs["v0"]; rf = np.zeros(3); n = np.array([0, 0, 1.0])
sc = C.Scales.build(r0, rf, m_wet=B.M_WET, t_max=B.TMAX); dyn = Dynamics(sc, field=B.ConstantField())
x0 = np.concatenate([r0 / sc.L, v0 * sc.T / sc.L, [np.log(B.M_WET)], n])
kw = dict(dyn=dyn, scales=sc, x_init=x0, rf_nd=rf / sc.L, n_hat=n, K=60, mode="fuel", D_nd=0.0, iters=150,
          verbose=False, slew_max_deg_s=None, a_init=None, term_point_deg=180.0, term_point_nodes=0,
          tol_obj=1e-6, tol_defect_m=1e-2)
if variant.startswith("R="):            # fixed radius sweep: R0 = value, Rs0 in the default 0.5/0.35 ratio
    Rv = float(variant[2:])
    kw.update(R0=Rv, Rs0=Rv * 0.5 / 0.35, R_bounds=(1e-4, max(1.0, 2 * Rv)), w_growth=1.0)
else:
    kw.update({"default": {}, "ratio": dict(ratio_test=True), "fixedR": dict(w_growth=1.0),
               "noTR": dict(R0=1e3, Rs0=1e3, R_bounds=(1e-4, 1e3))}[variant])
t = time.time(); r = ptr.run(**kw); wall = time.time() - t
H = r.history
bind = sum(1 for h in H if h["eta"] >= 0.99 * h["R"] or h["eta_s"] >= 0.99 * h["Rs"])
# first iteration at which the propellant is within 0.01 % of its final value
J = np.array([h["jphys"] for h in H]); fin = J[-1]
first = int(np.argmax(np.abs(J - fin) / fin < 1e-4)) if len(J) else -1
out = dict(case=case, variant=variant, status=r.status, it=r.iters, mp=r.fuel_kg, tf=r.tf_s, wall=wall,
           cap_binding_iters=bind, R_min=min(h["R"] for h in H), R_max=max(h["R"] for h in H),
           rejected=sum(1 for h in H if not h["accepted"]), within_1e4_at=first,
           max_eta_over_R_first5=[round(h["eta"] / h["R"], 3) for h in H[:5]],
           tf_first5=[round(h["tf_s"], 2) for h in H[:5]])
print(json.dumps(out), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{case}_{variant}.json"), "w"), indent=1)

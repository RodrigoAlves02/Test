"""Trace the cold start iteration by iteration: where does sigma go and why does it not settle?"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import nominal, run
import gen_dataset as G

X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
print(f"true optimum: sigma {sig:.1f} s\n")
for lab, extra in [("cold defaults", {}),
                   ("cold tr/10", dict(eta0=100.0, eta_s0=50.0, eta_u0=0.01))]:
    o = run(x0, m, warm=None, extra=extra, iters=40, ctx=ctx)
    print(f"--- {lab}: {o['r']['reason']}/{o['r'].get('exit_via')} sigma {o['r']['sigma']:.1f} "
          f"fuel {o['r']['fuel']:.4f}")
    print(f"{'it':>3}{'sig_bar':>9}{'sig_new':>9}{'dsig':>8}{'eta_s':>8}{'eta':>9}{'eta_u':>9}"
          f"{'trfrac':>7}{'rho':>11}{'dL':>10}{'vc':>10}{'A':>2}")
    for d in o["tr"]:
        print(f"{d['it']:>3}{d['sigma_bar']:>9.1f}{d['sigma_new']:>9.1f}{d['dsig_step']:>8.1f}"
              f"{d['eta_s']:>8.2f}{d['eta']:>9.2f}{d['eta_u']:>9.2e}{d['tr_sigma_frac']:>7.2f}"
              f"{d['rho']:>11.3f}{d['dL']:>10.3f}{d['vc']:>10.2e}{'Y' if d['accepted'] else '.':>2}")
    print()

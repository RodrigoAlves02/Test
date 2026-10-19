#!/usr/bin/env python
"""Diagnostic harness for the SCvx solver: run a case, report what the iteration actually did."""
from __future__ import annotations
import os, sys, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","TBB_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

SCVX = "/Users/rodrigoalves/Desktop/Thesis/scvx_landing"
if os.getcwd() != SCVX:
    os.chdir(SCVX)
sys.path.insert(0, SCVX)


def nominal():
    z = np.load(os.path.join(SCVX, "descent_nominal.npz"))
    return z["X"], z["U"], float(z["sigma"]), z["x_init"], np.asarray(z["rf"], float)


def run(x_init, mass, warm=None, extra=None, iters=120, ctx=None):
    """One solve. extra is merged into tune.EXTRA_CFG; module-level ETA_* are restored after."""
    import tune, gen_dataset as G
    prev_extra = dict(tune.EXTRA_CFG)
    saved = (tune.ETA0, tune.ETA_U0, tune.ETA_S0)
    # solver_cfg passes these as explicit keywords AND unpacks EXTRA_CFG into the same update(),
    # so anything it names has to be set on the tune module instead or update() raises
    # "multiple values for keyword argument".
    NAMED = {"eta0": "ETA0", "eta_u0": "ETA_U0", "eta_s0": "ETA_S0", "Wvc": "WVC",
             "r0t": "RHO_ACCEPT", "r1t": "RHO_SHRINK", "r2t": "RHO_GROW",
             "bsh": "SHRINK_FACTOR", "bex": "GROW_FACTOR", "K": "NODES_K"}
    extra = dict(extra or {})
    saved_named = {}
    for k, attr in NAMED.items():
        if k in extra:
            saved_named[attr] = getattr(tune, attr)
            setattr(tune, attr, extra.pop(k))
    tune.EXTRA_CFG = {**prev_extra, **extra}
    try:
        ctx = ctx or G.Ctx()
        t0 = time.perf_counter()
        cs, r, tr, cfg = G.solve(ctx, x_init, mass, warm=warm, iters=iters)
        wall = time.perf_counter() - t0
    finally:
        tune.EXTRA_CFG = prev_extra
        tune.ETA0, tune.ETA_U0, tune.ETA_S0 = saved
        for attr, v in saved_named.items():
            setattr(tune, attr, v)
    return dict(r=r, tr=tr, cfg=cfg, cs=cs, wall=wall)


def summary(out, label="", sig_ref=None):
    r, tr = out["r"], out["tr"]
    acc = sum(1 for d in tr if d["accepted"])
    grew = sum(1 for i, d in enumerate(tr[:-1])
               if tr[i + 1]["eta_s"] > d["eta_s"] + 1e-12)
    shr = sum(1 for i, d in enumerate(tr[:-1])
              if tr[i + 1]["eta_s"] < d["eta_s"] - 1e-12)
    rho_hi = sum(1 for d in tr if d["rho"] >= 0.9)
    s = dict(label=label, reason=r["reason"], exit_via=r.get("exit_via", ""),
             iters=len(tr), accepted=acc, rejected=len(tr) - acc,
             eta_s_grew=grew, eta_s_shrank=shr, rho_ge_0p9=rho_hi,
             sigma=float(r["sigma"]), fuel=float(r["fuel"]), miss=float(r["miss"]),
             lcvx_gap=float(r["lcvx_gap"]), wall=out["wall"],
             eta_s_first=float(tr[0]["eta_s"]) if tr else np.nan,
             eta_s_last=float(tr[-1]["eta_s"]) if tr else np.nan,
             dL_first=float(tr[0]["dL"]) if tr else np.nan,
             dL_last=float(tr[-1]["dL"]) if tr else np.nan)
    if sig_ref is not None:
        s["sigma_err"] = float(r["sigma"]) - sig_ref
    return s


HDR = (f"{'label':<26}{'reason':<11}{'via':<8}{'it':>4}{'acc':>5}{'rej':>5}"
       f"{'g':>3}{'s':>4}{'r>.9':>6}{'sigma':>9}{'dsig':>8}{'fuel':>8}{'miss':>9}{'wall':>7}")


def line(s):
    ds = s.get("sigma_err", float("nan"))
    return (f"{s['label']:<26}{s['reason']:<11}{s['exit_via']:<8}{s['iters']:>4}{s['accepted']:>5}"
            f"{s['rejected']:>5}{s['eta_s_grew']:>3}{s['eta_s_shrank']:>4}{s['rho_ge_0p9']:>6}"
            f"{s['sigma']:>9.1f}{ds:>8.1f}{s['fuel']:>8.4f}{s['miss']:>9.2e}{s['wall']:>7.1f}")

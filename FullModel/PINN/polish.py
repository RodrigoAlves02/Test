#!/usr/bin/env python
"""L-BFGS polish of an existing run's checkpoint, on a FIXED batch.

WHY A FIXED BATCH, AND WHY THE FIRST ATTEMPT DID NOTHING. L-BFGS builds curvature pairs from
successive gradients and takes a strong-Wolfe line search along them. Both require a DETERMINISTIC
objective. train.py's closure drew a fresh random minibatch on every call, so each evaluation
measured a different function: the line search saw noise, failed its conditions on the first probe,
and the optimiser stopped after a single iteration -- loss went UP, 1.1962e-02 to 1.3729e-02, and
the held-out position with it, 53.00 m to 53.44 m. L-BFGS and minibatching are incompatible, not
merely a poor fit.

So the batch is drawn ONCE, here, and every closure call sees exactly those points -- supervised
indices AND the collocation indices and their tau values. The subset is a fixed random draw from
the run's own training split, with its seed and size recorded, because the polish is then optimising
a slightly different objective than the Adam phase and that has to be visible rather than implied.

The result lands in runs/<run_id>/polish/, next to the Adam run it came from, with its own metrics
and checkpoint. The Adam checkpoint is left untouched so the comparison stays available.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
sys.path.insert(0, HERE)

from runcfg import Config                                            # noqa: E402
import data as D                                                     # noqa: E402
from model import Surrogate                                          # noqa: E402
from gravity_torch import PinnGravity                                # noqa: E402
import losses as L                                                   # noqa: E402
from train import pick_device, to_t                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<run_id> to polish")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=65536, help="FIXED supervised batch")
    ap.add_argument("--collocation", type=int, default=16384, help="FIXED collocation batch")
    ap.add_argument("--history", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345, help="draws the fixed batch; recorded")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    run = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    cfg = Config.load(os.path.join(run, "config.json"))
    out = os.path.join(run, "polish"); os.makedirs(out, exist_ok=True)
    dev = pick_device(a.device)
    log = open(os.path.join(out, "log.txt"), "w")

    def say(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()

    say(f"polishing {os.path.basename(run)}   device {dev}")
    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(HERE, cfg.data_dir)
    parents = D.load_parents(data_dir, defect_max=cfg.defect_max, verbose=False)
    sc = D.Scaling(**cfg.scaling)
    tr = D.build(data_dir, cfg.split, "train", j_max=cfg.j_max, parents=parents)
    va = D.build(data_dir, cfg.split, "val", j_max=cfg.j_max, parents=parents)
    TR = {k: to_t(v, dev) for k, v in tr.items() if k != "seed"}
    VA = {k: to_t(v, dev) for k, v in va.items() if k != "seed"}
    say(f"train pairs {TR['tau'].shape[0]:,}   val pairs {VA['tau'].shape[0]:,}")

    phys = L.Physics(SCVX, device=dev)
    phys.gravity = PinnGravity(phys.C.PINN_NPZ).to(dev)
    mdl = Surrogate(cfg, sc, phys.Tmin, phys.Tmax).to(dev)
    mdl.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev))
    site, nhat = to_t(sc.r_site, dev), to_t(sc.n_hat, dev)
    W = dict(r=cfg.w_r, v=cfg.w_v, m=cfg.w_m, d=cfg.w_d, T=cfg.w_T, tgo=cfg.w_tgo)

    # ---- THE BATCH, DRAWN ONCE -------------------------------------------------------------
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    n = TR["tau"].shape[0]
    idx = torch.randint(0, n, (min(a.batch, n),), generator=g).to(dev)
    jdx = torch.randint(0, n, (min(a.collocation, n),), generator=g).to(dev)
    tcol = torch.rand(jdx.shape[0], generator=g).to(dev)
    say(f"fixed batch: {idx.numel():,} supervised, {jdx.numel():,} collocation, seed {a.seed}")

    @torch.no_grad()
    def val():
        k = min(200_000, VA["tau"].shape[0]); i = torch.arange(k, device=dev)
        o = mdl(VA["r0"][i], VA["v0"][i], VA["m0"][i], VA["d0"][i], VA["tau"][i])
        return dict(val_r_m=float((o["r"]-VA["r"][i]).norm(dim=-1).mean()),
                    val_v_ms=float((o["v"]-VA["v"][i]).norm(dim=-1).mean()),
                    val_m_kg=float((o["m"]-VA["m"][i]).abs().mean()),
                    val_d_deg=float(torch.rad2deg(torch.arccos(
                        (o["d"]*VA["d"][i]).sum(-1).clamp(-1,1))).mean()),
                    val_T_N=float((o["T"]-VA["T"][i]).abs().mean()),
                    val_tgo_s=float((o["tgo"]-VA["tgo"][i]).abs().mean()))

    def objective():
        o = mdl(TR["r0"][idx], TR["v0"][idx], TR["m0"][idx], TR["d0"][idx], TR["tau"][idx])
        lab = {k: TR[k][idx] for k in ("r","v","m","d","T","tgo")}
        sup = L.supervised(o, lab, sc)
        loss = sum(W[k]*sup[k] for k in sup)
        if cfg.w_residual > 0 and jdx.numel():
            res = L.residual(mdl, phys, TR["r0"][jdx], TR["v0"][jdx], TR["m0"][jdx],
                             TR["d0"][jdx], tcol)
            rr = ((res["e_r"]/sc.V)**2).mean()
            rv = ((res["e_v"]/(sc.V/sc.TF))**2).mean()
            rm = ((res["e_m"]/sc.MDOT)**2).mean()
            hg = L.hinges(mdl, phys, res["out"], res["tau"], site, nhat, 0.1, sc)
            loss = loss + cfg.w_residual*(rr+rv+rm) + cfg.w_hinge*sum(hg.values())
        return loss

    before = val(); l0 = float(objective())
    say("before: loss %.6e  " % l0 + "  ".join(f"{k} {v:.4g}" for k, v in before.items()))

    opt = torch.optim.LBFGS(mdl.parameters(), max_iter=a.iters, history_size=a.history,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-14,
                            tolerance_change=1e-16)
    hist, t0 = [], time.perf_counter()

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        hist.append(float(loss))
        if len(hist) % 100 == 1:
            say(f"[lbfgs {len(hist):>5}] loss {hist[-1]:.6e}  [{time.perf_counter()-t0:.0f}s]")
        return loss
    opt.step(closure)

    after = val(); l1 = float(objective())
    say("after : loss %.6e  " % l1 + "  ".join(f"{k} {v:.4g}" for k, v in after.items()))
    say(f"closure evaluations {len(hist)}   loss {l0:.6e} -> {l1:.6e}  "
        f"({100*(l1/l0-1):+.2f}%)   [{time.perf_counter()-t0:.0f}s]")
    for k in before:
        say(f"  {k:<12} {before[k]:>12.5g} -> {after[k]:>12.5g}   {100*(after[k]/before[k]-1):+8.2f}%")

    torch.save({k: v.cpu() for k, v in mdl.state_dict().items()}, os.path.join(out, "model.pt"))
    json.dump(dict(run=os.path.basename(run), iters=a.iters, batch=int(idx.numel()),
                   collocation=int(jdx.numel()), batch_seed=a.seed, history=a.history,
                   closure_evals=len(hist), loss_before=l0, loss_after=l1,
                   before=before, after=after, wall_s=time.perf_counter()-t0,
                   loss_curve=hist[::max(1, len(hist)//500)]),
              open(os.path.join(out, "metrics.json"), "w"), indent=1)
    say(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""One run. Everything it used and everything it produced lands in runs/<run_id>/.

ADAM THEN L-BFGS. Adam finds the basin; L-BFGS polishes it. The split matters here because the
residual is stiff -- it is a second-order condition on a quantity (tf) the network is also learning
-- and first-order methods stall on it long before it is actually converged.

THE RESIDUAL IS RAMPED IN. d/dt = (1/tf) d/dtau, and tf is an output, so at step zero the residual
is being divided by a number the network invented. Weighting it from the start teaches the network
to make the residual small by making tf wrong, which is the one failure mode that looks like
success. The ramp lets the supervised tgo term pin tf first.

NOTHING IS CONFIGURED OUTSIDE config.py. If a number influenced this run it is in config.json.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
THESIS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from runcfg import Config                                            # noqa: E402
import data as D                                                     # noqa: E402
from model import Surrogate, check_structural, nominal_reference                        # noqa: E402
from gravity_torch import PinnGravity                                # noqa: E402
import losses as L                                                   # noqa: E402


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_t(a, dev, dt=None):
    return torch.tensor(np.asarray(a), dtype=dt or torch.get_default_dtype(), device=dev)


def main(argv=None):
    cfg = Config.from_args(argv)
    # SET BEFORE ANYTHING IS CONVERTED. to_t and every buffer follow the default
    # dtype, so switching it later would leave the data in one precision and the
    # model in another.
    if cfg.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = pick_device(cfg.device)

    # ---------------------------------------------------------------- data
    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(HERE, cfg.data_dir)
    parents = D.load_parents(data_dir, defect_max=cfg.defect_max)
    cfg.data_fingerprint = D.fingerprint(parents)
    cfg.split = D.split_by_parent(parents, cfg.frac_val, cfg.frac_test, cfg.split_seed,
                                  n_train=cfg.n_parents)
    tr_parents = [p for p in parents if int(p["seed"]) in set(cfg.split["train"])]
    sc = D.Scaling.from_parents(tr_parents)
    cfg.scaling = asdict(sc)

    run_dir = os.path.join(HERE, cfg.out_root, cfg.run_id)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "config.json"))
    log_f = open(os.path.join(run_dir, "log.txt"), "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    say(f"run_id {cfg.run_id}   device {dev}   out {run_dir}")
    say(f"parents {len(parents)}  train {len(cfg.split['train'])} "
        f"val {len(cfg.split['val'])} test {len(cfg.split['test'])}")
    tr = D.build(data_dir, cfg.split, "train", j_max=cfg.j_max, parents=parents)
    va = D.build(data_dir, cfg.split, "val", j_max=cfg.j_max, parents=parents)
    say(f"pairs  train {len(tr['tau']):,}   val {len(va['tau']):,}")

    TR = {k: to_t(v, dev) for k, v in tr.items() if k != "seed"}
    VA = {k: to_t(v, dev) for k, v in va.items() if k != "seed"}

    # ---------------------------------------------------------------- model + physics
    phys = L.Physics(SCVX, device=dev, dtype=torch.get_default_dtype())
    # THE GRAVITY NET FOLLOWS THE RUN'S DTYPE. PinnGravity defaults to float64, so a float32
    # run was evaluating its physics residual through a float64 sub-net and casting at every
    # call -- and MPS refuses a float64 buffer outright, which silently forced every local run
    # onto the CPU.
    phys.gravity = PinnGravity(phys.C.PINN_NPZ, dtype=torch.get_default_dtype()).to(dev)
    ref = None
    if cfg.reference == "nominal":
        ref = nominal_reference(tr_parents, device=dev)
        say("ansatz carries a nominal reference trajectory (C2 spline, mean training parent)")
    mdl = Surrogate(cfg, sc, phys.Tmin, phys.Tmax, ref=ref).to(dev)
    struct = check_structural(mdl, sc, device=dev)
    say("structural check: " + "  ".join(f"{k}={v:.2e}" for k, v in struct.items()))
    say(f"parameters {sum(p.numel() for p in mdl.parameters()):,}")

    site = to_t(sc.r_site, dev); nhat = to_t(sc.n_hat, dev)
    D_allow = 0.1

    json.dump(dict(code_sha1=cfg.code_hash(), config_sha1=cfg.hash(),
                   python=platform.python_version(), torch=torch.__version__,
                   device=str(dev), host=platform.node(), platform=platform.platform(),
                   cuda=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                   physics=phys.record(), structural=struct,
                   n_parameters=int(sum(p.numel() for p in mdl.parameters())),
                   started=time.strftime("%Y-%m-%dT%H:%M:%S")),
              open(os.path.join(run_dir, "env.json"), "w"), indent=1)

    # ---------------------------------------------------------------- loss assembly
    W = dict(r=cfg.w_r, v=cfg.w_v, m=cfg.w_m, d=cfg.w_d, T=cfg.w_T, tgo=cfg.w_tgo)

    def batch_idx(n, b, gen):
        return torch.randint(0, n, (b,), device=dev, generator=gen)

    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    _vg = torch.Generator(device=dev).manual_seed(cfg.val_seed)
    n_va = VA['tau'].shape[0]
    vi = (torch.arange(n_va, device=dev) if n_va <= cfg.val_sample
          else torch.randint(0, n_va, (cfg.val_sample,), device=dev, generator=_vg))
    n_tr = TR["tau"].shape[0]

    def total_loss(step, fixed=None):
        """fixed is None for Adam (resample every step) and the frozen draw for the L-BFGS polish,
        which needs the SAME points on every closure call or its line search has no function to
        search along."""
        i = batch_idx(n_tr, cfg.batch, gen) if fixed is None else fixed["i"]
        out = mdl(TR["r0"][i], TR["v0"][i], TR["m0"][i], TR["d0"][i], TR["tau"][i])
        lab = {k: TR[k][i] for k in ("r", "v", "m", "d", "T", "tgo")}
        sup = L.supervised(out, lab, sc)
        loss = sum(W[k] * sup[k] for k in sup)
        parts = {f"sup_{k}": float(v) for k, v in sup.items()}

        ramp = min(1.0, step / max(cfg.residual_ramp_steps, 1))
        if cfg.w_residual > 0 and cfg.n_collocation > 0:
            if fixed is None:
                j = batch_idx(n_tr, cfg.n_collocation, gen)
                tc = (torch.rand(cfg.n_collocation, device=dev, generator=gen)
                      if cfg.collocation_tau == "uniform" else TR["tau"][j])
            else:
                j, tc = fixed["j"], fixed["tc"]
            res = L.residual(mdl, phys, TR["r0"][j], TR["v0"][j], TR["m0"][j], TR["d0"][j], tc)
            # scaled so the three equations are comparable: length, speed, mass
            r_r = ((res["e_r"] / sc.V) ** 2).mean()
            r_v = ((res["e_v"] / (sc.V / sc.TF)) ** 2).mean()
            r_m = ((res["e_m"] / sc.MDOT) ** 2).mean()
            hg = L.hinges(mdl, phys, res["out"], res["tau"], site, nhat, D_allow, sc)
            loss = loss + cfg.w_residual * ramp * (r_r + r_v + r_m) \
                        + cfg.w_hinge * sum(hg.values())
            parts.update(res_r=float(r_r), res_v=float(r_v), res_m=float(r_m),
                         **{f"h_{k}": float(v) for k, v in hg.items()})
        parts["ramp"] = ramp
        return loss, parts

    @torch.no_grad()
    def val_metrics():
        out = mdl(VA["r0"][vi], VA["v0"][vi], VA["m0"][vi], VA["d0"][vi], VA["tau"][vi])
        dr = (out["r"] - VA["r"][vi]).norm(dim=-1)
        dv = (out["v"] - VA["v"][vi]).norm(dim=-1)
        dtgo = (out["tgo"] - VA["tgo"][vi]).abs()
        e = VA["tgo"][vi] <= cfg.val_tgo_end_s      # near touchdown, by time-to-go
        return dict(
            val_r_m=float(dr.mean()),
            val_v_ms=float(dv.mean()),
            val_m_kg=float((out["m"] - VA["m"][vi]).abs().mean()),
            val_d_deg=float(torch.rad2deg(torch.arccos(
                (out["d"] * VA["d"][vi]).sum(-1).clamp(-1, 1))).mean()),
            val_T_N=float((out["T"] - VA["T"][vi]).abs().mean()),
            val_tgo_s=float(dtgo.mean()),
            # the terminal band, where the landing actually happens
            val_r_m_end=float(dr[e].mean()),
            val_v_ms_end=float(dv[e].mean()),
            val_tgo_s_end=float(dtgo[e].mean()))

    # ---------------------------------------------------------------- Adam
    opt = torch.optim.Adam(mdl.parameters(), lr=cfg.adam_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.adam_steps, 1), eta_min=cfg.adam_lr_final)
    hist_path = os.path.join(run_dir, "history.csv")
    hist_f = open(hist_path, "w"); hist_cols = None
    t0 = time.perf_counter()
    for step in range(cfg.adam_steps):
        if cfg.sigma_head and cfg.sigma_anneal_steps > 0:
            f = min(1.0, step / cfg.sigma_anneal_steps)
            mdl.sharpness = (cfg.sigma_sharpness
                             + f * (cfg.sigma_sharpness_final - cfg.sigma_sharpness))
        opt.zero_grad(set_to_none=True)
        loss, parts = total_loss(step)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        if step % cfg.eval_every == 0 or step == cfg.adam_steps - 1:
            vm = val_metrics()
            row = dict(phase="adam", step=step, loss=float(loss), lr=sched.get_last_lr()[0],
                       wall_s=time.perf_counter()-t0, **parts, **vm)
            if hist_cols is None:
                hist_cols = list(row); hist_f.write(",".join(hist_cols) + "\n")
            hist_f.write(",".join(f"{row[c]}" for c in hist_cols) + "\n"); hist_f.flush()
            say(f"[adam {step:>6}] loss {float(loss):.4e}  r {vm['val_r_m']:8.2f} m  "
                f"v {vm['val_v_ms']:7.4f} m/s  d {vm['val_d_deg']:6.2f} deg  "
                f"T {vm['val_T_N']:6.3f} N  tgo {vm['val_tgo_s']:7.2f} s  "
                f"[{row['wall_s']:.0f}s]")

    # THE PRE-POLISH WEIGHTS ARE KEPT. A fixed batch is a subset, so L-BFGS can drive the training
    # loss down while the held-out error rises -- measured on the smoke run: loss -81.9 %, val_r
    # 646 -> 919 m. Keeping model_adam.pt means the polish can be judged rather than assumed, and a
    # run whose polish overfits is not a lost run.
    torch.save({k: v.cpu() for k, v in mdl.state_dict().items()},
               os.path.join(run_dir, "model_adam.pt"))

    # ---------------------------------------------------------------- L-BFGS polish
    # ON A FIXED BATCH, drawn once here and reused by every closure call. See Config.lbfgs_batch
    # for why: a resampled closure makes the line search search a different function each try.
    if cfg.lbfgs_steps > 0:
        g2 = torch.Generator(device=dev).manual_seed(cfg.lbfgs_batch_seed)
        nb, nc = min(cfg.lbfgs_batch, n_tr), min(cfg.lbfgs_collocation, n_tr)
        fixed = dict(i=torch.randint(0, n_tr, (nb,), device=dev, generator=g2),
                     j=torch.randint(0, n_tr, (nc,), device=dev, generator=g2))
        fixed["tc"] = (torch.rand(nc, device=dev, generator=g2)
                       if cfg.collocation_tau == "uniform" else TR["tau"][fixed["j"]])
        say(f"L-BFGS polish on a FIXED batch: {nb:,} supervised, {nc:,} collocation, "
            f"seed {cfg.lbfgs_batch_seed}")
        l0 = float(total_loss(cfg.adam_steps, fixed=fixed)[0])
        vm_pre = val_metrics()

        lb = torch.optim.LBFGS(mdl.parameters(), max_iter=cfg.lbfgs_steps,
                               history_size=cfg.lbfgs_history, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-12, tolerance_change=1e-14)
        state = {"n": 0}

        def closure():
            lb.zero_grad(set_to_none=True)
            loss, _ = total_loss(cfg.adam_steps, fixed=fixed)   # ramp fully on
            loss.backward()
            state["n"] += 1
            if state["n"] % 50 == 1:
                say(f"[lbfgs {state['n']:>5}] loss {float(loss):.6e}")
            return loss
        lb.step(closure)
        l1 = float(total_loss(cfg.adam_steps, fixed=fixed)[0])
        say(f"L-BFGS: {state['n']} closure evaluations, loss {l0:.6e} -> {l1:.6e} "
            f"({100*(l1/l0-1):+.2f} %)")
        vm = val_metrics()
        hist_f.write(",".join(f"{dict(phase='lbfgs', step=state['n'], loss=float('nan'), lr=0.0, wall_s=time.perf_counter()-t0, **{k: float('nan') for k in hist_cols if k not in ('phase','step','loss','lr','wall_s') and k not in vm}, **vm).get(c, '')}" for c in hist_cols) + "\n")
        say("after L-BFGS: " + "  ".join(f"{k} {v:.4g}" for k, v in vm.items()))
        json.dump(dict(closure_evals=state["n"], loss_before=l0, loss_after=l1,
                       batch=int(nb), collocation=int(nc), batch_seed=cfg.lbfgs_batch_seed,
                       val_before=vm_pre, val_after=vm),
                  open(os.path.join(run_dir, "polish.json"), "w"), indent=1)
    hist_f.close()

    torch.save({k: v.cpu() for k, v in mdl.state_dict().items()},
               os.path.join(run_dir, "model.pt"))
    say(f"wrote {run_dir}")
    log_f.close()
    return run_dir


if __name__ == "__main__":
    main()

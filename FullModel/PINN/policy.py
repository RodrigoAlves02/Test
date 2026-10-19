#!/usr/bin/env python
"""A direct guidance policy: the state now, the control now.

    (r, v, m, d)  ->  (omega, Gamma)

WHY THE RATE AND NOT THE DIRECTION. The commanded thrust direction is never far from the current
one -- the engine slews at most 2 deg/s and the measured demand peaks at 1.16 deg/s -- so a network
emitting a thrust VECTOR would learn to copy d, score a low loss and carry almost no signal, the
useful part being the small correction it had swamped. The rate omega is the new information, and
it is what the solver's own control is: u = [omega, Gamma] with d-dot = omega x d.

WHY TWO COMPONENTS AND NOT THREE. subproblem.py carries the gauge row d.omega = 0 because the
component of omega along d produces no motion: (omega + c d) x d = omega x d. So omega has exactly
two meaningful degrees of freedom, which are the two a pointing command has. Emitting them in the
tangent plane at d makes d.omega = 0 true by construction rather than by training, in the same way
||d|| = 1 and r(0) = r0 are true by construction in the trajectory surrogate.

WHAT IT GIVES UP, stated rather than glossed: this produces a command, not a plan. There is nothing
to check constraints against before flying it, and no terminal guarantee of the kind tau buys the
surrogate. That is the trade the comparison in this work is about.

THE LABELS NEED NO NEW GENERATION. Each case stores d at every node, and omega_k is the rotation
carrying d_k to d_{k+1} across the segment -- the same construction ptr.py uses to build its guess.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D


# ---------------------------------------------------------------- labels
def controls_from_parent(p):
    """(state, control) pairs for one trajectory, control as (omega, T) in SI."""
    r, v, m, d, T = (np.asarray(p[k], float) for k in ("r", "v", "m", "d", "T"))
    K = r.shape[0]
    dt = float(p["tf"]) / (K - 1)
    d0, d1 = d[:-1], d[1:]
    cosang = np.clip((d0 * d1).sum(1), -1.0, 1.0)
    ang = np.arccos(cosang)
    axis = np.cross(d0, d1)
    nrm = np.linalg.norm(axis, axis=1, keepdims=True)
    # a turn of zero has no axis; any perpendicular direction gives the same zero rate
    axis = np.where(nrm > 1e-12, axis / np.maximum(nrm, 1e-300), 0.0)
    omega = axis * (ang / dt)[:, None]
    n = T.shape[0]                                   # T has K-1 entries
    return dict(r=r[:n], v=v[:n], m=m[:n], d=d[:n], omega=omega[:n], T=T[:n])


def tangent_basis(d):
    """Two unit vectors spanning the plane normal to d, built the same way in numpy and torch."""
    lib = torch if isinstance(d, torch.Tensor) else np
    ref = lib.zeros_like(d)
    ref[..., 2] = 1.0
    alt = lib.zeros_like(d)
    alt[..., 0] = 1.0
    # pick whichever reference is less aligned with d, so the cross product never degenerates
    use_alt = (lib.abs(d[..., 2]) > 0.9)[..., None]
    ref = lib.where(use_alt, alt, ref)
    e1 = lib.cross(ref, d, dim=-1) if lib is torch else np.cross(ref, d)
    e1 = e1 / (lib.linalg.norm(e1, axis=-1, keepdims=True) if lib is np
               else e1.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    e2 = lib.cross(d, e1, dim=-1) if lib is torch else np.cross(d, e1)
    return e1, e2


def build(parents, seeds):
    want = set(int(s) for s in seeds)
    out = [controls_from_parent(p) for p in parents if int(p["seed"]) in want]
    cat = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    e1, e2 = tangent_basis(cat["d"])
    cat["a1"] = (cat["omega"] * e1).sum(1)
    cat["a2"] = (cat["omega"] * e2).sum(1)
    cat["w_par"] = (cat["omega"] * cat["d"]).sum(1)      # the gauge residual in the labels
    return cat


# ---------------------------------------------------------------- model
class Policy(nn.Module):
    def __init__(self, sc, t_min, t_max, w_max, width=256, depth=5, act=nn.Tanh):
        super().__init__()
        self.register_buffer("site", torch.tensor(sc.r_site, dtype=torch.get_default_dtype()))
        self.L, self.V, self.M = sc.L, sc.V, sc.M
        self.Tmin, self.Tmax, self.w_max = float(t_min), float(t_max), float(w_max)
        layers, k = [], 10
        for _ in range(depth):
            layers += [nn.Linear(k, width), act()]
            k = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(k, 3)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def forward(self, r, v, m, d):
        z = torch.cat([(r - self.site) / self.L, v / self.V, (m / self.M)[..., None], d], -1)
        h = self.head(self.trunk(z))
        # the rate lives in the tangent plane at d, so d.omega = 0 holds by construction, and it is
        # bounded by the slew limit the hardware actually has
        a = torch.tanh(h[..., :2]) * self.w_max
        e1, e2 = tangent_basis(d)
        omega = a[..., :1] * e1 + a[..., 1:2] * e2
        T = self.Tmin + (self.Tmax - self.Tmin) * torch.sigmoid(h[..., 2])
        return dict(omega=omega, a=a, T=T)


# ---------------------------------------------------------------- training
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../SCvx/Figures&data/pinn")
    ap.add_argument("--name", default="policy")
    ap.add_argument("--n-parents", type=int, default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1e-5)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--lbfgs-steps", dest="lbfgs_steps", type=int, default=2000,
                    help="L-BFGS polish iterations after Adam; 0 disables it")
    ap.add_argument("--lbfgs-batch", dest="lbfgs_batch", type=int, default=65536,
                    help="FIXED batch for the polish. L-BFGS builds curvature from differences of "
                         "gradients and its line search assumes a deterministic objective, so a "
                         "resampled closure makes every evaluation a different function and it "
                         "stops after one iteration.")
    ap.add_argument("--lbfgs-seed", dest="lbfgs_seed", type=int, default=12345)
    ap.add_argument("--resume", default=None,
                    help="load this run's policy.pt and continue. With --steps 0 it polishes an "
                         "already-trained model instead of retraining it from scratch.")
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)

    sys.path.insert(0, os.path.join(HERE, "..", "SCvx"))
    import config as C
    parents = D.load_parents(a.data_dir, defect_max=1e-3)
    split = D.split_by_parent(parents, n_train=a.n_parents)
    tr_p = [p for p in parents if int(p["seed"]) in set(split["train"])]
    sc = D.Scaling.from_parents(tr_p)
    TR, VA = build(parents, split["train"]), build(parents, split["val"])
    print(f"parents {len(parents)}  train {len(split['train'])} val {len(split['val'])}")
    print(f"pairs   train {TR['T'].size:,}  val {VA['T'].size:,}")
    print(f"label gauge residual |omega.d| max {np.abs(TR['w_par']).max():.3e} rad/s "
          f"(zero by construction of the label)")
    w_max = np.radians(C.SLEW_MAX_DEG_S)
    print(f"label rate |omega| max {np.abs(np.hypot(TR['a1'], TR['a2'])).max():.4f} rad/s "
          f"against a limit of {w_max:.4f}")

    t = lambda k, s: torch.as_tensor(s[k], dtype=torch.get_default_dtype())
    mdl = Policy(sc, C.TMIN, C.TMAX, w_max, width=a.width, depth=a.depth)
    print(f"parameters {sum(p.numel() for p in mdl.parameters()):,}")
    if a.resume:
        ck = os.path.join(HERE, "runs", a.resume, "policy.pt")
        mdl.load_state_dict(torch.load(ck, map_location="cpu"))
        print(f"resumed from {ck}", flush=True)
    opt = torch.optim.Adam(mdl.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps, 1), eta_min=a.lr_final)
    g = torch.Generator().manual_seed(a.seed)
    n = TR["T"].size
    TRt = {k: t(k, TR) for k in ("r", "v", "m", "d", "a1", "a2", "T")}
    VAt = {k: t(k, VA) for k in ("r", "v", "m", "d", "a1", "a2", "T")}

    def loss_on(S, idx):
        o = mdl(S["r"][idx], S["v"][idx], S["m"][idx], S["d"][idx])
        # both channels scaled by their own physical range, so neither can dominate by units
        la = ((o["a"][..., 0] - S["a1"][idx]) ** 2 + (o["a"][..., 1] - S["a2"][idx]) ** 2).mean()
        lt = ((o["T"] - S["T"][idx]) ** 2).mean()
        return la / w_max ** 2 + lt / (C.TMAX - C.TMIN) ** 2, o

    hist, t0 = [], time.perf_counter()
    for step in range(a.steps):
        i = torch.randint(0, n, (a.batch,), generator=g)
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_on(TRt, i)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step(); sch.step()
        if step % a.eval_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                j = torch.arange(min(200_000, VA["T"].size))
                _, o = loss_on(VAt, j)
                dw = torch.rad2deg(torch.sqrt((o["a"][..., 0] - VAt["a1"][j]) ** 2
                                              + (o["a"][..., 1] - VAt["a2"][j]) ** 2)).mean()
                dT = (o["T"] - VAt["T"][j]).abs().mean()
                gauge = (o["omega"] * VAt["d"][j]).sum(-1).abs().max()
            hist.append(dict(step=step, loss=float(loss), d_omega_deg_s=float(dw),
                             d_T_N=float(dT), gauge=float(gauge)))
            print(f"[{step:>6}] loss {float(loss):.4e}   d_omega {float(dw):.5f} deg/s   "
                  f"d_T {float(dT):.4f} N   |omega.d| {float(gauge):.2e}   "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)

    # ---------------------------------------------------------------- L-BFGS polish
    polish = None
    if a.lbfgs_steps > 0:
        g2 = torch.Generator().manual_seed(a.lbfgs_seed)
        fi = torch.randint(0, n, (min(a.lbfgs_batch, n),), generator=g2)
        print(f"L-BFGS polish on a FIXED batch of {fi.numel():,}, seed {a.lbfgs_seed}", flush=True)
        with torch.no_grad():
            l0 = float(loss_on(TRt, fi)[0])
        lb = torch.optim.LBFGS(mdl.parameters(), max_iter=a.lbfgs_steps, history_size=50,
                               line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                               tolerance_change=1e-14)
        state = {"n": 0}

        def closure():
            lb.zero_grad(set_to_none=True)
            loss, _ = loss_on(TRt, fi)
            loss.backward()
            state["n"] += 1
            return loss
        lb.step(closure)
        with torch.no_grad():
            l1 = float(loss_on(TRt, fi)[0])
            j = torch.arange(min(200_000, VA["T"].size))
            _, o = loss_on(VAt, j)
            dw = float(torch.rad2deg(torch.sqrt((o["a"][..., 0] - VAt["a1"][j]) ** 2
                                                + (o["a"][..., 1] - VAt["a2"][j]) ** 2)).mean())
            dT = float((o["T"] - VAt["T"][j]).abs().mean())
            gauge = float((o["omega"] * VAt["d"][j]).sum(-1).abs().max())
        polish = dict(closure_evals=state["n"], loss_before=l0, loss_after=l1,
                      batch=int(fi.numel()), d_omega_deg_s=dw, d_T_N=dT, gauge=gauge)
        print(f"L-BFGS: {state['n']} closure evaluations, loss {l0:.6e} -> {l1:.6e} "
              f"({100*(l1/l0-1):+.2f} %)", flush=True)
        print(f"after L-BFGS: d_omega {dw:.5f} deg/s   d_T {dT:.4f} N   |omega.d| {gauge:.2e}",
              flush=True)

    out = os.path.join(HERE, "runs", f"{a.name}")
    os.makedirs(out, exist_ok=True)
    torch.save({k: v.cpu() for k, v in mdl.state_dict().items()}, os.path.join(out, "policy.pt"))
    json.dump(dict(vars(a), history=hist, polish=polish, n_train=int(n), n_val=int(VA["T"].size),
                   parameters=int(sum(p.numel() for p in mdl.parameters()))),
              open(os.path.join(out, "policy.json"), "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

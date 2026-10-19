#!/usr/bin/env python
"""Supervised fit + physics residual + inequality hinges.

THE RESIDUAL IS WRITTEN IN THE ROTATING FRAME, because that is the frame the plan is solved in and
the only one in which the dynamics are autonomous -- which is also what makes the sub-trajectory
labels legal in the first place. Dropping the rotating terms would make the residual disagree with
the labels by exactly the Coriolis and centrifugal accelerations.

    dr/dt = v
    dv/dt = g(r) + T d / m - 2 w x v - w x (w x r)
    dm/dt = -T / (Isp g0)

with tau normalised, so d/dt = (1/tf) d/dtau and tf is itself a network output. That is why the
residual weight is RAMPED: until tgo is roughly right the residual is being scaled by a wrong
number and its gradient points nowhere useful.

WHAT IS A HINGE AND WHAT IS NOT. Only genuine inequalities appear here. The equalities and bounds
that must never be violated -- initial state, ||d|| = 1, the thrust band, the terminal plane -- are
structural in model.py and have no loss term at all. A constraint that appears in both places would
be double-counted and would also suggest it is negotiable, which it is not.
"""
from __future__ import annotations

import numpy as np
import torch


class Physics:
    """The constants the residual and the hinges need, read once from the SCvx config and RECORDED.

    Read from config.py rather than retyped, so a dataset and the network trained on it cannot drift
    apart silently: if TMIN changes, both move together and the run's config.json records what was
    in force.
    """

    def __init__(self, scvx_dir, device="cpu", dtype=torch.float32):
        # BY FILE PATH, NOT BY NAME. This package has its own config.py, which shadows the solver's
        # on sys.path -- `import config` here silently returned the PINN Config module and the
        # constants came back missing. Loading the file explicitly cannot be shadowed.
        import importlib.util, os, sys
        if scvx_dir not in sys.path:
            sys.path.append(scvx_dir)
        spec = importlib.util.spec_from_file_location("scvx_config",
                                                      os.path.join(scvx_dir, "config.py"))
        C = importlib.util.module_from_spec(spec)
        sys.modules["scvx_config"] = C
        spec.loader.exec_module(C)
        self.C = C
        t = lambda a: torch.tensor(np.asarray(a, float), dtype=dtype, device=device)
        self.w = t(C.W_BODY)                       # body rotation, rad/s, rotating frame
        self.alpha = float(1.0 / (C.ISP * C.G0))   # mdot = -alpha * T
        self.Tmin, self.Tmax = float(C.TMIN), float(C.TMAX)
        self.slew_max = float(np.radians(C.SLEW_MAX_DEG_S))
        self.vtouch = float(C.VTOUCH)
        self.gs = float(np.radians(C.GLIDESLOPE_DEG))
        self.point = float(np.radians(C.POINTING_DEG))
        self.mdry = float(C.MDRY)

    def record(self):
        return dict(w_body=[float(x) for x in self.w.cpu().numpy()], alpha=self.alpha,
                    Tmin=self.Tmin, Tmax=self.Tmax, slew_max_rad_s=self.slew_max,
                    vtouch=self.vtouch, glideslope_rad=self.gs, pointing_rad=self.point,
                    mdry=self.mdry)


def d_dtau(out_key, out, tau):
    """d(out[key])/dtau, keeping the graph so it can be differentiated through."""
    y = out[out_key]
    if y.dim() == 1:
        g, = torch.autograd.grad(y.sum(), tau, create_graph=True, retain_graph=True)
        return g
    cols = []
    for i in range(y.shape[-1]):
        g, = torch.autograd.grad(y[..., i].sum(), tau, create_graph=True, retain_graph=True)
        cols.append(g)
    return torch.stack(cols, -1)


def residual(model, phys, r0, v0, m0, d0, tau):
    """The three dynamics equations, in SI, evaluated at arbitrary tau. Returns per-equation terms."""
    tau = tau.clone().requires_grad_(True)
    out = model(r0, v0, m0, d0, tau)
    tf = out["tf"].unsqueeze(-1)

    drdt = d_dtau("r", out, tau) / tf
    dvdt = d_dtau("v", out, tau) / tf
    dmdt = d_dtau("m", out, tau) / tf.squeeze(-1)

    # IN THE GRAVITY NET'S OWN DTYPE, not unconditionally float64: the net is now built in the
    # run's dtype, and MPS has no float64 at all, so a hard .double() here forced every local
    # run onto the CPU. The cast back to the model's dtype is kept either way.
    g = phys.gravity.acceleration_si(
        out["r"].to(phys.gravity.We.dtype)).to(out["r"].dtype)
    w = phys.w
    cor = 2.0 * torch.cross(w.expand_as(out["v"]), out["v"], dim=-1)
    cen = torch.cross(w.expand_as(out["r"]), torch.cross(w.expand_as(out["r"]), out["r"], dim=-1), dim=-1)
    a_thrust = out["T"].unsqueeze(-1) * out["d"] / out["m"].unsqueeze(-1)

    e_r = drdt - out["v"]
    e_v = dvdt - (g + a_thrust - cor - cen)
    e_m = dmdt + phys.alpha * out["T"]
    return dict(e_r=e_r, e_v=e_v, e_m=e_m, out=out, tau=tau)


def hinges(model, phys, out, tau, site, n_hat, D_allow, sc):
    """Only the genuine inequalities, EACH NON-DIMENSIONALISED BY ITS OWN CHANNEL'S SCALE.

    The scaling is not cosmetic. Written in raw SI these are squared metres, squared m/s and squared
    kilograms, while every supervised term is dimensionless -- so a terminal point 300 m off scored
    (300 - 0.1)^2 = 9e4 against a position term of order 1e-3, ten orders of magnitude apart. The
    optimiser then did the rational thing and destroyed the trajectory to satisfy one terminal
    constraint: measured at step 4000 of the first run, h_miss was 99.4 % of the total loss and the
    held-out position error had gone from 1455 m to 9235 m. Divided through, w_hinge = 1 means what
    it says and the hinges are comparable with everything else.
    """
    rel = torch.relu
    h = {}
    # slew: |dd/dtau| / tf <= slew_max. ||d|| = 1, so |d-dot| IS the turn rate.
    dddt = d_dtau("d", out, tau) / out["tf"].unsqueeze(-1)
    h["slew"] = ((rel(dddt.norm(dim=-1) - phys.slew_max) / phys.slew_max) ** 2).mean()
    # terminal, applied where tau is at the end of the arc
    near = (tau > 0.995)
    if near.any():
        v_end = out["v"][near]
        h["vtouch"] = ((rel(v_end.norm(dim=-1) - phys.vtouch) / phys.vtouch) ** 2).mean()
        if model.cfg.terminal != "site":
            # with terminal == "site" the lateral miss is identically zero by construction, so this
            # term would only add a gradient of noise
            dr = out["r"][near] - site
            lat = (dr - (dr @ n_hat).unsqueeze(-1) * n_hat).norm(dim=-1)
            h["miss"] = ((rel(lat - D_allow) / sc.L) ** 2).mean()
    # pointing cone and glideslope hold along the whole arc
    h["point"] = (rel(np.cos(phys.point) - (out["d"] @ n_hat)) ** 2).mean()   # already O(1)
    dr = out["r"] - site
    up = dr @ n_hat
    latr = (dr - up.unsqueeze(-1) * n_hat).norm(dim=-1)
    h["glide"] = ((rel(latr - np.tan(phys.gs) * up) / sc.L) ** 2).mean()
    h["mdry"] = ((rel(phys.mdry - out["m"]) / sc.M) ** 2).mean()
    return h


def supervised(out, lab, sc):
    """Per-channel MSE in the FROZEN scaling, so no channel dominates by unit choice alone."""
    e = {}
    e["r"] = (((out["r"] - lab["r"]) / sc.L) ** 2).mean()
    e["v"] = (((out["v"] - lab["v"]) / sc.V) ** 2).mean()
    e["m"] = (((out["m"] - lab["m"]) / sc.M) ** 2).mean()
    e["d"] = ((out["d"] - lab["d"]) ** 2).mean()
    e["T"] = (((out["T"] - lab["T"]) / sc.T_N) ** 2).mean()
    e["tgo"] = (((out["tgo"] - lab["tgo"]) / sc.TF) ** 2).mean()
    return e

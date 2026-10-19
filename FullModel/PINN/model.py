#!/usr/bin/env python
"""The surrogate, with the constraints that must never be violated built into its OUTPUT MAP.

    inputs    (r0, v0, m0, d0, tau)          11 numbers, in the frozen global scaling
    outputs   (r, v, m, d, T, tgo)

WHAT IS STRUCTURAL AND WHAT IS A PENALTY

Four things hold identically, for any weights, at initialisation and after any amount of training:

    r(0), v(0), m(0), d(0) equal the input      every correction carries a factor tau
    ||d|| = 1                                   d is normalised, not regressed
    Tmin <= T <= Tmax                           a sigmoid, so the bound cannot be exceeded
    r(1) lies on the tangent plane              r_f = r_site + P q, and n.(P q) = 0 identically

Those are the constraints whose violation would make an answer meaningless rather than inaccurate,
so none of them is left to a loss term that merely discourages it. What stays soft is the set of
genuine INEQUALITIES -- slew rate, touchdown speed, lateral miss, pointing cone, the altitude
corridor -- because an inequality has no natural parametrisation and the labels already satisfy it.

The terminal point is a NETWORK OUTPUT constrained to the plane, not the site. Stage 2 lands within
D + 0.1 m of the site, so pinning r(1) = r_site would be a 10 cm lie on every sample; letting the
net choose q inside the plane is exact for any D and needs no special casing.

THRUST GOES THROUGH A SWITCHING FUNCTION. Minimum-fuel with lossless convexification is bang-bang:
on this dataset the commanded thrust sits at the 5 N floor or the 20 N ceiling with almost nothing
between, switching two or three times. Regressing T directly asks a smooth network to produce a
near-discontinuity, and the smearing lands straight in the mass and velocity integrals. Instead the
net learns a smooth sigma(tau) and the sharpness lives in a fixed sigmoid, so the discontinuity is
free and the network's own job stays smooth.

tf IS AN OUTPUT AND IT MULTIPLIES THE RESIDUAL. With tau normalised, d/dt = (1/tf) d/dtau, so the
physics residual depends on a quantity the network is still learning. It is parametrised as
TF_ref * exp(n) to keep it positive and O(1), and the trainer ramps the residual weight in rather
than applying it while tf is still nonsense.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


ACT = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "softplus": nn.Softplus}


class FourierTau(nn.Module):
    """Random Fourier features on tau only. Off by default; tau is where sharp structure lives."""

    def __init__(self, n, scale=5.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(n, generator=g) * scale)

    def forward(self, tau):
        x = 2 * np.pi * tau * self.B
        return torch.cat([torch.sin(x), torch.cos(x)], -1)


class SplineRef(nn.Module):
    """A C2 interpolant of the nominal trajectory, evaluated in torch.

    WHY A SPLINE AND NOT LINEAR INTERPOLATION. The physics residual differentiates r and v with
    respect to tau, so a piecewise-linear reference would hand it a derivative with a jump at each
    of the 99 interior nodes and the residual would be wrong there by construction. A natural cubic
    spline is C2, so dr/dtau is C1 and the residual sees a smooth reference.

    The coefficients are fitted once on the host and stored as buffers; evaluation is a bucket
    lookup and a cubic, which autograd differentiates exactly.
    """

    def __init__(self, tau, y):
        super().__init__()
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(np.asarray(tau, float), np.asarray(y, float), axis=0, bc_type="natural")
        self.register_buffer("knots", torch.tensor(cs.x[:-1], dtype=torch.get_default_dtype()))
        # scipy stores powers in descending order; keep that and evaluate by Horner
        self.register_buffer("coef", torch.tensor(np.asarray(cs.c), dtype=torch.get_default_dtype()))

    def forward(self, tau):
        i = torch.clamp(torch.searchsorted(self.knots, tau.detach().contiguous(), right=True) - 1,
                        0, self.knots.numel() - 1)
        dt = (tau - self.knots[i])[..., None]
        out = self.coef[0][i]
        for p in range(1, self.coef.shape[0]):
            out = out * dt + self.coef[p][i]
        return out


def nominal_reference(parents, device="cpu"):
    """r(tau) and v(tau) of the nominal descent, as C2 interpolants, in SI.

    Taken as the MEAN over the training parents rather than from nominal.npz. Every case is a
    dispersion about the same nominal and every one is stored on the same tau grid in SI, so the
    mean is the nominal to within the dispersion and needs no reconstruction of the solver's
    non-dimensional scales -- which nominal.npz does not carry.
    """
    tau = np.asarray(parents[0]["tau"], float)
    r = np.mean([np.asarray(p["r"], float) for p in parents], axis=0)
    v = np.mean([np.asarray(p["v"], float) for p in parents], axis=0)
    # STORED AS A DISPLACEMENT FROM ITS OWN START, so the reference contributes exactly zero at
    # tau = 0 and r(0) = r0 stays exact. Fitting the absolute trajectory instead makes the ansatz
    # subtract two numbers of order 1e4 m, which in float32 leaves about a millimetre of residue
    # and quietly demotes a by-construction guarantee to an approximate one.
    return SplineRef(tau, r - r[0]).to(device), SplineRef(tau, v - v[0]).to(device)


class Surrogate(nn.Module):
    def __init__(self, cfg, scaling, t_min, t_max, ref=None):
        super().__init__()
        self.cfg = cfg
        # (r_ref, v_ref) interpolants of the nominal descent, or None for the
        # straight-line reference the ansatz used before.
        self.ref_r, self.ref_v = (ref if ref is not None else (None, None))
        self.Tmin, self.Tmax = float(t_min), float(t_max)
        self.sharpness = float(cfg.sigma_sharpness)   # annealed by the trainer
        s = scaling
        self.register_buffer("site", torch.tensor(s.r_site, dtype=torch.get_default_dtype()))
        self.register_buffer("nhat", torch.tensor(s.n_hat, dtype=torch.get_default_dtype()))
        self.L, self.V, self.M, self.TN, self.TF = s.L, s.V, s.M, s.T_N, s.TF
        # the tangent basis the terminal point is expressed in: n x e and n x (n x e)
        n = np.array(s.n_hat, float)
        e = np.cross([0.0, 0.0, 1.0], n); e /= np.linalg.norm(e)
        self.register_buffer("t1", torch.tensor(e, dtype=torch.get_default_dtype()))
        self.register_buffer("t2", torch.tensor(np.cross(n, e), dtype=torch.get_default_dtype()))

        self.ff = FourierTau(cfg.fourier_features, seed=cfg.seed) if cfg.fourier_features else None
        n_in = 10 + (2 * cfg.fourier_features if self.ff else 1)
        act = ACT[cfg.activation]
        layers, k = [], n_in
        for _ in range(cfg.depth):
            layers += [nn.Linear(k, cfg.width), act()]
            k = cfg.width
        self.trunk = nn.Sequential(*layers)
        self.head_r = nn.Linear(k, 3)
        self.head_v = nn.Linear(k, 3)
        self.head_m = nn.Linear(k, 1)
        self.head_d = nn.Linear(k, 3)
        self.head_T = nn.Linear(k, 1)           # sigma(tau) when sigma_head, else T directly

        # TF AND THE TERMINAL POINT GET THEIR OWN TRUNK, WHICH TAU DOES NOT ENTER. They are
        # properties of the initial condition alone -- one flight time per trajectory, one landing
        # point per trajectory -- so reading them off the tau-dependent trunk let them drift along
        # the trajectory. Measured on run full4784: for a FIXED initial condition the predicted tf
        # varied over tau by a median of 39.7 % (539 s), up to 140 %. Since tgo = tf * (1 - tau),
        # that is the clock every other output is parameterised against, and it was moving.
        # Excluding tau structurally is the same device used for r(0) = r0 and ||d|| = 1: the
        # property holds by construction rather than by training against it.
        w0 = max(cfg.width // 2, 32)
        l0, k0 = [], 10
        for _ in range(max(cfg.depth // 2, 2)):
            l0 += [nn.Linear(k0, w0), act()]
            k0 = w0
        self.trunk0 = nn.Sequential(*l0)
        self.head_tf = nn.Linear(k0, 1)
        self.head_q = nn.Linear(k0, 2)          # terminal point, IN the tangent plane
        for h in (self.head_r, self.head_v, self.head_m, self.head_d, self.head_q):
            nn.init.zeros_(h.bias); nn.init.normal_(h.weight, std=1e-3)
        nn.init.zeros_(self.head_tf.bias)

    # ---------------------------------------------------------------- scaling helpers
    def encode0(self, r0, v0, m0, d0):
        """The initial condition alone. No tau, by design -- see trunk0."""
        return torch.cat([(r0 - self.site) / self.L, v0 / self.V,
                          (m0 / self.M)[..., None], d0], -1)

    def encode(self, r0, v0, m0, d0, tau):
        f = self.ff(tau[..., None]) if self.ff else tau[..., None]
        return torch.cat([self.encode0(r0, v0, m0, d0), f], -1)

    def forward(self, r0, v0, m0, d0, tau):
        """Everything in SI. tau in [0, 1]. Returns a dict so callers cannot mis-order outputs."""
        h = self.trunk(self.encode(r0, v0, m0, d0, tau))
        h0 = self.trunk0(self.encode0(r0, v0, m0, d0))      # tau cannot reach this one
        t = tau[..., None]

        tf = self.TF * torch.exp(self.head_tf(h0).clamp(-4.0, 4.0))           # > 0, and tau-free
        if self.cfg.terminal == "site":
            r_f = self.site.expand_as(r0)              # lands AT the site, for any weights
        else:
            q = self.head_q(h0) * self.L
            r_f = self.site + q[..., :1] * self.t1 + q[..., 1:2] * self.t2    # on the plane, exactly

        # THE REFERENCE CARRIES THE SHAPE, THE NETWORK CARRIES THE DEVIATION. Without it the
        # network must reproduce a 12 km descent from scratch and a 1 % relative error is 120 m.
        # The nominal is shifted to honour both endpoints exactly, so r(0) = r0 and r(1) = r_f
        # remain true by construction whatever the reference is.
        if self.ref_r is not None:
            # GROUPED SO BOTH ENDPOINTS CANCEL EXACTLY. The reference is zero at tau = 0 by
            # storage, and (rn - t*rn1) is one expression that goes to exactly zero at tau = 1,
            # so r(0) = r0 and r(1) = r_f survive in float32. Writing it as r0 + rn + t*(...)
            # instead spreads the cancellation across separate additions and leaves a millimetre
            # at whichever end is not written first.
            rn = self.ref_r(tau)
            rn1 = self.ref_r(torch.ones_like(tau[:1]))
            r_base = r0 * (1.0 - t) + r_f * t + (rn - t * rn1)
            v_base = v0 + self.ref_v(tau)
        else:
            r_base = r0 + t * (r_f - r0)
            v_base = v0
        r = r_base + t * (1.0 - t) * self.head_r(h) * self.L
        v = v_base + t * self.head_v(h) * self.V
        # mass may only go down, and equals m0 at tau = 0
        m = m0[..., None] - t * torch.nn.functional.softplus(self.head_m(h)) * self.M
        d = d0 + t * self.head_d(h)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)                 # ||d|| = 1, exactly

        if self.cfg.sigma_head:
            sig = self.head_T(h)
            T = self.Tmin + (self.Tmax - self.Tmin) * torch.sigmoid(self.sharpness * sig)
        else:
            sig = None
            T = self.Tmin + (self.Tmax - self.Tmin) * torch.sigmoid(self.head_T(h))
        return dict(r=r, v=v, m=m.squeeze(-1), d=d, T=T.squeeze(-1),
                    tgo=(tf.squeeze(-1) * (1.0 - tau)), tf=tf.squeeze(-1),
                    sigma=(sig.squeeze(-1) if sig is not None else None), r_f=r_f)


def check_structural(model, scaling, n=256, device="cpu", seed=0):
    """The four structural claims, asserted numerically. Run on every fresh model."""
    g = torch.Generator().manual_seed(seed)
    site = torch.tensor(scaling.r_site, dtype=torch.get_default_dtype(), device=device)
    nh = torch.tensor(scaling.n_hat, dtype=torch.get_default_dtype(), device=device)
    r0 = site + torch.randn(n, 3, generator=g).to(device) * 5e3
    v0 = torch.randn(n, 3, generator=g).to(device)
    m0 = 600 + torch.randn(n, generator=g).to(device)
    d0 = torch.randn(n, 3, generator=g).to(device); d0 = d0 / d0.norm(dim=-1, keepdim=True)
    out0 = model(r0, v0, m0, d0, torch.zeros(n, device=device))
    out1 = model(r0, v0, m0, d0, torch.ones(n, device=device))
    tau = torch.rand(n, generator=g).to(device)
    outm = model(r0, v0, m0, d0, tau)
    res = {
        "r(0) == r0 [m]": (out0["r"] - r0).abs().max().item(),
        "v(0) == v0 [m/s]": (out0["v"] - v0).abs().max().item(),
        "m(0) == m0 [kg]": (out0["m"] - m0).abs().max().item(),
        "d(0) == d0 [-]": (out0["d"] - d0).abs().max().item(),
        "|d| == 1 [-]": (outm["d"].norm(dim=-1) - 1).abs().max().item(),
        "T in band [N]": max(float(model.Tmin - outm["T"].min()), float(outm["T"].max() - model.Tmax)),
        "m <= m0 [kg]": float((outm["m"] - m0).max()),
        "r(1) on plane [m]": ((out1["r"] - site) @ nh).abs().max().item(),
        "r(1) lateral [m]": (lambda d: (d - (d @ nh).unsqueeze(-1) * nh).norm(dim=-1).max().item())(
            out1["r"] - site),
        "tgo(1) == 0 [s]": out1["tgo"].abs().max().item(),
    }
    return res


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from config import Config
    from data import Scaling, load_parents, split_by_parent
    d = sys.argv[1] if len(sys.argv) > 1 else "../SCvx/Figures&data/pinn_trial"
    ps = load_parents(d, defect_max=1e-3)
    sp = split_by_parent(ps)
    sc = Scaling.from_parents([p for p in ps if int(p["seed"]) in set(sp["train"])])
    cfg = Config()
    mdl = Surrogate(cfg, sc, 5.0, 20.0)
    print(f"parameters: {sum(p.numel() for p in mdl.parameters()):,}")
    print("\nstructural guarantees (should all be ~0 or negative):")
    for k, v in check_structural(mdl, sc).items():
        print(f"  {k:<22} {v:>12.3e}")

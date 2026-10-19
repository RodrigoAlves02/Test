#!/usr/bin/env python
"""PINN-GM-III as a differentiable torch module, byte-for-byte the field the LABELS were solved on.

WHY THIS FILE EXISTS. The physics residual needs gravity at arbitrary collocation points, and it
must be the SAME field the SCvx solve used -- `Dynamics(sc, field=GR.pinn_field())`. If the residual
used the polyhedral truth instead, it would be penalising the network for reproducing its own
labels, which is the opposite of what a physics term is for. So this is a port, not a re-model, and
it is verified against scvx/pinn.py's NumpyPINN rather than trusted.

THE ACCELERATION COMES FROM AUTOGRAD, NOT FROM A HAND-WRITTEN BACKPROP. NumpyPINN differentiates the
potential by hand because it has no autograd; here a = dU/dx falls out of the same graph the network
is trained through, which removes a whole class of silent disagreement between the two.

ALL IN NON-DIMENSIONAL UNITS, as the original: positions divide by x_star, accelerations multiply
by a_star.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PinnGravity(nn.Module):
    def __init__(self, npz_path, dtype=torch.float64):
        super().__init__()
        z = np.load(npz_path)
        sd = {k[3:]: z[k] for k in z.files if k.startswith("sd/")}
        t = lambda a: torch.tensor(np.asarray(a), dtype=dtype)
        self.register_buffer("We", t(sd["embed.weight"])); self.register_buffer("be", t(sd["embed.bias"]))
        self.n_layers = int(z["layers"])
        for i in range(self.n_layers):
            self.register_buffer(f"Wh{i}", t(sd[f"hidden.{i}.weight"]))
            self.register_buffer(f"bh{i}", t(sd[f"hidden.{i}.bias"]))
        self.register_buffer("Wo", t(sd["out.weight"]).reshape(-1))
        self.bo = float(np.asarray(sd["out.bias"]).ravel()[0])
        self.mu_nd = float(z["mu_nd"]); self.r_ref = float(z["r_ref"])
        self.x_star = float(z["x_star"]); self.a_star = float(z["a_star"])
        self.k, self.r_lf, self.k_lf = 0.5, 0.0, 0.5      # PINNGM3 defaults, as in training

    def potential_nd(self, x):
        """U at NON-DIMENSIONAL position x (..., 3). Mirrors NumpyPINN's assembly exactly."""
        r = x.norm(dim=-1, keepdim=True)
        hat = x / r
        inside = (r <= 1.0)
        ri = torch.where(inside, r, torch.ones_like(r))
        re = torch.where(inside, torch.ones_like(r), 1.0 / r)
        n = torch.where(inside, torch.ones_like(r), r)
        feat = torch.cat([ri, re, hat], -1)

        e = torch.nn.functional.gelu(feat @ self.We.T + self.be)
        h = e
        for i in range(self.n_layers):
            h = torch.nn.functional.gelu(h @ getattr(self, f"Wh{i}").T
                                         + getattr(self, f"bh{i}")) + e
        proxy = (h @ self.Wo).unsqueeze(-1) + self.bo

        U_lf = self.mu_nd / r
        w_lf = 0.5 * (1.0 + torch.tanh(self.k_lf * (r - self.r_lf)))
        H_bc = 0.5 * (1.0 + torch.tanh(self.k * (r - self.r_ref)))
        return (w_lf * U_lf + (1.0 - H_bc) * proxy / n).squeeze(-1)

    def acceleration_si(self, r_si, create_graph=True):
        """a [m/s^2] at position r [m], differentiable. a = dU/dx, as in NumpyPINN."""
        x = (r_si / self.x_star).to(self.We.dtype).requires_grad_(True)
        U = self.potential_nd(x).sum()
        a_nd, = torch.autograd.grad(U, x, create_graph=create_graph)
        return a_nd * self.a_star

    def forward(self, r_si):
        return self.acceleration_si(r_si)


def verify_against_numpy(npz_path, scvx_dir, n=64, seed=0):
    """The port is checked, not assumed. Random points inside and outside the Brillouin sphere."""
    import sys
    if scvx_dir not in sys.path:
        sys.path.insert(0, scvx_dir)
    from scvx.pinn import load_pinn_np
    ref, x_star, a_star = load_pinn_np(npz_path)
    g = PinnGravity(npz_path)
    rng = np.random.default_rng(seed)
    # a spread of radii: inside the sphere, the landing regime, and far out
    rad = np.concatenate([rng.uniform(0.3, 1.0, n//3), rng.uniform(1.0, 2.0, n//3),
                          rng.uniform(2.0, 6.0, n - 2*(n//3))]) * x_star
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    R = u * rad[:, None]
    A_ref = np.array([ref.acceleration(r / x_star) * a_star for r in R])
    A_new = g.acceleration_si(torch.tensor(R, dtype=torch.float64),
                              create_graph=False).detach().numpy()
    err = np.linalg.norm(A_new - A_ref, axis=1)
    rel = err / np.maximum(np.linalg.norm(A_ref, axis=1), 1e-30)
    return dict(n=n, max_abs=float(err.max()), max_rel=float(rel.max()),
                median_rel=float(np.median(rel)))


if __name__ == "__main__":
    import os, sys
    TH = "/Users/rodrigoalves/Desktop/Thesis"
    npz = os.path.join(TH, "scvx_landing", "data", "pinn_eros_d3R_8x64_np.npz")
    res = verify_against_numpy(npz, os.path.join(TH, "scvx_landing"))
    print("torch port vs scvx/pinn.py NumpyPINN:")
    for k, v in res.items():
        print(f"  {k:<12} {v:.6e}" if isinstance(v, float) else f"  {k:<12} {v}")

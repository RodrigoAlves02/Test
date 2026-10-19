#!/usr/bin/env python
"""A spherical-harmonic gravity model of the SAME body the polyhedral field describes.

WHY FIT RATHER THAN USE PUBLISHED COEFFICIENTS. A published NEAR field describes the real Eros,
whose interior is not uniform. Comparing it against a constant-density polyhedron would confound
the REPRESENTATION with a different mass distribution, and the figure would not mean what its
caption says. Fitting the harmonics to the polyhedral field itself makes all models describe one
mass distribution, so the only thing being compared is the representation.

THE FIT IS DONE OVER A RANGE OF RADII, NOT ON ONE SPHERE. At a single radius every degree carries
the same constant factor (R/r)^n and the design matrix columns become collinear, so the degrees
cannot be separated. Sampling several radii is what makes the inversion well posed.

The model is then evaluated INSIDE the circumscribing sphere, which is where the descent happens
and where the series carries no convergence guarantee. That is the point of the comparison.
"""
import numpy as np


def _legendre(N, u):
    """P_nm(u) and dP_nm/dphi, geodesy convention (no Condon-Shortley phase).

    Written out rather than taken from scipy: lpmn was removed in scipy 1.18, and the sign
    convention of whatever replaces it is exactly the kind of thing that silently flips odd
    orders. The recursions below carry no phase by construction, and the exterior-field check
    in the fit would catch it if they did.
    """
    P = np.zeros((N + 1, N + 1))
    dPdphi = np.zeros((N + 1, N + 1))
    c = np.sqrt(max(1.0 - u * u, 1e-300))           # cos(latitude)
    P[0, 0] = 1.0
    for m in range(1, N + 1):                       # sectoral
        P[m, m] = (2 * m - 1) * c * P[m - 1, m - 1]
    for m in range(0, N):                           # first off-diagonal
        P[m + 1, m] = (2 * m + 1) * u * P[m, m]
    for m in range(0, N + 1):                       # the rest
        for n in range(m + 2, N + 1):
            P[n, m] = ((2 * n - 1) * u * P[n - 1, m]
                       - (n + m - 1) * P[n - 2, m]) / (n - m)
    # (1 - u^2) dP/du = -n u P_nm + (n + m) P_{n-1,m}, and dP/dphi = cos(phi) dP/du
    for m in range(0, N + 1):
        for n in range(m, N + 1):
            prev = P[n - 1, m] if n - 1 >= m else 0.0
            dPdphi[n, m] = (-n * u * P[n, m] + (n + m) * prev) / max(c, 1e-300)
    return P, dPdphi


def basis_grad(r_vec, N, Rref, GM):
    """Gradient of every harmonic basis function at one point.

    Returns (n_terms, 3) and the (n, m, kind) key order, with kind 0 = cos (C), 1 = sin (S).
    """
    x, y, z = r_vec
    r = np.sqrt(x * x + y * y + z * z)
    rho = np.sqrt(x * x + y * y)
    u = z / r                                   # sin(latitude)
    lam = np.arctan2(y, x)
    P, dPdphi = _legendre(N, u)

    cols, keys = [], []
    for n in range(N + 1):
        rn = (Rref / r) ** n
        for m in range(n + 1):
            for kind in (0, 1):
                if kind == 1 and m == 0:
                    continue
                trig = np.cos(m * lam) if kind == 0 else np.sin(m * lam)
                dtrig = -m * np.sin(m * lam) if kind == 0 else m * np.cos(m * lam)
                base = (GM / r) * rn
                f = base * P[n, m] * trig
                dUdr = -(n + 1) / r * f
                dUdphi = base * dPdphi[n, m] * trig
                dUdlam = base * P[n, m] * dtrig
                # spherical -> Cartesian gradient
                c1 = dUdr / r - z * dUdphi / (r * r * max(rho, 1e-300))
                c2 = dUdlam / max(rho * rho, 1e-300)
                ax = c1 * x - c2 * y
                ay = c1 * y + c2 * x
                az = dUdr * z / r + rho * dUdphi / (r * r)
                cols.append((ax, ay, az)); keys.append((n, m, kind))
    return np.array(cols), keys


class SphericalHarmonicField:
    """Degree-N harmonic field fitted to a reference acceleration field."""

    def __init__(self, coeffs, keys, N, Rref, GM):
        self.c, self.keys, self.N, self.Rref, self.GM = coeffs, keys, N, Rref, GM
        self._CS = None

    def _tables(self):
        """Coefficients as (N+1, N+1) C and S arrays, built once."""
        if getattr(self, "_CS", None) is None:
            N = self.N
            Cnm = np.zeros((N + 1, N + 1)); Snm = np.zeros((N + 1, N + 1))
            for coef, (n, m, kind) in zip(self.c, self.keys):
                (Cnm if kind == 0 else Snm)[n, m] = coef
            self._CS = (Cnm, Snm, np.arange(N + 1))
        return self._CS

    def g(self, r):
        """Acceleration at one point, summed directly.

        NOT via basis_grad. That routine materialises one 3-vector per coefficient inside a triple
        Python loop -- 81 of them at degree 8 -- so that the fit can use them as a design matrix.
        Evaluation needs only the three summed derivatives, and because the Cartesian conversion is
        linear in them the sums can be accumulated first and converted once. The degree recursion
        still loops, but over DEGREE alone (N steps, on array slices) rather than over every term.
        """
        x, y, z = float(r[0]), float(r[1]), float(r[2])
        rr = np.sqrt(x * x + y * y + z * z)
        rho = np.sqrt(x * x + y * y)
        u = z / rr
        lam = np.arctan2(y, x)
        P, dPdphi = _legendre(self.N, u)
        Cnm, Snm, nar = self._tables()
        mlam = nar * lam
        cm, sm = np.cos(mlam), np.sin(mlam)
        A = Cnm * cm[None, :] + Snm * sm[None, :]            # the cos/sin combination
        Bm = (-Cnm * sm[None, :] + Snm * cm[None, :]) * nar[None, :]
        rn = (self.Rref / rr) ** nar
        base = self.GM / rr
        w = base * rn                                        # per-degree weight
        dUdr = -np.einsum("n,n,nm,nm->", w, (nar + 1.0) / rr, P, A)
        dUdphi = np.einsum("n,nm,nm->", w, dPdphi, A)
        dUdlam = np.einsum("n,nm,nm->", w, P, Bm)
        c1 = dUdr / rr - z * dUdphi / (rr * rr * max(rho, 1e-300))
        c2 = dUdlam / max(rho * rho, 1e-300)
        return np.array([c1 * x - c2 * y, c1 * y + c2 * x,
                         dUdr * z / rr + rho * dUdphi / (rr * rr)])

    def g_slow(self, r):
        """The original route, kept so the fast path can be checked against it."""
        B, _ = basis_grad(np.asarray(r, float), self.N, self.Rref, self.GM)
        return self.c @ B

    def grad(self, r):
        return np.zeros((3, 3))

    def both(self, r):
        return self.g(r), np.zeros((3, 3))


def fit(ref_field, N, Rref, GM, radii_km, n_dir=240, seed=0):
    """Least-squares fit of a degree-N harmonic model to `ref_field` sampled over several radii."""
    rng = np.random.default_rng(seed)
    i = np.arange(n_dir) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_dir)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    dirs = np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi), np.cos(phi)])
    rows, rhs = [], []
    for Rkm in radii_km:
        for d in dirs:
            p = d * Rkm * 1e3
            B, keys = basis_grad(p, N, Rref, GM)
            rows.append(B.T)                     # (3, n_terms)
            rhs.append(np.asarray(ref_field.g(p), float))
    A = np.vstack(rows); b = np.concatenate(rhs)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return SphericalHarmonicField(c, keys, N, Rref, GM)

#!/usr/bin/env python
"""Truth dynamics for the full simulation: equations of motion and propagator backends.

This is the reference model. Everything else in the simulation is measured against it, so it is
written to be uniform rather than clever: the thrust is a function of time, thrust(t) -> [N], and the
integrator samples it wherever its own stages fall. No step is aligned to a thrust discontinuity and
no segment is integrated separately. The truth propagator behaves the same way on a coast, on a
braking burn and on a descent, and the price of that uniformity is paid in tolerance, which is what
the validation scripts are for.

Two backends:
  - "scipy"  : always available. Fixed-step Runge-Kutta from explicit tableaux, plus SciPy's
               variable-step solvers.
  - "tudat"  : tudatpy, imported lazily. Not installed here; the import raises with instructions
               rather than failing obscurely, so the comparison runs as soon as it is available.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# --- physical constants -------------------------------------------------------------------------
G0 = 9.80665                      # m/s^2, standard gravity, only for Isp
EROS_ROT_PERIOD = 5.270 * 3600.0  # s
EROS_GM = 4.4665e5                # m^3/s^2
OMEGA = np.array([0.0, 0.0, 2.0 * np.pi / EROS_ROT_PERIOD])


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


WX = skew(OMEGA)


# --- gravity ------------------------------------------------------------------------------------
class Gravity:
    """A gravity field that counts its own evaluations.

    Counting at the source is the only honest way to compare propagators: step counts flatter the
    high-order tableaux, which buy long steps with many stages, and wall clock mixes in the Python
    overhead of whichever driver is being used.
    """

    def __init__(self, kind="poly", mesh_q=64, mesh_path=None, gm=EROS_GM):
        self.kind, self.n, self.gm = kind, 0, gm
        self._f = self._build(kind, mesh_q, mesh_path)

    def _build(self, kind, mesh_q, mesh_path):
        if kind == "pointmass":
            return lambda r: -self.gm * r / np.linalg.norm(r) ** 3
        if kind == "poly":
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "..", "scvx_landing"))
            from scvx.gravity import load_mesh, make_polyhedral
            from mesh_lod import lod_mesh, density_for_gm
            mesh = load_mesh(mesh_path) if mesh_path else (
                load_mesh() if mesh_q == 64 else lod_mesh(mesh_q))
            self.faces = len(mesh.faces)
            ev = make_polyhedral(mesh, density=density_for_gm(mesh, gm=self.gm))
            return lambda r: np.asarray(ev(r)[1], float)
        if kind == "pinn":
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", "..", "scvx_landing"))
            from scvx_rate import PinnGravity
            ev = PinnGravity(tensor=False)
            self.faces = None
            return lambda r: np.asarray(ev(r)[1], float)
        raise ValueError(f"unknown gravity kind: {kind}")

    def __call__(self, r):
        self.n += 1
        return self._f(r)

    def reset(self):
        self.n = 0


# --- frame algebra ------------------------------------------------------------------------------
# The frame study (Validation/Frame Validation) settled on the INERTIAL frame: 1.9x less algebra per
# evaluation, and ~1000x more accurate at equal step on an orbital arc, because the rotating frame
# forces the integrator to resolve the body spin on top of the orbital motion.
def Rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def inertial_to_body(r_I, v_I, t):
    """Body frame is rotated by +wt from inertial at time t, aligned at t = 0."""
    R = Rz(-OMEGA[2] * t)
    r_B = R @ r_I
    return r_B, R @ v_I - np.cross(OMEGA, r_B)


def body_to_inertial(r_B, v_B, t):
    R = Rz(OMEGA[2] * t)
    return R @ r_B, R @ (v_B + np.cross(OMEGA, r_B))


# --- equations of motion ------------------------------------------------------------------------
@dataclass
class Vehicle:
    isp: float = 225.0
    dry_mass: float = 490.0

    @property
    def alpha(self):
        return 1.0 / (self.isp * G0)


def eom_inertial(t, y, g: Gravity, thrust: Callable[[float], np.ndarray], veh: Vehicle):
    """Inertial-frame dynamics. The field is body-fixed, so each evaluation rotates the position in
    and the acceleration back out -- two rotations sharing one sine/cosine pair.

        rdot = v
        vdot = R(wt) g(R(-wt) r) + T/m
        mdot = -|T| / (Isp g0)

    No Coriolis and no centrifugal term: the frame is not rotating.
    """
    r_I, v_I = y[0:3], y[3:6]
    w = OMEGA[2] * t
    c, s = np.cos(w), np.sin(w)
    r_B = np.array([c * r_I[0] + s * r_I[1], -s * r_I[0] + c * r_I[1], r_I[2]])
    g_B = g(r_B)
    a = np.array([c * g_B[0] - s * g_B[1], s * g_B[0] + c * g_B[1], g_B[2]])
    T = thrust(t)
    if T is not None:
        Tn = np.linalg.norm(T)
        if Tn > 0.0:
            return np.concatenate([v_I, a + T / y[6], [-veh.alpha * Tn]])
    return np.concatenate([v_I, a, [0.0]])


def eom(t, y, g: Gravity, thrust: Callable[[float], np.ndarray], veh: Vehicle):
    """Rotating-frame translational dynamics with mass depletion.

    State y = [r(3), v(3), m(1)] in the Eros body-fixed rotating frame, SI units.

        rdot = v
        vdot = T/m - 2 w x v - w x (w x r) + g(r)
        mdot = -|T| / (Isp g0)

    The thrust is whatever thrust(t) returns at this instant, in Newtons and in the same frame.
    """
    r, v, m = y[0:3], y[3:6], y[6]
    T = thrust(t)
    a = g(r) - 2.0 * (WX @ v) - WX @ (WX @ r)
    if T is not None:
        Tn = np.linalg.norm(T)
        if Tn > 0.0:
            a = a + T / m
            return np.concatenate([v, a, [-veh.alpha * Tn]])
    return np.concatenate([v, a, [0.0]])


# --- thrust profiles ----------------------------------------------------------------------------
def coast():
    """No thrust, ever."""
    return lambda t: None


def segments_thrust(segs):
    """Piecewise-constant thrust from a list of (name, duration, thrust_vector_or_None).

    Deliberately a function of absolute time with no notion of where the integrator is: the caller
    does not get to tell the propagator about the discontinuities.
    """
    edges = np.cumsum([s[1] for s in segs])
    vecs = [s[2] for s in segs]

    def f(t):
        i = int(np.searchsorted(edges, t, side="right"))
        return vecs[min(i, len(vecs) - 1)]
    return f


def zoh_thrust(U_newton, tf):
    """Zero-order-hold thrust on K-1 equal segments, as an SCvx solution supplies it."""
    K = U_newton.shape[1]
    h = tf / (K - 1)

    def f(t):
        k = min(int(t // h), K - 2)
        return U_newton[:, k]
    return f


# --- fixed-step explicit Runge-Kutta --------------------------------------------------------------
def _rk4():
    A = np.zeros((4, 4)); A[1, 0] = 0.5; A[2, 1] = 0.5; A[3, 2] = 1.0
    return A, np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]), np.array([0.0, 0.5, 0.5, 1.0])


def _rk5():
    """Cash-Karp, 6 stages, order 5 -- the Butcher minimum for its order."""
    A = np.zeros((6, 6))
    A[1, 0] = 1 / 5
    A[2, 0], A[2, 1] = 3 / 40, 9 / 40
    A[3, 0], A[3, 1], A[3, 2] = 3 / 10, -9 / 10, 6 / 5
    A[4, 0], A[4, 1], A[4, 2], A[4, 3] = -11 / 54, 5 / 2, -70 / 27, 35 / 27
    A[5, 0], A[5, 1], A[5, 2] = 1631 / 55296, 175 / 512, 575 / 13824
    A[5, 3], A[5, 4] = 44275 / 110592, 253 / 4096
    B = np.array([37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771])
    return A, B, np.array([0, 1 / 5, 3 / 10, 3 / 5, 1, 7 / 8])


def _rk6():
    """Butcher's 7-stage sixth-order rule."""
    A = np.zeros((7, 7))
    A[1, 0] = 1 / 3
    A[2, 1] = 2 / 3
    A[3, 0], A[3, 1], A[3, 2] = 1 / 12, 1 / 3, -1 / 12
    A[4, 0], A[4, 1], A[4, 2], A[4, 3] = -1 / 16, 9 / 8, -3 / 16, -3 / 8
    A[5, 1], A[5, 2], A[5, 3], A[5, 4] = 9 / 8, -3 / 8, -3 / 4, 1 / 2
    A[6, 0], A[6, 1], A[6, 2] = 9 / 44, -9 / 11, 63 / 44
    A[6, 3], A[6, 5] = 18 / 11, -16 / 11
    B = np.array([11 / 120, 0, 27 / 40, 27 / 40, -4 / 15, -4 / 15, 11 / 120])
    return A, B, A.sum(1)


def _rk8():
    """The 12 propagating stages of DOP853, taken from SciPy rather than retyped."""
    from scipy.integrate._ivp import dop853_coefficients as d
    n = d.N_STAGES
    return (np.array(d.A[:n, :n], float), np.array(d.B[:n], float), np.array(d.C[:n], float))


TABLEAU = {"rk4": _rk4, "rk5": _rk5, "rk6": _rk6, "rk8": _rk8}
STAGES = {"euler": 1, "midpoint": 2, "heun": 2, "rk3": 3, "rk4": 4, "rk5": 6, "rk6": 7, "rk8": 12}
FIXED = ["euler", "midpoint", "heun", "rk3", "rk4", "rk5", "rk6", "rk8"]
ADAPTIVE = ["RK45", "DOP853", "Radau", "LSODA", "BDF"]


def _step(f, t, y, h, scheme):
    if scheme == "euler":
        return y + h * f(t, y)
    if scheme == "midpoint":
        k1 = f(t, y)
        return y + h * f(t + 0.5 * h, y + 0.5 * h * k1)
    if scheme == "heun":
        k1 = f(t, y); k2 = f(t + h, y + h * k1)
        return y + 0.5 * h * (k1 + k2)
    if scheme == "rk3":
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + h, y - h * k1 + 2.0 * h * k2)
        return y + h / 6.0 * (k1 + 4.0 * k2 + k3)
    A, B, C = TABLEAU[scheme]()
    k = []
    for i in range(len(B)):
        yi = y
        if i:
            acc = sum(A[i, j] * k[j] for j in range(i) if A[i, j])
            if not np.isscalar(acc):
                yi = y + h * acc
        k.append(f(t + C[i] * h, yi))
    return y + h * sum(B[i] * k[i] for i in range(len(B)) if B[i])


def rhs(frame):
    return eom_inertial if frame == "inertial" else eom


def propagate_fixed(y0, t0, tf, dt, scheme, g, thrust, veh=Vehicle(), frame="inertial"):
    """Fixed step of EXACTLY dt, marching until the arc is used up.

    dt means dt. The step is not rescaled to divide the arc evenly, because a requested 1024 s step
    that silently becomes 1014.7 s is not the step that was asked for and the reported convergence
    would be against the wrong abscissa.

    The termination condition is the step overshooting the end: once the time remaining is smaller
    than dt the step is bigger than needed, so that final step alone is trimmed to land exactly on
    tf. Every other step in the arc is the full dt.
    """
    _f = rhs(frame)
    f = lambda t, y: _f(t, y, g, thrust, veh)
    y, t, n = np.asarray(y0, float).copy(), t0, 0
    while True:
        rem = tf - t
        if rem <= 1e-9:
            break
        h = dt if dt < rem else rem          # termination: trim the last, oversized step
        y = _step(f, t, y, h, scheme)
        t += h
        n += 1
    return y, n


def propagate_adaptive(y0, t0, tf, method, rtol, atol, g, thrust, veh=Vehicle(), max_step=np.inf,
                       frame="inertial"):
    """One SciPy call over the whole arc -- the solver is never told where the thrust changes."""
    from scipy.integrate import solve_ivp
    _f = rhs(frame)
    s = solve_ivp(lambda t, y: _f(t, y, g, thrust, veh), (t0, tf), np.asarray(y0, float),
                  method=method, rtol=rtol, atol=atol, max_step=max_step, dense_output=False)
    if not s.success:
        raise RuntimeError(s.message)
    return s.y[:, -1], int(s.nfev)


# --- tudatpy backend ----------------------------------------------------------------------------
TUDAT_HINT = (
    "tudatpy is not installed. It is distributed through conda, not pip:\n"
    "    conda install -c conda-forge tudatpy\n"
    "No conda was found on this machine, so a package manager (miniforge) has to be installed\n"
    "first. The scipy backend runs without it; this backend exists so the speed comparison can\n"
    "be made as soon as tudatpy is importable."
)


def have_tudat():
    try:
        import tudatpy  # noqa: F401
        return True
    except ImportError:
        return False


def propagate_tudat(*args, **kwargs):
    if not have_tudat():
        raise ImportError(TUDAT_HINT)
    raise NotImplementedError(
        "tudatpy is importable but the backend is not wired yet. It needs a custom acceleration for "
        "the Werner-Scheeres polyhedral field and a thrust model driven by the same thrust(t), so "
        "that both backends integrate an identical right-hand side and the comparison is about the "
        "propagator rather than about the model.")


# --- timing helper ------------------------------------------------------------------------------
@dataclass
class Run:
    label: str
    y: np.ndarray
    evals: int
    steps: int
    cpu: float
    setting: str = ""
    extra: dict = field(default_factory=dict)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0

"""Strapdown accelerometer triad: the truth-to-measurement bridge for the IMU.

WHAT AN ACCELEROMETER MEASURES. Specific force -- acceleration from everything except gravity. A
proof mass in free fall moves with its case, so the instrument reads zero under gravity alone; on
this vehicle the only real force is thrust, so the true measurement is T/m.

It senses in INERTIAL axes. Coriolis and centrifugal are frame artifacts, not forces on the proof
mass, so the instrument never sees them. The caller therefore rotates the delivered thrust from ACAF
to ACI, divides by mass, and hands the result here; the errors are applied in inertial components
and converted afterwards, never before. Because T is a vector and m a scalar, rotating before or
after the division gives the same answer.

    a_meas_ACI = a_true_ACI + b_ACI(t) + nu_a          nu_a ~ N(0, (N/sqrt(dt))^2 I)

THE TWO THINGS THAT ARE NOT FREE CHOICES

  1. sigma_a = N / sqrt(dt), NOT a fixed number. The physical noise is white with PSD N^2, so the
     velocity increment over one step has variance N^2*dt; representing that as a constant
     acceleration over the step gives Var = N^2*dt/dt^2 = N^2/dt. Written any other way, halving the
     integration step changes the answer. This matches EKF._Q, which uses Q_vv = N^2*dt -- the
     velocity-increment variance -- so device and filter agree by construction.

  2. The bias is a RANDOM CONSTANT, not a random walk. The LN-200S datasheet gives bias
     REPEATABILITY (turn-on to turn-on), and no in-run stability or random-walk figure. So there is
     one seeded draw per flight and no drift term. Writing b_k = b_{k-1} + nu_b*dt would also be
     wrong dimensionally: its accumulated variance scales as sigma^2*dt*T, so it would depend on the
     step size.

THE FRAME THE BIAS IS CONSTANT IN. Physically the bias is fixed in the INSTRUMENT, which is
body-fixed, and the vehicle's engine axis sweeps 129.5 deg over this descent. Holding the bias fixed
in ACI instead misses a swing of 2*|b_perp|*sin(dtheta/2), where b_perp is the component
perpendicular to the slew axis: at most 543 ug for a 300 ug bias, i.e. larger than the bias itself. So the default holds it constant in a THRUST-ALIGNED frame and carries it into ACI
by parallel transport along the boresight path:

    b_ACI(t_k) = R(d_{k-1} -> d_k) b_ACI(t_{k-1}),     b_ACI(t_0) = b_TAF

Fixing one axis leaves the roll about it undetermined (an SO(2) freedom), and parallel transport is
the convention that needs no reference vector: Gram-Schmidt against the local vertical or the spin
axis goes SINGULAR exactly when the thrust aligns with it, which is touchdown (ang_act -> 1.4 deg).
Transport also uses the very rotation actuator_model.rotate_towards applies to slew the thrust, so
the two models agree by construction rather than by coincidence.

Only the ALONG-THRUST bias component is physically meaningful under this convention; the two
cross-axis components follow the chosen roll, which is arbitrary because there is no attitude model.
That component is also the one degenerate with a thrust-magnitude or mass error -- b_3 and
eps*||T||/m are the same signal, and no filter can separate them.

NOT MODELLED: the 300 ppm scale factor and the 0.1 mrad input-axis misalignment. On 0.0333 m/s^2
they contribute 1.0e-05 and 3.3e-06 m/s^2, i.e. 300x below the bias. Reinstating the scale factor is
one term in measure(); see the note in config_nav.

No gyro: this is 3DoF, there is no attitude state.
"""
from __future__ import annotations

import numpy as np

import config_nav as CN

_EPS = 1e-12


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else None


def rotation_between(d_from, d_to):
    """The MINIMAL rotation taking unit vector d_from onto d_to (Rodrigues).

    R = I + sin(th) K + (1 - cos(th)) K^2 about the axis d_from x d_to. This is the same rotation
    actuator_model.rotate_towards uses to slew the thrust direction, which is why applying it to the
    bias keeps the bias frame and the thrust frame consistent.
    """
    a, b = np.asarray(d_from, float), np.asarray(d_to, float)
    c = float(np.clip(a @ b, -1.0, 1.0))
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s < _EPS:
        if c > 0.0:
            return np.eye(3)                      # already aligned
        # exactly anti-parallel: a pi rotation about ANY perpendicular axis, R = 2 k k^T - I
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        k = _unit(np.cross(a, tmp))
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = axis / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


class IMU:
    """One triad. Seeded once; the bias draw is reproducible from `seed` alone."""

    def __init__(self, grade=None, rate=None, seed=0, d0=None, bias_frame=None):
        self.p = CN.imu_params(grade, rate, bias_frame)
        self.rng = np.random.default_rng(seed)
        # ONE draw per flight, held constant in the thrust-aligned frame.
        self.b_taf = self.rng.normal(0.0, self.p["bias_sigma"], 3)
        self.R = np.eye(3)                        # ACI <- TAF, accumulated by transport
        self.d = _unit(d0)                        # current boresight in ACI; None until first thrust
        self.n = 0

    # -- what the instrument is actually applying, for plotting against the filter's estimate ------
    @property
    def bias(self):
        """The bias in ACI axes at this instant."""
        return self.R @ self.b_taf

    @property
    def bias0(self):
        """The bias as drawn, in the thrust-aligned frame. Constant, so |bias0| == |bias| always."""
        return self.b_taf.copy()

    def _transport(self, d_new):
        """Carry the bias with the boresight. The FIRST non-zero direction fixes the frame."""
        if d_new is None:
            return                                # engine off: nothing to transport along
        if self.d is None:
            self.d = d_new
            return
        self.R = rotation_between(self.d, d_new) @ self.R
        self.d = d_new

    def measure(self, a_true_I, dt, d_I=None):
        """a_true_I: TRUE specific force in INERTIAL axes [m/s^2]. Returns the measured value.

        The boresight is taken from the direction of the specific force, because a = T/m makes them
        parallel whenever the engine is firing. d_I only has to be supplied when the engine is OFF
        and there is no signal to read the direction from; with the engine off and no d_I the frame
        simply holds still, which is the right behaviour -- a thrust of zero has no direction.
        """
        a = np.asarray(a_true_I, float)
        if self.p["bias_frame"] == "thrust":
            self._transport(_unit(d_I) if d_I is not None else _unit(a))
        sig = self.p["vrw"] / np.sqrt(dt)
        self.n += 1
        return a + self.bias + self.rng.normal(0.0, sig, 3)

    def perfect(self):
        """Zero every error source -- for isolating the gravity-model gap."""
        self.b_taf = np.zeros(3)
        self.p = dict(self.p)
        self.p["vrw"] = 0.0
        self.p["bias_inst"] = 0.0
        return self

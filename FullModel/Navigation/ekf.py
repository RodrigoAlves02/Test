"""EKF for the descent. Dimensional units (m, m/s, m/s^2), INERTIAL frame.

STATE  x = [r(3), v(3), b(3)]   all in ACI

WHY INERTIAL. The accelerometer senses in inertial axes -- Coriolis and centrifugal are frame
artifacts, not forces on a proof mass -- so a filter in ACI takes the IMU measurement with no
rotation at all, and the two fictitious-force terms disappear from the propagation entirely.
The price is that GRAVITY has to be rotated, because the field is defined body-fixed:

    g_I(r, t)   = Rz(w t) g_body( Rz(-w t) r )
    dg_I/dr     = Rz(w t) G_body Rz(-w t)

which is a similarity transform of the gradient tensor, exact and cheap. Eros turns 39.8 deg over
this 2098 s descent, so the rotation is not a detail that can be dropped.

WHAT EACH SENSOR HANDS IN, and in which frame:
    IMU     specific force, ACI          -- native, no conversion
    camera  position fix, rotated to ACI by the camera block (Camera.measure with t)
    lidar   slant range, a SCALAR and therefore frame-free; only its H is rotated

THE BIAS IS A RANDOM CONSTANT. The LN-200S datasheet gives bias REPEATABILITY -- one draw per
power-up -- and no in-run stability, so tau = inf, b_dot = 0 and Q_bb = 0. The covariance block on
b can then only shrink, which is correct for a constant and is also the less forgiving assumption:
if the real bias drifted, this filter would not notice.

Gravity is whatever field the caller passes. For flight that is PINN-GM-III, deliberately not the
truth's 49 152-facet polyhedral field, so the gravity-model gap is measured rather than assumed
away. The SHAPE model, used by the lidar's predicted range, is the 49k mesh in both -- that is a
statement about mission sequence, not a shortcut: the shape model is a product of the survey phase
that precedes the descent, exactly as it was for OSIRIS-REx and Hayabusa2.
"""
from __future__ import annotations

import numpy as np

import config_nav as CN
import frames as NFR
from imu import rotation_between, _unit


class EKF:
    def __init__(self, x0, P0, field, imu_params, wz=None, bias_frame=None, use_grad=None,
                 d0_I=None, gate_nis=None, dmc_sigma=None, dmc_frame=None):
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.field = field
        self.p = imu_params
        self.wz = CN.w_body_z() if wz is None else float(wz)
        self.bias_frame = bias_frame or CN.EKF_BIAS_FRAME
        self.use_grad = CN.EKF_USE_GRAD if use_grad is None else use_grad
        # B carries the bias from the frame it is CONSTANT in into ACI. Identity while the model
        # says the bias is inertially fixed; a transported rotation while it rides the boresight.
        self.B = np.eye(3)
        self.d = _unit(d0_I)
        self.n = 0
        self.n_upd = 0
        self._t_last = 0.0
        # Chi-square innovation gate, PER DEGREE OF FREEDOM, so one threshold serves the scalar
        # lidar range and the 3-vector camera fix alike. None disables it.
        self.gate_nis = CN.EKF_GATE_NIS if gate_nis is None else gate_nis
        self.dmc_sigma = CN.EKF_DMC_SIGMA if dmc_sigma is None else float(dmc_sigma)
        self.dmc_frame = dmc_frame or CN.EKF_DMC_FRAME
        self.n_rej = 0
        self.last_rej = None

    # -- gravity, rotated out of the body frame and back ------------------------------------------
    def g_inertial(self, r_I, t):
        Rz = NFR.Rz(self.wz * t)
        r_b = Rz.T @ np.asarray(r_I, float)
        return Rz @ self.field.g(r_b)

    def G_inertial(self, r_I, t):
        if not self.use_grad:
            return np.zeros((3, 3))
        Rz = NFR.Rz(self.wz * t)
        return Rz @ self.field.grad(Rz.T @ np.asarray(r_I, float)) @ Rz.T

    # -- the bias frame ---------------------------------------------------------------------------
    def set_boresight(self, d_I):
        """Carry B with the thrust axis, by the same minimal rotation the IMU device applies.

        The filter does not see the delivered thrust, so the caller passes the direction it
        COMMANDED. The two differ by the actuator lag and slew limit, which is a real and small
        mismatch and is left in rather than papered over. With nothing passed, propagate() falls
        back to the debiased measurement direction, which is what the filter can observe on its own.
        """
        if self.bias_frame != "thrust":
            return
        u = _unit(d_I)
        if u is None:
            return
        if self.d is None:
            self.d = u
            return
        self.B = rotation_between(self.d, u) @ self.B
        self.d = u

    # -- dynamics ----------------------------------------------------------------------------------
    def _dmc_rot(self, t):
        """ACAF -> ACI for the DMC states. Identity when they are modelled as inertially fixed."""
        return NFR.Rz(self.wz * t) if self.dmc_frame == "body" else np.eye(3)

    def _f(self, x, a_meas_I, t):
        v, b = x[CN.IV], x[CN.IB]
        a = np.asarray(a_meas_I, float) - self.B @ b + self._dmc_rot(t) @ x[CN.IA]
        return np.concatenate([v, a + self.g_inertial(x[CN.IR], t), np.zeros(3), np.zeros(3)])

    def _F(self, x, t):
        F = np.zeros((CN.NX, CN.NX))
        F[CN.IR, CN.IV] = np.eye(3)
        F[CN.IV, CN.IR] = self.G_inertial(x[CN.IR], t)
        F[CN.IV, CN.IB] = -self.B
        F[CN.IV, CN.IA] = self._dmc_rot(t)
        # F[b, b] = F[a_u, a_u] = 0: both are constants in their own frame, driven only by Q
        return F

    def _Q(self, dt):
        Q = np.zeros((CN.NX, CN.NX))
        # the velocity-increment variance of a white acceleration with PSD N^2, which is exactly
        # the convention IMU.measure uses when it draws at N/sqrt(dt)
        Q[CN.IV, CN.IV] = np.eye(3) * self.p["vrw"] ** 2 * dt
        beta = np.exp(-dt / self.p["tau"])
        Q[CN.IB, CN.IB] = np.eye(3) * self.p["bias_inst"] ** 2 * max(1.0 - beta ** 2, 0.0)
        # the random walk that lets the DMC states track a model error that CHANGES with altitude
        Q[CN.IA, CN.IA] = np.eye(3) * self.dmc_sigma ** 2 * dt
        return Q

    def propagate(self, a_meas_I, dt, t, d_I=None):
        """One IMU step. `t` is the time at the START of the step; gravity is rotated with it."""
        if self.bias_frame == "thrust":
            if d_I is not None:
                self.set_boresight(d_I)
            else:
                self.set_boresight(_unit(np.asarray(a_meas_I, float) - self.B @ self.x[CN.IB]))
        k1 = self._f(self.x, a_meas_I, t)
        k2 = self._f(self.x + 0.5 * dt * k1, a_meas_I, t + 0.5 * dt)
        k3 = self._f(self.x + 0.5 * dt * k2, a_meas_I, t + 0.5 * dt)
        k4 = self._f(self.x + dt * k3, a_meas_I, t + dt)
        self.x = self.x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self._t_last = t + dt
        F = self._F(self.x, t + dt)
        Phi = np.eye(CN.NX) + F * dt + 0.5 * (F @ F) * dt * dt
        self.P = Phi @ self.P @ Phi.T + self._Q(dt)
        self.P = 0.5 * (self.P + self.P.T)
        self.n += 1
        return self.x

    # -- measurement update -------------------------------------------------------------------------
    def update(self, z, H, R, h_pred):
        """One measurement update, Joseph form, with a chi-square innovation gate.

        Joseph rather than the short (I - KH)P because that form is only symmetric-positive-definite
        when K is exactly optimal; with a linearised H and thousands of updates, round-off
        accumulates and the short form can lose positive-definiteness. Joseph stays PSD for ANY
        gain, which also leaves room for a deliberately suboptimal one later.

        THE GATE IS NOT OPTIONAL, AND IT IS NOT A TUNING KNOB. Without it the filter accepted a
        lidar return of 11 112 m against a predicted 0.33 m, with an assumed sigma of 3 m -- a
        3700-sigma outlier -- and moved the state 414 m while its own reported sigma stayed at
        1.3 m. That return was not a sensor fault: the vehicle had flown INSIDE the shape (the
        contact test used to compare against the site's tangent plane rather than the mesh), and a
        ray cast downward from inside a 34 km body correctly exits through the far wall. The
        contact test is fixed separately; the gate is what makes the filter robust to the class of
        fault rather than to that one instance.

        The statistic is the normalised innovation squared, y' S^-1 y, which is chi-square with
        dim(z) degrees of freedom when the filter is consistent. GATE_NIS is the threshold; 16.3 is
        the 99.9th percentile for 3 dof, 10.8 for 1 dof. A rejected measurement leaves x and P
        untouched and is counted, because a filter that silently discards data is worse than one
        that accepts bad data -- at least the second kind is visible in the plots.
        """
        z = np.atleast_1d(np.asarray(z, float))
        y = z - np.atleast_1d(np.asarray(h_pred, float))
        S = H @ self.P @ H.T + R
        Sinv = np.linalg.inv(S)
        nis = float(y @ Sinv @ y)
        if self.gate_nis is not None and nis > self.gate_nis * len(y):
            self.n_rej += 1
            self.last_rej = (float(np.linalg.norm(y)), nis)
            return float(np.linalg.norm(y)), -1.0
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        A = np.eye(CN.NX) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.n_upd += 1
        return float(np.linalg.norm(y)), float(np.sqrt(np.abs(np.linalg.det(S)))) if S.size else 0.0

    # -- accessors ------------------------------------------------------------------------------------
    @property
    def dmc_inertial(self):
        """The estimated unmodelled acceleration as it acts on the propagation, in ACI."""
        return self._dmc_rot(self._t_last) @ self.x[CN.IA]

    @property
    def bias_inertial(self):
        """The bias as it acts on the propagation, in ACI -- the thing comparable to IMU.bias."""
        return self.B @ self.x[CN.IB]

    @property
    def sigma(self):
        return np.sqrt(np.clip(np.diag(self.P), 0.0, None))

    def sigma_pos(self):
        return float(np.linalg.norm(self.sigma[CN.IR]))

    def sigma_vel(self):
        return float(np.linalg.norm(self.sigma[CN.IV]))


def initial_state_and_cov(r_true, v_true, rng, pos_sigma=CN.OD_POS_SIGMA,
                          vel_sigma=CN.OD_VEL_SIGMA, bias_sigma=None, t=0.0, wz=None):
    """Draw the OD handover error and build the matching P0, in ACI.

    r_true and v_true arrive in ACAF, which is where the truth lives. At t = 0 the two frames
    coincide, so the rotation below is the identity for a nominal handover; it is written out
    anyway because a run that starts at t != 0 would otherwise be silently wrong. Note the VELOCITY
    conversion carries the frame-rate term -- v_I = Rz v_rot + w x r_I -- which position does not.

    Direction uniform on the sphere, magnitude normal: the same convention the truth dispersions
    use, so the two are comparable. The filter starts believing it is at r_true + dr while the
    vehicle is actually at r_true, and nothing in an inertial-only filter can ever discover that.
    """
    wz = CN.w_body_z() if wz is None else float(wz)
    Rz = NFR.Rz(wz * t)
    w = np.array([0.0, 0.0, wz])

    def sphere():
        u = rng.normal(size=3)
        return u / np.linalg.norm(u)

    r_I = Rz @ np.asarray(r_true, float)
    v_I = Rz @ np.asarray(v_true, float) + np.cross(w, r_I)
    dr = sphere() * rng.normal(0.0, pos_sigma)
    dv = sphere() * rng.normal(0.0, vel_sigma)
    bs = CN.imu_params()["bias_sigma"] if bias_sigma is None else bias_sigma
    # The DMC states start at zero: at handover (12 km) the PINN and the polyhedral agree to 0.17%
    # and the vertical component of their difference is ~1e-6 m/s^2, so there is nothing to
    # compensate yet. P0 is set to what the error GROWS to lower down, which is what lets the first
    # measurements below ~1 km actually move the states instead of being argued away by a tight
    # prior.
    x0 = np.concatenate([r_I + dr, v_I + dv, np.zeros(3), np.zeros(3)])
    P0 = np.diag(np.concatenate([np.full(3, pos_sigma ** 2),
                                 np.full(3, vel_sigma ** 2),
                                 np.full(3, bs ** 2),
                                 np.full(3, CN.EKF_DMC_P0 ** 2)]))
    return x0, P0, dr, dv

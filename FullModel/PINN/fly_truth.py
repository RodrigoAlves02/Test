#!/usr/bin/env python
"""Fly a control history through the TRUTH propagator and measure where the vehicle actually lands.

WHAT THIS MEASURES, AND WHY IT IS NOT THE VALIDATION ERROR. evaluate.py scores the network against
the labels channel by channel: 112 m of position, 1.68 N of thrust, 0.5 deg of direction. None of
those is the number the mission cares about. A surrogate whose POSITION output is perfect but whose
CONTROL output is slightly wrong is useless, and one whose position output is loose but whose
control is right is fine -- because the vehicle does not fly the predicted position, it flies the
predicted THRUST, and where it ends up is then decided by the real dynamics, not by the network.

So: take the commanded thrust the network produces, hand it to the engine, integrate the real
asteroid, and see where the vehicle stops. That is flyability, in metres.

THE SCVX LABEL IS FLOWN THROUGH THE SAME HARNESS, and that is the point of the comparison. The
label is not a 0 m reference: it is a plan made on a 100-node grid with PINN-GM-III gravity, and
flying it open-loop through the polyhedral truth with a real engine already misses. Anything the
network adds sits on top of that, and only the difference is attributable to the network.

THE HARNESS IS THE ONE THE TRUTH-DYNAMICS STUDY SELECTED.

  frame        inertial            (Truth Dynamics/Validation frame_test: the rotating frame makes
                                   the integrator resolve the body spin on top of the orbital motion)
  scheme       fixed-step RK8      (the 12 propagating stages of DOP853)
  step         dt = dt_act         (step_convergence_actx: eighth-order down to the round-off floor,
                                   touchdown converged to 2024.341561 s)
  gravity      polyhedral, 49 152 facets, the real shape
  engine       first-order lag tau = 20 ms, 2 deg/s slew limit, 5-20 N clip
  command      zero-order held every dt_act, as the flight loop holds it
  in-hold      the engine's own closed-form lag and turn, with a restart at the corner where the
               slew completes (--frozen reproduces closed_loop.py's convention instead, which
               freezes the near-end thrust across the hold -- wrong by up to 15 N for 0.44 s of
               every 0.5 s hold now that tau is 20 ms rather than 0.5 s)
  touchdown    height above the REAL mesh crosses zero, bracketed and bisected by re-flying the
               hold from its own start -- a fixed-step method has no event root-finder

OPEN LOOP, DELIBERATELY. The network is queried ONCE, at the handover state, and its control is
then flown blind for the whole 2045 s descent. That is the hardest possible use of it and the only
one that isolates dynamic consistency: with feedback, integration error, gravity-model error and
network error are all nulled together and the number stops saying anything about the network. It is
also the use the structural terminal constraint promises to support -- r(1) = r_site holds by
construction, so if the predicted control were dynamically consistent the vehicle would arrive.

Usage:
    fly_truth.py --run runs/full4784-ca2423087b-cdf88698 --cases 8
    fly_truth.py --run ... --treatments label          # the baseline alone
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
GNC = os.path.abspath(os.path.join(HERE, "..", "Guidance and Control"))
for _p in (HERE, SCVX, GNC):
    sys.path.insert(0, _p)

import config as C                                                      # noqa: E402
import gravity as GR                                                    # noqa: E402
from actuator_model import Actuator                                     # noqa: E402

WZ = float(C.W_BODY[2])
ALPHA = 1.0 / (C.ISP * C.G0)
R_BODY_MAX = 17_700.0        # m, above the largest body radius (17.629 km): contact is impossible
BISECT_TOL_S = 1e-9
BISECT_MAX = 40


# ------------------------------------------------------------------ the integrator
def _rk8_tableau():
    """The 12 propagating stages of DOP853, taken from SciPy rather than retyped."""
    from scipy.integrate._ivp import dop853_coefficients as _d
    n = _d.N_STAGES
    return (np.array(_d.A[:n, :n], float), np.array(_d.B[:n], float), np.array(_d.C[:n], float))


_RK8 = _rk8_tableau()


def rhs_truth(t, y, T_fn, g_body):
    """Seven-state INERTIAL truth dynamics, SI, with a time-varying thrust.

        y = [r_I(3) m, v_I(3) m/s, m kg]
        rdot = v
        vdot = R(wt) g(R(-wt) r) + T(t)/m
        mdot = -|T(t)| / (Isp g0)

    Identical to closed_loop.rhs_truth except that the thrust is a FUNCTION of t rather than a
    frozen vector, which is what lets the engine's closed-form in-hold profile be integrated
    instead of a sample of it. No Coriolis and no centrifugal term: the frame does not rotate. The
    polyhedral field wants a body-fixed position and returns a body-fixed acceleration, so each
    evaluation rotates in and back out -- two rotations sharing one sine/cosine pair.
    """
    r_I, v_I, m = y[0:3], y[3:6], y[6]
    c, s_ = np.cos(WZ * t), np.sin(WZ * t)
    r_B = np.array([c * r_I[0] + s_ * r_I[1], -s_ * r_I[0] + c * r_I[1], r_I[2]])
    gB = np.asarray(g_body(r_B), float)
    a = np.array([c * gB[0] - s_ * gB[1], s_ * gB[0] + c * gB[1], gB[2]])
    T = np.asarray(T_fn(t), float)
    Tn = float(np.linalg.norm(T))
    if Tn > 0.0:
        a = a + T / m
    return np.concatenate([v_I, a, [-ALPHA * Tn]])


def rk8_fixed(t0, t1, y0, dt, T_fn, g_body):
    """Fixed-step RK8 over [t0, t1], trimming ONLY the last step. dt means dt."""
    A, B, Cc = _RK8
    y = np.asarray(y0, float).copy()
    t, n = t0, 0
    while True:
        rem = t1 - t
        if rem <= 1e-12:
            break
        h = dt if dt < rem else rem
        k = []
        for j in range(len(B)):
            yj = y + h * sum(A[j, mm] * k[mm] for mm in range(j)) if j else y
            k.append(np.asarray(rhs_truth(t + Cc[j] * h, yj, T_fn, g_body), float))
        y = y + h * sum(B[j] * k[j] for j in range(len(B)))
        t += h
        n += 1
    return y, n


# ------------------------------------------------------------------ frame algebra
def body_to_inertial(r_B, v_B, t):
    """Body frame is rotated by +wt from inertial at time t, aligned at t = 0."""
    c, s_ = np.cos(WZ * t), np.sin(WZ * t)
    vb = np.asarray(v_B, float) + np.cross(C.W_BODY, r_B)
    R = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.asarray(r_B, float), R @ vb


def inertial_to_body(r_I, v_I, t):
    c, s_ = np.cos(WZ * t), np.sin(WZ * t)
    R = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]])
    r_B = R @ np.asarray(r_I, float)
    return r_B, R @ np.asarray(v_I, float) - np.cross(C.W_BODY, r_B)


def rot_to_inertial_vec(u, t):
    c, s_ = np.cos(WZ * t), np.sin(WZ * t)
    return np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1], u[2]])


# ------------------------------------------------------------------ the engine, in closed form
def applied_thrust_exact(t_i, h, mag_i, d_i, mag_cmd, d_cmd, tau, slew_max):
    """The thrust the engine ACTUALLY applies through one hold, as a function of time.

    Verbatim from Truth Dynamics/step_convergence.py, which is where it was derived and verified.
    Both actuator channels have closed forms over a zero-order-hold interval:

        mag(t)   = mag_cmd + (mag_i - mag_cmd) exp(-(t - t_i)/tau)
        theta(t) = min(theta_tot, slew_max (t - t_i))
        d(t)     = d_i cos(theta) + (k x d_i) sin(theta) + k (k . d_i)(1 - cos(theta))

    theta(t) is only piecewise smooth -- its derivative jumps where the turn completes -- so the
    caller is handed that corner t* and must restart the integrator there, exactly as it must at a
    thrust jump or at the surface. One extra restart per hold, no extra stages.

    Returns (T_rot(t), t*): the applied thrust in the ROTATING frame, and the corner.
    """
    c = float(np.clip(float(np.asarray(d_i) @ np.asarray(d_cmd)), -1.0, 1.0))
    th_tot = float(np.arccos(c))
    axis = np.cross(d_i, d_cmd)
    an = float(np.linalg.norm(axis))
    if th_tot <= 1e-12 or an < 1e-12 or slew_max <= 0.0:
        k_ax, th_tot = None, 0.0
    else:
        k_ax = axis / an
    t_star = t_i + (min(h, th_tot / slew_max) if th_tot > 0.0 else 0.0)

    def T_rot(t):
        dtl = max(t - t_i, 0.0)
        mag = mag_cmd + (mag_i - mag_cmd) * np.exp(-dtl / tau)
        if k_ax is None:
            return mag * d_i
        th = min(th_tot, slew_max * dtl)
        cs, sn = np.cos(th), np.sin(th)
        d = d_i * cs + np.cross(k_ax, d_i) * sn + k_ax * float(k_ax @ d_i) * (1.0 - cs)
        return mag * d

    return T_rot, t_star


# ------------------------------------------------------------------ the two command histories
def label_command(case):
    """The thrust the SCvx plan asks for at time t, in the ROTATING frame. Newtons.

    REBUILT FROM THE STORED SI LABELS, because the case files keep r, v, m, d, T and tf and not the
    non-dimensional (X, U, s) that closed_loop.plan_thrust reads. It is the same function:

      MAGNITUDE. The plan's control is a thrust ACCELERATION held across the segment, so the FORCE
      it asks for drifts with the mass inside the hold: T(t) = Gamma_k m(t), and the plan carries
      ln m linearly, so m(t) = m_k^(1-lam) m_{k+1}^lam. The stored T_k is Gamma_k m_k, hence
      T(t) = T_k (m_{k+1}/m_k)^lam. Worth 0.006 N on 20 N over a 20.66 s segment -- included
      because it is free, not because it matters.

      DIRECTION rotates continuously across the segment and is NOT interpolated: the plan's own
      kinematics are d-dot = omega x d with omega held, so the direction part-way through is d_k
      turned about omega_k, and the gauge makes omega_k perpendicular to d_k. A perpendicular axis
      is exactly the great-circle normal of (d_k, d_k+1), so the arc between the stored node
      directions IS the plan's rotation, to within the plan's own defect (< 1 mm). Interpolating
      and renormalising would take the CHORD instead and cut the corner by up to 0.66 deg.
    """
    K = int(case["K"])
    tf = float(case["tf"])
    seg = tf / (K - 1)
    T, m, d = np.asarray(case["T"], float), np.asarray(case["m"], float), np.asarray(case["d"], float)
    ax, th, rat = [], [], []
    for k in range(K - 1):
        a = np.cross(d[k], d[k + 1])
        an = float(np.linalg.norm(a))
        c = float(np.clip(d[k] @ d[k + 1], -1.0, 1.0))
        ax.append(a / an if an > 1e-14 else None)
        th.append(float(np.arccos(c)) if an > 1e-14 else 0.0)
        rat.append(float(m[k + 1] / m[k]))

    def cmd(t):
        kf = float(np.clip(t / seg, 0.0, K - 1 - 1e-12))
        k = int(np.floor(kf))
        lam = kf - k
        dk = d[k]
        if ax[k] is not None and th[k] > 1e-15:
            a = ax[k]
            cs, sn = np.cos(lam * th[k]), np.sin(lam * th[k])
            dk = dk * cs + np.cross(a, dk) * sn + a * float(a @ dk) * (1.0 - cs)
        return T[k] * (rat[k] ** lam) * dk / float(np.linalg.norm(dk))

    return cmd, tf


def pinn_command(mdl, ic, torch, tf_override=None):
    """The thrust the network asks for at time t, in the ROTATING frame. Newtons.

    OPEN LOOP: the network is queried at the HANDOVER state for every command epoch, with
    tau = t / tf. tf is the network's OWN predicted flight time by default -- it has no access to
    the label's -- and `tf_override` substitutes the label's instead, which separates an error in
    the clock from an error in the thrust profile. tau is clamped to [0, 1]; past tf the schedule
    holds the terminal command rather than extrapolating the network outside its domain.

    Also returns the SPREAD of the predicted tf over tau, which is zero for a model whose tf comes
    off the tau-free trunk and is the pathology itself for one whose does not.
    """
    r0 = torch.tensor(ic["r0"], dtype=torch.float32)[None]
    v0 = torch.tensor(ic["v0"], dtype=torch.float32)[None]
    m0 = torch.tensor([float(ic["m0"])], dtype=torch.float32)
    d0 = torch.tensor(ic["d0"], dtype=torch.float32)[None]
    with torch.no_grad():
        tg = torch.linspace(0.0, 1.0, 101)
        tfs = mdl(r0.expand(101, 3), v0.expand(101, 3), m0.expand(101),
                  d0.expand(101, 3), tg)["tf"].numpy().astype(float)
    tf0 = float(tfs[0])
    tf = float(tf_override) if tf_override is not None else tf0
    spread = dict(tf_at_tau0=tf0, tf_min=float(tfs.min()), tf_med=float(np.median(tfs)),
                  tf_max=float(tfs.max()),
                  tf_spread_frac=float((tfs.max() - tfs.min()) / max(np.median(tfs), 1e-9)))

    def cmd(t):
        tau = torch.tensor([min(max(t / tf, 0.0), 1.0)], dtype=torch.float32)
        with torch.no_grad():
            o = mdl(r0, v0, m0, d0, tau)
        return float(o["T"].item()) * o["d"][0].numpy().astype(float)

    return cmd, tf, spread


# ------------------------------------------------------------------ reading a pre-trunk0 run
def build_model(cfg, sc, state, torch, nn, Surrogate):
    """The checkpoint decides the architecture, not the current model.py.

    Runs made before the tau-free trunk was added -- full4784, n1000, w384d8 -- read head_tf and
    head_q off the TAU-DEPENDENT trunk, so their state_dict has no trunk0.* and its head_tf.weight
    is (1, width) rather than (1, width//2). The current Surrogate cannot load them. Rather than put
    a legacy branch inside model.py, the old output map is reconstructed here, in the one place that
    needs it.

    THE CONSEQUENCE IS NOT COSMETIC and is why these runs are being measured rather than trusted:
    with tf read off a trunk that tau enters, a single trajectory has a different predicted flight
    time at every point along itself -- measured on full4784, a median spread of 39.7 % (539 s) and
    up to 140 %. tgo = tf (1 - tau) is the clock every other output is parameterised against, so it
    was moving. Flying such a model open loop has to pick ONE tf; tf(tau = 0) is used, because that
    is the number an onboard user reads at the moment of the query, and the spread over tau is
    reported beside the result.
    """
    legacy = not any(k.startswith("trunk0.") for k in state)
    if not legacy:
        m = Surrogate(cfg, sc, C.TMIN, C.TMAX)
        m.load_state_dict(state)
        m.eval()
        return m, False

    class LegacySurrogate(Surrogate):
        def __init__(self):
            super().__init__(cfg, sc, C.TMIN, C.TMAX, ref=None)
            del self.trunk0                      # absent from the checkpoint
            self.trunk0 = None
            self.head_tf = nn.Linear(cfg.width, 1)
            self.head_q = nn.Linear(cfg.width, 2)

        def forward(self, r0, v0, m0, d0, tau):
            """Surrogate.forward with h0 = h: the only difference is where tf and r_f are read."""
            h = self.trunk(self.encode(r0, v0, m0, d0, tau))
            t = tau[..., None]
            tf = self.TF * torch.exp(self.head_tf(h).clamp(-4.0, 4.0))
            if self.cfg.terminal == "site":
                r_f = self.site.expand_as(r0)
            else:
                q = self.head_q(h) * self.L
                r_f = self.site + q[..., :1] * self.t1 + q[..., 1:2] * self.t2
            r_base = r0 + t * (r_f - r0)
            r = r_base + t * (1.0 - t) * self.head_r(h) * self.L
            v = v0 + t * self.head_v(h) * self.V
            m = m0[..., None] - t * torch.nn.functional.softplus(self.head_m(h)) * self.M
            d = d0 + t * self.head_d(h)
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if self.cfg.sigma_head:
                sig = self.head_T(h)
                T = self.Tmin + (self.Tmax - self.Tmin) * torch.sigmoid(self.sharpness * sig)
            else:
                sig = None
                T = self.Tmin + (self.Tmax - self.Tmin) * torch.sigmoid(self.head_T(h))
            return dict(r=r, v=v, m=m.squeeze(-1), d=d, T=T.squeeze(-1),
                        tgo=(tf.squeeze(-1) * (1.0 - tau)), tf=tf.squeeze(-1),
                        sigma=(sig.squeeze(-1) if sig is not None else None), r_f=r_f)

    m = LegacySurrogate()
    m.load_state_dict(state)
    m.eval()
    return m, True


# ------------------------------------------------------------------ the march
def fly(cmd, r0_B, v0_B, m0, site, n_hat, fld, act, dt_act, dt_prop, lag_exact, t_cap):
    """March the truth from handover to touchdown (or to t_cap), one command hold at a time."""
    edges = np.minimum(np.arange(int(np.ceil(t_cap / dt_act)) + 1) * dt_act, t_cap)
    g = fld.g

    def height(y, t):
        r_B, _ = inertial_to_body(y[0:3], y[3:6], t)
        if float(np.linalg.norm(r_B)) > R_BODY_MAX:
            return float("inf")                  # outside the largest body radius: cannot be down
        return float(GR.height_above_surface(r_B, n_hat))

    rI, vI = body_to_inertial(r0_B, v0_B, 0.0)
    y = np.concatenate([rI, vI, [float(m0)]])
    mag, d_app = act.clip_command(cmd(0.0))      # the engine is already lit at handover
    nst, t_td, y_td, landed = 0, float(edges[-1]), None, False

    def march(y_in, t0, t1, T_fn, split):
        if split is not None and t0 < split < t1:
            ya, na = rk8_fixed(t0, split, y_in, dt_prop, T_fn, g)
            yb, nb = rk8_fixed(split, t1, ya, dt_prop, T_fn, g)
            return yb, na + nb
        return rk8_fixed(t0, t1, y_in, dt_prop, T_fn, g)

    for i in range(len(edges) - 1):
        h_i = edges[i + 1] - edges[i]
        if h_i <= 0.0:
            break
        T_c = cmd(edges[i])
        mag_cmd, d_cmd = act.clip_command(T_c)
        if lag_exact:
            T_rot, split = applied_thrust_exact(edges[i], h_i, mag, d_app, mag_cmd, d_cmd,
                                                act.tau, act.slew_max)
            T_fn = lambda t, _f=T_rot: rot_to_inertial_vec(_f(t), t)
        else:
            # closed_loop.py's convention: the actuator state at the START of the hold, frozen in
            # the rotating frame and rotated into inertial once at the hold epoch.
            T_I = rot_to_inertial_vec(mag * d_app, edges[i])
            T_fn, split = (lambda t, _T=T_I: _T), None

        y_new, nn = march(y, edges[i], edges[i + 1], T_fn, split)
        nst += nn
        if height(y_new, edges[i + 1]) <= 0.0:
            lo, hi, y_hi = edges[i], edges[i + 1], y_new
            for _ in range(BISECT_MAX):
                mid = 0.5 * (lo + hi)
                y_m, _ = march(y, edges[i], mid, T_fn, split)
                if height(y_m, mid) <= 0.0:
                    hi, y_hi = mid, y_m
                else:
                    lo = mid
                if hi - lo < BISECT_TOL_S:
                    break
            t_td, y_td, landed = float(hi), y_hi, True
            break
        y = y_new
        mag, d_app, _ = act.step(mag, d_app, T_c, h_i)

    if y_td is None:
        y_td = y
    r_B, v_B = inertial_to_body(y_td[0:3], y_td[3:6], t_td)
    dr = r_B - np.asarray(site, float)
    vert = float(dr @ n_hat)
    lat = float(np.linalg.norm(dr - vert * np.asarray(n_hat, float)))
    return dict(landed=bool(landed), t_td=t_td, steps=nst,
                r_B=r_B.tolist(), v_B=v_B.tolist(), m=float(y_td[6]),
                miss_m=float(np.linalg.norm(dr)), lat_m=lat, vert_m=vert,
                alt_m=(float("nan") if landed else height(y_td, t_td)),
                speed_ms=float(np.linalg.norm(v_B)),
                v_vert_ms=float(v_B @ n_hat),
                fuel_kg=float(m0) - float(y_td[6]))


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: polish/model.pt else model.pt")
    ap.add_argument("--data-dir", dest="data_dir", default=None)
    ap.add_argument("--split", default="val", choices=("val", "test", "train"))
    ap.add_argument("--cases", type=int, default=8, help="how many trajectories to fly")
    ap.add_argument("--seeds", default=None, help="explicit comma-separated case seeds instead")
    ap.add_argument("--treatments", default="label,pinn")
    ap.add_argument("--pinn-tf", dest="pinn_tf", default="net", choices=("net", "label"),
                    help="the clock the network's control is flown on: its own predicted tf, or "
                         "the label's. 'label' separates a wrong flight time from a wrong profile")
    ap.add_argument("--dt-act", dest="dt_act", type=float, default=0.5,
                    help="command hold / actuator sub-step [s]; the flight default is 0.5")
    ap.add_argument("--dt-prop", dest="dt_prop", type=float, default=None,
                    help="truth integrator step [s]; default dt_act, one RK8 step per hold")
    ap.add_argument("--tau", type=float, default=0.02, help="thrust lag time constant [s]")
    ap.add_argument("--slew", type=float, default=C.SLEW_MAX_DEG_S, help="slew limit [deg/s]")
    ap.add_argument("--frozen", action="store_true",
                    help="freeze the applied thrust across each hold (closed_loop.py's "
                         "convention) instead of integrating the engine's exact in-hold profile")
    ap.add_argument("--t-margin", dest="t_margin", type=float, default=1.20,
                    help="fly to this multiple of the treatment's own tf before giving up")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.set_default_dtype(torch.float32)
    from runcfg import Config
    import data as D
    from model import Surrogate

    run = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    cfg = Config.load(os.path.join(run, "config.json"))
    ckpt = a.checkpoint or (os.path.join(run, "polish", "model.pt")
                            if os.path.exists(os.path.join(run, "polish", "model.pt"))
                            else os.path.join(run, "model.pt"))
    dd = a.data_dir or cfg.data_dir
    data_dir = dd if os.path.isabs(dd) else os.path.join(HERE, dd)
    treat = [t.strip() for t in a.treatments.split(",") if t.strip()]
    dt_prop = a.dt_prop if a.dt_prop is not None else a.dt_act

    parents = D.load_parents(data_dir, defect_max=cfg.defect_max, verbose=False)
    by_seed = {int(p["seed"]): p for p in parents}
    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",")]
    else:
        pool = sorted(s for s in cfg.split[a.split] if s in by_seed)
        # evenly spaced across the split, so the sample is not the low-seed end of it
        idx = np.linspace(0, len(pool) - 1, min(a.cases, len(pool))).round().astype(int)
        seeds = [pool[i] for i in sorted(set(idx.tolist()))]

    sc = D.Scaling(**cfg.scaling)
    mdl, legacy = None, False
    if "pinn" in treat:
        mdl, legacy = build_model(cfg, sc, torch.load(ckpt, map_location="cpu"),
                                  torch, nn, Surrogate)

    fld = GR.field()
    act = Actuator(tau=a.tau, slew_max_deg_s=a.slew, t_min=C.TMIN, t_max=C.TMAX)
    site = np.asarray(sc.r_site, float)
    n_hat = np.asarray(sc.n_hat, float)

    print(f"run        {os.path.basename(run)}   checkpoint {os.path.relpath(ckpt, run)}")
    print(f"data       {data_dir}   {len(parents)} parents, split '{a.split}'")
    print(f"cases      {len(seeds)}: " + ", ".join(str(s) for s in seeds))
    print(f"truth      inertial, RK8, dt {dt_prop:g} s, polyhedral {fld.n_facets:,} facets")
    print(f"engine     tau {a.tau:g} s, slew {a.slew:g} deg/s, clip {C.TMIN:g}-{C.TMAX:g} N, "
          f"command held every {a.dt_act:g} s")
    print(f"in-hold    {'FROZEN at the hold epoch' if a.frozen else 'EXACT (closed-form lag and turn, restart at the corner)'}")
    print(f"site       {site} m,  n_hat {n_hat}")
    print(f"treatments {', '.join(treat)}   mode OPEN LOOP"
          + (f"   pinn clock: {a.pinn_tf}" if "pinn" in treat else ""))
    if legacy:
        print("ARCHITECTURE  pre-trunk0 checkpoint: tf and the terminal point are read off the "
              "TAU-DEPENDENT\n              trunk, so tf varies along a single trajectory. "
              "tf(tau = 0) is flown; the spread is reported per case.")
    print()

    rows = []
    for s in seeds:
        p = by_seed[s]
        ic = dict(r0=np.asarray(p["r0"], float), v0=np.asarray(p["v0"], float),
                  m0=float(p["m0"]), d0=np.asarray(p["d0"], float))
        rec = dict(seed=int(s), tf_label=float(p["tf"]), fuel_label=float(p["fuel"]),
                   D_label=float(p["D"]), miss_label_plan=float(p["miss_m"]),
                   range0_m=float(np.linalg.norm(ic["r0"] - site)))
        print(f"case {s:>5d}   tf_label {rec['tf_label']:8.2f} s   range {rec['range0_m']/1e3:6.3f} km"
              f"   plan miss {rec['miss_label_plan']:.4f} m", flush=True)
        for name in treat:
            if name == "label":
                cmd, tf = label_command(p)
            elif name == "pinn":
                cmd, tf, spread = pinn_command(
                    mdl, ic, torch,
                    tf_override=(rec["tf_label"] if a.pinn_tf == "label" else None))
                rec["pinn_tf_spread"] = spread
                print(f"   pinn   tf(tau=0) {spread['tf_at_tau0']:8.2f} s   "
                      f"tf over tau: min {spread['tf_min']:8.2f}  med {spread['tf_med']:8.2f}  "
                      f"max {spread['tf_max']:8.2f}  spread {100*spread['tf_spread_frac']:6.1f} %",
                      flush=True)
            else:
                raise SystemExit(f"unknown treatment: {name}")
            fld.n_calls = 0
            w0 = time.perf_counter()
            r = fly(cmd, ic["r0"], ic["v0"], ic["m0"], site, n_hat, fld, act,
                    a.dt_act, dt_prop, not a.frozen, t_cap=a.t_margin * tf)
            r.update(treatment=name, tf_cmd=float(tf), gravity_calls=int(fld.n_calls),
                     wall_s=time.perf_counter() - w0)
            rec[name] = r
            tag = "touchdown" if r["landed"] else "NO CONTACT"
            print(f"   {name:<6} tf_cmd {tf:8.2f} s   {tag} at {r['t_td']:9.3f} s   "
                  f"miss {r['miss_m']:10.3f} m  (lat {r['lat_m']:9.3f}, vert {r['vert_m']:+9.3f})   "
                  f"|v| {r['speed_ms']:7.4f} m/s   fuel {r['fuel_kg']:7.4f} kg   "
                  f"[{r['gravity_calls']:,} g, {r['wall_s']:.0f} s]", flush=True)
            if not r["landed"]:
                print(f"          altitude at t_cap {r['alt_m']:.1f} m", flush=True)
        rows.append(rec)

    print("\n" + "=" * 100)
    for name in treat:
        mi = np.array([r[name]["miss_m"] for r in rows if name in r])
        sp = np.array([r[name]["speed_ms"] for r in rows if name in r])
        fu = np.array([r[name]["fuel_kg"] for r in rows if name in r])
        nl = sum(1 for r in rows if name in r and not r[name]["landed"])
        print(f"{name:<6}  flown miss  median {np.median(mi):10.3f} m   p95 {np.percentile(mi,95):10.3f} m"
              f"   max {mi.max():10.3f} m")
        print(f"        contact speed median {np.median(sp):8.4f} m/s  (VTOUCH {C.VTOUCH:g})"
              f"   fuel median {np.median(fu):7.4f} kg"
              + (f"   NO CONTACT in {nl}/{len(mi)}" if nl else ""))
    if len(treat) == 2 and all(t in ("label", "pinn") for t in treat):
        dm = np.array([r["pinn"]["miss_m"] - r["label"]["miss_m"] for r in rows])
        print(f"\npinn - label  flown miss  median {np.median(dm):+10.3f} m   "
              f"min {dm.min():+.3f}   max {dm.max():+.3f} m")

    out = a.out or os.path.join(run, "fly_truth.json")
    json.dump(dict(run=os.path.basename(run), checkpoint=os.path.relpath(ckpt, run),
                   split=a.split, seeds=seeds, treatments=treat, mode="open_loop",
                   dt_act=a.dt_act, dt_prop=dt_prop, tau=a.tau, slew=a.slew,
                   pinn_tf=a.pinn_tf, legacy_architecture=bool(legacy),
                   lag_exact=not a.frozen, t_margin=a.t_margin,
                   site=site.tolist(), n_hat=n_hat.tolist(), rows=rows),
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

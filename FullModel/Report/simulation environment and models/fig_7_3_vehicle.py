#!/usr/bin/env python
"""Figure for section 7.3 -- the vehicle and its actuator.

(a) WHY THE ENGINE CANNOT HOVER. The minimum thrust the engine can deliver produces more
    acceleration than the local effective gravity at every altitude on the descent, so at minimum
    throttle pointed at the vertical the vehicle is pushed upward and cannot come down.

(b, c) THE ACTUATOR IN ACTION. The plan is a sequence of zero-order-held commands; the engine
    answers them through a first-order lag in magnitude and a rate limit in direction. Both are
    driven here with the nominal plan at the actuator sub-step, so what is plotted is the command
    the guidance issues against the thrust the vehicle actually applies.
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TH = "/Users/rodrigoalves/Desktop/Thesis"
SCVX = os.path.join(TH, "FullModel", "SCvx")
sys.path.insert(0, SCVX)
sys.path.insert(0, os.path.join(TH, "FullModel", "Guidance and Control"))
HERE = os.path.dirname(os.path.abspath(__file__))
import config as C, gravity as GR, problem_setup as PS
from closed_loop import plan_thrust
from actuator_model import Actuator

z = np.load(os.path.join(SCVX, "Figures&data", "pinn", "nominal.npz"), allow_pickle=True)
X, U, s = z["X"], z["U"], float(z["s"])
sc = PS.build(r0_override=z["r0"], v0_override=z["v0"], m_wet=float(z["m0"]))["scales"]
tf = sc.t_dim(s)
r_site, n_hat = np.asarray(z["r_site"], float), np.asarray(z["n_hat"], float)

# ---------------------------------------------------------------- (a) hover check
fld = GR.field()
h = np.linspace(0.0, C.ALTITUDE_KM * 1e3, 140)
geff = np.empty_like(h)
for i, hi in enumerate(h):
    p = r_site + hi * n_hat
    g = np.asarray(fld.g(p), float) - np.cross(C.W_BODY, np.cross(C.W_BODY, p))
    geff[i] = np.linalg.norm(g)

# ---------------------------------------------------------------- (b,c) drive the actuator
DT = 0.5                                        # actuator sub-step
act = Actuator(tau=0.5, slew_max_deg_s=C.SLEW_MAX_DEG_S, t_min=C.TMIN, t_max=C.TMAX)
n = int(np.floor(tf / DT))
t = np.arange(n) * DT
cmd_mag = np.empty(n); out_mag = np.empty(n); rate = np.empty(n); mis = np.empty(n)
T0, _ = plan_thrust(X, U, s, 0.0, sc)
mag, d = act.clip_command(T0)
for i in range(n):
    T_cmd, _ = plan_thrust(X, U, s, sc.t_nd(t[i]), sc)
    cmd_mag[i] = np.linalg.norm(T_cmd)
    _, d_cmd = act.clip_command(T_cmd)
    # STEP FIRST, THEN RECORD, so the delivered value is the one the command produced. Recording
    # the pre-step state against the current command reports the lag as a full-step error.
    mag, d, w = act.step(mag, d, T_cmd, DT)
    out_mag[i] = mag
    rate[i] = np.degrees(w)
    mis[i] = np.degrees(np.arccos(np.clip(d @ d_cmd, -1, 1)))

k = int(np.argmax(np.abs(np.diff(cmd_mag)))) + 1      # the terminal step, floor to ceiling
w0, w1 = max(0, k - 10), min(n, k + 40)
err = np.abs(out_mag - cmd_mag)
print(f"command steps {cmd_mag[k-1]:.2f} -> {cmd_mag[k]:.2f} N at t = {t[k]:.1f} s", flush=True)
print(f"thrust tracking error: peak {err.max():.2f} N  median {np.median(err):.4f} N", flush=True)
print(f"slew rate demanded: peak {rate.max():.2f} deg/s against a {C.SLEW_MAX_DEG_S:g} deg/s limit"
      f"  ({100*rate.max()/C.SLEW_MAX_DEG_S:.0f} % of it)", flush=True)
print(f"pointing misalignment: peak {mis.max():.2f} deg", flush=True)

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})


def save(fig, stem):
    # Fixed margins rather than a tight box, so every panel in the chapter is the same size.
    fig.subplots_adjust(left=0.215, right=0.965, bottom=0.185, top=0.965)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"{stem}.{ext}"), dpi=300)
    print("wrote", stem, flush=True)


# ---- the engine cannot hover
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(h / 1e3, np.full_like(h, C.TMAX / C.MWET), color="#1d3557", lw=1.3,
        label=r"$T_{\max}/m_0$")
ax.plot(h / 1e3, np.full_like(h, C.TMIN / C.MWET), color="#d94801", lw=1.5,
        label=r"$T_{\min}/m_0$")
ax.plot(h / 1e3, geff, color="0.25", lw=1.3, ls="--", label=r"$\|\mathbf{g}_{\rm eff}\|$")
ax.fill_between(h / 1e3, geff, C.TMIN / C.MWET, color="#d94801", alpha=0.10, lw=0)
ax.set_yscale("log")
ax.set_xlabel("altitude  [km]")
ax.set_ylabel(r"acceleration  [m s$^{-2}$]")
ax.set_xlim(0, C.ALTITUDE_KM)
ax.grid(alpha=.25, lw=.5)
ax.legend(frameon=False, loc="lower left", fontsize=7.5)
save(fig, "fig_hover"); plt.close(fig)

# ---- magnitude lag at the terminal step
# THE DELIVERED VALUE IS PLOTTED AT THE END OF ITS INTERVAL. act.step advances from t to t+dt, so
# out_mag[i] is the thrust at t[i]+DT; plotting it at t[i] makes the engine appear to respond
# before it was told to.
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.step(np.append(t[w0:w1], t[w1 - 1] + DT), np.append(cmd_mag[w0:w1], cmd_mag[w1 - 1]),
        where="post", color="0.35", lw=1.2, ls="--", label="commanded")
ax.plot(t[w0:w1] + DT, out_mag[w0:w1], color="#1d3557", lw=1.5, label="delivered")
ax.set_xlabel("time  [s]")
ax.set_ylabel("thrust  [N]")
ax.grid(alpha=.25, lw=.5)
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
save(fig, "fig_thrust_lag"); plt.close(fig)

# ---- slew rate, as a fraction of what the actuator can deliver
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(t, 100.0 * rate / C.SLEW_MAX_DEG_S, color="#d94801", lw=1.0)
ax.set_xlabel("time  [s]")
ax.set_ylabel("slew rate demanded  [% of limit]")
ax.set_xlim(0, tf)
ax.set_ylim(0, 100)
ax.grid(alpha=.25, lw=.5)
save(fig, "fig_slew"); plt.close(fig)

#!/usr/bin/env python
"""Figures for the actuator model: thrust response and slew rate limiting.

Two figures, one per limit, each showing the guidance COMMAND against what the engine actually
DELIVERS. The command is a zero-order hold that changes at every guidance cycle, which is what the
guidance really produces.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # the actuator model lives one level up
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "scvx_landing"))

import thesis_style as ts                                            # noqa: E402
from actuator_model import Actuator, zoh, foh                        # noqa: E402

OUT = os.path.join(HERE, "Figures&data")


def command_history(dt_guid=10.0, n=9, seed=20260921):
    """A guidance-like command: magnitude stepping between the thrust bounds, direction turning."""
    rng = np.random.default_rng(seed)
    times = np.arange(n) * dt_guid
    mags = np.array([20.0, 5.0, 14.0, 20.0, 8.0, 5.0, 17.0, 11.0, 20.0])[:n]
    # directions that step by a few degrees to tens of degrees, so the rate limit binds on some
    # transitions and not on others
    angs = np.deg2rad(np.array([0.0, 8.0, 45.0, 52.0, 120.0, 126.0, 60.0, 62.0, 95.0])[:n])
    vecs = np.stack([mags * np.cos(angs), mags * np.sin(angs), np.zeros(n)], axis=1)
    return times, vecs, mags, angs


def _hold(a, times, vecs):
    return (foh(times, vecs), "first-order hold, ramped") if a.hold == "foh" \
        else (zoh(times, vecs), "zero-order hold, stepped")


def fig_thrust(a):
    """Commanded versus delivered thrust magnitude, for several response time constants."""
    times, vecs, _, _ = command_history(a.dt_guid, a.n)
    cmd, hold_lab = _hold(a, times, vecs)
    t_end = times[-1] + a.dt_guid

    ts.use()
    fig, ax = plt.subplots(figsize=ts.figsize())
    act0 = Actuator(tau=a.taus[0], slew_max_deg_s=a.slew)
    ref = act0.run(cmd, t_end, a.dt_sim, mag0=0.0)
    if a.hold == "foh":
        ax.plot(ref["t"], ref["cmag"], color="0.35", lw=2.0, label=f"commanded ({hold_lab})")
    else:
        ax.step(ref["t"], ref["cmag"], where="post", color="0.35", lw=2.0,
                label=f"commanded ({hold_lab})")

    cols = [ts.BLUE, ts.ORANGE, ts.GREEN, ts.PURPLE]
    for tau, c in zip(a.taus, cols):
        r = Actuator(tau=tau, slew_max_deg_s=a.slew).run(cmd, t_end, a.dt_sim, mag0=0.0)
        ax.plot(r["t"], r["mag"], "-", color=c, lw=1.8,
                label=rf"delivered, $\tau$ = {tau:g} s")

    for y, lab in ((act0.t_max, r"$T_{max}$"), (act0.t_min, r"$T_{min}$")):
        ax.axhline(y, color=ts.RED, ls=":", lw=1.2)
        ax.text(0.995, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="bottom" if y < 10 else "top", color=ts.RED)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust Magnitude [N]")
    ax.set_xlim(0, t_end)
    ax.set_ylim(0, act0.t_max * 1.18)
    ax.grid(True, alpha=.25)
    ax.legend(loc="lower right", ncol=2, columnspacing=1.2, handletextpad=0.5)
    # a first-order lag tracking a ramp settles at a CONSTANT offset r*tau, not a decaying
    # transient -- annotate the largest one so the mechanism is visible, not just asserted
    if a.hold == "foh":
        r_max = np.abs(np.diff(ref["cmag"])).max() / a.dt_sim
        ax.text(0.02, 0.97, f"ramp rate up to {r_max:.3f} N/s\n"
                            f"steady-state lag $r\\tau$ = {r_max*a.taus[0]:.3f} N at "
                            f"$\\tau$={a.taus[0]:g} s",
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.35")
    save(fig, f"actuator_thrust_response_{a.hold}")


def fig_slew(a):
    """Commanded versus delivered thrust DIRECTION, and the rate that produced it."""
    times, vecs, _, angs = command_history(a.dt_guid, a.n)
    cmd, hold_lab = _hold(a, times, vecs)
    t_end = times[-1] + a.dt_guid
    act = Actuator(tau=a.taus[0], slew_max_deg_s=a.slew)
    r = act.run(cmd, t_end, a.dt_sim, mag0=0.0, d0=vecs[0])

    ang = np.degrees(np.arctan2(r["dirs"][:, 1], r["dirs"][:, 0]))
    cang = np.degrees(np.arctan2(r["cdirs"][:, 1], r["cdirs"][:, 0]))
    ang, cang = np.unwrap(np.radians(ang)) * 180 / np.pi, np.unwrap(np.radians(cang)) * 180 / np.pi

    ts.use()
    fig, (ax, bx) = plt.subplots(2, 1, sharex=True, figsize=ts.figsize(ratio=0.85),
                                 gridspec_kw=dict(hspace=0.12))
    if a.hold == "foh":
        ax.plot(r["t"], cang, color="0.35", lw=2.0, label=f"commanded ({hold_lab})")
    else:
        ax.step(r["t"], cang, where="post", color="0.35", lw=2.0,
                label=f"commanded ({hold_lab})")
    ax.plot(r["t"], ang, "-", color=ts.BLUE, lw=1.8, label="delivered")
    ax.set_ylabel("Thrust Direction [deg]")
    ax.grid(True, alpha=.25)
    ax.legend(loc="upper left", ncol=2)

    bx.plot(r["t"], np.degrees(r["rate"]), "-", color=ts.ORANGE, lw=1.6,
            label="slew rate used")
    bx.axhline(a.slew, color=ts.RED, ls="-.", lw=1.4,
               label=rf"limit, {a.slew:g}$^\circ$/s")
    bx.set_xlabel("Time [s]")
    bx.set_ylabel("Slew Rate [deg/s]")
    bx.set_xlim(0, t_end)
    bx.set_ylim(0, a.slew * 1.35)
    bx.grid(True, alpha=.25)
    bx.legend(loc="upper right", ncol=2)
    save(fig, f"actuator_slew_limit_{a.hold}")


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        f = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(f, dpi=200, bbox_inches="tight")
        print("wrote", f)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="*", default=[0.5, 2.0, 5.0])
    ap.add_argument("--slew", type=float, default=2.0, help="slew limit [deg/s]")
    ap.add_argument("--dt-guid", type=float, default=10.0, help="guidance cycle [s]")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--dt-sim", type=float, default=0.01)
    ap.add_argument("--hold", choices=("zoh", "foh"), default="zoh",
                    help="how the guidance command is held between updates")
    a = ap.parse_args()
    fig_thrust(a)
    fig_slew(a)


if __name__ == "__main__":
    main()

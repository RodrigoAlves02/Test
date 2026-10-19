#!/usr/bin/env python
"""Actuator model: what the engine actually does with the thrust the guidance asks for.

Sits between the guidance output and the truth dynamics. The guidance produces a commanded thrust
vector and holds it until the next cycle; the engine cannot follow it exactly, for two independent
reasons that this block imposes separately.

MAGNITUDE - first-order lag (thrust response dynamics)

    d|T|/dt = (|T_cmd| - |T|) / tau

  With the command held constant across a guidance interval this has a closed form,

    |T|(t) = |T_cmd| + (|T_0| - |T_cmd|) exp(-t/tau)

  so the model is stepped EXACTLY over a zero-order-hold interval rather than integrated. There is
  no discretisation error in this block.

DIRECTION - rate limiter (slew constraint)

  The thrust direction can only rotate so fast. Over an interval the applied direction turns toward
  the commanded one about their common normal, by at most slew_max * dt:

    dtheta = min( angle(d, d_cmd), slew_max * dt )

  and once aligned it stops. This is a bang-bang rotation, which is what a rate limit produces: the
  actuator turns at full rate until it arrives, never faster and never part-way.

  In 3DoF there is no attitude state, so this limit is an ATTITUDE PROXY. It stands in for the
  vehicle's rotational dynamics without modelling them, which is worth stating plainly rather than
  leaving a reader to wonder where the attitude went.

The two limits are independent: the magnitude lag does not slow the slew, and the slew does not
delay the throttle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v, fallback=None):
    n = np.linalg.norm(v)
    if n < 1e-14:
        return np.array([0.0, 0.0, 1.0]) if fallback is None else np.asarray(fallback, float)
    return np.asarray(v, float) / n


def rotate_towards(d, d_cmd, max_angle):
    """Rotate unit vector d toward unit vector d_cmd by at most max_angle radians (Rodrigues)."""
    d = _unit(d)
    d_cmd = _unit(d_cmd, fallback=d)
    c = float(np.clip(d @ d_cmd, -1.0, 1.0))
    theta = np.arccos(c)
    if theta <= 1e-12 or max_angle <= 0.0:
        return d, 0.0
    step = min(theta, max_angle)
    axis = np.cross(d, d_cmd)
    an = np.linalg.norm(axis)
    if an < 1e-12:                      # exactly anti-parallel: any perpendicular axis will do
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(d @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(d, tmp)
        an = np.linalg.norm(axis)
    k = axis / an
    cs, sn = np.cos(step), np.sin(step)
    return _unit(d * cs + np.cross(k, d) * sn + k * (k @ d) * (1.0 - cs)), step


@dataclass
class Actuator:
    """Engine limits. Defaults follow the vehicle used throughout the thesis."""

    tau: float = 0.02             # thrust response time constant [s]
    #  20 ms, not the 0.5 s first assumed. The modelled engine is a 20 N monopropellant hydrazine
    #  thruster -- ArianeGroup's unit of that class gives 7.9-24.6 N at Isp 222-230 s and 3.2-10.4
    #  g/s, against the 5-20 N, 225 s and 2.27-9.06 g/s modelled here. Thrusters of this class
    #  report valve opening in 6-10 ms and chamber pressure rise in tens of milliseconds, so a
    #  half-second time constant was wrong by one to two orders of magnitude.
    slew_max_deg_s: float = 2.0   # maximum thrust-direction rotation rate [deg/s]
    t_min: float = 5.0            # minimum deliverable thrust [N]
    t_max: float = 20.0           # maximum deliverable thrust [N]

    @property
    def slew_max(self):
        return np.radians(self.slew_max_deg_s)

    def clip_command(self, T_cmd):
        """Guidance may ask for anything; the engine delivers between t_min and t_max."""
        mag = float(np.linalg.norm(T_cmd))
        if mag < 1e-14:
            return 0.0, np.array([0.0, 0.0, 1.0])
        return float(np.clip(mag, self.t_min, self.t_max)), _unit(T_cmd)

    def step(self, mag, d, T_cmd, dt):
        """Advance the applied thrust over one zero-order-hold interval.

        Returns (new magnitude, new direction, slew rate actually used [rad/s]).
        """
        mag_cmd, d_cmd = self.clip_command(T_cmd)
        mag_new = mag_cmd + (mag - mag_cmd) * np.exp(-dt / self.tau)
        d_new, turned = rotate_towards(d, d_cmd, self.slew_max * dt)
        return mag_new, d_new, turned / dt if dt > 0 else 0.0

    def run(self, command_fn, t_end, dt_sim, mag0=0.0, d0=None):
        """Fly a command history through the actuator on a fine simulation grid.

        command_fn(t) returns the commanded thrust vector, already zero-order held by the caller.
        dt_sim only sets the resolution at which the output is sampled; the magnitude update is
        exact over each sub-interval, and the slew is a rate limit, so refining dt_sim does not
        change the answer except by resolving the corners more finely.
        """
        n = int(round(t_end / dt_sim))
        t = np.linspace(0.0, t_end, n + 1)
        mag = np.zeros(n + 1)
        rate = np.zeros(n + 1)
        dirs = np.zeros((n + 1, 3))
        cmag = np.zeros(n + 1)
        cdirs = np.zeros((n + 1, 3))

        d = _unit(d0 if d0 is not None else command_fn(0.0))
        mag[0] = mag0
        dirs[0] = d
        cmag[0], cdirs[0] = self.clip_command(command_fn(0.0))

        for i in range(n):
            Tc = command_fn(t[i])
            cmag[i], cdirs[i] = self.clip_command(Tc)
            mag[i + 1], d, rate[i + 1] = self.step(mag[i], d, Tc, dt_sim)
            dirs[i + 1] = d
        cmag[n], cdirs[n] = self.clip_command(command_fn(t[n]))
        return dict(t=t, mag=mag, dirs=dirs, rate=rate, cmag=cmag, cdirs=cdirs)


def foh(times, vectors):
    """A first-order-hold command history: the guidance value interpolated LINEARLY between updates.

    This is what a plan carrying thrust as a STATE produces -- the rate is held, so the thrust
    itself ramps. It matters because of how a first-order lag answers each:

      step  of size dT : error starts at dT and decays as exp(-t/tau)   -- a large transient
      ramp  of rate r  : error settles at the CONSTANT offset r*tau     -- small and bounded

    With tau = 0.5 s, the terminal burn ramping 5 -> 20 N over ~140 s is r = 0.107 N/s, so the lag
    is 0.054 N. The same 15 N delivered as a step starts 15 N in error. Measured in the closed
    loop: peak thrust tracking error 3.71 N held, 0.18 N ramped.
    """
    times = np.asarray(times, float)
    vectors = np.asarray(vectors, float)

    def f(t):
        if t <= times[0]:
            return vectors[0]
        if t >= times[-1]:
            return vectors[-1]
        i = int(np.searchsorted(times, t, side="right") - 1)
        i = max(0, min(i, len(vectors) - 2))
        lam = (t - times[i]) / (times[i + 1] - times[i])
        return (1.0 - lam) * vectors[i] + lam * vectors[i + 1]
    return f


def zoh(times, vectors):
    """A zero-order-hold command history: the guidance value in force until the next update."""
    times = np.asarray(times, float)
    vectors = np.asarray(vectors, float)

    def f(t):
        i = int(np.searchsorted(times, t, side="right") - 1)
        return vectors[max(0, min(i, len(vectors) - 1))]
    return f

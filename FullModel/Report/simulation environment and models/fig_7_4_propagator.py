#!/usr/bin/env python
"""Figure for section 7.4 -- choosing the truth propagator step.

Each curve is the position error of the propagated trajectory against a reference computed at a
much finer step, as a function of the step taken. Fourth-order convergence holds while truncation
error dominates; below a step of about half a second accumulated round-off takes over and the
error rises again, so refining further makes the answer worse rather than better.
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TH = "/Users/rodrigoalves/Desktop/Thesis"
SRC = os.path.join(TH, "FullModel", "Truth Dynamics", "Figures&Data")
HERE = os.path.dirname(os.path.abspath(__file__))

plan = json.load(open(os.path.join(SRC, "step_convergence_rk4.json")))
act = json.load(open(os.path.join(SRC, "step_convergence_act_rk4.json")))
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})

fig, ax = plt.subplots(figsize=(3.4, 2.6))
for d, c, m, lab in ((plan, "#1d3557", "o", "commanded plan"),
                     (act, "#d94801", "s", "through the actuator")):
    dt = np.array([r["dt"] for r in d["rows"]])
    er = np.array([r["max_r"] for r in d["rows"]])
    ax.loglog(dt, er, color=c, marker=m, ms=3.5, lw=1.3, label=lab)
# the fourth-order slope the scheme should show while truncation dominates
dref = np.array([10.0, 1.0])
ax.loglog(dref, 5.4847e-05 * (dref / 10.0) ** 4, color="0.55", lw=0.9, ls=":")
ax.text(3.0, 2.5e-6, r"$\Delta t^{4}$", fontsize=8, color="0.45")
ax.axvline(0.5, color="0.75", lw=0.8, zorder=0)
ax.set_xlabel(r"integrator step  $\Delta t$  [s]")
ax.set_ylabel("position error  [m]")
ax.grid(alpha=.25, lw=.5, which="both")
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
ax.invert_xaxis()
fig.subplots_adjust(left=0.215, right=0.965, bottom=0.185, top=0.965)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "figures", f"fig_step.{ext}"), dpi=300)
print("wrote fig_step", flush=True)
for nm, d in (("plan", plan), ("actuator", act)):
    r = min(d["rows"], key=lambda x: x["max_r"])
    print(f"  {nm:9s} best dt {r['dt']:.3f} s -> {r['max_r']:.3e} m, {r['evals']:,} evaluations",
          flush=True)

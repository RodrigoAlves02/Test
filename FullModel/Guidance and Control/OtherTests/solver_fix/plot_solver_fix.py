"""The two measurements that carry the diagnosis, on one figure.

Left  -- guess invariance. Same x_init, same problem, only the seeded flight time differs. An
         optimiser must return one answer, i.e. a flat line.
Right -- why it did not. dL, the predicted improvement, against the trust-region radius, taken from
         a run of consecutive REJECTED steps so the reference (and therefore d_ref) is frozen. A
         trust-region method needs dL -> 0 as the radius shrinks.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from harness import nominal, run, summary
import gen_dataset as G

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "Figures&data")
X, U, sig, x0, rf = nominal(); m = float(np.exp(x0[6])); ctx = G.Ctx()
BEFORE = dict(eta0=1000., eta_s0=500., eta_u0=0.1,  r1t=0.36, r2t=0.90,
              cap_clamp=False)   # the ORIGINAL caps, so this really is "before"
AFTER  = dict(eta0=300.,  eta_s0=400., eta_u0=1e-3, r1t=0.0,  r2t=0.50)
FACT = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)

CACHE = os.path.join(OUT, "solver_fix.json")
if os.path.exists(CACHE) and "--refresh" not in sys.argv:
    data = json.load(open(CACHE))                      # reuse the measured runs; only redraw
else:
    data = {"seeds": [sig*f for f in FACT], "sigma_true_ref": sig}
    for tag, cfg in (("before", BEFORE), ("after", AFTER)):
        rows = []
        for f in FACT:
            o = run(x0, m, warm=(X, U, sig*f), extra=dict(cfg), iters=60, ctx=ctx)
            s_ = summary(o, "", sig_ref=sig); rows.append(s_)
            print(f"{tag:<7} seed {sig*f:7.1f} -> {s_['sigma']:7.1f}  {s_['reason']:<10}"
                  f"{s_['iters']:>3} it  fuel {s_['fuel']:.4f}", flush=True)
        data[tag] = rows; data[tag + "_cfg"] = cfg
    o = run(x0, m, warm=None, extra=dict(BEFORE), iters=40, ctx=ctx)
    best, cur = [], []
    for d in o["tr"]:
        cur = cur + [d] if not d["accepted"] else []
        if len(cur) > len(best): best = list(cur)
    data["frozen"] = [dict(it=d["it"], eta_s=d["eta_s"], eta=d["eta"], dL=d["dL"], rho=d["rho"])
                      for d in best]
    json.dump(data, open(CACHE, "w"), indent=1)

import matplotlib.ticker as mt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
sd = np.array(data["seeds"])
after_y = np.array([r["sigma"] for r in data["after"]])
best_sig = float(np.median(after_y[np.abs(after_y - np.median(after_y)) < 20]))
for tag, c, mk, lab in (("before", "#B8351F", "o", "before"), ("after", "#1B5E9C", "s", "after")):
    y = np.array([r["sigma"] for r in data[tag]])
    cf = data[tag + "_cfg"]
    a1.plot(sd, y, mk + "-", color=c, ms=6.5, lw=1.9, mfc=c, zorder=3,
            label=(f"{lab}  $\\eta_{{s0}}$={cf['eta_s0']:.0f} s, $\\eta_{{u0}}$={cf['eta_u0']:g}, "
                   f"$\\rho$ {cf['r1t']:.2f}/{cf['r2t']:.2f}   "
                   f"| spread {y.max()-y.min():.0f} s"))
a1.plot(sd, sd, ":", color="#9A9A9A", lw=1.3, zorder=1, label="returns its seed (no optimisation)")
a1.axhline(sig, color="#5A5A5A", lw=1.0, ls="--", zorder=1)
a1.axhline(best_sig, color="#1B5E9C", lw=1.0, ls="--", alpha=0.55, zorder=1)
a1.set_ylim(1400, 2260)
a1.text(sd[-1], sig - 26, f"shipped nominal, {sig:.0f} s", ha="right", va="top",
        fontsize=8, color="#5A5A5A")
a1.text(sd[0], best_sig + 12, f"better optimum the fix reaches, {best_sig:.0f} s  "
        f"(2.2% less fuel)", ha="left", va="bottom", fontsize=8, color="#1B5E9C")
a1.annotate("the one seed placed exactly on\nthe shipped nominal stays there",
            xy=(sd[3], after_y[3]), xytext=(sd[4], 1455), fontsize=7.8, color="#1B5E9C",
            ha="left", arrowprops=dict(arrowstyle="->", color="#1B5E9C", lw=0.9))
a1.set_xlabel("seeded flight time  $\\sigma_{seed}$  [s]")
a1.set_ylabel("returned flight time  $\\sigma$  [s]")
a1.set_title("Guess invariance: identical problem, seven seeds", fontsize=10.5)
a1.legend(fontsize=8, loc="upper left", framealpha=0.96)
a1.grid(alpha=0.25, lw=0.6)

fz = data["frozen"]
es = np.array([d["eta_s"] for d in fz]); dl = np.array([d["dL"] for d in fz])
a2.plot(es, dl, "o-", color="#B8351F", ms=6.5, lw=1.9, zorder=3,
        label=f"measured $\\Delta L$   ({len(fz)} consecutive rejected steps)")
a2.plot(es, dl[0]*(es/es[0])**2, "--", color="#1B5E9C", lw=1.7, zorder=2,
        label="what a trust-region method needs:  $\\Delta L \\propto \\eta_s^{2}$")
a2.set_xscale("log", base=2); a2.set_yscale("log"); a2.invert_xaxis()
a2.set_xticks(es); a2.get_xaxis().set_major_formatter(mt.FormatStrFormatter("%g"))
a2.get_xaxis().set_minor_formatter(mt.NullFormatter())
a2.set_ylim(dl.min()*0.02, dl.max()*4)
a2.set_xlabel("time trust-region radius  $\\eta_s$  [s]      (shrinking $\\rightarrow$)")
a2.set_ylabel("predicted improvement  $\\Delta L$")
a2.set_title("$\\Delta L$ does not vanish with the trust region", fontsize=10.5)
a2.annotate(f"flat to 3 d.p. ({dl[0]:.3f}) while $\\eta_s$ falls {es[0]/es[-1]:.0f}$\\times$.\n"
            f"$\\Delta L \\simeq W_{{vc}}\\,\\|d_{{ref}}\\|_1$, and the subproblem can always\n"
            f"zero its virtual control inside the region, so $\\rho$'s\n"
            f"denominator carries no step-size information.",
            xy=(es[2], dl[2]), xytext=(0.035, 0.06), textcoords="axes fraction",
            fontsize=8, color="#B8351F",
            arrowprops=dict(arrowstyle="->", color="#B8351F", lw=0.9))
a2.legend(fontsize=8, loc="upper right", framealpha=0.96)
a2.grid(alpha=0.25, lw=0.6, which="major")
fig.suptitle("SCvx trust-region defect and fix \u2014 nominal Eros descent, 11 771 m, $K$=80",
             fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"solver_fix.{ext}"), dpi=300, bbox_inches="tight")
print("\nwrote " + OUT + "/solver_fix.{json,pdf,png}")

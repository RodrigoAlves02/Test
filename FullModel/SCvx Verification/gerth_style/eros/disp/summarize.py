"""Two tables from res/*.json.
A: solver behaviour per (seed, formulation, variant).
B: consistency of the RETURNED control (ptr re-gauges omega after its test) vs the JUDGED one (pre-gauge Un):
   per-segment defects and open-loop flights (rate = omega integrated from d0; att = d re-anchored per node)."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["default", "noslew", "ratio", "fixedR", "R0.7"]
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "res", "*.json")))]
R.sort(key=lambda r: (r["seed"], r["formulation"] != "10", ORDER.index(r["variant"])))
nan = float("nan")
print("A  solver")
print(f"{'seed':>4} {'form':>4} {'variant':8} {'stage1':10} {'it1':>4} {'stage2':10} {'it2':>4} {'fuel kg':>9} {'tf s':>7} {'rej':>4} {'wall min':>8} {'|d| lo':>8}")
for r in R:
    a, b = r["stage1"], r["stage2"] or {}
    print(f"{r['seed']:>4} {r['formulation']:>4} {r['variant']:8} {a['status'][:10]:10} {a['it']:>4} {b.get('status','-')[:10]:10} "
          f"{b.get('it',0):>4} {b.get('fuel',nan):9.4f} {b.get('tf',nan):7.1f} {a['rejected']+b.get('rejected',0):>4} "
          f"{r['wall_s']/60:8.1f} {b.get('dnorm',[nan])[0]:8.5f}")
print("\nB  returned vs judged control (stage 2)")
print(f"{'seed':>4} {'form':>4} {'variant':8} | {'r def mm':>8} {'v def':>8} {'d def':>8} {'rate miss':>9} {'rate alt':>8} {'att miss':>8} {'att alt':>8} "
      f"| {'r def mm':>8} {'v def':>8} {'d def':>8} {'rate miss':>9} {'rate alt':>8} {'att miss':>8} {'att alt':>8}")
for r in R:
    b = r["stage2"] or {}
    g = lambda k: b.get(k, nan)
    print(f"{r['seed']:>4} {r['formulation']:>4} {r['variant']:8} | {g('defect_m')*1e3:8.3f} {nan:8.1e} {g('d_defect_max_rad'):8.1e} "
          f"{g('rate_miss_m'):9.3f} {g('rate_alt_m'):8.3f} {g('att_miss_m'):8.3f} {g('att_alt_m'):8.3f} "
          f"| {g('j_defect_m')*1e3:8.3f} {g('j_defect_v'):8.1e} {g('j_d_defect_max_rad'):8.1e} {g('j_rate_miss_m'):9.3f} "
          f"{g('j_rate_alt_m'):8.3f} {g('j_att_miss_m'):8.3f} {g('j_att_alt_m'):8.3f}")

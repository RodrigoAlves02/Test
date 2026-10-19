#!/usr/bin/env python
"""THE acceptance test: same problem, different seeds, one answer.

Fix x_init and the landing site -- nothing physical changes -- and seed the flight time across a
wide band. A correct solver returns one trajectory. The previous solver spanned 363 s in t_go on
exactly this test, and every pathology it had showed up here first, so this runs before fuel or
iteration count are even looked at.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, ptr, problem_setup as PS
from dynamics import NX, NU

HERE = os.path.dirname(os.path.abspath(__file__))


def main(K=60, iters=35, factors=(0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)):
    P = PS.build(); dyn, sc = P["dyn"], P["scales"]
    v0n = float(np.linalg.norm(P["x_init"][3:6])) * sc.v
    rng = float(np.linalg.norm(P["x_init"][0:3] - P["rf_nd"])) * sc.L
    t_lo, t_hi = C.tf_bounds(v0n, rng)
    print(f"tf bounds [{t_lo:.1f}, {t_hi:.1f}] s     K={K}")
    print(f"{'seed tf':>10}{'stage':>8}{'status':>12}{'it':>4}{'tf [s]':>10}"
          f"{'fuel [kg]':>11}{'miss [m]':>10}{'defect [m]':>12}{'|nu|':>10}")
    rows = []
    for fct in factors:
        s_seed = sc.t_nd(float(np.clip(t_lo * 2.0 * fct, t_lo, t_hi)))
        Xg, Ug, _ = ptr.initial_guess(dyn, P["x_init"], P["rf_nd"], P["n_hat"], sc, K,
                                      sc.t_nd(t_lo), sc.t_nd(t_hi))
        common = dict(dyn=dyn, scales=sc, x_init=P["x_init"], rf_nd=P["rf_nd"],
                      n_hat=P["n_hat"], K=K, iters=iters, verbose=False)
        r1 = ptr.run(mode="error", warm=(Xg, Ug, s_seed), **common)
        D = r1.D_nd + 1.0 / sc.L
        r2 = ptr.run(mode="fuel", D_nd=D, warm=(r1.X, r1.U, r1.s), **common)
        for nm, r in (("error", r1), ("fuel", r2)):
            print(f"{sc.t_dim(s_seed):>10.1f}{nm:>8}{r.status:>12}{r.iters:>4}{r.tf_s:>10.1f}"
                  f"{r.fuel_kg:>11.4f}{r.miss_m:>10.4f}{r.defect_m:>12.2e}{r.vc:>10.1e}", flush=True)
        rows.append(dict(seed_tf=sc.t_dim(s_seed), tf=r2.tf_s, fuel=r2.fuel_kg, miss=r2.miss_m,
                         defect=r2.defect_m, status=r2.status, iters=r2.iters))
    tf = np.array([r["tf"] for r in rows]); fu = np.array([r["fuel"] for r in rows])
    print(f"\nSEED LADDER RESULT")
    print(f"  seed tf spanned {min(r['seed_tf'] for r in rows):.0f} .. "
          f"{max(r['seed_tf'] for r in rows):.0f} s")
    print(f"  returned tf     {tf.min():.1f} .. {tf.max():.1f} s    spread {tf.max()-tf.min():.1f} s"
          f"  ({(tf.max()-tf.min())/tf.mean()*100:.2f}%)")
    print(f"  returned fuel   {fu.min():.4f} .. {fu.max():.4f} kg  spread {fu.max()-fu.min():.4f} kg"
          f"  ({(fu.max()-fu.min())/fu.mean()*100:.2f}%)")
    os.makedirs(os.path.join(HERE, "Figures&data"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "Figures&data", "seed_ladder.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--K",type=int,default=60)
    ap.add_argument("--iters",type=int,default=35)
    a=ap.parse_args(); main(K=a.K, iters=a.iters)

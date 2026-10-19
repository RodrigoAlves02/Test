#!/usr/bin/env python
"""Fly the network closed loop against the polyhedral truth. MPC, with a forward pass as the planner.

THE COMMANDS COME FROM tau > 0, NOT FROM tau = 0. d(0) = d_input is structural in the ansatz, so a
query at tau = 0 returns the direction it was handed and carries no information. What the network
produces is a TRAJECTORY, and a cycle follows it:

    once per cycle   query at tau = 0 from the current truth state   -> tgo
    each sub-step    tau_t = t / tgo, query -> T(tau_t), d(tau_t), command T*d

which is what closed_loop.py does with plan_thrust(X, U, s, dt_ahead, sc); the plan is a forward pass
instead of a stored (X, U, s).

THE ENDGAME IS SINGULAR. tau = t/tgo with tgo -> 0 means a fixed cycle eventually spans the whole
remaining trajectory and then runs past it. Below --commit-alt the loop stops re-planning and flies
the last trajectory it has, which is what the SCvx loop does for the same reason.

TRUTH IS THE POLYHEDRAL FIELD IN THE INERTIAL FRAME, the reference model, while the network plans in
the rotating frame on PINN-GM-III -- the same model error the real vehicle has. Thrust is held
constant across each actuator sub-step in the ROTATING frame, because that is the frame the actuator
points in, and rotated into the inertial frame once per sub-step.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
GNC = os.path.abspath(os.path.join(HERE, "..", "Guidance and Control"))
VALID = os.path.join(HERE, "..", "Truth Dynamics", "Validation")
# ORDER MATTERS: insert(0) means the LAST one wins, so HERE must be last or SCvx/config.py
# shadows this package's config.py -- the same collision that bit losses.Physics.
for p in (VALID, GNC, SCVX, HERE):
    sys.path.insert(0, p)

from runcfg import Config                                            # noqa: E402
import data as D                                                     # noqa: E402
from model import Surrogate                                          # noqa: E402
import losses as L                                                   # noqa: E402
from train import pick_device                                        # noqa: E402
import gravity as GR                                                 # noqa: E402
from actuator_model import Actuator                                  # noqa: E402
import truth_dynamics as TD                                          # noqa: E402


def fly(mdl, sc, phys, g, veh, act, r0, v0, m0, d0, dt_guid=20.0, dt_act=0.5,
        commit_alt=20.0, t_max=4000.0, dev="cpu", trace=None):
    """One descent. Returns the outcome and, optionally, a trace."""
    n_hat = np.asarray(sc.n_hat, float); site = np.asarray(sc.r_site, float)
    WZ = float(TD.OMEGA[2])
    rot2in = lambda vec, t: np.array([np.cos(WZ*t)*vec[0] - np.sin(WZ*t)*vec[1],
                                      np.sin(WZ*t)*vec[0] + np.cos(WZ*t)*vec[1], vec[2]])
    height = lambda y, t: float(GR.height_above_surface(
        TD.inertial_to_body(y[0:3], y[3:6], t)[0], n_hat))

    rI, vI = TD.body_to_inertial(r0, v0, 0.0)
    y = np.concatenate([rI, vI, [m0]])
    mag, d_app = act.clip_command(np.zeros(3) + d0 * 10.0)      # engine starts pointing at d0
    t = 0.0
    tgo_pred, committed = None, False
    n_query = 0

    def query(r_rot, v_rot, m, d_now, taus):
        nonlocal n_query
        n = len(taus); n_query += n
        with torch.no_grad():
            o = mdl(torch.tensor(np.repeat(r_rot[None], n, 0), dtype=torch.float32, device=dev),
                    torch.tensor(np.repeat(v_rot[None], n, 0), dtype=torch.float32, device=dev),
                    torch.tensor(np.full(n, m), dtype=torch.float32, device=dev),
                    torch.tensor(np.repeat(d_now[None], n, 0), dtype=torch.float32, device=dev),
                    torch.tensor(np.asarray(taus), dtype=torch.float32, device=dev))
        return o["T"].cpu().numpy(), o["d"].cpu().numpy(), o["tgo"].cpu().numpy()

    while t < t_max:
        rb, vb = TD.inertial_to_body(y[0:3], y[3:6], t)
        h = float(GR.height_above_surface(rb, n_hat))
        if h <= commit_alt and not committed:
            committed = True                       # stop re-planning; fly what we have
        if not committed or tgo_pred is None:
            _, _, tg = query(rb, vb, y[6], d_app, [0.0])
            tgo_pred = float(tg[0])
            cycle_t0 = t
        # one guidance cycle, sampled at the actuator rate
        n_sub = max(1, int(round(dt_guid / dt_act)))
        for i in range(n_sub):
            if t >= t_max:
                break
            tau = np.clip((t - cycle_t0 + (0.0 if committed else 0.0)) / max(tgo_pred, 1e-6),
                          0.0, 1.0 - 1e-6) if not committed else \
                  np.clip((t - cycle_t0) / max(tgo_pred, 1e-6), 0.0, 1.0 - 1e-6)
            Tq, dq, _ = query(rb, vb, y[6], d_app, [tau])
            T_cmd_rot = float(Tq[0]) * dq[0]
            T_I = rot2in(mag * d_app, t)                     # applied = actuator state, held
            y_new, _ = TD.propagate_fixed(y, t, t + dt_act, dt_act, "rk4", g,
                                          lambda _t, _T=T_I: _T, veh, frame="inertial")
            if height(y_new, t + dt_act) <= 0.0:
                lo, hi, y_hi = t, t + dt_act, y_new
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    y_m, _ = TD.propagate_fixed(y, t, mid, dt_act, "rk4", g,
                                                lambda _t, _T=T_I: _T, veh, frame="inertial")
                    if height(y_m, mid) <= 0.0:
                        hi, y_hi = mid, y_m
                    else:
                        lo = mid
                    if hi - lo < 1e-9:
                        break
                y, t = y_hi, hi
                rb, vb = TD.inertial_to_body(y[0:3], y[3:6], t)
                return dict(landed=True, t=t, miss_m=float(np.linalg.norm(
                                (rb - site) - ((rb - site) @ n_hat) * n_hat)),
                            h_m=height(y, t), speed=float(np.linalg.norm(vb)),
                            vert=float(abs(vb @ n_hat)), m_end=float(y[6]),
                            fuel=float(m0 - y[6]), n_query=n_query)
            mag, d_app, _ = act.step(mag, d_app, T_cmd_rot, dt_act)
            y, t = y_new, t + dt_act
            rb, vb = TD.inertial_to_body(y[0:3], y[3:6], t)
            if trace is not None:
                trace.append(dict(t=t, h=height(y, t), m=float(y[6]), T=float(mag),
                                  speed=float(np.linalg.norm(vb))))
    return dict(landed=False, t=t, miss_m=float("nan"), h_m=height(y, t),
                speed=float(np.linalg.norm(vb)), m_end=float(y[6]),
                fuel=float(m0 - y[6]), n_query=n_query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--nominal", required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--dt-guid", dest="dt_guid", type=float, default=20.0)
    ap.add_argument("--dt-act", dest="dt_act", type=float, default=0.5)
    ap.add_argument("--commit-alt", dest="commit_alt", type=float, default=20.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    run = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    cfg = Config.load(os.path.join(run, "config.json"))
    sc = D.Scaling(**cfg.scaling)
    dev = pick_device(a.device)
    phys = L.Physics(SCVX, device=dev)
    ck = os.path.join(run, "polish", "model.pt")
    ck = ck if os.path.exists(ck) else os.path.join(run, "model.pt")
    mdl = Surrogate(cfg, sc, phys.Tmin, phys.Tmax).to(dev)
    mdl.load_state_dict(torch.load(ck, map_location=dev)); mdl.eval()

    sys.path.insert(0, SCVX)
    import pinn_case as PC
    nom = np.load(a.nominal)
    g = TD.Gravity(kind="poly"); veh = TD.Vehicle(isp=phys.C.ISP, dry_mass=phys.C.MDRY)
    act = Actuator(tau=0.5, slew_max_deg_s=2.0, t_min=phys.Tmin, t_max=phys.Tmax)

    class A:
        pos_sigma = 1000.0; vel_sigma = 1.0; mass_sigma = 2.0; point_sigma_deg = 3.0
        seed = 0
    print(f"model {os.path.basename(run)}   cycle {a.dt_guid:g} s   actuator {a.dt_act:g} s   "
          f"commit below {a.commit_alt:g} m\n")
    print(f"{'seed':>5}{'landed':>8}{'miss [m]':>11}{'h [m]':>9}{'|v| [m/s]':>11}"
          f"{'vert':>8}{'fuel [kg]':>11}{'tf [s]':>9}{'queries':>9}{'wall':>7}")
    for s in (int(x) for x in a.seeds.split(",")):
        A.seed = s
        r0, v0, m0, d0 = PC.draw(A, nom)
        t0 = time.perf_counter(); g.reset()
        r = fly(mdl, sc, phys, g, veh, act, r0, v0, m0, d0, dt_guid=a.dt_guid,
                dt_act=a.dt_act, commit_alt=a.commit_alt, dev=dev)
        print(f"{s:>5}{str(r['landed']):>8}{r['miss_m']:>11.2f}{r['h_m']:>9.3f}"
              f"{r['speed']:>11.4f}{r.get('vert', float('nan')):>8.4f}{r['fuel']:>11.4f}"
              f"{r['t']:>9.1f}{r['n_query']:>9}{time.perf_counter()-t0:>7.0f}s")


if __name__ == "__main__":
    main()

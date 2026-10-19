#!/usr/bin/env python
"""THE battery. Identical for every run, versioned, and the thing that decides what is promoted.

WHY THE OBVIOUS METRICS ARE MISLEADING HERE, and what is done instead.

tau = 1 IS THE SAME POINT FOR EVERY TRAJECTORY. Every descent ends at the site, and r(1) = r_site is
structural in the model, so absolute position error must go to zero as tau -> 1 no matter how bad the
network is. A "terminal band" in metres therefore flatters everything, including a baseline that
ignores the initial condition entirely. Absolute error is reported, but never alone: every length is
also given RELATIVE to the range still to be flown, and every time relative to the time still to go.

BANDS ARE IN tgo, NOT tau. tau is a fraction of a flight whose length varies from 1685 to 2532 s, so
tau = 0.5 is not a comparable place across trajectories. Seconds-to-go are.

tau = 0 IS THE REAL TEST, and it is a whole section of its own. The sub-trajectory expansion makes
every node of every parent a tau = 0 input somewhere, so the tau = 0 inputs span the entire corridor
from 12 km down to metres. "Given where I am now, what is the rest of the flight" IS the question
the solver answers, and it is the only one the network has to answer in flight.

THE SAMPLING IS LOPSIDED AND NO SINGLE NUMBER IS QUOTED. The final node appears in all K-1 tails and
the first in one, so a plain mean over pairs mostly measures the endgame, which is also the easiest
regime. Everything is reported per band.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
import torch

EVAL_VERSION = "1.0.0"      # bump on ANY change to what is measured; compare.py refuses to mix

HERE = os.path.dirname(os.path.abspath(__file__))
SCVX = os.path.abspath(os.path.join(HERE, "..", "SCvx"))
sys.path.insert(0, HERE)
from runcfg import Config                                            # noqa: E402
import data as D                                                     # noqa: E402
from model import Surrogate, check_structural                        # noqa: E402
from gravity_torch import PinnGravity                                # noqa: E402
import losses as L                                                   # noqa: E402
from train import pick_device, to_t                                  # noqa: E402

TGO_BANDS = [(1500, 1e9, ">1500s"), (500, 1500, "500-1500s"),
             (60, 500, "60-500s"), (0, 60, "<60s")]


def _stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                max=float(x.max()), mean=float(x.mean()))


# ---------------------------------------------------------------- A/B: per channel, banded by tgo
def channels_by_tgo(mdl, P, dev, site, nhat):
    """Absolute error AND error relative to what is still to be flown, per tgo band."""
    with torch.no_grad():
        o = mdl(P["r0"], P["v0"], P["m0"], P["d0"], P["tau"])
    rem = (P["r"] - site).norm(dim=-1)                  # range still to go, at the LABEL point
    e = dict(
        r_m=(o["r"] - P["r"]).norm(dim=-1),
        v_ms=(o["v"] - P["v"]).norm(dim=-1),
        m_kg=(o["m"] - P["m"]).abs(),
        d_deg=torch.rad2deg(torch.arccos((o["d"] * P["d"]).sum(-1).clamp(-1, 1))),
        T_N=(o["T"] - P["T"]).abs(),
        tgo_s=(o["tgo"] - P["tgo"]).abs())
    rel = dict(
        r_frac=e["r_m"] / rem.clamp_min(1.0),           # 1 m floor: at the site "relative" is moot
        tgo_frac=e["tgo_s"] / P["tgo"].clamp_min(1.0))
    tgo = P["tgo"].cpu().numpy()
    out = {}
    for lo, hi, name in TGO_BANDS:
        sel = (tgo >= lo) & (tgo < hi)
        if not sel.any():
            continue
        out[name] = {k: _stats(v.cpu().numpy()[sel]) for k, v in {**e, **rel}.items()}
    out["ALL"] = {k: _stats(v.cpu().numpy()) for k, v in {**e, **rel}.items()}
    return out


# ---------------------------------------------------------------- C: physics residual
def residuals(mdl, phys, P, dev, n=20000, seed=0):
    """The three equations, at the NODES and at random tau -- the latter is unsupervised territory."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, P["tau"].shape[0], (min(n, P["tau"].shape[0]),), generator=g).to(dev)
    out = {}
    for tag, tau in (("at_nodes", P["tau"][i]),
                     ("random_tau", torch.rand(i.shape[0], generator=g).to(dev))):
        res = L.residual(mdl, phys, P["r0"][i], P["v0"][i], P["m0"][i], P["d0"][i], tau)
        out[tag] = dict(e_r_ms=_stats(res["e_r"].norm(dim=-1).detach().cpu().numpy()),
                        e_v_ms2=_stats(res["e_v"].norm(dim=-1).detach().cpu().numpy()),
                        e_m_kgs=_stats(res["e_m"].abs().detach().cpu().numpy()))
    return out


# ---------------------------------------------------------------- D: constraints
def constraints(mdl, phys, P, dev, site, nhat, n=20000, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    i = torch.randint(0, P["tau"].shape[0], (min(n, P["tau"].shape[0]),), generator=g).to(dev)
    tau = torch.rand(i.shape[0], generator=g).to(dev).requires_grad_(True)
    o = mdl(P["r0"][i], P["v0"][i], P["m0"][i], P["d0"][i], tau)
    rate = (L.d_dtau("d", o, tau) / o["tf"].unsqueeze(-1)).norm(dim=-1)
    dr = o["r"] - site
    up = dr @ nhat
    lat = (dr - up.unsqueeze(-1) * nhat).norm(dim=-1)
    gs = lat - np.tan(phys.gs) * up
    with torch.no_grad():
        viol = lambda x: dict(rate=float((x > 0).float().mean()), worst=float(x.max()))
        return dict(
            slew=dict(limit_deg_s=np.degrees(phys.slew_max),
                      **viol((rate - phys.slew_max).detach()),
                      max_used_deg_s=float(np.degrees(rate.max().item()))),
            glideslope=dict(limit_deg=np.degrees(phys.gs), **viol(gs.detach())),
            mass_above_dry=viol((phys.mdry - o["m"]).detach()),
            thrust_in_band=dict(min_N=float(o["T"].min()), max_N=float(o["T"].max()),
                                Tmin=phys.Tmin, Tmax=phys.Tmax),
            dnorm=dict(lo=float(o["d"].norm(dim=-1).min()), hi=float(o["d"].norm(dim=-1).max())))


# ---------------------------------------------------------------- E: switches and fuel
def switches_and_fuel(mdl, parents, sc, dev, phys, n_traj=100):
    """T error AT the switches, and fuel from integrating the predicted thrust.

    A mean thrust error says nothing about whether a switch is late; the switch is where the error
    enters the mass and velocity integrals, so it is measured where it lives.
    """
    sw_err, mid_err, fuel_rel, fuel_abs = [], [], [], []
    for p in parents[:n_traj]:
        K = int(p["K"]); tau = torch.tensor(p["tau"], dtype=torch.float32, device=dev)
        r0 = to_t(np.repeat(p["r"][0][None], K, 0), dev); v0 = to_t(np.repeat(p["v"][0][None], K, 0), dev)
        m0 = to_t(np.full(K, p["m"][0]), dev); d0 = to_t(np.repeat(p["d"][0][None], K, 0), dev)
        with torch.no_grad():
            o = mdl(r0, v0, m0, d0, tau)
        Tp = o["T"].cpu().numpy()[:K-1]; Tl = p["T"]
        jump = np.abs(np.diff(Tl)); thr = 0.25 * (Tl.max() - Tl.min() + 1e-9)
        ks = np.where(jump > thr)[0]
        near = np.zeros(K-1, bool)
        for k in ks:
            near[max(0, k-2):min(K-1, k+3)] = True
        if near.any():
            sw_err.append(np.abs(Tp - Tl)[near].mean())
        if (~near).any():
            mid_err.append(np.abs(Tp - Tl)[~near].mean())
        dt = float(p["tf"]) / (K - 1)
        f_pred = phys.alpha * Tp.sum() * dt
        f_lab = float(p["m"][0] - p["m"][-1])
        fuel_abs.append(f_pred - f_lab); fuel_rel.append(100 * (f_pred / f_lab - 1))
    return dict(n_traj=min(n_traj, len(parents)),
                T_err_at_switches_N=_stats(sw_err), T_err_elsewhere_N=_stats(mid_err),
                fuel_err_kg=_stats(fuel_abs), fuel_err_pct=_stats(fuel_rel))


# ---------------------------------------------------------------- the tau=0 section
def from_each_state(mdl, parents, sc, dev, site, n_traj=200, j_stride=7):
    """THE OPERATIONAL TEST. From every node, predict the whole remaining flight.

    This is what the network is for: not "interpolate a trajectory you were given", but "given the
    state I am in now, what is the rest of the descent". Every node of every parent is a legitimate
    starting state -- that is the Bellman argument the sub-trajectories rest on -- so the inputs here
    span the whole corridor, 12 km down to metres, not just the handover region.
    """
    rows = []
    for p in parents[:n_traj]:
        K = int(p["K"]); tf = float(p["tf"]); tau = p["tau"]
        for j in range(0, K - 1, j_stride):
            tf_j = tf * (1.0 - tau[j])
            kk = np.arange(j, K)
            tp = (tau[kk] - tau[j]) / (1.0 - tau[j])
            nq = kk.size
            with torch.no_grad():
                o = mdl(to_t(np.repeat(p["r"][j][None], nq, 0), dev),
                        to_t(np.repeat(p["v"][j][None], nq, 0), dev),
                        to_t(np.full(nq, p["m"][j]), dev),
                        to_t(np.repeat(p["d"][j][None], nq, 0), dev),
                        to_t(tp, dev))
            er = np.linalg.norm(o["r"].cpu().numpy() - p["r"][kk], axis=1)
            alt0 = float(np.linalg.norm(p["r"][j] - p["r_site"]))
            rows.append(dict(alt_m=alt0, tgo_s=tf_j,
                             err_max_m=float(er.max()), err_end_m=float(er[-1]),
                             err_max_frac=float(er.max() / max(alt0, 1.0)),
                             tf_err_s=float(abs(o["tgo"].cpu().numpy()[0] - tf_j)),
                             tf_err_frac=float(abs(o["tgo"].cpu().numpy()[0] - tf_j) / max(tf_j, 1.0))))
    out = {}
    for lo, hi, name in TGO_BANDS:
        sel = [r for r in rows if lo <= r["tgo_s"] < hi]
        if sel:
            out[name] = {k: _stats([r[k] for r in sel])
                         for k in ("err_max_m", "err_end_m", "err_max_frac", "tf_err_s", "tf_err_frac")}
    out["n_starts"] = len(rows)
    return out


# ---------------------------------------------------------------- G: baselines
def baselines(parents_tr, parents_te, dev, n_traj=100):
    """Without these, "44 m" has no referent.

    nearest-neighbour  the closest TRAINING handover state, its trajectory used verbatim. A network
                       that cannot beat this has memorised rather than generalised.
    nominal            the undispersed trajectory, flown whatever the initial condition. The floor.
    """
    X = np.array([np.concatenate([p["r"][0], p["v"][0], [p["m"][0]], p["d"][0]])
                  for p in parents_tr])
    sd = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0, 2.0, 0.05, 0.05, 0.05])
    nn_err, nom_err = [], []
    nom = parents_tr[0]
    for p in parents_te[:n_traj]:
        q = np.concatenate([p["r"][0], p["v"][0], [p["m"][0]], p["d"][0]])
        k = int(np.argmin(((X - q) / sd).__abs__().sum(1)))
        nn_err.append(float(np.linalg.norm(parents_tr[k]["r"] - p["r"], axis=1).mean()))
        nom_err.append(float(np.linalg.norm(nom["r"] - p["r"], axis=1).mean()))
    return dict(nearest_neighbour_r_m=_stats(nn_err), nominal_r_m=_stats(nom_err))


# ---------------------------------------------------------------- H: cost
def cost(mdl, dev, K=100, reps=50):
    r0 = torch.zeros(K, 3, device=dev); v0 = torch.zeros(K, 3, device=dev)
    m0 = torch.full((K,), 600.0, device=dev); d0 = torch.zeros(K, 3, device=dev); d0[:, 2] = 1.0
    tau = torch.linspace(0, 1, K, device=dev)
    with torch.no_grad():
        for _ in range(5):
            mdl(r0, v0, m0, d0, tau)
        t0 = time.perf_counter()
        for _ in range(reps):
            mdl(r0, v0, m0, d0, tau)
        dt = (time.perf_counter() - t0) / reps
    return dict(per_trajectory_s=dt, per_query_us=1e6 * dt / K,
                parameters=int(sum(p.numel() for p in mdl.parameters())),
                scvx_solve_s=200.0, speedup=200.0 / max(dt, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--checkpoint", default=None, help="default: polish/model.pt if present")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", dest="data_dir", default=None,
                    help="override config.json's data_dir; the fingerprint is checked either way")
    ap.add_argument("--n-traj", dest="n_traj", type=int, default=200)
    a = ap.parse_args()

    run = a.run if os.path.isabs(a.run) else os.path.join(HERE, a.run)
    cfg = Config.load(os.path.join(run, "config.json"))
    dev = pick_device(a.device)
    ckpt = a.checkpoint or (os.path.join(run, "polish", "model.pt")
                            if os.path.exists(os.path.join(run, "polish", "model.pt"))
                            else os.path.join(run, "model.pt"))
    dd = a.data_dir or cfg.data_dir
    data_dir = dd if os.path.isabs(dd) else os.path.join(HERE, dd)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"data_dir does not exist: {data_dir}\n"
                         f"  the run was trained on '{cfg.data_dir}'. If that directory was renamed,"
                         f" pass --data-dir explicitly.")
    parents = D.load_parents(data_dir, defect_max=cfg.defect_max, verbose=False)
    # A RUN MUST NOT BE SCORED AGAINST DATA IT DID NOT TRAIN ON. Renaming a dataset directory, or
    # regenerating one in place, silently repoints every run that referenced it -- which is exactly
    # what happened when pinn/ became pinn_cone140/. The fingerprint is over the seeds, fuel and tf,
    # so a changed label set cannot pass unnoticed.
    fp = D.fingerprint(parents)
    want = cfg.data_fingerprint or {}
    if want and fp["sha1"] != want.get("sha1"):
        print(f"  WARNING: data fingerprint differs from training.\n"
              f"    trained on sha1 {want.get('sha1')}  ({want.get('n_parents')} parents)\n"
              f"    found      sha1 {fp['sha1']}  ({fp['n_parents']} parents)")
        missing = set(cfg.split[a.split]) - {int(p["seed"]) for p in parents}
        if missing:
            raise SystemExit(f"  {len(missing)} of the run's {a.split} seeds are absent here -- "
                             f"this is not the dataset the run was trained on.")
        print("    the split's seeds are all present, continuing")
    sc = D.Scaling(**cfg.scaling)
    phys = L.Physics(SCVX, device=dev); phys.gravity = PinnGravity(phys.C.PINN_NPZ).to(dev)
    mdl = Surrogate(cfg, sc, phys.Tmin, phys.Tmax).to(dev)
    mdl.load_state_dict(torch.load(ckpt, map_location=dev)); mdl.eval()
    site, nhat = to_t(sc.r_site, dev), to_t(sc.n_hat, dev)

    want = set(cfg.split[a.split]); tr = set(cfg.split["train"])
    P_te = [p for p in parents if int(p["seed"]) in want]
    P_tr = [p for p in parents if int(p["seed"]) in tr]
    pairs = D.build(data_dir, cfg.split, a.split, j_max=cfg.j_max, parents=parents)
    P = {k: to_t(v, dev) for k, v in pairs.items() if k != "seed"}
    print(f"evaluating {os.path.basename(run)} on {a.split}: {len(P_te)} trajectories, "
          f"{P['tau'].shape[0]:,} pairs   checkpoint {os.path.basename(os.path.dirname(ckpt))}")

    M = dict(eval_version=EVAL_VERSION, run=os.path.basename(run), split=a.split,
             checkpoint=os.path.relpath(ckpt, run), n_trajectories=len(P_te),
             n_pairs=int(P["tau"].shape[0]),
             structural=check_structural(mdl, sc, device=dev))
    M["A_channels_by_tgo"] = channels_by_tgo(mdl, P, dev, site, nhat)
    M["C_residual"] = residuals(mdl, phys, P, dev)
    M["D_constraints"] = constraints(mdl, phys, P, dev, site, nhat)
    M["E_switches_fuel"] = switches_and_fuel(mdl, P_te, sc, dev, phys, n_traj=a.n_traj)
    M["F_from_each_state"] = from_each_state(mdl, P_te, sc, dev, site, n_traj=a.n_traj)
    M["G_baselines"] = baselines(P_tr, P_te, dev, n_traj=min(100, len(P_te)))
    M["H_cost"] = cost(mdl, dev)

    out = os.path.join(run, f"metrics_{a.split}.json")
    json.dump(M, open(out, "w"), indent=1)
    print(f"wrote {out}\n")
    al = M["A_channels_by_tgo"]
    print(f"{'tgo band':<12}{'r [m]':>10}{'r/range':>10}{'v [m/s]':>10}{'d [deg]':>9}"
          f"{'T [N]':>8}{'tgo [s]':>9}{'m [kg]':>8}")
    for b in [x[2] for x in TGO_BANDS] + ["ALL"]:
        if b in al:
            g = al[b]
            print(f"{b:<12}{g['r_m']['median']:>10.2f}{g['r_frac']['median']:>10.4f}"
                  f"{g['v_ms']['median']:>10.4f}{g['d_deg']['median']:>9.3f}"
                  f"{g['T_N']['median']:>8.3f}{g['tgo_s']['median']:>9.2f}"
                  f"{g['m_kg']['median']:>8.4f}")
    b = M["G_baselines"]
    print(f"\nbaselines (mean |dr| over the arc):  nearest-neighbour "
          f"{b['nearest_neighbour_r_m']['median']:.1f} m   nominal {b['nominal_r_m']['median']:.1f} m")
    print(f"cost: {M['H_cost']['per_trajectory_s']*1e3:.2f} ms per trajectory, "
          f"{M['H_cost']['speedup']:.0f}x a solve")


if __name__ == "__main__":
    main()

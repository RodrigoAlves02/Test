#!/usr/bin/env python
"""Labels -> training pairs. Parents in, sub-trajectories and a frozen scaling out.

THE SUB-TRAJECTORIES ARE DERIVED HERE, NOT STORED. The tail of an optimal trajectory is optimal for
its own initial condition -- the plan's dynamics are autonomous in the rotating frame, so the
principle of optimality transfers -- and the tail is the SAME physical arc as its parent, just
re-anchored. Storing all K of them would repeat each arc about K/2 times over; deriving them costs
an index shift.

    tail j, for k = j .. K-1:
        tau'  = (tau_k - tau_j) / (1 - tau_j),   tau_j = j/(K-1)
        tf_j  = tf * (1 - tau_j)
        tgo   = tf_j * (1 - tau')
        input = (r_j, v_j, m_j, d_j)

j runs to K-2. At j = K-1 the tail is a single point with no thrust label, and the thrust label
only exists for k <= K-2 anyway because U[:, K-1] is the ZOH dummy.

SPACING, NOT COUNT, IS WHAT MAKES A TAIL AS GOOD AS ITS PARENT. A tail has fewer nodes but the same
node spacing (tf/(K-1) seconds), so it is the same resolution over a shorter arc, not a coarser
discretisation. There is therefore no accuracy reason to cap j.

THE SAMPLING IS NOT UNIFORM IN tau, AND THAT IS A CHOICE. Counting pairs, the final node appears in
all K-1 tails while the handover node appears in one, so the endgame is weighted ~K times more
heavily than handover. On this problem that is arguably right -- the terminal corridor and pointing
constraints bite there and precision matters most -- but it is recorded as `tau_weighting` so it can
be turned off, and `flat` reweights by 1/(K-j) to undo it.

THE SPLIT IS BY PARENT SEED. Every tail of a parent is that parent's arc; splitting over pairs
would put the same trajectory in train and validation and report a score that means nothing.

THE SCALING IS FROZEN FROM THE TRAINING SPLIT AND REUSED EVERYWHERE. The solver's own scales are
PER SOLVE (L = |r0 - r_site|), so the same physical state has different coordinates depending on
which case it came from, and a tail differs again from its parent. Those must never reach the
network. One scaling, computed once, written into the run.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np


# ------------------------------------------------------------------ the frozen scaling
@dataclass
class Scaling:
    """One global non-dimensionalisation for the whole dataset. Written into every run.

    EVERY CHANNEL IS SCALED BY ITS RANGE, NOT ITS MAGNITUDE, and mass is why. Scaled by the maximum
    mass (601.69 kg) the channel's entire dynamic range is the fuel burned, ~5.6 kg, i.e. 0.0093 in
    scaled units -- so a 0.69 kg error registers as 0.00115 and the loss effectively cannot see it.
    Measured: the L-BFGS polish improved five channels by 12-39 % and made mass 42 % WORSE, because
    in those units mass accuracy was the cheapest thing to sell. Scaled by the range instead, the
    same error registers as 0.12, about 86x larger.

    The mass RESIDUAL had the same defect: dm/dt = -alpha*T is about 9.1e-3 kg/s while it was being
    divided by M/TF = 0.238 kg/s, under-weighting it ~26x. MDOT scales it by the actual rate.

    r, v and T need no such correction -- each genuinely spans its scale. Mass is the one channel
    whose offset dwarfs its variation.
    """
    r_site: list; n_hat: list
    L: float; V: float; M: float; T_N: float; TF: float
    MDOT: float = 1.0          # kg/s, for the mass residual
    M_ABS: float = 1.0         # the magnitude, kept for reference and for reading old runs

    @staticmethod
    def from_parents(parents, floor=1e-12, alpha=None, t_max=None):
        """Length from the handover range, the rest from the training split's own spread.

        Deliberately NOT per-case and NOT per-batch: a surrogate asked about a state must give the
        same answer however that state arrived, which fails the moment the coordinates depend on
        the trajectory the state was drawn from.
        """
        r = np.concatenate([p["r"] for p in parents])
        v = np.concatenate([p["v"] for p in parents])
        T = np.concatenate([p["T"] for p in parents])
        tf = np.array([p["tf"] for p in parents])
        # the RANGE of mass, i.e. the fuel actually burned, not the vehicle's mass
        burn = np.array([float(p["m"][0] - p["m"][-1]) for p in parents])
        m_abs = float(max(np.abs(np.concatenate([p["m"] for p in parents])).max(), floor))
        site, n = parents[0]["r_site"], parents[0]["n_hat"]
        a = alpha if alpha is not None else 1.0 / (225.0 * 9.80665)
        tmx = t_max if t_max is not None else float(np.abs(T).max())
        return Scaling(r_site=list(map(float, site)), n_hat=list(map(float, n)),
                       L=float(max(np.linalg.norm(r - site, axis=1).max(), floor)),
                       V=float(max(np.abs(v).max(), floor)),
                       M=float(max(burn.max(), floor)),
                       T_N=float(max(np.abs(T).max(), floor)),
                       TF=float(max(tf.max(), floor)),
                       MDOT=float(max(a * tmx, floor)),
                       M_ABS=m_abs)

    def to_json(self, path):
        json.dump(asdict(self), open(path, "w"), indent=1)

    @staticmethod
    def from_json(path):
        return Scaling(**json.load(open(path)))


# ------------------------------------------------------------------ reading the labels
PARENT_KEYS = ("tau", "r", "v", "m", "d", "T", "tf", "r0", "v0", "m0", "d0",
               "seed", "K", "D", "fuel", "miss_m", "defect_m", "r_site", "n_hat")


def load_parents(data_dir, defect_max=None, verbose=True):
    """Every accepted case, as plain dicts. Rejected cases carry a reason and are skipped.

    `defect_max` filters on the stored nonlinear defect, because NO STORED defect_m IS THE NUMBER
    ITS CASE WAS JUDGED ON. ptr tests convergence on defect(Xn, Un, sn) at ptr.py:474, the UNGAUGED
    candidate, then re-gauges the control at ptr.py:556 and recomputes the defect on the gauged
    control at ptr.py:661 -- and that recomputed value is what is written to the case file.

    The gauge is NOT physically inert, contrary to _gauge's own docstring. (omega + c*d) x d =
    omega x d holds exactly AT A NODE, but omega is held constant across the segment under ZOH
    while d rotates, so a component parallel to d at the node is no longer parallel partway
    through and does produce rotation there. Measured on the nominal, adding a component of omega
    along d and repropagating:

        +1 % of ||omega||   defect 3.45e-4 -> 3.89e-4 m
        +3 %                              -> 1.17e-3 m     already past the 1e-3 tolerance
        +10 %                             -> 3.90e-3 m

    which is exactly the scale of the outliers: 14 of 4798 land between 1.02x and 5.32x the
    tolerance. So the 1.8x in the earlier version of this comment was not a bound, and the cases
    are not marginal. Filtering here is exact and costs 0.3 % of the data. The real fix is to gauge
    BEFORE the tested defect so the number reported is the number tested; that changes solver
    behaviour and has not been made.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "case_*[0-9].npz")))
    out, dropped = [], 0
    for f in files:
        z = np.load(f)
        if defect_max is not None and float(z["defect_m"]) > defect_max:
            dropped += 1
            continue
        out.append({k: (z[k] if z[k].ndim else z[k].item()) for k in PARENT_KEYS})
    if verbose:
        print(f"{len(out)} parents from {data_dir}"
              + (f"  ({dropped} dropped on defect > {defect_max:g} m)" if defect_max else ""))
    return out


def fingerprint(parents):
    """Identifies the exact label set a run was trained on, so a result can be traced to its data."""
    h = hashlib.sha1()
    for p in sorted(parents, key=lambda q: int(q["seed"])):
        h.update(np.int64(p["seed"]).tobytes())
        h.update(np.float64(p["fuel"]).tobytes())
        h.update(np.float64(p["tf"]).tobytes())
    return dict(n_parents=len(parents), sha1=h.hexdigest(),
                seed_lo=int(min(int(p["seed"]) for p in parents)),
                seed_hi=int(max(int(p["seed"]) for p in parents)))


# ------------------------------------------------------------------ the split
def _bucket(seed, split_seed, n=1_000_000):
    """A seed's position in [0,1), from a hash of (seed, split_seed) alone.

    DELIBERATELY NOT A SHUFFLE OF THE PRESENT PARENT LIST. A shuffle depends on which cases happen
    to have finished, so val and test would silently change every time more labels arrive -- and a
    data-scaling study compares runs made at different dataset sizes, which is exactly when that
    happens. Hashing the seed itself makes a case's assignment a property of the case.
    """
    h = hashlib.sha1(f"{int(seed)}:{int(split_seed)}".encode()).digest()
    return int.from_bytes(h[:8], "big") % n / n


def split_by_parent(parents, frac_val=0.2, frac_test=0.0, seed=0, n_train=None):
    """n_train parents, then split WITHIN them. Nested in n_train, stable as the dataset grows.

    `n_parents = 1000` with `frac_val = 0.2` means 1000 trajectories are used in total, 800 to train
    on and 200 to validate against -- not 1000 training plus a separate validation set.

    THE SUBSET IS TAKEN BY SEED NUMBER. Ordering by hash and taking a prefix would be nested but not
    stable: a newly finished case can hash below an existing one and displace it, so the N=1000 set
    would change every time more labels arrive -- and a scaling study compares runs made as the
    dataset grows, which is exactly when that happens. Seeds are i.i.d. draws with no ordering
    structure, so numeric order costs nothing.

    THE TRAIN/VAL ASSIGNMENT IS BY HASH RANK WITHIN THE SUBSET, which buys exact counts at the cost
    of nesting: val(N=1000) is NOT a subset of val(N=2000), because a low-hash case from the second
    thousand can displace one from the first. So runs at different N are scored on different (though
    equally unbiased) validation sets, and a strictly like-for-like comparison has to re-score the
    larger run on the smaller run's validation seeds -- which the battery does, from the seed lists
    recorded here.
    """
    seeds = sorted(int(p["seed"]) for p in parents)
    used = seeds if n_train is None else seeds[:int(n_train)]
    # EXACT COUNTS, by hash RANK rather than a hash threshold. Thresholding at frac_val gives only
    # approximately that fraction -- 190 of 1000 rather than 200 -- which makes the split depend on
    # the hash's luck and the reported "80/20" a lie by a few percent.
    order = sorted(used, key=lambda s: _bucket(s, seed))
    n_v = int(round(frac_val * len(used)))
    n_t = int(round(frac_test * len(used)))
    val, test, train = order[:n_v], order[n_v:n_v + n_t], order[n_v + n_t:]
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    return dict(train=sorted(train), val=sorted(val), test=sorted(test),
                n_used=len(used), n_available=len(seeds))


# ------------------------------------------------------------------ pairs
def pairs_from_parent(p, j_max=None, with_thrust_only=True):
    """Every (input, tau) -> (output) pair this parent yields, in SI.

    Returns a dict of arrays, one row per pair. The thrust label is absent at the final node, so
    `with_thrust_only` drops those rows rather than inventing a value for them.
    """
    K = int(p["K"]); tau = p["tau"]; tf = float(p["tf"])
    jm = (K - 2) if j_max is None else min(int(j_max), K - 2)
    R, V, M, Dd, T = p["r"], p["v"], p["m"], p["d"], p["T"]
    cols = {k: [] for k in ("r0", "v0", "m0", "d0", "tau", "r", "v", "m", "d", "T", "tgo",
                            "tf_j", "j", "k", "seed")}
    for j in range(0, jm + 1):
        tj = tau[j]
        tf_j = tf * (1.0 - tj)
        if tf_j <= 0:
            continue
        kk = np.arange(j, K - 1 if with_thrust_only else K)
        if kk.size == 0:
            continue
        tp = (tau[kk] - tj) / (1.0 - tj)
        cols["r0"].append(np.repeat(R[j][None], kk.size, 0))
        cols["v0"].append(np.repeat(V[j][None], kk.size, 0))
        cols["m0"].append(np.full(kk.size, M[j]))
        cols["d0"].append(np.repeat(Dd[j][None], kk.size, 0))
        cols["tau"].append(tp)
        cols["r"].append(R[kk]); cols["v"].append(V[kk]); cols["m"].append(M[kk])
        cols["d"].append(Dd[kk]); cols["T"].append(T[kk])
        cols["tgo"].append(tf_j * (1.0 - tp))
        cols["tf_j"].append(np.full(kk.size, tf_j))
        cols["j"].append(np.full(kk.size, j)); cols["k"].append(kk)
        cols["seed"].append(np.full(kk.size, int(p["seed"])))
    return {k: np.concatenate(v) for k, v in cols.items()}


def build(data_dir, split_seeds, which, j_max=None, defect_max=None, parents=None):
    """All pairs for one split. Returns the arrays plus the per-row weight for `flat` sampling."""
    if parents is None:
        parents = load_parents(data_dir, defect_max=defect_max, verbose=False)
    want = set(split_seeds[which])
    sel = [p for p in parents if int(p["seed"]) in want]
    out = [pairs_from_parent(p, j_max=j_max) for p in sel]
    cat = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    K = int(sel[0]["K"])
    cat["w_flat"] = 1.0 / (K - cat["j"]).astype(float)       # undoes the endgame over-weighting
    return cat


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "../SCvx/Figures&data/pinn"
    ps = load_parents(d, defect_max=1e-3)
    fp = fingerprint(ps); print("fingerprint:", fp)
    sp = split_by_parent(ps)
    print({k: len(v) for k, v in sp.items()})
    sc = Scaling.from_parents([p for p in ps if int(p["seed"]) in set(sp["train"])])
    print("scaling:", {k: (round(v, 4) if isinstance(v, float) else "...")
                       for k, v in asdict(sc).items()})
    tr = build(d, sp, "train", parents=ps)
    print(f"train pairs: {len(tr['tau']):,}   per parent: {len(tr['tau'])/len(sp['train']):.0f}")
    print("columns:", ", ".join(f"{k}{list(v.shape[1:]) or ''}" for k, v in tr.items()))

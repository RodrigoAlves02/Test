#!/usr/bin/env python
"""Every knob of a run, in one dataclass, serialised into runs/<run_id>/config.json.

NAMED runcfg.py, NOT config.py, DELIBERATELY. SCvx has its own config.py holding the vehicle and
mission constants, and both directories end up on sys.path together. Whichever was inserted last
won, so `import config` silently returned the wrong module in whichever direction -- losses.Physics
got this package's Config instead of the solver's constants, and later SCvx/gravity.py got this
package's Config instead of MESH_TAB. Renaming removes the collision instead of fighting path order.

NOTHING THAT AFFECTS A RESULT MAY LIVE OUTSIDE THIS. A run that cannot be rebuilt from its own
config.json is a run that cannot be written up, so command-line arguments populate this object and
the object is what the trainer reads. The run_id is derived from the content, so two runs with
identical settings collide by design rather than quietly producing two entries in the log.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict, fields


@dataclass
class Config:
    # ---- identity -------------------------------------------------------------------------
    name: str = "baseline"
    notes: str = ""                      # why this run exists; shows up in compare.py

    # ---- data -----------------------------------------------------------------------------
    data_dir: str = "../SCvx/Figures&data/pinn"
    defect_max: float = 1e-3             # drop labels whose stored defect exceeds their tolerance
    j_max: int | None = None             # None = every sub-trajectory, j = 0 .. K-2
    # How many TRAINING parents to use. None = the whole pool. The subsets are nested, so a sweep
    # over this measures dataset size and nothing else -- which is the first experiment.
    n_parents: int | None = None
    frac_val: float = 0.20          # 80/20 within the n_parents actually used
    frac_test: float = 0.0
    split_seed: int = 0

    # ---- sampling -------------------------------------------------------------------------
    # "none" is the deliberate starting point. The node cloud is dense along the terminal corridor
    # because the optimal trajectory genuinely sits there, so an unweighted fit is the honest
    # baseline; "flat" (1/(K-j)) and "density" exist to be compared against it, not assumed better.
    weighting: str = "none"              # none | flat | density
    batch: int = 8192
    n_collocation: int = 4096            # residual points per step
    # The residual may be evaluated at ANY tau, so it is sampled UNIFORMLY rather than from the
    # node distribution -- filling in where labels are thin is the whole reason it is there.
    collocation_tau: str = "uniform"     # uniform | nodes

    # ---- model ----------------------------------------------------------------------------
    width: int = 256
    depth: int = 6
    activation: str = "tanh"             # tanh keeps second derivatives usable for the residual
    fourier_features: int = 0            # 0 = off; tau-only random features if > 0
    # "site"    r(1) = r_site exactly, by construction. The labels land within 0.0997 m of the
    #           site (measured max over 3028 cases), so this is a <= 10 cm modelling bias in
    #           exchange for a terminal guarantee that holds for any weights -- and it removes the
    #           miss hinge entirely, which is the term that destabilised the first run.
    # "learned" r(1) = r_site + P q with q a network output: matches the labels exactly, but the
    #           landing point is then only as good as the network.
    terminal: str = "site"
    # The ansatz may carry a REFERENCE TRAJECTORY, so the network predicts a deviation rather than
    # a 12 km descent from scratch; "nominal" uses a C2 spline through the mean training parent.
    reference: str = "none"              # none | nominal
    sigma_head: bool = True              # learn a switching function, pass it through a steep
    sigma_sharpness: float = 12.0        # sigmoid, instead of regressing a near-discontinuous T
    # THE SHARPNESS IS ANNEALED UPWARD. A steep sigmoid has a vanishing derivative away from its
    # centre, so starting steep starves the thrust channel; starting soft lets gradients flow and
    # finishing steep makes the output commit to a rail. Measured on run full4784 at a fixed
    # sharpness of 12: 82.9 % of predictions sit on a rail against 97.6 % of labels, i.e. the
    # switch is smeared across a band the truth never occupies.
    sigma_sharpness_final: float = 40.0
    sigma_anneal_steps: int = 20000      # 0 disables the anneal and holds sigma_sharpness

    # ---- loss -----------------------------------------------------------------------------
    w_r: float = 1.0
    w_v: float = 1.0
    w_m: float = 1.0
    w_d: float = 1.0
    w_T: float = 1.0
    w_tgo: float = 1.0
    w_residual: float = 1.0
    w_hinge: float = 1.0
    # tf multiplies the residual (d/dt = (1/tf) d/dtau), so the residual is meaningless until tgo
    # is roughly right. It is ramped in rather than applied from step zero.
    residual_ramp_steps: int = 2000

    # ---- optimisation ---------------------------------------------------------------------
    adam_steps: int = 40000
    adam_lr: float = 1e-3
    adam_lr_final: float = 1e-5          # cosine decay to this
    lbfgs_steps: int = 2000              # 0 disables the polish
    lbfgs_history: int = 50
    # THE POLISH RUNS ON A FIXED BATCH. L-BFGS builds curvature pairs from differences of gradients
    # and its strong-Wolfe line search assumes a DETERMINISTIC objective; a fresh minibatch per
    # closure call makes every evaluation a different function, the line search fails on the first
    # try and the optimiser stops after one iteration. Measured: 1 closure evaluation, loss
    # 1.372876e-02, val_r 53.00 -> 53.44 m. So the batch is drawn once and every closure sees
    # exactly those points, supervised AND collocation indices AND their tau values.
    lbfgs_batch: int = 65536             # fixed supervised batch for the polish
    lbfgs_collocation: int = 16384       # fixed collocation batch
    lbfgs_batch_seed: int = 12345        # draws that fixed batch; recorded like every other knob
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"                 # auto | cpu | cuda | mps
    # FLOAT64 is available and matters for more than the hard constraints. The residual takes
    # derivatives of the network output with respect to tau, and the outputs are of order 1e4 m,
    # so in float32 the derivative carries roughly seven significant digits before the residual
    # even forms. Whether that limits the physics term is a measurement, not a principle.
    dtype: str = "float32"               # float32 | float64
    amp: bool = False                    # off: the residual needs the precision

    # ---- bookkeeping ----------------------------------------------------------------------
    eval_every: int = 2000
    # VALIDATION SAMPLING. The metric used to take a CONTIGUOUS PREFIX of the validation pairs,
    # and data.build concatenates parent by parent, so 200k of 4.74M pairs was the first ~40 of
    # 957 parents -- 4 % of the set, always the same 4 %. It is a random draw now, seeded so the
    # number is still reproducible run to run.
    val_sample: int = 200_000
    val_seed: int = 7
    # The mean over all tau is diluted by the start of the trajectory, where the ansatz pins
    # r(0) = r0 and the error is zero by construction, so a terminal band is reported separately.
    # THE BAND IS ON tgo, NOT tau. Sub-trajectories are re-anchored, so tau only measures position
    # within whichever sub-trajectory a pair came from: a pair at tau = 0 can be a tail that starts
    # a few metres above the surface, and is then every bit as much "near landing" as one at
    # tau = 1. Time-to-go says how close to touchdown a query actually is; tau does not.
    val_tgo_end_s: float = 60.0
    out_root: str = "runs"

    # filled in by the trainer, never by hand
    data_fingerprint: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    scaling: dict = field(default_factory=dict)

    # ---- derived --------------------------------------------------------------------------
    def hash(self) -> str:
        """Content hash over the settings only -- not over fields the trainer fills in."""
        skip = {"notes", "data_fingerprint", "split", "scaling", "out_root", "eval_every", "device"}
        payload = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]

    @staticmethod
    def code_hash(files=("runcfg.py", "data.py", "model.py", "losses.py",
                         "train.py", "gravity_torch.py")) -> str:
        """SHA1 over the sources that define a run's behaviour.

        THE CONFIG HASH IS NOT ENOUGH. A loss-term fix changes what a run means while leaving every
        config field identical, so two different experiments would share a run_id and the second
        would overwrite the first -- which happened: the hinge non-dimensionalisation changed the
        objective completely and nothing in config.json moved. Folding the source into the id makes
        a code change a different run, which is what it is.
        """
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.sha1()
        for f in sorted(files):
            fp = os.path.join(here, f)
            if os.path.exists(fp):
                h.update(open(fp, "rb").read())
        return h.hexdigest()[:8]

    @property
    def run_id(self) -> str:
        return f"{self.name}-{self.hash()}-{self.code_hash()}"

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(asdict(self), open(path, "w"), indent=1)

    @staticmethod
    def load(path):
        return Config(**json.load(open(path)))

    @staticmethod
    def from_args(argv=None):
        """Every field is a --flag, so nothing can be set that is not recorded."""
        import argparse
        ap = argparse.ArgumentParser()
        for f in fields(Config):
            if f.name in ("data_fingerprint", "split", "scaling"):
                continue
            t = f.type if not isinstance(f.type, str) else f.type
            kind = {int: int, float: float, str: str, bool: int}.get(
                type(f.default), str if f.default is None else type(f.default))
            ap.add_argument(f"--{f.name.replace('_','-')}", dest=f.name, type=kind,
                            default=f.default)
        a = ap.parse_args(argv)
        return Config(**{k: v for k, v in vars(a).items()})


if __name__ == "__main__":
    c = Config()
    print("run_id:", c.run_id)
    print(json.dumps(asdict(c), indent=1)[:900])

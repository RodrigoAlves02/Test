# PINN — descent guidance surrogate

Learns the map the SCvx solver computes, so that in flight a forward pass replaces a ~200 s solve.

    inputs    (r, v, m, d, tau)
    outputs   (r(tau), v(tau), m(tau), d(tau), T(tau), tgo)

Labels come from `SCvx/pinn_case.py`: one dispersed handover per seed, one converged two-stage
solve, stored as the parent trajectory in SI. Sub-trajectories are derived here at load time, not
stored — see `data.py`.

## The protocol

This folder is run as an experiment log, not as a program. Three rules, and they are the point:

1. **Every run is recorded in full.** `runs/<run_id>/config.json` holds every knob, the data
   fingerprint, and the seeds. Nothing is passed on a command line that is not written to that file.
   A run that cannot be reproduced from its own config.json is a bug.

2. **Every run is tested identically.** `evaluate.py` is the single test battery and it is
   versioned (`EVAL_VERSION`). Two runs are comparable only if their `eval_version` matches; the
   comparison tooling refuses to rank across versions rather than quietly mixing them.

3. **The battery decides what is promoted.** Only runs that pass it go to the closed-loop Monte
   Carlo, which is the real test and is expensive. The battery exists to make that shortlist
   honest, so it is deliberately harsher than training loss.

## Layout

    config.py     every knob, one dataclass, serialised into each run
    data.py       parents -> sub-trajectories -> pairs, global scaling, split BY PARENT
    model.py      the network and the hard-constraint ansatz
    losses.py     supervised + physics residual + inequality hinges
    train.py      one run; writes runs/<run_id>/
    evaluate.py   THE battery; identical for every run
    compare.py    tabulates runs/ so results can be ranked and written up
    runs/         one directory per run, never edited by hand

## The split is BY PARENT SEED, not by sample

Every sub-trajectory of a parent is the same physical arc re-anchored, so a random split over
pairs would put the same trajectory in train and validation and report a meaningless score. The
split is on the seed, and the seed lists are written into config.json.

## Planned comparison: a 7-state supervised policy

If time allows, train a plain supervised network that outputs only the COMMANDED THRUST, on the
7-state formulation, called every dt_act with the actuator doing the rest. Same labels, so nothing
is wasted either way.

    trajectory surrogate (this)   replaces the SOLVE   physics-informed, verifiable, heavier
    guidance policy (comparison)  replaces the LOOP    simpler, directly flyable, supervised only

Two things settled about it in advance, so they are not rediscovered:

`d` STAYS AN INPUT even though it stops being an output. Two vehicles at the same (r, v, m) with the
engine pointing 40 deg apart need different commands, because the second must slew before it can
thrust usefully. Dropping `d` from the input is precisely the instant-pointing assumption the
10-state formulation exists to avoid -- it would make the thrust direction free to jump, which is the
3DoF idealisation, not a simplification of it.

IT IS NOT A PINN, and that is why it is the comparison rather than the thesis. The physics residual
is a condition on a FUNCTION of tau -- dr/dt = v, dv/dt = g + T d/m - ... -- and a policy emits one
command at one instant, so there is nothing to differentiate. Recovering the residual would mean
rolling the policy out and imposing the dynamics on the rollout, which costs per training step
everything the simpler architecture saved. As a supervised regression of the solver's output it is a
fair and useful baseline; it just answers a different question.

It also cannot be checked before it is flown. A trajectory surrogate lets the glideslope, the
terminal corridor and the fuel be verified ahead of committing; a policy can only be verified by
flying it, which is what the closed-loop Monte Carlo is for.

# Trust-region ablation

Question: does the loop in `ptr.run` need its trust-region machinery, and would classic PTR, the
SCvx ratio test or a different radius do better? TFC benchmarks (min-max, max-min-max), free final
time, K = 60, 150-iteration budget, tol_obj = 1e-6. Raw rows are in `ablation_results.json`.

Variants:
* `default`: the repo loop. Every step accepted; hard cap R adapted from the measured defect.
* `fixedR`, `R=x`: same with the radius frozen at x (no adaptation).
* `noTR`: cap removed (R = 1e3).
* `ratio`: `ratio_test=True` (SCvx accept/reject plus rho-band radius update).
* `ptr<w>`: classic PTR. Quadratic trust-region price with fixed weight w, no cap, on a patched
  copy of `subproblem.py`; `_nostop` disables objective-based stopping so the J history is judged.

## 10-state solver (`SCvx/`)

    case         variant         status     it   propellant [kg]
    min-max      default         converged   61  180.371
    min-max      fixedR (0.35)   converged   57  180.369
    min-max      R=0.1/0.2/0.7/1.4/3.0  converged  106/78/50/62/92   180.358-180.398
    min-max      noTR            converged   69  180.872   (+0.28 %, worse point)
    min-max      ratio           converged   41  180.339
    min-max      ptr0.1/1/10     stuck at 215.0 / 217.2 / 206.2 for 140 iterations (|d| = 0.958)
    max-min-max  default         converged   81  275.560
    max-min-max  fixedR (0.35)   converged  103  275.559
    max-min-max  R=0.1/0.2/0.7/1.4/3.0  maxiter/121/58/53/46   275.550-275.595
    max-min-max  noTR            maxiter    150  275.794
    max-min-max  ratio           stalled     62  276.055  (29 rejections, radius at its floor)
    max-min-max  ptr0.1/1/10     stuck at 308.8 / 295.6 / 303.9

    min-max from flight-time seeds 15 / 40 / 60 s:
      default  converged 116 / 66 / 21 it, 180.367 / 180.346 / 180.387 kg
      ratio    stalled at 180.332 / 180.359 / 212.598 kg
      ptr1     maxiter / maxiter / "converged" at 222.4 / 209.3 / 267.9 kg

Classic PTR fails on the 10-state form for a structural reason. Without the cap, the first step
pushes the direction d off the unit sphere, and nothing in the subproblem pulls it back: d_0 is free
inside ||d|| <= 1 and the kinematics conserve its norm. The solver then converges to a consistent
"short-direction" solution (||d|| = 0.958 at every node) that pays about 4 % more propellant per unit
of thrust. On this formulation the hard cap is what keeps ||d|| = 1.

## 7-state LCvx solver (`SCvx/_v1_zoh_backup`)

    case         variant   status     it   propellant [kg]
    min-max      default   converged  12   180.331
    min-max      ratio     converged   5   180.330
    min-max      ptr0.1    maxiter   150   180.345  (within 0.1 % at it 26)
    min-max      ptr1/10   maxiter   150   180.643 / 182.206  (still descending)
    max-min-max  default   converged   7   275.551
    max-min-max  ratio     converged   6   275.550
    max-min-max  ptr0.1    maxiter   150   275.564  (within 0.1 % at it 36)
    max-min-max  ptr1/10   maxiter   150   276.602 / 280.589  (still descending)

Classic PTR does not get stuck here, because there is no direction state, but it is an order of
magnitude slower than the hard cap.

## Reading

* Keep the hard cap in both solvers. Removing it gives a worse point or no convergence (10-state),
  and replacing it with a quadratic price is slower (7-state) or wrong (10-state).
* The defect-driven radius rule adds little over a fixed radius. R around 0.7 cut iterations on both
  10-state benchmarks (61 -> 50, 81 -> 58). Tune it on dispersed Eros cases before freezing it.
* The ratio test is the fastest option on the 7-state solver (5-6 iterations, best propellant) but
  stalls in 4 of 5 runs on the 10-state solver. A candidate default for 7-state only.
* The 10-state solver needs 5-10 times the iterations of the 7-state one (61 vs 12, 81 vs 7) and
  stops slightly higher (+0.02 %). That is the iteration cost of carrying the slew limit.
* For 10-state datasets, check ||d|| on every stored case (`dnorm_lo`/`dnorm_hi`): it is held at 1
  by the cap and the kinematics, not by a constraint.

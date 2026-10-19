# Eros tests

System-level tests of `../../../SCvx` on the Eros problem (49 152-facet Gaskell polyhedral field,
rotating body frame), run without modifying the solver. Configuration is patched before import
where needed; `run_descent.solve` and `ptr.run` are wrapped, never edited.

| script | what it does |
|---|---|
| `nominal_audit.py` | regenerates the 10-state nominal (two-stage, K = 60, config defaults) and runs `verify.audit` on it |
| `profile_eros.py` | time and gravity-call split of one Eros PTR iteration, one pinned core |
| `opt3.py 7` | the 7-state LCvx solver (`_v1_zoh_backup`) on the same handover, exact field, plus `verify.audit` |
| `fly_check.py` | flies the stored nominal control four ways: direction integrated from d0 or re-anchored per node, Gamma or thrust in N held |
| `disp/grid_gravity.py` | the polyhedral field tabulated on a grid in the site frame: a fast stand-in for PINN-GM-III (`build`, `check`) |
| `disp/fill_grid.py` | evaluates the rest of the grid box, in parts, and merges |
| `disp/disp_case.py` | one dispersed handover, one formulation, one trust-region variant; dataset settings of `pinn_case.py`; open-loop flights of the returned and of the judged control |
| `disp/worker.sh`, `jobs_*.txt` | runs a job list on one pinned core (`PY=/path/to/python ./worker.sh jobs_0.txt 2`) |
| `disp/regauge_test.py` | one converged label, flown with the control `ptr.run` returns and with the control its convergence test judged |
| `disp/diag_out.py` | where out-of-grid gravity queries land during stage 1 |

The grid (`disp/grid_gravity.npz`, 19 MB) is not committed: `python grid_gravity.py build`, then
`fill_grid.py part i 2 core` for i = 0, 1 and `fill_grid.py merge 2` (about 40 min per part on one core).

## Results (cloud container, 4 cores, Python 3.13, CVXPY 1.9.3, CLARABEL)

### Cost of one Eros iteration (`results/profile_eros.json`)

    K = 60, stage 1, one pinned core, 2 iterations: 371 s
    discretization 146.5 s (9 145 gravity calls)   defect 220.9 s (13 924 calls)
    CVXPY build 0.28 s                              compile + solve 2.2 s
    15.5 ms per polyhedral call on one core (about 4.7 ms on four); gravity is 96.7 % of the time

About 11 600 gravity calls per iteration. On Eros the cost is the gravity model, not the solver.

### Grid stand-in (`results/grid_check.json`, `fly_check.txt`)

    61 x 61 x 73 nodes, east/north +-7.5 km at 250 m, up -0.2..13 km (50 m below 1 km, 250 m above)
    28 us per call (exact: 15.5 ms on one core)
    g relative error: max 0.16 % below 1 km, 0.04 % above, median 0.015 %  (600 points in the cone)
    gradient: median 0.08 %, max 11 % near the surface (PINN-GM-III returns a zero gradient)
    the stored nominal control flown on the grid vs the exact field: 52.94 vs 52.84 m miss, -33.47 vs -33.58 m altitude
    7-state nominal: grid 5.5247 kg / 2036.9 s vs exact 5.5238 kg / 2037.0 s (pointing cone on vs off, not binding)

The first version covered only the 30 deg cone + 1 km. Early stage-1 iterates, with virtual control
still active and the glide-slope apex at a landing point kilometres from the site, sat up to 7 km
laterally at 2-5 km altitude, so 84 000 of 730 000 calls fell back to the exact field. The full box
is now tabulated.

### 10-state nominal, `verify.audit(K=60, iters=35)` (`results/nominal_audit.json`)

    stage 1  converged in 14 it, D = 0.000 m, tf 1616.1 s
    stage 2  MAXITER at 35 it: 5.5453 kg, tf 2058.5 s, still creeping (dJ/J 7e-4, tf +0.5 s/it)
    audit    lateral miss 0.094 m solver, 52.84 m flown; altitude -33.58 m flown; 0.3645 m/s at touchdown
             ALL CONSTRAINTS SATISFIED (nodes and between): False

The failure is in the direction channel. `verify.fly` integrates omega from d0 over the whole flight
(a rate command). The plan carries per-segment direction defects of up to 6.4e-3 rad during the
large turn (segments 24-29, 23-36 deg per segment), and `ptr`'s convergence test does not check the
direction defect. Flying the same control with d re-anchored at every node, as
`closed_loop.plan_thrust` does:

    flight                                         lateral miss   altitude   speed
    omega, Gamma held; d integrated from d0         52.941 m      -33.465    0.3638
    omega, Gamma held; d re-anchored per node        0.049 m       -1.725    0.3014
    thrust in N held;  d integrated from d0         53.015 m      -33.332    0.3626
    thrust in N held;  d re-anchored per node        0.676 m       -1.592    0.3001

Holding Gamma or thrust in newtons changes the result by under a metre; integrating the direction
open loop is the whole 60 m. The residual -1.7 m is this plan stopping at maxiter.

### Why the returned control is less consistent than the converged one (`results/regauge_test.json`)

`ptr.run` sets the new reference to `(Xn, _gauge(Un))` (ptr.py:575-579) and judges convergence on the
defects of `(Xn, Un)` (ptr.py:474), then returns the gauged control. Removing omega's component along
d is inert at a node but not across a 0.5 rad ZOH turn. Converged label, nominal handover, dataset
settings, grid field:

    control                  d defect   r defect   v defect     rate flight         attitude flight
    judged (Un)              3.1e-5     0.08 mm    7.6e-6 m/s   0.07 m, -0.22 m     0.09 m, +0.002 m
    returned (_gauge(Un))    4.4e-3     3.9 mm     4.5e-4 m/s   23.6 m, -4.34 m     0.41 m, -0.40 m

The stored labels (r, v, m, d, T) come from X and Gamma and are unaffected. The stored `defect_m`, the
rate audit and the within-segment rotation in `plan_thrust` use the returned omega. Fix without
changing the iteration: keep `U_out = Un` before the gauge and return it.

### 7-state, exact field (`results/opt3_7state_exact.json`)

    stage 1 converged in 5 it; stage 2 converged in 10 it: 5.5238 kg, tf 2037.0 s, 28 min
    flown miss 0.013 m, altitude +0.0005 m, 0.2999 m/s; all constraints satisfied at and between nodes

### Other findings

* `subproblem.py:410` sets the omega trust-region weight from `config.SLEW_MAX_DEG_S`, not from the
  limit passed to `ptr.run`. `SLEW_MAX_DEG_S = None` therefore crashes `subproblem.build`, although
  `verify.py:112` and `plot_descent.py:77` accept None; and a slew sweep passed by argument keeps the
  2 deg/s trust scaling. `opt3.py` and `disp_case.py` remove the limit through `ptr.run` instead.
* Stage 1 from a cold start spends about 9 iterations with virtual control above tolerance
  (vc/tol 6.5e5 -> 1.5e-4) before the defect-driven phase.

### Dispersed handovers: trust-region variants and 7 vs 10 states (`disp/res/*.json`, `results/disp_summary.txt`)

Five handovers (seed 0 = nominal; seeds 1-4 dispersed by 1 km, 1 m/s, 2 kg, as `pinn_case.py` draws
them), grid field, dataset settings (tol_obj 1e-7, defect 1 mm / 1e-4 m/s, 200 iterations, pointing
cone off, terminal 1 deg, align corridor 45 deg / 100 m for 10-state), d(0) free, two-stage, K = 60.

    stage 2, propellant [kg] (iterations)       seed 0          1               2               3               4
    10-state default                            5.5238 (44)     5.3699 (83)     5.2259 (89)     6.0268 (46)     5.1773 (29)
    10-state fixed R 0.35                       5.5253 (70)     5.3705 (122)    5.2257 (69)     6.0260 (57)     5.1768 (25)
    10-state fixed R 0.7                        5.5230 (94)     5.3652 (43)     5.2264 (63)     6.0135 (54)     5.1822 (71)
    10-state ratio test                         stalled in all five; 4 of 5 already stall in stage 1; 26-46 rejections
    10-state, slew limit off                    5.5225 (42)     5.3650 (122)    5.2294 (62)     6.0239 (59)     5.1802 (19)
    7-state default                             5.5247 (14)     5.3650 (11)     5.2308 (12)     6.0062 (18)     5.1701 (9)
    7-state ratio test                          maxiter/stalled in all five, 8.27-10.88 kg (LCvx gap up to 88 %: not physical)

Reading:

* Keep the repo default loop. Fixed radii converge too and land within 0.02 % (R 0.35) or -0.22 % to
  +0.09 % (R 0.7), with no consistent iteration gain: R 0.7 was faster on two handovers and slower on
  three. The 0.7 suggestion from the TFC benchmarks does not carry over.
* The ratio test fails on Eros for BOTH formulations, although it was the fastest option on the 7-state
  TFC benchmarks (5-6 iterations). Tune on the target problem, not on benchmarks.
* OPT-3, same optimum: 10-state without slew vs 7-state differ by 0.00-0.29 % (sign varies). Slew limit
  on vs off: -0.07 % to +0.09 %. Two "slew off" runs end slightly ABOVE "slew on", which a global
  optimum forbids, so ~0.1 % is the local-optimum / tolerance floor and the 2 deg/s limit is free here.
* Cost: the 7-state converges in 9-18 stage-2 iterations (0.3-0.5 min), the 10-state in 29-89
  (1.7-5.2 min). Handover 3 (all variants) and 1 (R 0.7) sent some calls outside the grid box to the
  exact field, so their wall times are not comparable.

Returned vs judged control, 10-state (table B in `disp_summary.txt`): the control the convergence test
judged has direction defects 1e-5 to 2e-4 rad and r defects 0.04-0.95 mm, and flown with attitude
commands lands 0.03-0.10 m from the site in all 20 converged runs. The returned (re-gauged) control
has direction defects up to 2e-2 rad and stored r defects up to 16.9 mm (9 of 20 above the 1 mm
tolerance), and flies 0.04-1.18 m (attitude) or up to 82 m (rate command).

### 7-state: can a slew-limited engine fly the plan? (`disp/disp7_slew.py`, `disp/fly7.py`, `disp/res7/`)

The 7-state direction is ZOH: constant over a hold, so the pointing cone at the node covers the hold
exactly, but it STEPS at every node and nothing bounds the step. Measured on the plans above:
largest step 28.7 deg on the nominal (the turn to vertical into the last hold), 101.7 deg on handover 3
(124 -> 149 -> 63 deg from vertical on the 5 N floor, a 51 s turn in a 32 s hold).

`disp7_slew.py` adds two affine rows of the pointing-row form to a patched copy of the 7-state
subproblem: a slew budget between holds, ah_ref[k].a[k+1] >= cos(w beta h) G[k+1] (and the mirror),
and a reachable first hold about the engine direction, both with an L1-priced slack. beta = 0.25,
w = 2 deg/s, i.e. each turn fits in a quarter of a hold.

    handover                          0           1           2           3           4
    propellant, baseline [kg]         5.5247      5.3650      5.2308      6.0062      5.1701
    propellant, slew rows [kg]        5.5295      5.3811      5.2390      6.0957      5.1798
                                      +0.09 %     +0.30 %     +0.16 %     +1.49 %     +0.19 %
    stage-2 iterations                14 -> 200   11 -> 56    12 -> 37    18 -> 82    9 -> 22

`fly7.py` flies each plan open loop with the engine delivering the charged Gamma along a/|a|,
starting at the handover attitude: ideal (instant turns), lag (2 deg/s geodesic tracker, turn starts at
the node) and lead (the same tracker, turn centred on the node). Lateral miss / altitude:

    plan                ideal              lag                  lead
    baseline 0          7.9 / +6.3         45.4 / -17.0         8.2 / +8.5
    baseline 1          4.8 / +2.2         68.3 / -10.7         33.6 / +50.5
    baseline 2          17.2 / +6.9        107.6 / -23.6        56.6 / +67.4
    baseline 3          0.9 / -0.3         279.6 / -510.2       64.5 / -385.9
    baseline 4          3.4 / +3.2         18.1 / -62.9         11.2 / -52.8
    slew rows 0         0.7 / +1.3         39.4 / -22.9         1.6 / +3.4
    slew rows 1         13.7 / +1.8        61.6 / -29.4         24.4 / +23.2
    slew rows 2         0.5 / -0.1         57.6 / -35.1         10.8 / +22.2
    slew rows 3         0.1 / -0.0         51.7 / -122.8        17.6 / -69.6
    slew rows 4         1.4 / +1.6         17.2 / -64.7         11.9 / -54.5

* The "ideal" column is not the solver's 2 mm: the 7-state relaxation is slack by 0.2-0.5 % (10-23 mN,
  58-59 of 59 holds, mostly on the 5 N floor), so an engine that delivers the Gamma it burns flies
  0.01-0.016 m/s of delta-v the plan did not count. `verify.fly` flies a, not Gamma a/|a|, and does
  not see it.
* Without a slew budget the plans are not flyable by a 2 deg/s engine (up to 280 m and a 0.83 m/s
  impact). Bounded steps plus centred turns bring it to 1.6-24 m laterally, but up to 70 m in altitude:
  open-loop error the re-planning has to absorb every cycle.
* The 10-state plan contains the rotation at <= 2 deg/s; its judged control flown with attitude
  commands lands 0.03-0.10 m from the site on the same handovers (tracker lag tau ignored in both).

SCvx FOR EROS POWERED DESCENT -- new implementation
====================================================
Self-contained. Nothing outside FullModel/SCvx is imported except the Gaskell shape model, which
is data. The previous solver in scvx_landing/ is untouched and remains available as a fallback.

Two-stage guidance after Blackmore et al. (2010): minimum landing error first, then minimum fuel
constrained to the range that stage returned. Stage 1 asks the only question answerable
unconditionally -- how close can this vehicle physically get -- and returns D. Stage 2 minimises
fuel subject to landing within D, which is feasible BY CONSTRUCTION. That is what makes
"infeasible" an answer the solver can give instead of buying a shortfall with slack.


RESULT  (K = 60, handover 12 km above the site at 8.61 m/s, upright touchdown enforced)
----------------------------------------------------------------------------------------
  flight time        2068.0 s
  propellant         5.5640 kg  (solver)  5.5640 kg (flown)   agree to 5e-7 kg
  lateral miss       0.0146 m   flown through DOP853 at rtol 1e-12
  terminal altitude -0.0001 m
  terminal speed     0.3000 m/s (limit 0.300)
  thrust at touchdown 0.000 deg from the local vertical
  max divergence     0.0009 m   between the solver's nodes and the flown trajectory
  every constraint satisfied at the nodes AND between them

  Guess invariance (test_seed_ladder.py), seeds spanning 555 - 2221 s:
      returned flight time  2048.4 .. 2075.7 s    spread 27.3 s   (1.32%)
      returned propellant   5.5229 .. 5.5277 kg   spread 0.0049 kg (0.09%)


TERMINAL THRUST POINTING -- UPRIGHT TOUCHDOWN
----------------------------------------------
Tc(tf) parallel to the local vertical, so the vehicle sets down on its legs with the engine
opposing effective gravity. Against the lossless-convexification slack this is LINEAR,
a == n_hat * Gamma, and at 0 deg it also forces ||a|| = Gamma, so the relaxation is tight there by
construction. n_hat is -g_eff rather than the radial direction, so "vertical" really does oppose
gravity at the site.

What it costs (test_terminal_pointing.py, K=60):

    terminal cone   intervals   flight time   propellant   vs free   angle achieved
       free             0          2063.1 s     5.5230 kg    0.00%      26.020 deg
       20 deg           1          2067.3 s     5.5258 kg    0.05%      19.982 deg
       10 deg           1          2061.8 s     5.5445 kg    0.39%       9.970 deg
        5 deg           1          2060.4 s     5.5545 kg    0.57%       4.990 deg
        0 deg           1          2068.0 s     5.5638 kg    0.74%       0.000 deg
        0 deg           3          2119.9 s     5.6483 kg    2.27%       0.000 deg

Exactly vertical over the LAST hold interval costs 0.74%. Holding it vertical for three intervals
costs 2.27%, so one interval is the efficient choice unless the attitude needs longer to settle.
Default: TERMINAL_POINT_DEG = 0, TERMINAL_POINT_NODES = 1.


REACHABILITY -- WHAT STAGE 1 IS ACTUALLY FOR
---------------------------------------------
On the undispersed nominal stage 1 always returns D = 0, so it looks like a no-op. Its value is in
closed loop, where a deviation can put the site out of reach and the guidance still has to produce
a FEASIBLE trajectory instead of claiming a landing the thrust cannot deliver. Dispersing the state
laterally at 1317 m altitude (test_reachability.py):

    cross-range   stage 1      D            stage 2
       600 m      converged      0.000 m    converged, 2.0274 kg
      1500 m      converged    414.704 m    site out of reach, lands at the best point it can
      3000 m      converged   1902.489 m    converged, 2.5295 kg

This only works because the glideslope cone's apex is at the ACHIEVED landing point, not the
nominal site. Anchored at the site it forces a trajectory that diverts 400 m to stay inside a cone
about a point it never reaches, and every dispersed case failed instead of returning a miss.

A dispersion that attacks STOPPING ability rather than divert distance (extra velocity into the
surface) is a different matter: no amount of lateral miss fixes "I cannot stop", and those cases are
genuinely infeasible. The loop reports status "infeasible" for them -- a virtual control that never
goes away -- rather than "maxiter" with a meaningless objective.


COMPARISON WITH THE PREVIOUS SOLVER
------------------------------------
Same initial condition (the old nominal's, range 11771 m), both control histories flown OPEN LOOP
through DOP853 at rtol 1e-12 (compare_old.py):

                             OLD (K=30)     NEW (K=60)
    propellant, flown          5.7855 kg      5.4822 kg     -5.24%
    lateral miss, flown        0.0007 m       0.0081 m
    open-loop divergence       0.0011 m       0.0088 m
    t_go spread across seeds     363 s          27.3 s
    fuel spread across seeds     0.30 kg        0.0049 kg

The fuel advantage is independently confirmed: the fixed-time sweep, which uses neither solver's
free-time machinery, puts the optimum at 2100 s / 5.5296 kg and the free-time solver lands on that
curve at its minimum. The 2 deg/s slew limit the old solver enforced is NOT binding on either
solution (0.98 and 0.46 deg/s), so its absence here does not explain the difference.

CAVEAT worth carrying forward: part of the saving is bought by riding the glideslope boundary --
29.99 deg against a 30 deg limit, where the old solution stayed at 19.12 deg. That leaves no margin
against dispersion, and under any deviation the cone is immediately violated. Flying the nominal at
a tighter cone (27 deg, say) would buy that margin back at some fuel cost.


WHAT IS DIFFERENT, AND WHY
---------------------------
1. NO TRUST-REGION RATIO. The old solver adapted a hard radius from rho = dJ/dL. dL is dominated by
   Wvc*||d_ref||_1, which the subproblem can always zero out, so it was measured CONSTANT to three
   decimals while the radius fell 16x -- the controller was steering on a signal carrying no
   step-size information. Here the step is bounded by a hard cap and the cap is moved by the
   measured linearisation error, which is a distance in metres and cannot be faked.

2. THE TRUST RADIUS IS *NOT* PRICED IN THE COST. This was tried, and failed for a reason specific
   to this problem: 58% of the propellant is burned idling at the 5 N floor, so fuel is dominated by
   flight duration, d(fuel)/d(tf) = 2.3e-3 kg/s. In objective units 100 s of flight time is worth
   0.042 while the trust price for it at w_ts = 1 is 0.167 -- FOUR TIMES the fuel it saves. The
   optimiser then refuses to move the flight time and reports convergence at the seed. Measured:
   returned tf spread 418.8 s (31.9%), fuel spread 2.40 kg (26.9%). With w_tr = w_ts = 0 and the
   hard cap alone, the same test gives 27.3 s and 0.005 kg.

3. CONVERGENCE ON THE PHYSICAL OBJECTIVE, absolute OR relative. A relative test alone never settles
   on an objective going to zero (stage 1's miss reaches ~3e-5 m and the relative change between two
   such numbers stays at 4e-3 forever). The old six-way step-size conjunction was satisfiable by a
   collapsed trust region; a miss in metres and a defect in metres are not.

4. PHYSICALLY MEANINGFUL FLIGHT-TIME BOUNDS. t_lo is the larger of "cannot null v0 faster than max
   thrust allows" and "cannot cross the range faster than max acceleration allows". The second is
   essential: with v0 = 0 the first is ZERO, and that was measured letting the optimiser drive tf to
   0.1 s and pay the virtual control to teleport the vehicle 12 km.

5. GRAVITY IS THE ONLY NONLINEARITY and it is kept, not approximated. The rotating-frame Coriolis
   and centrifugal terms are linear in the state and enter the Jacobian exactly -- which matters,
   because Coriolis is 5.1x gravity at handover on a body that turns 34 deg during the descent.


THINGS MEASURED ALONG THE WAY THAT CHANGED THE DESIGN
------------------------------------------------------
* A 45 deg pointing cone makes the descent IMPOSSIBLE. Tmin/m = 0.00833 m/s^2 exceeds gravity
  (0.0011 at 12 km to 0.0056 at the surface) everywhere and the engine cannot be switched off, so
  with the thrust within 45 deg of the vertical the net vertical acceleration is upward at every
  altitude. A feasible descent points the engine down-and-retrograde for most of the flight --
  133 deg from the vertical at handover, 26 deg at touchdown. POINTING_DEG is 140.
  OPEN ISSUE: in 3DoF the thrust direction is the only attitude proxy, so "engine retrograde" and
  "body-fixed sensor nadir" are the same constraint pulling opposite ways. A nadir sensor
  requirement cannot be expressed here without a gimbal, another face, or 6DoF attitude states.

* A POSITION-ONLY DEFECT IS BLIND. Multiple shooting restarts each segment from its own node, so a
  velocity inconsistency never shows up in position. A position-only measure read 1.09e-3 m --
  "converged" -- on a trajectory whose per-node velocity error was 2.7 m/s. Both are reported now.

* K = 30 IS NOT ENOUGH for an open-loop 2000 s trajectory. Re-measured with the current 5 mm defect
  tolerance: flown open loop it diverges 46.4 m, misses by 39.8 m, is left 50.2 m in the air, and
  breaches the glideslope by 10.8 m between nodes. At K = 60 the same audit gives 0.9 mm divergence
  and a 14.6 mm miss with every constraint satisfied.
  (This says nothing against K = 30 in a RECEDING HORIZON, where the mesh refines itself as t_go
  shrinks -- the closed-loop run below uses K = 40 and lands. It is a statement about flying a
  single 2000 s plan open loop.)

* A ZOH HAS K-1 CONTROL INTERVALS, NOT K. U[:, K-1] drives nothing, so the optimiser left it
  anywhere -- reporting 3.3 N against a 5 N floor and an apparent lossless-convexification gap of
  13.25 N that was not real. Tied to the last real interval. The true gap is <= 0.013 N.

* THE GLIDESLOPE ANGLE IS MEANINGLESS AT THE CONE APEX. At a terminal altitude of 8.6 mm with 8 mm
  of lateral offset the angle reads 43 deg while the constraint residual is 3 mm. The audit reports
  the residual in metres, which is what the constraint actually states.


VERIFICATION
-------------
  test_seed_ladder.py   same problem, seven flight-time seeds -- the acceptance test
  test_tf_sweep.py      fuel vs flight time from FIXED-time solves; the ground-truth curve the
                        free-time answer must lie on. Unimodal, minimum at 2100 s / 5.5296 kg on a
                        150 s grid; the free-time solver returns 2065 s / 5.5247 kg. Fuel moves
                        0.02% between 1950 and 2100 s, so the flat optimum -- not the solver -- is
                        what leaves a 27 s spread in the returned time.
  verify.py             flies the control through DOP853 at rtol 1e-12 and audits every constraint
                        at the nodes and between them, including the LCvx gap.
  test_reachability.py  disperses the state and checks stage 1 reports the site out of reach
  test_terminal_pointing.py   what an upright touchdown costs
  compare_old.py        like-for-like against the previous solver, same initial condition


FILES
------
  config.py          vehicle, body, site, scaling, flight-time bounds
  gravity.py         49 152-facet polyhedral field, memoised bit-exactly on r
  dynamics.py        non-dimensional rotating-frame EOM, Jacobians, ZOH discretisation, defect
  subproblem.py      the SOCP: LCvx thrust, glideslope, pointing, virtual control, trust region
  ptr.py             the successive loop and the initial guess
  problem_setup.py   geometry: site, local vertical, handover state
  run_descent.py     the two-stage driver
  verify.py          independent audit
  plot_descent.py    figures
  Figures&data/      descent_K60.{npz,pdf,png}, audit_K60.json, seed_ladder.json, tf_sweep.json

# Gerth-style verification tests

Unit and system tests modelled on Gerth (2014), *Convex Optimization for Constrained and Unified
Landing Guidance*, §6.6, run against the solver in `../../SCvx` without modifying it. The
configuration is patched before import, the same way `../run_verification.py` does it.

| script | Gerth analogue | what it checks |
|---|---|---|
| `unit_dynamics.py` | UNIT TEST 6.1 (DYN-1, DYN-2) | free fall and hover against closed form; analytic Jacobians vs finite differences; Taylor remainder of one discretized segment (slope 2); Van Loan sub-step convergence (order 2) |
| `vertical_drop.py` | UNIT TEST 6.4, CVX-1 | free-final-time vertical drop against the exact coast + full-thrust optimum (Meditch 1964), on a K ladder 31 / 61 / 121 / 241 |
| `gerth_cases.py` | FT-1, §7.2, §7.3 | Açıkmeşe & Ploen (2007) Mars case 1 at Gerth's flight times and Δt = 1 s: unconstrained, 4° glide slope, plus 35° thrust pointing |
| `profile_iter.py` | §8.4 | time split of one PTR iteration: discretization, CVXPY build, CVXPY compile, solve, defect |

None of these needs the Eros shape model; all use constant or point-mass gravity.

## Results (cloud container, Python 3.13, CVXPY 1.9.3, CLARABEL)

`unit_dynamics.py` → `unit_dynamics.json`

    DYN-1 free fall   N=10 dt=1 s : max node error h 1.4e-14 m, v 3.6e-15 m/s
    DYN-2 hover       N=100 dt=1 s: m_f 94.626631830 kg (exact 94.626631830), rel. error 1.6e-14
    JAC   analytic vs FD (point mass + rotating frame): max rel err A 3.5e-11, B 3.2e-11
    TAYLOR remainder slope 2.01 (2 expected)
    SUBSTEP Ad/Bd/zd error vs n_sub 1..16: 1.9e-06 -> 7.3e-09, order 2.00

`vertical_drop.py` → `vertical_drop.json` (analytic: t_s 20.4754 s, t_go 59.4926 s, m_p 3.2323 kg)

    K   dt[s]  status     t_go            m_p              t_s (impulse)
    31  1.986  converged  +0.131 %        +0.129 %         -0.0050 s
    61  0.992  converged  +0.053 %        +0.052 %         -0.0051 s
    121 0.496  converged  +0.023 %        +0.024 %         -0.0034 s
    241 0.248  converged  +0.011 %        +0.010 %         -0.0018 s

`profile_iter.py` (min-max benchmark, K = 60): discretization 77 %, CVXPY compile 14 %, build 3 %,
defect 4 %, solve 2 % of one iteration.

## Notes

* `config.tf_bounds` divides by `t_min`, so T_min = 0 (Gerth's CVX-1) raises ZeroDivisionError.
  `vertical_drop.py` keeps the lower bound and caps the upper bound at 300 s.
* The switch time is reported two ways. Gerth's rule (first node above 0.9 T_h) is limited by the
  node spacing. The impulse-equivalent switch, t_go - sum(T_k dt)/T_h, is accurate to 5 ms at every K.
* The Taylor test subtracts the zero-perturbation residual. Otherwise the floor set by the RK4
  reference vs the 10-substep Van Loan model flattens the fitted slope to about 1.3.

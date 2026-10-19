# Label sets

One directory per dataset. Each carries a `manifest.json` written at generation time saying what
produced it — dispersion, constraints, solver chain, tolerances, gates, nominal, and a hash of the
generating sources. Never rely on the directory name alone.

| directory | n | pointing cone | status |
|---|---|---|---|
| `pinn_cone140` | 4800 attempted, 4740 accepted, 60 rejected | **140 deg, ACTIVE** | **superseded** |
| `pinn` | regenerating | **disabled (180 deg)** | current |

## Why `pinn_cone140` was superseded

`C.POINTING_DEG = 140` was carried as a global thrust-pointing cone. It was never a hardware limit —
the landing attitude is set by the terminal pointing ramp and the slew rate — but it was active at
interior nodes on **27.5 %** of trajectories, pressed exactly on the limit.

Measured on 20 seeds at 140 deg against disabled:

- where it binds, the optimum wants **150–177 deg**; holding it at 140 costs **4.8 % of the fuel on
  median, 10.4 % at worst**, with **6–30x** more solver iterations
- where it does not bind, removing it is a no-op: fuel within 0.08 %, same iteration count
- it is the root of **59 of the 60 generation failures**. All 46 `infeasible` cases have a handover
  attitude outside the cone (`ang0 > 140`), which contradicts the `d_init` pin at node 0 and empties
  the feasible set. Seven of eight failing seeds converge in 6–9 iterations with it off, against 400
  and `infeasible` with it on.

So 27.5 % of `pinn_cone140`'s labels are shaped by an arbitrary constant and are suboptimal by up to
10 %. Usable for infrastructure work, not for a result.

## Known caveats that apply to both

- ~0.3 % of accepted cases store a `defect_m` above the tolerance they were tested against: `ptr`
  re-gauges omega AFTER its convergence test and reports the defect of the re-gauged control.
  Filter on `defect_m` at load.
- the thrust label stops at node `K-2`; `U[:, K-1]` is a tied dummy.

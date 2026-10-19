#!/usr/bin/env python
"""Write the Monte Carlo case list: one 'pos_sigma vel_sigma seed' per line.

THE DESIGN. Ten handover-dispersion configurations, 50 cases each, 500 runs.

    (0,    0    )   no truth dispersion at all -- the vehicle really is at the nominal handover
                    state.  The ONLY error is navigation, so this level is the landing-accuracy
                    distribution of the filter alone and the reference every other level is read
                    against.
    (10,   0.01 )   position and velocity dispersed together, three decades
    (100,  0.1  )
    (1000, 1    )
    (10,   0    )   position only, three decades -- separates a handover POSITION error from
    (100,  0    )   a velocity one, which the coupled cases cannot
    (1000, 0    )
    (0,    0.01 )   velocity only, three decades
    (0,    0.1  )
    (0,    1    )

The OD error is drawn on top of EVERY case, from the dispersed truth state, never from the nominal:
closed_loop builds the truth at (nominal + dispersion) and then hands that state to
ekf.initial_state_and_cov, which adds the 10 m / 0.01 m/s draw to it.  The filter is therefore
wrong about a vehicle that is itself not where the plan assumed.

The seed INDEX is common across configurations: case (10, 0.01, 907) and case (1000, 0, 907) share
a dispersion direction and share a navigation realisation, and differ only in the dispersion
magnitudes.  A difference between configurations is then attributable to the configuration rather
than to a different draw, which is worth far more than 500 independent draws at n = 50.
"""
import os

CONFIGS = [(0.0, 0.0),
           (10.0, 0.01), (100.0, 0.1), (1000.0, 1.0),
           (10.0, 0.0), (100.0, 0.0), (1000.0, 0.0),
           (0.0, 0.01), (0.0, 0.1), (0.0, 1.0)]
SEED0, NSEED = 901, 50

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    lines = [f"{p:g} {v:g} {SEED0 + j}" for p, v in CONFIGS for j in range(NSEED)]
    with open(os.path.join(here, "mc_cases.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} cases -> mc_cases.txt")
    print(f"first: {lines[0]}   last: {lines[-1]}")

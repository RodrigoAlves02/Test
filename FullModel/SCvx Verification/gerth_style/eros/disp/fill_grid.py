"""Fill the NaN nodes of grid_gravity.npz (the box outside the cone + 1 km).
usage: fill_grid.py part <i> <n> <core>   evaluate every n-th missing node starting at i
       fill_grid.py merge <n>"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); NPZ = os.path.join(HERE, "grid_gravity.npz")
if sys.argv[1] == "part":
    i, n, core = map(int, sys.argv[2:5]); os.sched_setaffinity(0, {core})
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "SCvx")))
    import time, gravity as GR
    z = np.load(NPZ); G = z["G"]; ex, nx, ux = z["ex"], z["nx"], z["ux"]
    miss = np.argwhere(np.isnan(G[..., 0]))[i::n]
    pts = (z["r_site"] + ex[miss[:, 0]][:, None] * z["east"] + nx[miss[:, 1]][:, None] * z["north"]
           + ux[miss[:, 2]][:, None] * z["n_hat"])
    t = time.time(); res = GR.field()._ev(np.ascontiguousarray(pts))
    V = np.array([np.concatenate([np.asarray(r[1], float), np.asarray(r[2], float)]) for r in res])
    np.savez(os.path.join(HERE, f"fill_{i}.npz"), idx=miss, V=V)
    print(f"part {i}: {len(pts)} nodes in {time.time()-t:.0f} s", flush=True)
else:
    n = int(sys.argv[2]); z = dict(np.load(NPZ)); G = z["G"]
    for i in range(n):
        p = np.load(os.path.join(HERE, f"fill_{i}.npz")); G[tuple(p["idx"].T)] = p["V"]
    assert not np.isnan(G).any()
    z["G"] = G; z["n_eval"] = G[..., 0].size
    np.savez_compressed(NPZ, **z); print("merged; grid complete", G.shape)

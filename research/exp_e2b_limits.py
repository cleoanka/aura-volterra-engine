"""E2b -- practical window limits: long binary windows + chaotic triple, and the
damping / Anderson ablation (w in {0.6, 1.0}; picard vs anderson)."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary, triple
from aura_engine.physics import VolterraSystem

def solve(p, W, nodes, accel, w, iters=600, tol=1e-12):
    s = VolterraSystem(p.bodies(), G=p.G, c=p.c, softening=p.softening, t_span=W,
                       n_samples=nodes, relaxation=w, scheme="chebyshev",
                       accelerator=accel, retardation=p.retardation, pn="none")
    with np.errstate(all="ignore"):
        return s.relax(iters, record_history=False, tol=tol)

rows = []
cases = [("binary", binary(c=8.0, retardation="naive"), (2.0, 3.0, 4.0, 6.0)),
         ("triple", triple(c=18.0, retardation="naive"), (0.25, 0.5, 1.0, 2.0, 3.0, 4.0))]
for name, p, Ws in cases:
    for Wu in Ws:
        W = Wu * (2 * math.pi / p.omega) if name == "binary" else Wu
        nodes = int(max(24, 14 * p.omega * W)) if name == "binary" else int(max(24, 40 * W))
        for accel, w in (("picard", 1.0), ("picard", 0.6), ("anderson", 1.0), ("anderson", 0.6)):
            r = solve(p, W, nodes, accel, w)
            ok = bool(r.converged and np.isfinite(r.trajectory).all())
            rows.append(dict(problem=name, W=W, W_units=Wu, nodes=nodes, accel=accel, w=w,
                             converged=ok, iters=r.iterations))
            print(f"{name} W={Wu} ({'orbits' if name=='binary' else 't'}) M={nodes} {accel:8s} w={w} conv={ok} it={r.iterations}")
json.dump(rows, open(os.path.join(ROOT, "research/results/e2b_limits.json"), "w"), indent=1)

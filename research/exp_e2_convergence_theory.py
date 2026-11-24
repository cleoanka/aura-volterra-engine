"""E2 -- Picard convergence on one window vs the Volterra (quasi-nilpotent) bound.

Theory (NOTES.md sec. 3): with causal (retarded, tau >= 0) delays the Picard map
stays a Volterra operator, so with Lipschitz constant L of the force field

    e_n <= e_0 * x^(2n) / (2n)!,      x = sqrt(L_eff) * W,

i.e. no convergence radius in exact arithmetic, a transient amplification
max_n x^(2n)/(2n)! ~ e^x / sqrt(2 pi x), and an iteration count that grows
~linearly in x. For the equal-mass binary L = 2 Omega^2 -> x = sqrt(2) Omega W.

We measure, for plain (undamped, w=1) Picard and for Anderson, on instantaneous
and retarded gravity: iterations to tol, the peak transient of the iterate
error, and the per-iteration contraction ratio vs x^2/((2n+1)(2n+2)).
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary  # noqa: E402
from aura_engine.physics import VolterraSystem  # noqa: E402

OUT = os.path.join(ROOT, "research", "results")
os.makedirs(OUT, exist_ok=True)


def solve(p, W, nodes, accel, iters=400, tol=1e-12):
    s = VolterraSystem(p.bodies(), G=p.G, c=p.c, softening=p.softening, t_span=W,
                       n_samples=nodes, relaxation=1.0, scheme="chebyshev",
                       accelerator=accel, retardation=p.retardation, pn="none")
    r = s.relax(iters, record_history=True, tol=tol)
    return r


def bound_iters(x, ratio):
    """Smallest n with x^(2n)/(2n)! <= ratio (log-space)."""
    for n in range(1, 2000):
        if 2 * n * math.log(max(x, 1e-300)) - math.lgamma(2 * n + 1) <= math.log(ratio):
            return n
    return None


def main():
    rows = []
    for model, c in (("none", 8.0), ("naive", 8.0), ("aberration", 8.0)):
        p = binary(c=c, retardation=model)
        L = 2.0 * p.omega ** 2
        for orbits in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            W = orbits * 2 * math.pi / p.omega
            x = math.sqrt(L) * W
            nodes = int(max(24, 14 * p.omega * W))
            # converged solution (Anderson, tight) as the fixed point R*
            ref = solve(p, W, nodes, "anderson", iters=600, tol=1e-14)
            for accel in ("picard", "anderson"):
                r = solve(p, W, nodes, accel)
                errs = [float(np.max(np.abs(h - ref.trajectory))) for h in r.history] \
                    if ref.converged else []
                e0 = errs[0] if errs else float("nan")
                peak = max(errs) / e0 if errs else float("nan")
                ratios = [errs[k + 1] / errs[k] for k in range(len(errs) - 1)
                          if errs[k] > 1e-11]
                pred_ratios = [x * x / ((2 * k + 1) * (2 * k + 2)) for k in range(len(ratios))]
                rows.append(dict(
                    model=model, c=c, orbits=orbits, W=W, omegaW=p.omega * W, x=x,
                    nodes=nodes, accel=accel, converged=bool(r.converged and ref.converged),
                    iters=r.iterations, iters_bound=bound_iters(x, 1e-12),
                    peak_amplification=peak,
                    peak_bound=math.exp(max(2 * k * math.log(x) - math.lgamma(2 * k + 1) for k in range(0, 400))),
                    ratios=ratios[:12], pred_ratios=pred_ratios[:12],
                    final_residual=r.residuals[-1] if r.residuals else None))
                print(f"{model:10s} {accel:8s} orbits={orbits:4.2f} x={x:6.2f} "
                      f"it={r.iterations:4d} (bound {rows[-1]['iters_bound']}) "
                      f"conv={rows[-1]['converged']} peak={peak:9.3g} "
                      f"(bound {rows[-1]['peak_bound']:.3g})")
    with open(os.path.join(OUT, "e2_convergence.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

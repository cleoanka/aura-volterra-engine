"""E7 -- scaling with N (bound random clusters, naive retardation, c=10).

For each N: reference = DelayDiffEq Vern9 (Float64, tol 1e-14, with dependent_lags);
competitor = DelayDiffEq Vern9 at tol 1e-10 (+lags). Engine variants (W=T/4, M=16):
  A  ballistic history, fixed windows (no error control, no tracking)
  B  C^2 history, accuracy_tol=1e-10 (a-posteriori Chebyshev control)
  C  C^2 history, accuracy_tol=1e-10, track_breaks=1
Reports error, RHS evaluations, windows and warm wall time.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, cluster, reference, run_stream

T = 1.0
rows = []
Ns = [int(a) for a in sys.argv[1:]] or [3, 4, 6, 8, 12, 16]
for N in Ns:
    p = cluster(N)
    tout = np.linspace(0.0, T, 9)
    for hist in ("ballistic", "quadratic"):
        ref = reference(p, T, tout, tol=1e-14, history=hist, dependent_lags=True)["R"][-1]
        comp = reference(p, T, tout, tol=1e-10, history=hist, dependent_lags=True)
        e = float(np.max(np.abs(comp["R"][-1] - ref)))
        rows.append(dict(N=N, solver="DelayDiffEq-Vern9+lags tol1e-10", history=hist, err=e,
                         rhs=comp["nf"], wall=comp["wall"]))
        print(f"N={N:3d} {hist:9s} DDE tol1e-10 err={e:8.1e} nf={comp['nf']:6d} wall={comp['wall']:.3f}s", flush=True)
        variants = [("A", 0, None)] if hist == "ballistic" else [("B", 0, 1e-10), ("C", 1, 1e-10)]
        M = 16
        for tag, L, at in variants:
            r = run_stream(p, T, T / 4, M, history=hist, track_breaks=L, accuracy_tol=at,
                           min_span_frac=1e-3)
            err = float(np.max(np.abs(r["R"] - ref)))
            rows.append(dict(N=N, solver=f"WR-{tag}", history=hist, track=L, err=err,
                             rhs=r["iters"] * M, windows=r["n_win"], wall=r["wall"],
                             converged=r["converged"]))
            print(f"N={N:3d} {hist:9s} WR-{tag} L={L} err={err:8.1e} rhs={r['iters']*M:6d} "
                  f"windows={r['n_win']:4d} wall={r['wall']:.3f}s conv={r['converged']}", flush=True)
    json.dump(rows, open(os.path.join(ROOT, "research/results/e7_scaling.json"), "w"), indent=1)

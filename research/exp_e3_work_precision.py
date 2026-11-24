"""E3 -- work-precision: windowed Picard-Chebyshev vs DelayDiffEq.jl.

Error = max |r(T) - r_ref(T)|, r_ref = DelayDiffEq Vern9 in 128-bit BigFloat
at tol 1e-18. Cost = (a) RHS evaluations (one full N-body force field at one
time point, light-cone solves included -- comparable across codes) and (b) warm
wall time (Python/NumPy vs Julia: implementation-dependent, report, do not lean on).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary, triple, reference, run_stream

rows = []
cases = [("binary-naive", binary(c=8.0, retardation="naive"), 2 * (2 * np.pi / np.sqrt(2)), 2),
         ("binary-aberration", binary(c=8.0, retardation="aberration"), 2 * (2 * np.pi / np.sqrt(2)), 3),
         ("triple-naive", triple(c=18.0, retardation="naive"), 6.0, 2)]
for name, p, T, L in cases:
    tout = np.linspace(0.0, T, 65)
    ref = reference(p, T, tout, tol=1e-18, bigfloat=True)["R"][-1]
    for alg in ("Vern9", "Vern7", "Tsit5"):
        for tol in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
            r = reference(p, T, tout, tol=tol, alg=alg)
            err = float(np.max(np.abs(r["R"][-1] - ref)))
            rows.append(dict(case=name, solver=f"DelayDiffEq-{alg}", tol=tol, err=err,
                             rhs=r["nf"], wall=r["wall"], retcode=r["retcode"]))
            print(f"{name:18s} DDE-{alg:5s} tol={tol:.0e} err={err:8.1e} nf={r['nf']:7d} wall={r['wall']:.3f}s", flush=True)
    for nwin in (4, 8, 16):
        for M in (12, 16, 24, 32, 48):
            for ptol in (1e-8, 1e-11, 1e-14):
                r = run_stream(p, T, T / nwin, M, track_breaks=L, tol=ptol)
                err = float(np.max(np.abs(r["R"] - ref)))
                rows.append(dict(case=name, solver="WR-Chebyshev", W=T / nwin, M=M, picard_tol=ptol,
                                 track=L, err=err, rhs=r["iters"] * M, iters=r["iters"],
                                 windows=r["n_win"], wall=r["wall"]))
                print(f"{name:18s} WR W=T/{nwin:<2d} M={M:2d} ptol={ptol:.0e} err={err:8.1e} "
                      f"rhs={r['iters']*M:6d} wall={r['wall']:.3f}s", flush=True)
json.dump(rows, open(os.path.join(ROOT, "research/results/e3_work_precision.json"), "w"), indent=1)

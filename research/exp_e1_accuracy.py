"""E1 -- accuracy of windowed Picard-Chebyshev vs an independent SD-DDE solver.

Reference: DelayDiffEq.jl MethodOfSteps(Vern9) in 128-bit BigFloat at tol 1e-18;
`ref_err` = error of the Float64 DelayDiffEq run at tol 1e-13. Engine: PiecewiseVolterraStream,
undamped Picard, tol 1e-13. Error = max |r_engine(T) - r_ref(T)| over bodies.

Sweeps: nodes per window M (spectral convergence?), window length W, and initial
history (ballistic vs C^2 'quadratic') to expose the DDE breaking-point effect.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary, triple, reference, run_stream

OUT = os.path.join(ROOT, "research", "results")
os.makedirs(OUT, exist_ok=True)


def main(quick=False):
    rows = []
    cases = [("binary-naive", binary(c=8.0, retardation="naive"), 2 * (2 * np.pi / np.sqrt(2))),
             ("binary-aberration", binary(c=8.0, retardation="aberration"), 2 * (2 * np.pi / np.sqrt(2))),
             ("binary-none", binary(c=8.0, retardation="none"), 2 * (2 * np.pi / np.sqrt(2))),
             ("triple-naive", triple(c=18.0, retardation="naive"), 6.0)]
    Ms = (8, 12, 16, 24, 32, 48, 64)
    for name, p, T in cases:
        tout = np.linspace(0.0, T, 65)
        for hist in ("ballistic", "quadratic"):
            if p.retardation == "none" and hist == "quadratic":
                continue
            t0 = time.perf_counter()
            ref = reference(p, T, tout, tol=1e-18, history=hist, bigfloat=True)
            ref_lo = reference(p, T, tout, tol=1e-13, history=hist)
            # error of the double-precision DelayDiffEq run at tol 1e-13
            ref_err = float(np.max(np.abs(ref["R"][-1] - ref_lo["R"][-1])))
            print(f"[{name} {hist}] ref nf={ref['nf']} wall={ref['wall']:.2f}s "
                  f"DDE(f64,1e-13) err={ref_err:.2e} ({time.perf_counter()-t0:.1f}s)")
            for nwin in (4, 8, 16, 32):
                W = T / nwin
                for M in Ms:
                    r = run_stream(p, T, W, M, history=hist)
                    err = float(np.max(np.abs(r["R"] - ref["R"][-1])))
                    rows.append(dict(case=name, history=hist, T=T, n_windows=nwin, W=W, M=M,
                                     err=err, ref_err=ref_err, iters=r["iters"],
                                     rhs_evals=r["iters"] * M, ref_nf=ref["nf"],
                                     wall=r["wall"], ref_wall=ref["wall"],
                                     converged=r["converged"], clamped=r["clamped"]))
                    print(f"   W=T/{nwin:<3d} M={M:3d} err={err:9.2e} iters={r['iters']:5d} "
                          f"rhs={r['iters']*M:7d} conv={r['converged']} wall={r['wall']:.2f}s")
            with open(os.path.join(OUT, "e1_accuracy.json"), "w") as f:
                json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

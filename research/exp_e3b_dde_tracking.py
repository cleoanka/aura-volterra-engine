"""E3b -- DelayDiffEq WITH dependent_lags discontinuity tracking (fairness check)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary, triple, reference
rows = []
cases = [("binary-naive", binary(c=8.0, retardation="naive"), 2 * (2 * np.pi / np.sqrt(2))),
         ("binary-aberration", binary(c=8.0, retardation="aberration"), 2 * (2 * np.pi / np.sqrt(2))),
         ("triple-naive", triple(c=18.0, retardation="naive"), 6.0)]
for name, p, T in cases:
    tout = np.linspace(0.0, T, 65)
    ref = reference(p, T, tout, tol=1e-18, bigfloat=True)["R"][-1]
    for alg in ("Vern9", "Vern7"):
        for tol in (1e-8, 1e-10, 1e-12, 1e-14):
            try:
                r = reference(p, T, tout, tol=tol, alg=alg, dependent_lags=True)
            except Exception as e:
                print(name, alg, tol, "FAILED", str(e)[-300:]); continue
            err = float(np.max(np.abs(r["R"][-1] - ref)))
            rows.append(dict(case=name, solver=f"DelayDiffEq-{alg}+lags", tol=tol, err=err,
                             rhs=r["nf"], wall=r["wall"], retcode=r["retcode"]))
            print(f"{name:18s} {alg}+lags tol={tol:.0e} err={err:8.1e} nf={r['nf']:6d} wall={r['wall']:.3f}s {r['retcode']}", flush=True)
json.dump(rows, open(os.path.join(ROOT, "research/results/e3b_dde_tracking.json"), "w"), indent=1)

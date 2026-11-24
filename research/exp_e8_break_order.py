"""E8 -- empirical check of the breaking-point order rule.

One window [0, W] that contains exactly one breaking point (xi_1 ~ tau, W < 2 tau),
no tracking, M swept; error at W vs the 128-bit DelayDiffEq reference. A jump in
x^(k) inside a Chebyshev window gives algebraic convergence; the fitted slope of
log(err) vs log(M) should be ordered like the predicted jump order:

    naive + ballistic phi    : x'''' jumps  (k = 4)
    naive + C^2 phi          : k = 5
    aberration + ballistic   : k = 3
    aberration + C^2         : k = 4
(rule: k = k_phi + (2 - q), k_phi = 2 ballistic / 3 C^2, q = 0 naive / 1 aberration)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary, reference, run_stream

W = 0.2                                   # tau ~ 0.1245 (c = 8): one break inside
Ms = (8, 12, 16, 24, 32, 48, 64, 96)
rows = []
for model in ("naive", "aberration"):
    p = binary(c=8.0, retardation=model)
    for hist, k_phi in (("ballistic", 2), ("quadratic", 3)):
        ref = reference(p, W, np.linspace(0, W, 3), tol=1e-20, history=hist, bigfloat=True)["R"][-1]
        errs = []
        for M in Ms:
            r = run_stream(p, W, W, M, history=hist, tol=1e-15)
            errs.append(float(np.max(np.abs(r["R"] - ref))))
        ok = [(m, e) for m, e in zip(Ms, errs) if e > 1e-14]      # above round-off
        slope = float(np.polyfit(np.log([m for m, _ in ok]), np.log([e for _, e in ok]), 1)[0]) \
            if len(ok) >= 3 else float("nan")
        k_pred = k_phi + (2 - (1 if model == "aberration" else 0))
        rows.append(dict(model=model, history=hist, k_pred=k_pred, slope=slope,
                         Ms=list(Ms), errs=errs))
        print(f"{model:10s} {hist:9s} predicted k={k_pred}  fitted slope={slope:6.2f}  "
              + " ".join(f"{e:.1e}" for e in errs), flush=True)
json.dump(rows, open(os.path.join(ROOT, "research/results/e8_break_order.json"), "w"), indent=1)

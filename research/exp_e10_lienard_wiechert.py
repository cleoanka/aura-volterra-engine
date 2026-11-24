"""E10 -- classical electrodynamic two-body problem (Lienard-Wiechert, relativistic).

'Classical positronium': charges +1/-1, unit masses, Coulomb-circular start, c = 8.
The radiation field depends on the source ACCELERATION at the retarded time, so
this is a NEUTRAL state-dependent-delay system (q = 2 in the break-order rule:
jumps never smooth out -> every level must be tracked).

(a) engine error vs tracking level L and M (vs 128-bit DelayDiffEq, neutral=true)
(b) DelayDiffEq (Float64, Vern9/Vern7, with and without dependent_lags) best error
(c) c -> infinity: identical to Newtonian gravity binary (checked in tests)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, lw_binary, reference, run_stream

T = 1.0
p = lw_binary(c=8.0)
tout = np.linspace(0, T, 5)
ref = reference(p, T, tout, tol=1e-18, bigfloat=True)["R"][-1]
rows = []
print("(a) engine: error vs tracking level (W = T/4)")
for L in (0, 1, 2, 4, 6, 8, 50):
    line = []
    for M in (16, 24, 32, 48):
        r = run_stream(p, T, T / 4, M, track_breaks=L)
        err = float(np.max(np.abs(r["R"] - ref)))
        rows.append(dict(part="a", L=L, M=M, err=err, rhs=r["iters"] * M, windows=r["n_win"],
                         wall=r["wall"]))
        line.append(f"M={M}:{err:.1e}")
    print(f"   L={L:3d} windows={r['n_win']:3d}  " + "  ".join(line), flush=True)
print("(b) DelayDiffEq neutral (Float64)")
for alg in ("Vern9", "Vern7"):
    for lags in (False, True):
        for tol in (1e-8, 1e-10, 1e-12, 1e-14):
            r = reference(p, T, tout, tol=tol, alg=alg, dependent_lags=lags)
            err = float(np.max(np.abs(r["R"][-1] - ref)))
            rows.append(dict(part="b", alg=alg, lags=lags, tol=tol, err=err, rhs=r["nf"],
                             wall=r["wall"]))
            print(f"   {alg}{'+lags' if lags else '     '} tol={tol:.0e} err={err:.1e} nf={r['nf']} "
                  f"wall={r['wall']:.3f}s", flush=True)
json.dump(rows, open(os.path.join(ROOT, "research/results/e10_lw.json"), "w"), indent=1)

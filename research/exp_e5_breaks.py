"""E5 -- breaking-point-aligned windows. Error vs reference for track_breaks levels."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary, triple, reference, run_stream
rows = []
cases = [("binary-naive", binary(c=8.0, retardation="naive"), 2 * (2 * np.pi / np.sqrt(2))),
         ("binary-aberration", binary(c=8.0, retardation="aberration"), 2 * (2 * np.pi / np.sqrt(2))),
         ("triple-naive", triple(c=18.0, retardation="naive"), 6.0)]
for name, p, T in cases:
    for hist in ("ballistic", "quadratic"):
        ref = reference(p, T, np.linspace(0.0, T, 65), tol=1e-18, history=hist, bigfloat=True)["R"][-1]
        for nwin in (4, 8, 16):
            for M in (16, 32, 64):
                line = []
                for tb in (0, 1, 2, 3):
                    r = run_stream(p, T, T / nwin, M, history=hist, track_breaks=tb)
                    err = float(np.max(np.abs(r["R"] - ref)))
                    rows.append(dict(case=name, history=hist, n_windows=nwin, M=M, track=tb,
                                     err=err, iters=r["iters"], rhs=r["iters"] * M,
                                     windows=r["n_win"], splits=r["splits"]))
                    line.append(f"L{tb}:{err:8.1e}({r['n_win']}w,{r['iters']*M}rhs)")
                print(f"{name:18s} {hist:9s} W=T/{nwin:<2d} M={M:2d} " + "  ".join(line), flush=True)
json.dump(rows, open(os.path.join(ROOT, "research/results/e5_breaks.json"), "w"), indent=1)

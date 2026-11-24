"""E4 -- ablation: v0.3.0 history (linearly resampled delay buffer; ballistic past
per chained segment) vs the spectral segment history. Same problem, same windows,
same reference. Usage: python exp_e4_history_ablation.py <path-to-old-checkout>"""
import importlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from problems import ROOT, binary, triple, reference, consistent_A0

def load_engine(path):
    for k in [k for k in sys.modules if k.startswith("aura_engine")]:
        del sys.modules[k]
    sys.path.insert(0, path)
    mod = importlib.import_module("aura_engine.physics")
    sys.path.pop(0)
    return mod

def run(mod, p, T, W, M, old):
    kw = dict(G=p.G, c=p.c, softening=p.softening, use_retardation=True, use_pn=False,
              relaxation=1.0, window_span=W, window_samples=M, iterations=200,
              scheme="chebyshev", accelerator="picard", retardation=p.retardation,
              pn="none", tol=1e-13, adaptive=False)
    s = mod.PiecewiseVolterraStream(p.bodies(), **kw)
    for _ in range(int(round(T / W))):
        s.step()
    return s.R_epoch.copy()

def batch(mod, p, T, M, nseg):
    s = mod.VolterraSystem(p.bodies(), G=p.G, c=p.c, softening=p.softening, t_span=T,
                           n_samples=M * nseg, relaxation=1.0, scheme="chebyshev",
                           accelerator="picard", retardation=p.retardation, pn="none")
    return s.propagate(200, n_segments=nseg, record_history=False, tol=1e-13).trajectory[-1]

old_path = sys.argv[1]
rows = []
cases = [("binary-naive", binary(c=8.0, retardation="naive"), 2 * (2 * np.pi / np.sqrt(2))),
         ("triple-naive", triple(c=18.0, retardation="naive"), 6.0)]
for name, p, T in cases:
    ref = reference(p, T, np.linspace(0.0, T, 65), tol=1e-18, bigfloat=True)["R"][-1]
    for label, path in (("v0.3.0", old_path), ("spectral-history", ROOT)):
        mod = load_engine(path)
        for nwin in (8, 16, 32):
            for M in (16, 32, 48):
                e_s = float(np.max(np.abs(run(mod, p, T, T / nwin, M, label) - ref)))
                e_b = float(np.max(np.abs(batch(mod, p, T, M, nwin) - ref)))
                rows.append(dict(case=name, engine=label, n_windows=nwin, M=M,
                                 stream_err=e_s, batch_segmented_err=e_b))
                print(f"{name} {label:16s} W=T/{nwin:<2d} M={M:2d} stream={e_s:9.2e} "
                      f"segmented-batch={e_b:9.2e}")
json.dump(rows, open(os.path.join(ROOT, "research/results/e4_history_ablation.json"), "w"), indent=1)

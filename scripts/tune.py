"""Diagnostic sweep: relax every preset and report convergence + conservation.

Run:  python scripts/tune.py
Prints per-preset residual decay, Newtonian energy drift, and trajectory bounds
so preset parameters (t_span, n_samples, relaxation, iterations) can be tuned.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura_engine import presets  # noqa: E402


def report(name: str) -> None:
    cfg = presets.get_preset(name)
    sysm = cfg.build()
    t0 = time.perf_counter()
    res = sysm.propagate(cfg.iterations, n_segments=cfg.n_segments, record_history=False)
    dt = time.perf_counter() - t0

    R = res.trajectory / cfg.length_unit
    span = np.ptp(R.reshape(-1, 3), axis=0)  # bounding-box extent per axis
    r0res = res.residuals[0] if res.residuals else float("nan")
    print(f"\n=== {name} ===")
    print(f"  bodies={sysm.N}  n={sysm.n}  segments={cfg.n_segments}"
          f"  relax={sysm.w}  retard={sysm.use_retardation}  pn={sysm.use_pn}")
    print(f"  time: {dt*1000:6.1f} ms   converged={res.converged}")
    print(f"  residual: first={r0res:.3e}  last(worst-seg)={res.residuals[-1]:.3e}")
    print(f"  energy drift (max-min)/|mean| = {res.energy_drift:.3e}")
    print(f"  bbox extent [{cfg.unit_label}]: "
          f"x={span[0]:.3f} y={span[1]:.3f}  (view_radius={cfg.view_radius})")
    finite = np.isfinite(R).all()
    print(f"  all-finite={finite}")


if __name__ == "__main__":
    names = sys.argv[1:] or presets.PRESET_NAMES
    for nm in names:
        report(nm)

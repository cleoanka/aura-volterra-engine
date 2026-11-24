"""E9 -- long-time behaviour (20 orbits): naive vs aberration retardation.

Binary, c in {8, 32}. Engine: breaking-point tracking (L2 naive / L3 aberration),
C^2-free ballistic history, error control accuracy_tol=1e-12. Measures the secular
rates of total energy E and angular momentum L (linear fit over the run) and the
final-state difference to DelayDiffEq (Vern9 + dependent_lags, tol 1e-14, Float64).

Expectation: naive -> Laplace secular growth dL/dt ~ Omega/(2c) (E6); aberration
x_j(t-tau) + tau v_j(t-tau) = x_j(t) - tau^2 a_j/2 + O(tau^3): correction is O(beta^2)
and mostly radial -> no O(beta) secular torque.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary, reference, stream

ORBITS = 20
rows = []
for c in (8.0, 32.0):
    for model, L in (("naive", 2), ("aberration", 3)):
        p = binary(c=c, retardation=model)
        P = 2 * np.pi / p.omega
        T = ORBITS * P
        s = stream(p, P / 4, 24, track_breaks=L, accuracy_tol=1e-12, min_span_frac=1e-3)
        ts, Es, Ls = [], [], []
        t0 = time.perf_counter()
        while s.t_epoch < T - 1e-12:
            s.window_span = min(P / 4, T - s.t_epoch)
            w = s.step()
            R, V = w.trajectory, w.velocity
            ke = 0.5 * np.einsum("knd,knd->k", V, V)              # unit masses
            d = np.linalg.norm(R[:, 0] - R[:, 1], axis=1)
            Es.append(ke - 1.0 / d)
            Ls.append(np.cross(R, V)[:, :, 2].sum(1))
            ts.append(w.t)
        wall = time.perf_counter() - t0
        t, E, Lm = np.concatenate(ts), np.concatenate(Es), np.concatenate(Ls)
        dE = float(np.polyfit(t, E, 1)[0])
        dL = float(np.polyfit(t, Lm, 1)[0])
        ref = reference(p, T, np.linspace(0, T, 9), tol=1e-14, dependent_lags=True)["R"][-1]
        err = float(np.max(np.abs(s.R_epoch - ref)))
        rows.append(dict(c=c, model=model, T=T, dE_dt=dE, dL_dt=dL,
                         rel_E_change=float((E[-1] - E[0]) / abs(E[0])),
                         rel_L_change=float((Lm[-1] - Lm[0]) / abs(Lm[0])),
                         err_vs_dde=err, wall=wall, windows=s.window_index))
        print(f"c={c:4.0f} {model:10s} dE/dt={dE:+.3e} dL/dt={dL:+.3e} "
              f"E change={rows[-1]['rel_E_change']:+.2e} L change={rows[-1]['rel_L_change']:+.2e} "
              f"|engine-DDE|={err:.1e} windows={s.window_index} wall={wall:.1f}s", flush=True)
json.dump(rows, open(os.path.join(ROOT, "research/results/e9_longtime.json"), "w"), indent=1)

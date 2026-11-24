"""Benchmark problems + shared harness for the SD-DDE study (see research/NOTES.md).

All problems are nondimensional (G = 1). The retarded model is a *model problem*
for state-dependent-delay (SD-DDE) numerics, not a claim about real gravity.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aura_engine.physics import Body, PiecewiseVolterraStream  # noqa: E402

JULIA_ENV = os.path.expanduser("~/.julia-envs/dde")
JL_SCRIPT = os.path.join(ROOT, "research", "reference_dde.jl")
CACHE = os.path.join(ROOT, "research", "cache")


@dataclass
class Problem:
    name: str
    masses: list
    R0: np.ndarray
    V0: np.ndarray
    c: float
    softening: float
    retardation: str = "naive"
    G: float = 1.0
    omega: float = 1.0            # characteristic angular frequency (for Omega*W)

    def bodies(self):
        return [Body(f"b{k}", m, self.R0[k], self.V0[k]) for k, m in enumerate(self.masses)]

    def with_(self, **kw):
        d = dict(self.__dict__); d.update(kw)
        return Problem(**d)


def binary(c=8.0, retardation="naive", d=1.0):
    """Equal-mass binary, circular for instantaneous gravity. Omega = sqrt(2/d^3)."""
    v = 0.5 * np.sqrt(2.0 / d)
    R0 = np.array([[d / 2, 0, 0], [-d / 2, 0, 0]], float)
    V0 = np.array([[0, v, 0], [0, -v, 0]], float)
    return Problem("binary", [1.0, 1.0], R0, V0, c, 0.0, retardation, omega=np.sqrt(2.0 / d**3))


def triple(c=18.0, retardation="naive", softening=0.12):
    """Chaotic equal-mass triple (the engine's 'Deep Chaos' preset ICs)."""
    s3 = np.sqrt(3.0)
    R0 = np.array([[0, 2 / s3, 0], [-1, -1 / s3, 0], [1, -1 / s3, 0]], float)
    V0 = np.array([[-0.5868986283848345, 0, 0],
                   [0.29344931419241704, -0.5082691216275094, 0],
                   [0.29344931419241743, 0.5082691216275094, 0]], float)
    return Problem("triple", [1.0, 1.0, 1.0], R0, V0, c, softening, retardation, omega=0.6)


# --------------------------------------------------------------------------- #
def consistent_A0(p: Problem, sweeps=60):
    """Force at t0 under the retarded model *with the quadratic history it
    induces* (fixed point A0 = f(phi_A0)), so the solution is C^2 at t0."""
    from aura_engine.history import BallisticHistory
    N = len(p.masses)
    A0 = np.zeros((N, 3))
    for _ in range(sweeps):
        phi = BallisticHistory(p.R0, p.V0, 0.0, A0=A0)
        A = np.zeros((N, 3))
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if p.retardation == "none":
                    s = p.R0[j]
                else:
                    tau = np.linalg.norm(p.R0[j] - p.R0[i]) / p.c
                    for _ in range(30):
                        rj, vj = phi(j, np.array([-tau]))
                        tau = np.linalg.norm(p.R0[i] - rj[0]) / p.c
                    rj, vj = phi(j, np.array([-tau]))
                    s = rj[0] + (tau * vj[0] if p.retardation == "aberration" else 0.0)
                d = s - p.R0[i]
                A[i] += p.G * p.masses[j] * d / (d @ d + p.softening ** 2) ** 1.5
        if np.max(np.abs(A - A0)) < 1e-16:
            break
        A0 = A
    return A0


def reference(p: Problem, T: float, tout, tol=1e-13, alg="Vern9", use_cache=True,
              history="ballistic", bigfloat=False, dependent_lags=False):
    """Solve with DelayDiffEq.jl (independent code, independent method)."""
    os.makedirs(CACHE, exist_ok=True)
    spec = dict(masses=p.masses, R0=p.R0.tolist(), V0=p.V0.tolist(), G=p.G, c=p.c,
                softening=p.softening, retardation=p.retardation, t0=0.0, T=float(T),
                tol=tol, alg=alg, tout=[float(x) for x in tout], history=history)
    if bigfloat:
        spec["bigfloat"] = True
    if dependent_lags:
        spec["dependent_lags"] = True
    if history == "quadratic":
        spec["A0"] = consistent_A0(p).tolist()
    import hashlib
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    out = os.path.join(CACHE, f"ref_{key}.json")
    if not (use_cache and os.path.exists(out)):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
        subprocess.run(["julia", f"--project={JULIA_ENV}", JL_SCRIPT, f.name, out],
                       check=True, capture_output=True, text=True)
    with open(out) as f:
        r = json.load(f)
    r["R"] = np.array(r["R"]).reshape(len(tout), -1, 3)
    r["V"] = np.array(r["V"]).reshape(len(tout), -1, 3)
    return r


def stream(p: Problem, W: float, nodes: int, *, iterations=200, tol=1e-13,
           accelerator="picard", relaxation=1.0, scheme="chebyshev", pn="none",
           history="ballistic", track_breaks=0, accuracy_tol=None, min_span_frac=0.05):
    return PiecewiseVolterraStream(
        p.bodies(), G=p.G, c=p.c, softening=p.softening, use_retardation=True,
        use_pn=False, relaxation=relaxation, window_span=W, window_samples=nodes,
        iterations=iterations, scheme=scheme, accelerator=accelerator,
        retardation=p.retardation, pn=pn, tol=tol, adaptive=accuracy_tol is not None,
        initial_A0=consistent_A0(p) if history == "quadratic" else None,
        track_breaks=track_breaks, accuracy_tol=accuracy_tol, min_span_frac=min_span_frac)


def run_stream(p: Problem, T: float, W: float, nodes: int, **kw):
    """March windows to T; return final (R, V), total iterations, wall, converged."""
    s = stream(p, W, nodes, **kw)
    iters, ok, n_win = 0, True, 0
    t0 = time.perf_counter()
    while s.t_epoch < T - 1e-12 * max(1.0, T):
        s.window_span = min(W, T - s.t_epoch)       # land exactly on T
        w = s.step()
        n_win += 1
        iters += w.iterations
        ok = ok and w.converged
    wall = time.perf_counter() - t0
    return dict(R=s.R_epoch.copy(), V=s.V_epoch.copy(), t=s.t_epoch, iters=iters,
                wall=wall, converged=ok, n_win=n_win, clamped=s.history.clamped,
                splits=s.break_splits, M=nodes)


def pair_evals_engine(iters, nodes, N):
    """Right-hand-side evaluations counted like DelayDiffEq's ``nf``: one full
    force field per node per iteration (+1 clean-velocity pass per window is
    ignored, < 1/iters). Each one costs N(N-1) pair forces + light-cone solves."""
    return iters * nodes




def cluster(N, c=10.0, retardation="naive", softening=0.05, seed=1):
    """Bound random cluster: N equal masses (total 1) in the unit ball, small
    random velocities (virial ratio ~0.5), zero total momentum."""
    rng = np.random.default_rng(seed)
    R0 = rng.uniform(-1, 1, (N * 4, 3))
    R0 = R0[np.linalg.norm(R0, axis=1) <= 1.0][:N]
    R0 -= R0.mean(0)
    V0 = rng.normal(0, 0.35, (N, 3))
    V0 -= V0.mean(0)
    return Problem(f"cluster{N}", [1.0 / N] * N, R0, V0, c, softening, retardation, omega=1.0)

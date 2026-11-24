"""
aura_engine.history
====================

History storage for the *state-dependent-delay* (retarded) Volterra solve.

A retarded force needs the source state at ``t - tau(r)`` where ``tau`` depends on
the (unknown) solution itself. Before the current window that past is already
converged; this module stores it so it can be read back **at the same order of
accuracy as the solver** (spectral for Chebyshev windows) and in ``O(1)`` memory:

* :class:`BallisticHistory` -- the *initial* history ``phi(t)`` for ``t < t0``
  (a DDE needs one; we use the free-motion extrapolation ``r0 + v0 (t - t0)``).
* :class:`SegmentHistory` -- the converged windows themselves, kept as their own
  nodal data and interpolated with the window's own interpolant (barycentric on
  CGL nodes, cubic Hermite on uniform nodes). Windows that end before
  ``t_now - span`` are pruned. No resampling, so no accuracy is lost at window
  joins (the previous implementation linearly resampled the buffer, which
  capped retarded lookups at second order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import barycentric_interpolate


def _hermite(t_nodes, f, df, tq):
    idx = np.clip(np.searchsorted(t_nodes, tq) - 1, 0, len(t_nodes) - 2)
    t0, t1 = t_nodes[idx], t_nodes[idx + 1]
    h = t1 - t0
    s = ((tq - t0) / h)[:, None]
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return (h00 * f[idx] + h10 * h[:, None] * df[idx]
            + h01 * f[idx + 1] + h11 * h[:, None] * df[idx + 1])


class BallisticHistory:
    """Initial history ``phi`` for ``t < t0``.

    Default: free motion ``r0 + v0 (t - t0)``. With ``A0`` (the force at ``t0``)
    it is the constant-acceleration arc ``r0 + v0 s + A0 s^2 / 2``, which makes
    the solution ``C^2`` at ``t0`` and pushes the DDE breaking-point jumps two
    derivatives higher (see research/NOTES.md, "breaking points").
    """

    def __init__(self, R0, V0, t0: float = 0.0, A0=None):
        self.R0 = np.asarray(R0, dtype=np.float64)
        self.V0 = np.asarray(V0, dtype=np.float64)
        self.A0 = None if A0 is None else np.asarray(A0, dtype=np.float64)
        self.t0 = float(t0)

    def __call__(self, j: int, tq: np.ndarray):
        dt = (np.asarray(tq) - self.t0)[:, None]
        if self.A0 is None:
            return self.R0[j] + self.V0[j] * dt, np.broadcast_to(self.V0[j], (len(tq), 3)).copy()
        return (self.R0[j] + self.V0[j] * dt + 0.5 * self.A0[j] * dt**2,
                self.V0[j] + self.A0[j] * dt)


@dataclass
class _Segment:
    ta: float
    tb: float
    t: np.ndarray           # (n,) global node times, ascending
    RV: np.ndarray          # (n, N, 6)
    weights: np.ndarray | None   # barycentric weights (CGL) or None (Hermite)


class SegmentHistory:
    """Converged past stored window-by-window, read back with each window's
    own interpolant. ``initial`` answers queries before the first window."""

    def __init__(self, initial: BallisticHistory):
        self.initial = initial
        self.segments: list[_Segment] = []
        self.clamped = 0                     # queries that fell into pruned past

    @property
    def t_start(self) -> float:
        return self.segments[0].ta if self.segments else self.initial.t0

    def append(self, t_nodes, R, V, weights=None) -> None:
        t_nodes = np.asarray(t_nodes, dtype=np.float64)
        RV = np.concatenate([R, V], axis=2).astype(np.float64, copy=True)
        self.segments.append(_Segment(float(t_nodes[0]), float(t_nodes[-1]),
                                      t_nodes.copy(), RV, weights))

    def prune(self, t_keep: float) -> None:
        """Drop windows that end before ``t_keep`` (keep at least one)."""
        while len(self.segments) > 1 and self.segments[0].tb < t_keep:
            self.segments.pop(0)

    def span(self) -> float:
        return (self.segments[-1].tb - self.segments[0].ta) if self.segments else 0.0

    def __call__(self, j: int, tq: np.ndarray):
        tq = np.asarray(tq, dtype=np.float64)
        r = np.empty((len(tq), 3))
        v = np.empty((len(tq), 3))
        done = np.zeros(len(tq), dtype=bool)
        # newest segment first: a join time belongs to the newer window
        for seg in reversed(self.segments):
            m = (~done) & (tq >= seg.ta) & (tq <= seg.tb)
            if not m.any():
                continue
            if seg.weights is not None:
                rv = barycentric_interpolate(seg.t, seg.RV[:, j, :], seg.weights, tq[m])
            else:
                rv = np.concatenate([
                    _hermite(seg.t, seg.RV[:, j, :3], seg.RV[:, j, 3:], tq[m]),
                    _hermite_linear_v(seg.t, seg.RV[:, j, 3:], tq[m])], axis=1)
            r[m], v[m] = rv[:, :3], rv[:, 3:]
            done |= m
        rest = ~done
        if rest.any():
            before_all = rest & (tq < self.initial.t0)
            if before_all.any():
                r[before_all], v[before_all] = self.initial(j, tq[before_all])
            pruned = rest & ~before_all
            if pruned.any():                 # past was pruned: clamp to oldest kept
                self.clamped += int(pruned.sum())
                seg = self.segments[0]
                r[pruned] = seg.RV[0, j, :3]
                v[pruned] = seg.RV[0, j, 3:]
        return r, v


def _hermite_linear_v(t_nodes, V, tq):
    out = np.empty((len(tq), V.shape[1]))
    for d in range(V.shape[1]):
        out[:, d] = np.interp(tq, t_nodes, V[:, d])
    return out

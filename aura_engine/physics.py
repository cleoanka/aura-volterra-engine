"""
aura_engine.physics
====================

The Volterra N-Body engine: a *global waveform-relaxation* orbital propagator.

The equations of motion ``r'' = a(r)`` are recast as a Volterra integral equation

    r_i(t) = r_i(0) + v_i(0) t + \\int_0^t (t - s) a_i[r](s) ds

and solved by relaxation over a whole time window at once. This module provides a
research-grade implementation with several selectable methods:

Quadrature (``scheme``)
    * ``"chebyshev"`` -- CGL nodes + spectral Chebyshev integration (MCPI):
      exponential accuracy, large convergence radius (default).
    * ``"trapezoid"`` -- uniform grid, 2nd order (kept for comparison).

Fixed-point solver (``accelerator``)
    * ``"anderson"`` -- Anderson acceleration of the Picard map (default): far
      fewer iterations than plain relaxation.
    * ``"picard"``   -- classic damped Picard, ``R <- (1-w)R + w G(R)``.

Retarded gravity (``retardation``)
    * ``"aberration"`` -- finite-``c`` with the leading velocity term
      ``r_ret + tau v_ret`` that cancels aberration (Carlip 2000). Nearly
      momentum-conserving -- the *physical* finite-speed model (default when on).
    * ``"naive"`` -- retarded position only. Breaks Newton's 3rd law on purpose;
      use it to *see* the spurious momentum/energy drift.
    * ``"none"`` -- instantaneous.
    The light-cone equation ``t_ret = t - |r_i - r_j(t_ret)|/c`` is solved by
    Newton iteration.

Post-Newtonian (``pn``)
    * ``"eih"``     -- full Einstein-Infeld-Hoffmann 1PN n-body (momentum
      conserving, correct for comparable masses).
    * ``"reduced"`` -- cheap Schwarzschild pairwise term.
    * ``"none"``.

Interpolation of the retarded past is barycentric-Lagrange (spectral) on CGL
nodes and cubic-Hermite ``(r, v)`` elsewhere.

Every solve reports conserved-quantity drift (energy, **linear momentum**,
**angular momentum**) so the fidelity of each model is measurable, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.polynomial.chebyshev as _cheb

from .operators import (
    apply_operator,
    barycentric_interpolate,
    barycentric_weights_cgl,
    build_scheme,
    chebyshev_tail,
)
from .history import BallisticHistory, SegmentHistory

C_LIGHT = 2.99792458e8


# --------------------------------------------------------------------------- #
# Conserved-quantity diagnostics
# --------------------------------------------------------------------------- #
def newtonian_energy(masses, R, V, G, eps2) -> np.ndarray:
    """Total softened Newtonian energy per time sample for ``(n, N, 3)`` fields."""
    ke = 0.5 * np.einsum("m,kmd,kmd->k", masses, V, V)
    pe = np.zeros(R.shape[0], dtype=np.float64)
    N = R.shape[1]
    for i in range(N):
        for j in range(i + 1, N):
            d = R[:, j, :] - R[:, i, :]
            r = np.sqrt(np.einsum("kd,kd->k", d, d) + eps2)
            pe -= G * masses[i] * masses[j] / r
    return ke + pe


def linear_momentum(masses, V) -> np.ndarray:
    """Total linear momentum per sample, ``(n, 3)``."""
    return np.einsum("m,kmd->kd", masses, V)


def angular_momentum(masses, R, V) -> np.ndarray:
    """Total angular momentum about the origin per sample, ``(n, 3)``."""
    return np.einsum("m,kmd->kd", masses, np.cross(R, V))


def _scalar_drift(e: np.ndarray) -> float:
    mean_e = float(np.mean(e))
    if mean_e != 0.0:
        return float((e.max() - e.min()) / abs(mean_e))
    return float(e.max() - e.min())


def _vector_drift(q: np.ndarray, scale: float) -> float:
    """Max deviation of a (near-)conserved vector series from its initial value,
    normalised by ``scale``."""
    dev = np.linalg.norm(q - q[0], axis=1).max()
    return float(dev / (scale + 1e-30))


# --------------------------------------------------------------------------- #
# Interpolation helpers (retarded-past lookups)
# --------------------------------------------------------------------------- #
def hermite_interpolate(t_nodes, f_vals, df_vals, tq) -> np.ndarray:
    """Cubic-Hermite interpolation of ``(n, 3)`` values with derivatives at ``tq``."""
    tq = np.clip(tq, t_nodes[0], t_nodes[-1])
    idx = np.clip(np.searchsorted(t_nodes, tq) - 1, 0, t_nodes.size - 2)
    ta, tb = t_nodes[idx], t_nodes[idx + 1]
    h = (tb - ta)
    s = (tq - ta) / h
    s2, s3 = s * s, s * s * s
    h00 = (2 * s3 - 3 * s2 + 1)[:, None]
    h10 = (s3 - 2 * s2 + s)[:, None]
    h01 = (-2 * s3 + 3 * s2)[:, None]
    h11 = (s3 - s2)[:, None]
    hc = h[:, None]
    return (h00 * f_vals[idx] + h10 * df_vals[idx] * hc
            + h01 * f_vals[idx + 1] + h11 * df_vals[idx + 1] * hc)


def linear_interpolate(t_nodes, f_vals, tq) -> np.ndarray:
    out = np.empty((tq.size, 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(tq, t_nodes, f_vals[:, d])
    return out


# --------------------------------------------------------------------------- #
# Force models (given already-resolved source states)
# --------------------------------------------------------------------------- #
def newtonian_all_pairs(R, masses, G, eps2) -> np.ndarray:
    """Instantaneous softened Newtonian acceleration field, fully vectorised."""
    diff = R[:, None, :, :] - R[:, :, None, :]        # (n, i, j, 3) = r_j - r_i
    r2 = np.einsum("kijd,kijd->kij", diff, diff)
    N = R.shape[1]
    idx = np.arange(N)
    r2[:, idx, idx] = np.inf                          # no self-interaction
    inv_r3 = (r2 + eps2) ** -1.5
    return G * np.einsum("j,kijd,kij->kid", masses, diff, inv_r3)


def eih_acceleration(R, V, masses, G, c, eps2) -> np.ndarray:
    """Full Einstein-Infeld-Hoffmann 1PN n-body acceleration (harmonic gauge)."""
    n, N, _ = R.shape
    c2 = c * c
    aN = newtonian_all_pairs(R, masses, G, eps2)      # Newtonian a_b for 1PN terms
    A = np.zeros((n, N, 3), dtype=np.float64)

    # pairwise 1/distance cache
    dist = np.zeros((N, N, n), dtype=np.float64)
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            dd = R[:, a, :] - R[:, b, :]
            dist[a, b] = np.sqrt(np.einsum("kd,kd->k", dd, dd) + eps2)

    for a in range(N):
        ra, va = R[:, a, :], V[:, a, :]
        va2 = np.einsum("kd,kd->k", va, va)
        for b in range(N):
            if b == a:
                continue
            rb, vb = R[:, b, :], V[:, b, :]
            rab = ra - rb                              # from b to a
            d = dist[a, b]
            d3 = d * d * d
            sum_ac = sum(masses[cc] / dist[a, cc] for cc in range(N) if cc != a)
            sum_bc = sum(masses[cc] / dist[b, cc] for cc in range(N) if cc != b)
            vb2 = np.einsum("kd,kd->k", vb, vb)
            vavb = np.einsum("kd,kd->k", va, vb)
            rab_vb = np.einsum("kd,kd->k", rab, vb) / d
            rba_ab = np.einsum("kd,kd->k", (rb - ra), aN[:, b, :])
            bracket = (1.0
                       - (4.0 * G / c2) * sum_ac
                       - (G / c2) * sum_bc
                       + va2 / c2 + 2.0 * vb2 / c2 - (4.0 / c2) * vavb
                       - (3.0 / (2.0 * c2)) * rab_vb ** 2
                       + (1.0 / (2.0 * c2)) * rba_ab)
            term1 = (G * masses[b]) * (rb - ra) / d3[:, None] * bracket[:, None]
            rab_dot = np.einsum("kd,kd->k", rab, (4.0 * va - 3.0 * vb))
            term2 = (G * masses[b] / c2) / d3[:, None] * rab_dot[:, None] * (va - vb)
            term3 = (3.5 / c2) * G * masses[b] * aN[:, b, :] / d[:, None]
            A[:, a, :] += term1 + term2 + term3
    return A


def reduced_pn_acceleration(R, V, masses, G, c, eps2) -> np.ndarray:
    """Cheap Schwarzschild pairwise 1PN term (prograde precession)."""
    n, N, _ = R.shape
    A = np.zeros((n, N, 3), dtype=np.float64)
    for i in range(N):
        ri, vi = R[:, i, :], V[:, i, :]
        for j in range(N):
            if i == j:
                continue
            rvec = ri - R[:, j, :]
            r2 = np.einsum("kd,kd->k", rvec, rvec) + eps2
            r = np.sqrt(r2)
            vrel = vi - V[:, j, :]
            v2 = np.einsum("kd,kd->k", vrel, vrel)
            rv = np.einsum("kd,kd->k", rvec, vrel)
            gm = G * masses[j]
            pref = gm / (c ** 2 * r ** 3)
            A[:, i, :] += pref[:, None] * (
                ((4.0 * gm / r) - v2)[:, None] * rvec + 4.0 * rv[:, None] * vrel
            )
    return A


# --------------------------------------------------------------------------- #
# Anderson acceleration of the Picard fixed point
# --------------------------------------------------------------------------- #
class _Anderson:
    """Type-II Anderson acceleration for x = G(x) on flattened state."""

    def __init__(self, depth: int = 5, reg: float = 1e-10):
        self.m = int(depth)
        self.reg = float(reg)
        self._x: list[np.ndarray] = []
        self._g: list[np.ndarray] = []

    def reset(self) -> None:
        self._x.clear()
        self._g.clear()

    def step(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """Given current iterate ``x`` and fixed-point value ``gx = G(x)``."""
        self._x.append(x.copy())
        self._g.append(gx.copy())
        if len(self._x) > self.m + 1:
            self._x.pop(0)
            self._g.pop(0)
        k = len(self._x)
        if k == 1:
            return gx
        F = np.stack([self._g[i] - self._x[i] for i in range(k)], axis=1)  # residuals
        dF = np.diff(F, axis=1)                          # (dim, k-1)
        fk = F[:, -1]
        # solve min || fk - dF gamma ||  (regularised least squares)
        A = dF.T @ dF + self.reg * np.eye(dF.shape[1])
        rhs = dF.T @ fk
        try:
            gamma = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return gx
        G = np.stack(self._g, axis=1)
        dG = np.diff(G, axis=1)
        return self._g[-1] - dG @ gamma


# --------------------------------------------------------------------------- #
# Body / results
# --------------------------------------------------------------------------- #
@dataclass
class Body:
    name: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    color: str = "#7de3ff"
    radius: float = 6.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)


@dataclass
class RelaxationResult:
    t: np.ndarray
    trajectory: np.ndarray
    velocity: np.ndarray
    history: list[np.ndarray] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    energy: Optional[np.ndarray] = None
    energy_drift: float = 0.0
    momentum_drift: float = 0.0
    angmom_drift: float = 0.0
    converged: bool = False
    iterations: int = 0

    @property
    def n_bodies(self) -> int:
        return self.trajectory.shape[1]


@dataclass
class WindowResult:
    index: int
    t: np.ndarray
    trajectory: np.ndarray
    velocity: np.ndarray
    residual: float
    energy_drift: float
    momentum_drift: float
    buffer_span: float
    converged: bool
    span: float
    iterations: int = 0          # Picard/Anderson iterations (all attempts)


# --------------------------------------------------------------------------- #
# Shared engine core (retardation + force assembly + relaxation)
# --------------------------------------------------------------------------- #
class _EngineCore:
    """Physics/method configuration and the acceleration assembler.

    Concrete engines (:class:`VolterraSystem`, :class:`PiecewiseVolterraStream`)
    supply the *time grid*, the *predictor*, and the *past* (how to look up
    source positions before the window start). Everything else is shared.
    """

    def __init__(self, *, G, c, softening, relaxation, scheme, accelerator,
                 retardation, pn, anderson_depth=5):
        self.G = float(G)
        self.c = float(c)
        self.softening = float(softening)
        self.eps2 = float(softening) ** 2
        self.w = float(relaxation)
        self.scheme = scheme
        self.accelerator = accelerator
        self.retardation = retardation      # "none" | "naive" | "aberration"
        self.pn = pn                         # "none" | "reduced" | "eih"
        self.anderson_depth = int(anderson_depth)

    # -- source-state lookup: subclasses implement the "past" branch --------- #
    def _source_state(self, j, R, V, tq):
        raise NotImplementedError

    def _in_window(self, R, V, j, tq):
        """Interpolate source ``j`` inside the current window at ``tq``."""
        if self.scheme == "chebyshev":
            rv = barycentric_interpolate(
                self.t, np.concatenate([R[:, j, :], V[:, j, :]], axis=1), self._bw, tq)
            return rv[:, :3], rv[:, 3:]
        return (hermite_interpolate(self.t, R[:, j, :], V[:, j, :], tq),
                linear_interpolate(self.t, V[:, j, :], tq))

    def _retarded_state(self, R, V, i, j):
        """Resolve the source position body ``i`` feels from body ``j``."""
        ri = R[:, i, :]
        # initial delay from the instantaneous separation
        d0 = R[:, j, :] - ri
        tau = np.sqrt(np.einsum("kd,kd->k", d0, d0)) / self.c
        for _ in range(4):                                # Newton on the light cone
            rj, vj = self._source_state(j, R, V, self.t - tau)
            rel = ri - rj
            dist = np.sqrt(np.einsum("kd,kd->k", rel, rel)) + 1e-30
            g = tau - dist / self.c
            ndir = rel / dist[:, None]
            dgdtau = 1.0 - np.einsum("kd,kd->k", ndir, vj) / self.c
            tau = tau - g / np.where(np.abs(dgdtau) < 1e-6, 1.0, dgdtau)
            tau = np.clip(tau, 0.0, None)
        rj, vj = self._source_state(j, R, V, self.t - tau)
        if self.retardation == "aberration":
            rj = rj + tau[:, None] * vj                   # cancel leading aberration
        return rj

    def _node_times(self):
        """Global times of the current window's nodes."""
        return self.t

    def _sources_at(self, R, V, TQ):
        """Source states for a query-time array ``TQ (n, N_i, N_j)``: body ``j``
        evaluated at ``TQ[:, :, j]``. One interpolation call per *source* body."""
        n, N = TQ.shape[0], TQ.shape[2]
        RJ = np.empty((n, N, N, 3))
        VJ = np.empty((n, N, N, 3))
        for j in range(N):
            rq, vq = self._source_state(j, R, V, TQ[:, :, j].reshape(-1))
            RJ[:, :, j, :] = rq.reshape(n, N, 3)
            VJ[:, :, j, :] = vq.reshape(n, N, 3)
        return RJ, VJ

    def _retarded_field(self, R, V):
        """Retarded Newtonian field, vectorised over all ordered pairs.

        Same light-cone Newton solve and source model as :meth:`_retarded_state`
        (kept as the per-pair reference), but every pair and node at once.
        """
        n, N = R.shape[0], R.shape[1]
        tn = self._node_times()[:, None, None]
        Ri = R[:, :, None, :]                                     # receiver i
        D0 = R[:, None, :, :] - Ri                                # r_j - r_i
        tau = np.sqrt(np.einsum("kijd,kijd->kij", D0, D0)) / self.c
        for _ in range(4):                                        # Newton on the light cone
            RJ, VJ = self._sources_at(R, V, tn - tau)
            rel = Ri - RJ
            dist = np.sqrt(np.einsum("kijd,kijd->kij", rel, rel)) + 1e-30
            g = tau - dist / self.c
            dg = 1.0 - np.einsum("kijd,kijd->kij", rel / dist[..., None], VJ) / self.c
            tau = np.clip(tau - g / np.where(np.abs(dg) < 1e-6, 1.0, dg), 0.0, None)
        RJ, VJ = self._sources_at(R, V, tn - tau)
        if self.retardation == "aberration":
            RJ = RJ + tau[..., None] * VJ
        d = RJ - Ri
        r2 = np.einsum("kijd,kijd->kij", d, d) + self.eps2
        idx = np.arange(N)
        r2[:, idx, idx] = 1.0                                     # self pair: d = 0 anyway
        w = (self.G * self.masses)[None, None, :] * r2 ** -1.5
        w[:, idx, idx] = 0.0                                      # no self-force
        cache = getattr(self, "_tau_cache", None)
        if cache is not None:
            for i in range(N):
                for j in range(N):
                    if i != j:
                        cache[(i, j)] = tau[:, i, j]
        return np.einsum("kij,kijd->kid", w, d)

    def _source_acc(self, j, tq, side=None):
        """Source acceleration at global times ``tq``: the current iterate's
        acceleration inside the window (lagged one Picard step, exact at the fixed
        point), the stored past before it. ``side`` picks the one-sided limit at
        a join (see :meth:`SegmentHistory.acc`)."""
        A = getattr(self, "_A_cur", None)
        tn = self._node_times()
        t0 = tn[0]
        tol = 1e-8 * (tn[-1] - tn[0])
        side = np.zeros(len(tq)) if side is None else side
        near = np.abs(tq - t0) <= tol
        inside = ((tq >= t0) & ~(near & (side < 0))) | (near & (side > 0))
        out = np.zeros((len(tq), 3))
        if inside.any() and A is not None:
            out[inside] = barycentric_interpolate(tn, A[:, j, :], self._bw, tq[inside])
        past = ~inside
        if past.any():
            h = getattr(self, "history", None)
            if h is not None and hasattr(h, "acc"):
                out[past] = h.acc(j, tq[past], side[past], tol)
        return out

    def _lw_field(self, R, V):
        """Liénard–Wiechert N-body (Gaussian units, Coulomb constant 1),
        relativistic mechanics. Source position, velocity AND acceleration are
        taken on the light cone -> a neutral state-dependent-delay system."""
        n, N = R.shape[0], R.shape[1]
        c = self.c
        tn = self._node_times()[:, None, None]
        Ri = R[:, :, None, :]
        D0 = R[:, None, :, :] - Ri
        tau = np.sqrt(np.einsum("kijd,kijd->kij", D0, D0)) / c
        for _ in range(4):
            RJ, VJ = self._sources_at(R, V, tn - tau)
            rel = Ri - RJ
            dist = np.sqrt(np.einsum("kijd,kijd->kij", rel, rel)) + 1e-30
            g = tau - dist / c
            dg = 1.0 - np.einsum("kijd,kijd->kij", rel / dist[..., None], VJ) / c
            tau = np.clip(tau - g / np.where(np.abs(dg) < 1e-6, 1.0, dg), 0.0, None)
        TQ = tn - tau
        RJ, VJ = self._sources_at(R, V, TQ)
        AJ = np.empty_like(RJ)
        node_side = np.zeros(n)
        node_side[0], node_side[-1] = 1.0, -1.0          # right / left limits at the ends
        side = np.repeat(node_side, N)
        for j in range(N):
            AJ[:, :, j, :] = self._source_acc(j, TQ[:, :, j].reshape(-1), side).reshape(n, N, 3)
        idx = np.arange(N)
        rel = Ri - RJ                                              # source -> receiver
        Rn2 = np.einsum("kijd,kijd->kij", rel, rel) + self.eps2
        Rn2[:, idx, idx] = 1.0
        Rn = np.sqrt(Rn2)
        nv = rel / Rn[..., None]
        beta = VJ / c
        bdot = AJ / c
        if getattr(self, "_lw_no_radiation", False):     # diagnostics only
            bdot = np.zeros_like(bdot)
        kappa = 1.0 - np.einsum("kijd,kijd->kij", nv, beta)
        nb = nv - beta
        b2 = np.einsum("kijd,kijd->kij", beta, beta)
        q = self.charges
        E = (q[None, None, :, None] * (
            nb * ((1.0 - b2) / (kappa**3 * Rn2))[..., None]
            + np.cross(nv, np.cross(nb, bdot)) / (c * kappa**3 * Rn)[..., None]))
        E[:, idx, idx, :] = 0.0
        B = np.cross(nv, E)
        vi = V[:, :, None, :] / c
        F = q[None, :, None] * (E + np.cross(np.broadcast_to(vi, E.shape), B)).sum(axis=2)
        v2 = np.einsum("knd,knd->kn", V, V) / c**2
        gamma = 1.0 / np.sqrt(np.clip(1.0 - v2, 1e-12, None))
        vF = np.einsum("knd,knd->kn", V, F)
        A = (F - (vF / c**2)[..., None] * V) / (gamma * self.masses[None, :])[..., None]
        cache = getattr(self, "_tau_cache", None)
        if cache is not None:
            for i in range(N):
                for j in range(N):
                    if i != j:
                        cache[(i, j)] = tau[:, i, j]
        return A

    def accelerations(self, R, V):
        """Full acceleration field ``(n, N, 3)`` for the configured models.

        EIH is a *complete* 1PN acceleration (it already contains the Newtonian
        term and its own finite-``c`` treatment), so it replaces -- not augments
        -- the Newtonian/retarded force. The reduced-PN term is a genuine additive
        correction on top of (optionally retarded) Newtonian gravity.
        """
        if self.pn == "eih":
            return eih_acceleration(R, V, self.masses, self.G, self.c, self.eps2)
        if self.retardation == "lw":
            return self._lw_field(R, V)

        if self.retardation == "none":
            A = newtonian_all_pairs(R, self.masses, self.G, self.eps2)
        elif getattr(self, "vectorized", True):
            A = self._retarded_field(R, V)
        else:
            n, N = R.shape[0], R.shape[1]
            A = np.zeros((n, N, 3), dtype=np.float64)
            for i in range(N):
                ri = R[:, i, :]
                for j in range(N):
                    if i == j:
                        continue
                    rj = self._retarded_state(R, V, i, j)
                    d = rj - ri
                    r2 = np.einsum("kd,kd->k", d, d) + self.eps2
                    A[:, i, :] += (self.G * self.masses[j]) * d * (r2 ** -1.5)[:, None]
        if self.pn == "reduced":
            A += reduced_pn_acceleration(R, V, self.masses, self.G, self.c, self.eps2)
        return A

    # -- the relaxation solve (shared) -------------------------------------- #
    def _relax_window(self, base, predictor, iterations, tol, record_history):
        """Relax one window given its ballistic ``base`` and ``predictor``.

        Self-consistent: the velocity used inside the force model is the Volterra
        velocity ``V = V0 + C @ A`` (not a finite difference), lagged one sub-step
        during iteration and exact at the converged fixed point.
        """
        v0 = self.V0[None, :, :]
        R = predictor.copy()
        V = np.broadcast_to(v0, R.shape).copy()           # initial velocity guess
        history = [R.copy()] if record_history else []
        residuals: list[float] = []
        scale = max(np.linalg.norm(R) / np.sqrt(R.size), 1e-30)
        anderson = _Anderson(self.anderson_depth) if self.accelerator == "anderson" else None
        converged, used = False, 0

        self._A_cur = None                                 # neutral models: source accel
        for used in range(1, iterations + 1):
            A = self.accelerations(R, V)
            self._A_cur = A
            gR = base + apply_operator(self.K2, A)         # undamped Picard map G(R)
            V = v0 + apply_operator(self.C, A)             # self-consistent velocity
            if anderson is not None:
                acc = anderson.step(R.reshape(-1), gR.reshape(-1)).reshape(R.shape)
                R_new = acc if used > 1 else (1.0 - self.w) * R + self.w * gR
            else:
                R_new = (1.0 - self.w) * R + self.w * gR
            res = float(np.linalg.norm(R_new - R) / np.sqrt(R.size) / scale)
            residuals.append(res)
            R = R_new
            if record_history:
                history.append(R.copy())
            if res < tol:
                converged = True
                break

        self._count_clamps(True)                           # judge only the fixed point
        A = self.accelerations(R, V)                       # clean final velocity
        self._A_cur = A
        self._count_clamps(False)
        V = v0 + apply_operator(self.C, A)
        return R, V, history, residuals, converged, used

    def _count_clamps(self, on):
        """Pruned-history lookups are counted (and warned about) only on the
        converged solution: intermediate Picard iterates may wander far."""
        h = getattr(self, "history", None)
        if h is not None and hasattr(h, "count_clamps"):
            h.count_clamps = on


# --------------------------------------------------------------------------- #
# Batch engine: one (or a few chained) global window(s)
# --------------------------------------------------------------------------- #
class VolterraSystem(_EngineCore):
    def __init__(self, bodies, *, G=6.67430e-11, c=C_LIGHT, softening=0.0,
                 t_span=1.0, n_samples=600, use_retardation=True, use_pn=False,
                 relaxation=0.6, scheme="chebyshev", accelerator="anderson",
                 retardation=None, pn=None, anderson_depth=5, t0=0.0, history=None):
        retardation = retardation or ("aberration" if use_retardation else "none")
        pn = pn or ("eih" if use_pn else "none")
        super().__init__(G=G, c=c, softening=softening, relaxation=relaxation,
                         scheme=scheme, accelerator=accelerator,
                         retardation=retardation, pn=pn, anderson_depth=anderson_depth)
        if len(bodies) < 2:
            raise ValueError("need at least two bodies")
        self.bodies = list(bodies)
        self.N = len(bodies)
        self.masses = np.array([b.mass for b in bodies], dtype=np.float64)
        self.R0 = np.array([b.position for b in bodies], dtype=np.float64)
        self.V0 = np.array([b.velocity for b in bodies], dtype=np.float64)
        self.t_span = float(t_span)
        self.n = int(n_samples)
        self.use_retardation = self.retardation != "none"
        self.use_pn = self.pn != "none"

        self.t0 = float(t0)
        self.t, self.C, self.K2 = build_scheme(self.scheme, self.t0, self.t0 + self.t_span, self.n)
        self._bw = barycentric_weights_cgl(self.n - 1) if self.scheme == "chebyshev" else None
        # DDE initial history phi(t), t < t0 (free motion unless a past is given)
        self.history = history if history is not None else BallisticHistory(self.R0, self.V0, self.t0)

    # source lookup: inside window -> interpolate; before t0 -> history
    def _source_state(self, j, R, V, tq):
        rq, vq = self._in_window(R, V, j, tq)
        past = tq < self.t[0]
        if past.any():
            rp, vp = self.history(j, tq[past])
            rq[past] = rp
            vq[past] = vp
        return rq, vq

    def predictor(self):
        """Taylor predictor r0 + v0 t + 1/2 a0 t^2 (a0 = instantaneous IC accel)."""
        t = (self.t - self.t0)[:, None, None]
        base = self.R0[None, :, :] + t * self.V0[None, :, :]
        a0 = newtonian_all_pairs(self.R0[None], self.masses, self.G, self.eps2)[0]
        return base + 0.5 * (t ** 2) * a0[None, :, :]

    def energy(self, R, V):
        return newtonian_energy(self.masses, R, V, self.G, self.eps2)

    def _diagnostics(self, R, V):
        e = self.energy(R, V)
        p = linear_momentum(self.masses, V)
        L = angular_momentum(self.masses, R, V)
        vscale = float(np.sum(self.masses * np.linalg.norm(self.V0, axis=1)))
        return (e, _scalar_drift(e),
                _vector_drift(p, vscale),
                _vector_drift(L, float(np.linalg.norm(L[0]))))

    def relax(self, iterations=40, *, record_history=True, tol=1e-9):
        t_col = (self.t - self.t0)[:, None, None]
        base = self.R0[None, :, :] + t_col * self.V0[None, :, :]
        R, V, history, residuals, converged, used = self._relax_window(
            base, self.predictor(), iterations, tol, record_history)
        e, ed, pd, ld = self._diagnostics(R, V)
        return RelaxationResult(
            t=self.t, trajectory=R, velocity=V, history=history, residuals=residuals,
            energy=e, energy_drift=ed, momentum_drift=pd, angmom_drift=ld,
            converged=converged, iterations=used)

    # -- segmented (chained) propagation ------------------------------------ #
    def _child(self, R0, V0, t_span, n, t0, history):
        bodies = [Body(b.name, b.mass, R0[k], V0[k], color=b.color, radius=b.radius)
                  for k, b in enumerate(self.bodies)]
        return VolterraSystem(bodies, G=self.G, c=self.c, softening=self.softening,
                              t_span=t_span, n_samples=n, relaxation=self.w,
                              scheme=self.scheme, accelerator=self.accelerator,
                              retardation=self.retardation, pn=self.pn,
                              anderson_depth=self.anderson_depth, t0=t0, history=history)

    def propagate(self, iterations=40, *, n_segments=1, record_history=True, tol=1e-9):
        if n_segments <= 1:
            return self.relax(iterations, record_history=record_history, tol=tol)
        seg_span = self.t_span / n_segments
        seg_n = max(8, self.n // n_segments)
        R0, V0 = self.R0.copy(), self.V0.copy()
        t_parts, R_parts, V_parts, residuals = [], [], [], []
        converged_all, t_off = True, self.t0
        # the retarded past of segment k is the *converged* trajectory of the
        # earlier segments (plus the initial history before t0) -- not a
        # ballistic extrapolation from the segment's own epoch.
        past = SegmentHistory(self.history)
        for s in range(n_segments):
            child = self._child(R0, V0, seg_span, seg_n, t_off, past)
            r = child.relax(iterations, record_history=False, tol=tol)
            past.append(r.t, r.trajectory, r.velocity, child._bw)
            start = 0 if s == 0 else 1
            t_parts.append(r.t[start:])
            R_parts.append(r.trajectory[start:])
            V_parts.append(r.velocity[start:])
            residuals.append(r.residuals[-1] if r.residuals else float("nan"))
            converged_all = converged_all and r.converged
            R0, V0 = r.trajectory[-1].copy(), r.velocity[-1].copy()
            t_off += seg_span
        t = np.concatenate(t_parts); R = np.concatenate(R_parts); V = np.concatenate(V_parts)
        e = newtonian_energy(self.masses, R, V, self.G, self.eps2)
        vscale = float(np.sum(self.masses * np.linalg.norm(self.V0, axis=1)))
        L = angular_momentum(self.masses, R, V)
        return RelaxationResult(
            t=t, trajectory=R, velocity=V, history=[], residuals=residuals, energy=e,
            energy_drift=_scalar_drift(e),
            momentum_drift=_vector_drift(linear_momentum(self.masses, V), vscale),
            angmom_drift=_vector_drift(L, float(np.linalg.norm(L[0]))),
            converged=converged_all, iterations=iterations)


# --------------------------------------------------------------------------- #
# Streaming engine: Piecewise Volterra windowing (infinite horizon)
# --------------------------------------------------------------------------- #
class PiecewiseVolterraStream(_EngineCore):
    def __init__(self, bodies, *, G, c, softening, use_retardation, use_pn,
                 relaxation, window_span, window_samples, iterations,
                 scheme="chebyshev", accelerator="anderson", retardation=None,
                 pn=None, anderson_depth=5, buffer_safety=2.5,
                 max_buffer_samples=None, tol=1e-9, adaptive=True, min_span_frac=0.05,
                 initial_A0=None, track_breaks=0,
                 accuracy_tol=None, charges=None):
        retardation = retardation or ("aberration" if use_retardation else "none")
        pn = pn or ("eih" if use_pn else "none")
        super().__init__(G=G, c=c, softening=softening, relaxation=relaxation,
                         scheme=scheme, accelerator=accelerator,
                         retardation=retardation, pn=pn, anderson_depth=anderson_depth)
        if window_samples < 8:
            raise ValueError("window_samples must be >= 8")
        self.bodies = list(bodies)
        self.N = len(bodies)
        self.masses = np.array([b.mass for b in bodies], dtype=np.float64)
        self._R0 = np.array([b.position for b in bodies], dtype=np.float64)
        self._V0 = np.array([b.velocity for b in bodies], dtype=np.float64)
        self.window_span = float(window_span)
        self.m = int(window_samples)
        self.iterations = int(iterations)
        self.tol = float(tol)
        self.buffer_safety = float(buffer_safety)
        self.max_buffer_samples = max_buffer_samples  # unused (kept for API compat)
        self.use_retardation = self.retardation != "none"
        self.use_pn = self.pn != "none"
        self.adaptive = bool(adaptive)
        self.min_span_frac = float(min_span_frac)
        self.initial_A0 = initial_A0      # None -> ballistic initial history
        self.charges = None if charges is None else np.asarray(charges, dtype=np.float64)
        if self.retardation == "lw" and (self.charges is None or self.scheme != "chebyshev"):
            raise ValueError("retardation='lw' needs charges=[...] and scheme='chebyshev'")
        # DDE breaking-point tracking: number of propagation levels to align
        # window ends with (0 = off). See research/NOTES.md, "breaking points".
        self.track_breaks = int(track_breaks)
        # a-posteriori error control (Chebyshev only): halve the window while the
        # trailing Chebyshev coefficients of the positions exceed this (relative to
        # the position scale). None = off (previous behaviour).
        self.accuracy_tol = accuracy_tol
        self.accuracy_halvings = 0
        self.warm_start = True
        self.reset()

    # -- operators for the current span (rebuilt when the span adapts) ------- #
    def _build_ops(self, span):
        self.t, self.C, self.K2 = build_scheme(self.scheme, 0.0, span, self.m)
        self._bw = barycentric_weights_cgl(self.m - 1) if self.scheme == "chebyshev" else None
        self._span = span

    def reset(self):
        self.R_epoch = self._R0.copy()
        self.V_epoch = self._V0.copy()
        self.t_epoch = 0.0
        self.window_index = 0
        self._build_ops(self.window_span)
        self._seed_buffer()
        # discontinuity set: (time, level, body). Level 0 = t0 (body -1 = all),
        # where the derivative jumps because the initial history phi is not the
        # solution.
        self.breaks = [(self.t_epoch, 0, -1, None)]
        self.break_splits = 0
        self._tau_cache = {}
        self._span_hint = None
        self._prev_end_derivs = None

    @classmethod
    def from_config(cls, cfg, iterations, relaxation, use_retardation, use_pn,
                    window_samples=48):
        # Spectral CGL windows need far fewer nodes than trapezoid. Each streaming
        # window spans one chunk of the horizon (>= 8 windows across it).
        scheme = getattr(cfg, "scheme", "chebyshev")
        window_samples = int(min(getattr(cfg, "window_samples", window_samples) or window_samples,
                                 cfg.n_samples))
        n_win = max(getattr(cfg, "n_segments", 1), 8)
        window_span = cfg.t_span / n_win
        return cls(cfg.bodies, G=cfg.G, c=cfg.c, softening=cfg.softening,
                   use_retardation=use_retardation, use_pn=use_pn, relaxation=relaxation,
                   window_span=window_span, window_samples=window_samples,
                   iterations=iterations, scheme=scheme,
                   retardation=getattr(cfg, "retardation", None),
                   pn=getattr(cfg, "pn", None))

    # -- V0 for the current window is the epoch velocity -------------------- #
    @property
    def V0(self):
        return self.V_epoch

    # -- delay buffer ------------------------------------------------------- #
    def _max_delay(self, R):
        if self.retardation == "none" or self.N < 2:
            return 0.0
        maxd = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                maxd = max(maxd, float(np.linalg.norm(R[i] - R[j])))
        return maxd / self.c

    def _buffer_length(self, R_epoch):
        """Retained past: ``buffer_safety * tau_max`` (at least one window)."""
        return max(self.buffer_safety * self._max_delay(R_epoch), self._span)

    def _seed_buffer(self):
        # spectral segment history: converged windows kept as their own nodal data
        self.history = SegmentHistory(BallisticHistory(self.R_epoch, self.V_epoch, self.t_epoch,
                                                       A0=self.initial_A0))
        self.buffer_span = self._buffer_length(self.R_epoch)

    def _update_buffer(self, tg, R, V, new_epoch):
        self.history.append(tg, R, V, self._bw, A=getattr(self, "_A_cur", None))
        keep = self._buffer_length(R[-1])
        self.history.prune(new_epoch - keep)
        self.buffer_span = self.history.span()

    def _node_times(self):
        return self.t_epoch + self.t

    # source lookup: inside window -> interpolate; before epoch -> history
    def _source_state(self, j, R, V, tq):
        tg = self.t_epoch + self.t
        rq, vq = self._in_window_global(R, V, j, tq, tg)
        past = tq < self.t_epoch
        if past.any():
            rp, vp = self.history(j, tq[past])
            rq[past] = rp
            vq[past] = vp
        return rq, vq

    def _in_window_global(self, R, V, j, tq, tg):
        local = tq - self.t_epoch                          # map to local window time
        if self.scheme == "chebyshev":
            rv = barycentric_interpolate(
                self.t, np.concatenate([R[:, j, :], V[:, j, :]], axis=1), self._bw, local)
            return rv[:, :3], rv[:, 3:]
        return (hermite_interpolate(self.t, R[:, j, :], V[:, j, :], local),
                linear_interpolate(self.t, V[:, j, :], local))

    # override: retardation uses GLOBAL query times -> handled in _source_state
    def _retarded_state(self, R, V, i, j):
        ri = R[:, i, :]
        tg = self.t_epoch + self.t
        d0 = R[:, j, :] - ri
        tau = np.sqrt(np.einsum("kd,kd->k", d0, d0)) / self.c
        for _ in range(4):
            rj, vj = self._source_state(j, R, V, tg - tau)
            rel = ri - rj
            dist = np.sqrt(np.einsum("kd,kd->k", rel, rel)) + 1e-30
            g = tau - dist / self.c
            ndir = rel / dist[:, None]
            dgdtau = 1.0 - np.einsum("kd,kd->k", ndir, vj) / self.c
            tau = np.clip(tau - g / np.where(np.abs(dgdtau) < 1e-6, 1.0, dgdtau), 0.0, None)
        self._tau_cache[(i, j)] = tau
        rj, vj = self._source_state(j, R, V, tg - tau)
        if self.retardation == "aberration":
            rj = rj + tau[:, None] * vj
        return rj

    def _solve_span(self, span, record_history=False):
        self._tau_cache = {}
        # only the accepted (last) solve of a step may count pruned-history lookups
        self.history.clamped = getattr(self, "_clamp_base", self.history.clamped)
        if span != self._span:
            self._build_ops(span)
        t_local = self.t
        base = self.R_epoch[None] + t_local[:, None, None] * self.V_epoch[None]
        tl = t_local[:, None, None]
        derivs = self._prev_end_derivs if self.warm_start else None
        if span < 1e-6 * self.window_span:
            derivs = None                       # sliver window: plain predictor
        if derivs is not None:
            # 4th-order Taylor predictor from the previous window's own Chebyshev
            # polynomial (a, jerk, snap at its end): the true retarded force
            a0, j0, s0 = derivs
            predictor = base + tl**2 / 2 * a0[None] + tl**3 / 6 * j0[None] + tl**4 / 24 * s0[None]
        else:
            a0 = newtonian_all_pairs(self.R_epoch[None], self.masses, self.G, self.eps2)[0]
            predictor = base + 0.5 * tl**2 * a0[None]
        return self._relax_window(base, predictor, self.iterations, self.tol, record_history)

    def _end_derivatives(self, V):
        """(a, jerk, snap) at the window end from the Chebyshev polynomial of V."""
        if self.scheme != "chebyshev":
            return None
        M = V.shape[0] - 1
        from .operators import _cheb_value_to_coeff
        Vinv = _cheb_value_to_coeff(M)[0]
        C = Vinv @ V.reshape(M + 1, -1)                     # (M+1, N*3)
        scale = 2.0 / self._span                             # d/dt = (2/span) d/dx
        out = []
        for k in (1, 2, 3):
            Ck = _cheb.chebder(C, m=k, axis=0)
            out.append((_cheb.chebval(1.0, Ck) * scale**k).reshape(V.shape[1], 3))
        return tuple(out)

    def _first_break(self):
        """Earliest breaking point strictly inside the just-solved window.

        Breaks are tracked per body: ``(d, level, body)`` says that ``body``'s
        trajectory has a derivative jump at ``d`` (``body = -1``: every body, the
        initial-history jump at ``t0``). A jump of body ``j`` at ``d`` reaches body
        ``i`` -- and only ``i`` -- through pair ``(i, j)`` when ``t - tau_ij(t) = d``.
        ``g(t) = t - tau_ij(t)`` is increasing (``g' = 1 - tau' > 0`` for
        ``|v| < c``), so a sign change of ``g - d`` over the window's nodes
        brackets a unique root, refined by bisection on the window's own
        interpolant of ``tau``.

        Returns ``(xi_local, [(level, body), ...])`` -- every body whose break
        coincides with the earliest root (within the merge tolerance) -- or
        ``None``.
        """
        if not self._tau_cache or self.track_breaks <= 0:
            return None
        tg = self.t_epoch + self.t
        span = self._span
        # Roots are identified by their ORIGIN (receiver i, source j, parent break
        # d), not by time: two different roots may be arbitrarily close (nearly
        # symmetric systems) and must both be aligned, while the same root
        # re-found a hair after the window end (re-solve drift) is not new.
        # Roots closer than `same` are aligned together (error ~ jump * gap, i.e.
        # negligible); each keeps its own origin so its descendants are tracked.
        # Splitting at gaps ~1e-10 would create windows so short that Chebyshev
        # derivatives (warm start, Newton landing) drown in round-off.
        same = getattr(self, "_same_rel", 1e-9) * self.window_span
        margin = min(same, 1e-9 * span)
        roots = []
        self._boundary_breaks = []
        for (i, j), tau in self._tau_cache.items():
            g = tg - tau
            for d, lvl, body, _origin in self.breaks:
                if lvl >= self.track_breaks or body not in (-1, j):
                    continue
                h = g - d
                if not (h[0] < 0.0 < h[-1]):
                    continue
                k = int(np.argmax(h > 0.0))
                lo, hi = self.t[k - 1], self.t[k]

                def hval(tl):
                    if self.scheme == "chebyshev":
                        tq = barycentric_interpolate(self.t, tau, self._bw, np.array([tl]))[0]
                    else:
                        tq = float(np.interp(tl, self.t, tau))
                    return self.t_epoch + tl - tq - d
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if hval(mid) > 0.0:
                        hi = mid
                    else:
                        lo = mid
                xi = 0.5 * (lo + hi)
                if self._known_origin(i, j, d):
                    continue
                if margin < xi < span - margin:
                    roots.append((xi, lvl + 1, i, j, d))
                elif xi <= margin:
                    # at the window start: nothing to split, but it must be
                    # registered or its descendants are never tracked
                    self._boundary_breaks.append((self.t_epoch, lvl + 1, i, (j, d)))
        for br in self._boundary_breaks:
            if not self._known_origin(br[2], *br[3]):
                self.breaks.append(br)
        if not roots:
            return None
        x0 = min(r[0] for r in roots)
        first = min(roots, key=lambda r: r[0])
        self._break_pair = (first[2], first[3], first[4])
        return x0, sorted({(r[1], r[2], (r[3], r[4])) for r in roots if r[0] - x0 < same})

    def _refine_break_end(self, span, R, V):
        """Newton step on g(t_end) = t_end - tau_ij(t_end) - d. The bisection root
        came from a window that straddled the kink, so it is only ~1e-7 accurate.
        Uses the closed-form light-cone derivative
            tau' = n.(v_i - v_j(t-tau)) / (c - n.v_j(t-tau)),  n = (x_i - x_j)/|x_i - x_j|,
        evaluated on the converged end state -- no differentiation of nodal data,
        so it stays well conditioned on arbitrarily short (sliver) windows.
        Returns a new span, or None if already on the root."""
        i, j, d = self._break_pair
        tau = self._tau_cache.get((i, j))
        if tau is None:
            return None
        t_end = self.t_epoch + span
        g = t_end - tau[-1] - d
        if abs(g) <= 1e-15 * max(1.0, abs(t_end)):
            return None
        rj, vj = self._source_state(j, R, V, np.array([t_end - tau[-1]]))
        rel = R[-1, i] - rj[0]
        n = rel / max(np.linalg.norm(rel), 1e-300)
        dtau = float(n @ (V[-1, i] - vj[0])) / (self.c - float(n @ vj[0]))
        return span - g / (1.0 - dtau)

    def _trunc_err(self, V):
        # velocities, not positions: positions are one integral smoother, so their
        # tail under-estimates the error by 2-3 orders near close encounters
        # (research/NOTES.md, E7). Measured relative to the velocity spread about
        # the centre of mass, so a uniform boost does not change the control.
        Vrel = V - (np.einsum("n,knd->kd", self.masses, V) / self.masses.sum())[:, None, :]
        scale = max(float(np.abs(Vrel).max()), 1e-30)
        return chebyshev_tail(Vrel) / scale

    def _known_origin(self, i, j, d):
        return any(o == (j, d) and b == i for _, _, b, o in self.breaks)

    def _add_break(self, t_break, sources):
        for level, body, origin in sources:
            if not self._known_origin(body, *origin):
                self.breaks.append((t_break, level, body, origin))
        # a jump at d can only be reached while t - tau_max <= d
        horizon = t_break - 2.0 * self.buffer_safety * self._max_delay(self.R_epoch) - self._span
        self.breaks = [br for br in self.breaks if br[0] >= horizon]

    def step(self) -> WindowResult:
        """Relax the next window (adaptively shrinking on non-convergence)."""
        span = self.window_span
        if self.accuracy_tol is not None and getattr(self, "_span_hint", None):
            # start from the last accepted span (grown by 1.5x), not the nominal
            # one, so a hard stretch is not re-discovered by halving every window
            span = min(self.window_span, self._span_hint)
        min_span = self.window_span * self.min_span_frac
        self._clamp_base = self.history.clamped
        R, V, hist, res, converged, used = self._solve_span(span)
        total = used
        while self.adaptive and not converged and span > min_span:
            span *= 0.5
            R, V, hist, res, converged, used = self._solve_span(span)
            total += used
        span_pre_break = span
        # align the window end with the earliest breaking point inside it (the
        # root moves slightly with the truncated solve: repeat to self-consistency)
        new_break = None
        if self.track_breaks > 0 and self.retardation != "none":
            for _ in range(4):
                fb = self._first_break()
                if fb is None:
                    break
                span, new_break = fb
                R, V, hist, res, converged, used = self._solve_span(span)
                total += used
                self.break_splits += 1
            if new_break is not None:
                for _ in range(3):                         # land exactly on the root
                    new_span = self._refine_break_end(span, R, V)
                    if new_span is None or not (0.0 < new_span < 1.5 * span):
                        break
                    span = new_span
                    R, V, hist, res, converged, used = self._solve_span(span)
                    total += used
            # a truncated window is shorter, so it normally converges; if not,
            # fall back to halving (the break then lies beyond this window end
            # and is re-detected by the next one)
            while self.adaptive and not converged and span > min_span:
                span *= 0.5
                new_break = None
                R, V, hist, res, converged, used = self._solve_span(span)
                total += used
        # error control AFTER break alignment: a kink inside the window inflates
        # the Chebyshev tail, and cutting at the break is the right fix, not halving
        halved = False
        if self.accuracy_tol is not None and self.scheme == "chebyshev":
            while converged and span > min_span and self._trunc_err(V) > self.accuracy_tol:
                span *= 0.5
                new_break = None                       # the break now lies beyond the end
                halved = True
                R, V, hist, res, converged, used = self._solve_span(span)
                total += used
                self.accuracy_halvings += 1
        self.residual_history = res
        if self.accuracy_tol is not None:
            self._span_hint = 1.5 * (span if halved else span_pre_break)
        if self.history.clamped > self._clamp_base:
            import warnings
            warnings.warn("retarded lookup reached pruned history on the accepted solution; "
                          "clamped to the oldest retained sample (increase buffer_safety)",
                          RuntimeWarning, stacklevel=2)

        tg = self.t_epoch + self.t
        e = newtonian_energy(self.masses, R, V, self.G, self.eps2)
        vscale = float(np.sum(self.masses * np.linalg.norm(self.V_epoch, axis=1))) + 1e-30
        result = WindowResult(
            index=self.window_index, t=tg.copy(), trajectory=R.copy(), velocity=V.copy(),
            residual=res[-1] if res else float("nan"), energy_drift=_scalar_drift(e),
            momentum_drift=_vector_drift(linear_momentum(self.masses, V), vscale),
            buffer_span=self.buffer_span, converged=converged, span=span,
            iterations=total)

        new_epoch = self.t_epoch + span
        if new_break is not None:
            self._add_break(new_epoch, new_break)
        self._update_buffer(tg, R, V, new_epoch)
        self._prev_end_derivs = (self._end_derivatives(V)
                                 if span >= 1e-6 * self.window_span else None)
        self.R_epoch, self.V_epoch = R[-1].copy(), V[-1].copy()
        self.t_epoch = new_epoch
        self.window_index += 1
        return result

    def __iter__(self):
        return self

    def __next__(self) -> WindowResult:
        return self.step()

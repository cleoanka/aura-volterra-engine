"""
aura_engine.physics
====================

The Volterra N-Body engine: a *global waveform-relaxation* orbital propagator.

Instead of marching the state forward one step at a time (RK4 / Verlet), the
whole trajectory ``R(t)`` over the entire time grid is treated as a single
unknown and solved by **damped Picard iteration** of the Volterra integral
equation

    R_i(t) = R_i(0) + V_i(0) t + \\int_0^t (t - s) A_i[R](s) ds .

Each iteration:

1.  evaluates the acceleration field ``A[R]`` for the *current* estimate of the
    full trajectory (optionally using retarded, finite-``c`` gravity and a
    reduced post-Newtonian term), then
2.  applies the double-integral operator ``K2`` (see :mod:`aura_engine.operators`)
    to the whole time axis at once, then
3.  under-relaxes: ``R <- (1 - w) R + w * Picard(R)``.

The relaxation factor ``w`` widens the radius of convergence so a *single global
window* can span a full orbit (plain Picard would need many short segments).
Watching ``R`` evolve across iterations is the "waveform relaxation" the UI
animates: a straight ballistic guess literally bends into curved orbits.

Physics honesty
---------------
* **Softened Newtonian gravity** is the always-on core and is well posed.
* **Retarded potentials** use ``t_ret = t - |r_i - r_j| / c`` with one fixed-point
  refinement of the delay. This is a *toy* finite-speed model: naive retarded
  Newtonian gravity is **not** general relativity (real GR cancels aberration
  through velocity-dependent terms). For Solar-System ``c`` the delay is a tiny,
  stable perturbation; the Chaos preset deliberately lowers ``c`` to make the
  delay visible.
* **Post-Newtonian** is an optional, reduced 1PN pairwise term (Schwarzschild
  precession), not the full Einstein-Infeld-Hoffmann cross terms. It is OFF by
  default and validated in the tests to give a *prograde* perihelion advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .operators import apply_operator, build_integral_operators


def newtonian_energy(
    masses: np.ndarray, R: np.ndarray, V: np.ndarray, G: float, eps2: float
) -> np.ndarray:
    """Total softened Newtonian energy per time sample for ``(n, N, 3)`` fields."""
    n = R.shape[0]
    ke = 0.5 * np.einsum("m,kmd,kmd->k", masses, V, V)
    pe = np.zeros(n, dtype=np.float64)
    N = R.shape[1]
    for i in range(N):
        for j in range(i + 1, N):
            d = R[:, j, :] - R[:, i, :]
            r = np.sqrt(np.einsum("kd,kd->k", d, d) + eps2)
            pe -= G * masses[i] * masses[j] / r
    return ke + pe


def _energy_drift(e: np.ndarray) -> float:
    mean_e = float(np.mean(e))
    if mean_e != 0.0:
        return float((e.max() - e.min()) / abs(mean_e))
    return float(e.max() - e.min())


def pair_acceleration(
    ri: np.ndarray,
    rj: np.ndarray,
    mj: float,
    G: float,
    eps2: float,
    *,
    vi: Optional[np.ndarray] = None,
    vj: Optional[np.ndarray] = None,
    c: float = 2.99792458e8,
    use_pn: bool = False,
) -> np.ndarray:
    """Acceleration on body ``i`` due to body ``j`` along a whole time axis.

    ``ri``/``rj`` are ``(n, 3)`` position histories (``rj`` already resolved to the
    retarded position if finite-``c`` gravity is active). Softened Newtonian core
    plus an optional reduced 1PN (Schwarzschild) term. Shared by the batch
    :class:`VolterraSystem` and the streaming :class:`PiecewiseVolterraStream` so
    both use *identical* physics.
    """
    d = rj - ri                                   # vector from i toward j
    r2 = np.einsum("kd,kd->k", d, d) + eps2
    inv_r3 = r2 ** -1.5
    a = (G * mj) * d * inv_r3[:, None]
    if use_pn:
        rvec = -d                                 # source -> target
        r = np.sqrt(r2)
        vrel = vi - vj
        v2 = np.einsum("kd,kd->k", vrel, vrel)
        rv = np.einsum("kd,kd->k", rvec, vrel)
        gm = G * mj
        pref = gm / (c ** 2 * r ** 3)
        a = a + pref[:, None] * (
            ((4.0 * gm / r) - v2)[:, None] * rvec + 4.0 * rv[:, None] * vrel
        )
    return a


# --------------------------------------------------------------------------- #
# Body
# --------------------------------------------------------------------------- #
@dataclass
class Body:
    """A point mass with an initial state and cosmetic attributes."""

    name: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    color: str = "#7de3ff"
    radius: float = 6.0  # marker size hint for the UI (not a physical radius)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)


# --------------------------------------------------------------------------- #
# Solver result
# --------------------------------------------------------------------------- #
@dataclass
class RelaxationResult:
    """Everything a caller (tests, UI, GIF renderer) needs after solving."""

    t: np.ndarray                      # (n,)
    trajectory: np.ndarray             # (n, N, 3) converged positions
    velocity: np.ndarray               # (n, N, 3) Volterra velocities
    history: list[np.ndarray] = field(default_factory=list)  # per-iteration (n,N,3)
    residuals: list[float] = field(default_factory=list)     # per-iteration rel. change
    energy: Optional[np.ndarray] = None                       # (n,) Newtonian energy
    energy_drift: float = 0.0          # (max-min)/|mean| over the window
    converged: bool = False
    iterations: int = 0

    @property
    def n_bodies(self) -> int:
        return self.trajectory.shape[1]


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class VolterraSystem:
    """Global-relaxation N-body propagator.

    Parameters
    ----------
    bodies : list[Body]
    G, c, softening : float
        Gravitational constant, speed of gravity, and Plummer softening length.
    t_span : float
        Total propagation time (grid runs ``0 .. t_span``).
    n_samples : int
        Number of time samples on the global window (>= 8).
    use_retardation, use_pn : bool
        Toggle finite-speed gravity and the reduced 1PN term.
    relaxation : float
        Damping factor ``w`` in ``(0, 1]``. Smaller = more stable / slower.
    """

    def __init__(
        self,
        bodies: list[Body],
        *,
        G: float = 6.67430e-11,
        c: float = 2.99792458e8,
        softening: float = 0.0,
        t_span: float = 1.0,
        n_samples: int = 600,
        use_retardation: bool = True,
        use_pn: bool = False,
        relaxation: float = 0.6,
    ) -> None:
        if len(bodies) < 2:
            raise ValueError("need at least two bodies")
        if n_samples < 8:
            raise ValueError("n_samples must be >= 8")
        if not (0.0 < relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1]")

        self.bodies = list(bodies)
        self.G = float(G)
        self.c = float(c)
        self.softening = float(softening)
        self.eps2 = float(softening) ** 2
        self.t_span = float(t_span)
        self.n = int(n_samples)
        self.use_retardation = bool(use_retardation)
        self.use_pn = bool(use_pn)
        self.w = float(relaxation)

        self.N = len(bodies)
        self.masses = np.array([b.mass for b in bodies], dtype=np.float64)
        self.R0 = np.array([b.position for b in bodies], dtype=np.float64)  # (N,3)
        self.V0 = np.array([b.velocity for b in bodies], dtype=np.float64)  # (N,3)

        self.t = np.linspace(0.0, self.t_span, self.n)
        self.dt = float(self.t[1] - self.t[0])
        # Precompute the global integral operators once (reused every iteration).
        self.C, self.K2 = build_integral_operators(self.t)

    # ---------------------------------------------------------------- #
    # Predictor: a fast ballistic baseline path (straight, inertial).
    # ---------------------------------------------------------------- #
    def predictor(self) -> np.ndarray:
        """Return the ballistic predictor ``R0 + V0 * t`` shaped ``(n, N, 3)``."""
        return self.R0[None, :, :] + self.t[:, None, None] * self.V0[None, :, :]

    # ---------------------------------------------------------------- #
    # Retarded-position interpolation of source body j along query times.
    # ---------------------------------------------------------------- #
    def _sample_body(self, R: np.ndarray, j: int, tq: np.ndarray) -> np.ndarray:
        """Interpolate body ``j``'s position at query times ``tq`` (shape (n,)).

        For ``tq`` before the window start we fall back to a straight-line
        (ballistic) extrapolation into the past using the body's initial state,
        which keeps the retarded field well defined near ``t = 0``.
        """
        out = np.empty((tq.size, 3), dtype=np.float64)
        past = tq < self.t[0]
        for d in range(3):
            vals = np.interp(tq, self.t, R[:, j, d])
            if past.any():
                vals = np.where(
                    past, self.R0[j, d] + self.V0[j, d] * (tq - self.t[0]), vals
                )
            out[:, d] = vals
        return out

    def _retarded_position(
        self, R: np.ndarray, i: int, j: int, ri: np.ndarray
    ) -> np.ndarray:
        """Position of source ``j`` as *seen* by target ``i`` (finite ``c``).

        Solves ``t_ret = t - |r_i(t) - r_j(t_ret)| / c`` with one fixed-point
        refinement -- accurate enough for the mildly relativistic regimes here.
        """
        c = self.c
        rj_now = R[:, j, :]
        dist = np.linalg.norm(rj_now - ri, axis=1)
        tq = self.t - dist / c
        rj = self._sample_body(R, j, tq)
        # one refinement using the freshly delayed position
        dist = np.linalg.norm(rj - ri, axis=1)
        tq = self.t - dist / c
        return self._sample_body(R, j, tq)

    # ---------------------------------------------------------------- #
    # Acceleration field over the whole time grid.
    # ---------------------------------------------------------------- #
    def accelerations(self, R: np.ndarray, V: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorised acceleration field ``A[R]`` of shape ``(n, N, 3)``.

        Loops over the (small) set of ordered body pairs; every operation inside
        the loop is vectorised across the full time axis.
        """
        n, N = self.n, self.N
        A = np.zeros((n, N, 3), dtype=np.float64)
        need_v = self.use_pn
        if need_v and V is None:
            V = np.gradient(R, self.t, axis=0)

        for i in range(N):
            ri = R[:, i, :]
            vi = V[:, i, :] if need_v else None
            for j in range(N):
                if i == j:
                    continue
                rj = self._retarded_position(R, i, j, ri) if self.use_retardation \
                    else R[:, j, :]
                vj = V[:, j, :] if need_v else None
                A[:, i, :] += pair_acceleration(
                    ri, rj, self.masses[j], self.G, self.eps2,
                    vi=vi, vj=vj, c=self.c, use_pn=self.use_pn,
                )
        return A

    # ---------------------------------------------------------------- #
    # Newtonian energy diagnostic (softened, instantaneous).
    # ---------------------------------------------------------------- #
    def energy(self, R: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Total (softened, non-retarded) mechanical energy per time sample."""
        return newtonian_energy(self.masses, R, V, self.G, self.eps2)

    # ---------------------------------------------------------------- #
    # The global waveform-relaxation solve.
    # ---------------------------------------------------------------- #
    def relax(
        self,
        iterations: int = 40,
        *,
        record_history: bool = True,
        tol: float = 1e-9,
    ) -> RelaxationResult:
        """Run damped Picard iteration over the whole window.

        Parameters
        ----------
        iterations : int
            Maximum number of relaxation sweeps.
        record_history : bool
            Keep every intermediate trajectory (for the relaxation animation).
        tol : float
            Relative-change stopping tolerance.
        """
        R = self.predictor()
        history: list[np.ndarray] = [R.copy()] if record_history else []
        residuals: list[float] = []

        t_col = self.t[:, None, None]
        base = self.R0[None, :, :] + t_col * self.V0[None, :, :]  # ballistic term
        scale = max(np.linalg.norm(R) / np.sqrt(R.size), 1e-30)

        converged = False
        used = 0
        for used in range(1, iterations + 1):
            V = np.gradient(R, self.t, axis=0) if self.use_pn else None
            A = self.accelerations(R, V)
            picard = base + apply_operator(self.K2, A)
            R_new = (1.0 - self.w) * R + self.w * picard

            res = float(np.linalg.norm(R_new - R) / np.sqrt(R.size) / scale)
            residuals.append(res)
            R = R_new
            if record_history:
                history.append(R.copy())
            if res < tol:
                converged = True
                break

        V_final = self.V0[None, :, :] + apply_operator(self.C, self.accelerations(R))
        e = self.energy(R, V_final)

        return RelaxationResult(
            t=self.t,
            trajectory=R,
            velocity=V_final,
            history=history,
            residuals=residuals,
            energy=e,
            energy_drift=_energy_drift(e),
            converged=converged,
            iterations=used,
        )

    # ---------------------------------------------------------------- #
    # Segmented propagation (chained global windows -> longer horizons).
    # ---------------------------------------------------------------- #
    def _child(
        self, R0: np.ndarray, V0: np.ndarray, t_span: float, n_samples: int
    ) -> "VolterraSystem":
        """A sibling engine on a sub-window, inheriting all physics settings."""
        bodies = [
            Body(b.name, b.mass, R0[k], V0[k], color=b.color, radius=b.radius)
            for k, b in enumerate(self.bodies)
        ]
        return VolterraSystem(
            bodies,
            G=self.G,
            c=self.c,
            softening=self.softening,
            t_span=t_span,
            n_samples=n_samples,
            use_retardation=self.use_retardation,
            use_pn=self.use_pn,
            relaxation=self.w,
        )

    def propagate(
        self,
        iterations: int = 40,
        *,
        n_segments: int = 1,
        record_history: bool = True,
        tol: float = 1e-9,
    ) -> RelaxationResult:
        """Relax over ``n_segments`` chained global windows.

        ``n_segments == 1`` is exactly :meth:`relax` (with its full morph
        history). For ``n_segments > 1`` each sub-window is relaxed to
        convergence and the converged end-state seeds the next -- this is the
        segmented Picard--Chebyshev (MCPI) strategy that extends the method to
        many-orbit and chaotic horizons that a single window cannot reach.
        Successive windows still solve their whole time axis globally.
        """
        if n_segments <= 1:
            return self.relax(iterations, record_history=record_history, tol=tol)

        seg_span = self.t_span / n_segments
        seg_n = max(8, self.n // n_segments)
        R0, V0 = self.R0.copy(), self.V0.copy()

        t_parts, R_parts, V_parts = [], [], []
        residuals: list[float] = []
        converged_all = True
        t_offset = 0.0
        for s in range(n_segments):
            child = self._child(R0, V0, seg_span, seg_n)
            r = child.relax(iterations, record_history=False, tol=tol)
            start = 0 if s == 0 else 1  # avoid duplicating the shared boundary
            t_parts.append(r.t[start:] + t_offset)
            R_parts.append(r.trajectory[start:])
            V_parts.append(r.velocity[start:])
            residuals.append(r.residuals[-1] if r.residuals else float("nan"))
            converged_all = converged_all and r.converged
            R0 = r.trajectory[-1].copy()
            V0 = r.velocity[-1].copy()
            t_offset += seg_span

        t = np.concatenate(t_parts)
        R = np.concatenate(R_parts)
        V = np.concatenate(V_parts)
        e = newtonian_energy(self.masses, R, V, self.G, self.eps2)

        return RelaxationResult(
            t=t,
            trajectory=R,
            velocity=V,
            history=[],  # multi-segment: UI reveals the drawn trajectory instead
            residuals=residuals,
            energy=e,
            energy_drift=_energy_drift(e),
            converged=converged_all,
            iterations=iterations,
        )


# --------------------------------------------------------------------------- #
# Window result (one slice of a continuous stream)
# --------------------------------------------------------------------------- #
@dataclass
class WindowResult:
    """One relaxed window ``[T_k, T_{k+1}]`` emitted by the streaming engine."""

    index: int
    t: np.ndarray               # (m,) global time samples
    trajectory: np.ndarray      # (m, N, 3)
    velocity: np.ndarray        # (m, N, 3)
    residual: float             # final relaxation residual on this window
    energy_drift: float         # Newtonian drift across this window
    buffer_span: float          # length of the delay buffer carried forward
    converged: bool


# --------------------------------------------------------------------------- #
# Piecewise Volterra streaming engine  (continuous / infinite run)
# --------------------------------------------------------------------------- #
class PiecewiseVolterraStream:
    """Infinite-horizon propagator via **Piecewise Volterra Windowing**.

    Rather than solving one huge global window (``O(t^2)`` memory/compute), time
    is cut into fixed windows ``W_k = [T_k, T_{k+1}]``. Each window is relaxed
    globally by damped Picard, then:

    1. the **final state** ``(r(T_{k+1}), v(T_{k+1}))`` becomes the next epoch;
    2. a **sliding delay buffer** of length ``~c*tau_max`` (the tail of the
       just-computed trajectory) is retained so the retarded potentials
       ``r(t - tau)`` in the next window read *real* past positions, not a
       ballistic guess;
    3. everything older than the buffer is discarded.

    This bounds cost per window to ``O(W^2)`` and memory to ``O(1)`` while
    preserving the holistic, drift-resistant delay resolution of the global
    Volterra solve -- the numerics of waveform relaxation with the runtime
    profile of a real-time flight computer.

    Call :meth:`step` repeatedly (it never stops) or iterate the object.
    """

    def __init__(
        self,
        bodies: list[Body],
        *,
        G: float,
        c: float,
        softening: float,
        use_retardation: bool,
        use_pn: bool,
        relaxation: float,
        window_span: float,
        window_samples: int,
        iterations: int,
        buffer_safety: float = 2.5,
        max_buffer_samples: int = 256,
        tol: float = 1e-9,
    ) -> None:
        if window_samples < 8:
            raise ValueError("window_samples must be >= 8")
        self.bodies = list(bodies)
        self.N = len(bodies)
        self.masses = np.array([b.mass for b in bodies], dtype=np.float64)
        self.G, self.c = float(G), float(c)
        self.softening = float(softening)
        self.eps2 = float(softening) ** 2
        self.use_retardation = bool(use_retardation)
        self.use_pn = bool(use_pn)
        self.w = float(relaxation)
        self.window_span = float(window_span)
        self.m = int(window_samples)
        self.iterations = int(iterations)
        self.tol = float(tol)
        self.buffer_safety = float(buffer_safety)
        self.max_buffer_samples = int(max_buffer_samples)

        # Local window grid [0, window_span] and its Volterra operators (built once).
        self.local_t = np.linspace(0.0, self.window_span, self.m)
        self.dt = float(self.local_t[1] - self.local_t[0])
        self.C, self.K2 = build_integral_operators(self.local_t)

        self._R0 = np.array([b.position for b in bodies], dtype=np.float64)
        self._V0 = np.array([b.velocity for b in bodies], dtype=np.float64)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Rewind the stream to the initial epoch and re-seed the buffer."""
        self.R_epoch = self._R0.copy()
        self.V_epoch = self._V0.copy()
        self.t_epoch = 0.0
        self.window_index = 0
        self._seed_buffer()

    @classmethod
    def from_config(cls, cfg, iterations, relaxation, use_retardation, use_pn,
                    window_samples: int = 64) -> "PiecewiseVolterraStream":
        """Build a stream from a :class:`~aura_engine.presets.PresetConfig`.

        The window keeps the batch grid's time resolution (``dt``) but only
        ``window_samples`` points per slice, so every scenario streams at the
        same fidelity it was tuned for.
        """
        dt = cfg.t_span / (cfg.n_samples - 1)
        window_samples = int(min(window_samples, cfg.n_samples))
        window_span = dt * (window_samples - 1)
        return cls(
            cfg.bodies, G=cfg.G, c=cfg.c, softening=cfg.softening,
            use_retardation=use_retardation, use_pn=use_pn, relaxation=relaxation,
            window_span=window_span, window_samples=window_samples,
            iterations=iterations,
        )

    # ------------------------------------------------------------------ #
    # Delay buffer management
    # ------------------------------------------------------------------ #
    def _max_delay(self, R: np.ndarray) -> float:
        """Largest light-travel delay tau_max = max pairwise distance / c."""
        if not self.use_retardation or self.N < 2:
            return 0.0
        maxd = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                maxd = max(maxd, float(np.linalg.norm(R[i] - R[j])))
        return maxd / self.c

    def _buffer_length(self, R_epoch: np.ndarray) -> tuple[float, int]:
        """Choose the buffer span (~safety*tau_max) and its sample count."""
        tau_max = self._max_delay(R_epoch)
        span = self.buffer_safety * tau_max
        span = max(span, 2.0 * self.dt)                 # never degenerate
        n = int(np.clip(round(span / self.dt) + 1, 8, self.max_buffer_samples))
        return span, n

    def _seed_buffer(self) -> None:
        """Initialise the buffer with a ballistic pre-history before t=0."""
        span, n = self._buffer_length(self.R_epoch)
        bt = np.linspace(self.t_epoch - span, self.t_epoch, n)
        # ballistic: r(t) = r0 + v0 (t - t_epoch)
        self.buf_t = bt
        self.buf_R = (self.R_epoch[None, :, :]
                      + (bt - self.t_epoch)[:, None, None] * self.V_epoch[None, :, :])
        self.buffer_span = span

    def _update_buffer(self, tg: np.ndarray, R: np.ndarray, new_epoch: float) -> None:
        """Retain only the last ``buffer_span`` of history, resampled to fixed size."""
        span, n = self._buffer_length(R[-1])
        # concatenate old buffer with the new window (drop duplicated boundary)
        all_t = np.concatenate([self.buf_t, tg[1:]])
        all_R = np.concatenate([self.buf_R, R[1:]])
        lo = new_epoch - span
        keep = all_t >= lo - 1e-12
        ct, cR = all_t[keep], all_R[keep]
        new_t = np.linspace(lo, new_epoch, n)
        new_R = np.empty((n, self.N, 3), dtype=np.float64)
        for i in range(self.N):
            for d in range(3):
                new_R[:, i, d] = np.interp(new_t, ct, cR[:, i, d])
        self.buf_t, self.buf_R, self.buffer_span = new_t, new_R, span

    def _sample_source(self, R_win: np.ndarray, j: int, tq: np.ndarray) -> np.ndarray:
        """Position of body ``j`` at global query times ``tq``.

        Interpolates across the concatenation of the *delay buffer* (the real
        past) and the *current window iterate*. Queries older than the buffer are
        clamped to its oldest sample (delay exceeded the retained history).
        """
        tg = self.t_epoch + self.local_t
        # buffer ends exactly at t_epoch == tg[0]; drop the duplicate join point
        cat_t = np.concatenate([self.buf_t[:-1], tg])
        out = np.empty((tq.size, 3), dtype=np.float64)
        for d in range(3):
            cat = np.concatenate([self.buf_R[:-1, j, d], R_win[:, j, d]])
            out[:, d] = np.interp(tq, cat_t, cat)
        return out

    def _retarded_position(self, R_win, i, j, ri) -> np.ndarray:
        tg = self.t_epoch + self.local_t
        dist = np.linalg.norm(R_win[:, j, :] - ri, axis=1)
        rj = self._sample_source(R_win, j, tg - dist / self.c)
        dist = np.linalg.norm(rj - ri, axis=1)                 # one refinement
        return self._sample_source(R_win, j, tg - dist / self.c)

    # ------------------------------------------------------------------ #
    def _accelerations(self, R_win: np.ndarray, V_win: Optional[np.ndarray]) -> np.ndarray:
        A = np.zeros((self.m, self.N, 3), dtype=np.float64)
        for i in range(self.N):
            ri = R_win[:, i, :]
            vi = V_win[:, i, :] if self.use_pn else None
            for j in range(self.N):
                if i == j:
                    continue
                rj = (self._retarded_position(R_win, i, j, ri)
                      if self.use_retardation else R_win[:, j, :])
                vj = V_win[:, j, :] if self.use_pn else None
                A[:, i, :] += pair_acceleration(
                    ri, rj, self.masses[j], self.G, self.eps2,
                    vi=vi, vj=vj, c=self.c, use_pn=self.use_pn,
                )
        return A

    # ------------------------------------------------------------------ #
    def step(self) -> WindowResult:
        """Relax the next window, advance the epoch, slide the buffer. Never stops."""
        t_col = self.local_t[:, None, None]
        base = self.R_epoch[None, :, :] + t_col * self.V_epoch[None, :, :]
        R = base.copy()                                   # ballistic predictor
        scale = max(np.linalg.norm(R) / np.sqrt(R.size), 1e-30)

        residual, converged = float("nan"), False
        A = np.zeros_like(R)
        for _ in range(self.iterations):
            V = np.gradient(R, self.local_t, axis=0) if self.use_pn else None
            A = self._accelerations(R, V)
            R_new = (1.0 - self.w) * R + self.w * (base + apply_operator(self.K2, A))
            residual = float(np.linalg.norm(R_new - R) / np.sqrt(R.size) / scale)
            R = R_new
            if residual < self.tol:
                converged = True
                break

        V_final = self.V_epoch[None, :, :] + apply_operator(self.C, self._accelerations(R, None))
        tg = self.t_epoch + self.local_t
        e = newtonian_energy(self.masses, R, V_final, self.G, self.eps2)

        result = WindowResult(
            index=self.window_index, t=tg.copy(), trajectory=R.copy(),
            velocity=V_final.copy(), residual=residual, energy_drift=_energy_drift(e),
            buffer_span=self.buffer_span, converged=converged,
        )

        # advance epoch + slide the delay buffer, then discard the window
        new_epoch = self.t_epoch + self.window_span
        self._update_buffer(tg, R, new_epoch)
        self.R_epoch = R[-1].copy()
        self.V_epoch = V_final[-1].copy()
        self.t_epoch = new_epoch
        self.window_index += 1
        return result

    def __iter__(self):
        return self

    def __next__(self) -> WindowResult:
        return self.step()

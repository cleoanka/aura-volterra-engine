"""
aura_engine.operators
======================

Discrete Volterra integral operators over a uniform time grid.

Project AURA replaces step-by-step marching (RK4, Verlet, ...) with a *global*
integral formulation. The equations of motion

    r''(t) = a(t)

are rewritten as a **Volterra integral equation of the second kind**

    r(t) = r(0) + v(0) t + \\int_0^t (t - s) a(s) ds
                              \\_______________________/
                                the double-integral (Green's) kernel  K(t, s) = (t - s)

    v(t) = v(0) + \\int_0^t a(s) ds

The kernel is *causal* (lower-triangular): the state at time ``t`` depends only
on the acceleration history ``s <= t``. That is exactly what makes this a
Volterra rather than a Fredholm operator, and it is what lets a single fixed
matrix encode "integrate the whole trajectory at once".

Two interchangeable quadrature *schemes* build the operators:

* ``"trapezoid"`` -- uniform grid, cumulative trapezoid. 2nd-order (``O(dt^2)``),
  simple and robust. ``K2 = C @ C``.
* ``"chebyshev"`` -- Chebyshev-Gauss-Lobatto (CGL) nodes + the spectral Chebyshev
  integration matrix (Modified Chebyshev-Picard Iteration, MCPI). **Spectral /
  exponential** accuracy for smooth trajectories and a far larger Picard
  convergence radius per window; nodes cluster at the window ends.

Either way the operators satisfy

* ``v = v0 + C @ a``
* ``r = r0 + v0 * (t - t0) + K2 @ a``

so applying ``K2 @ a`` is a single BLAS ``gemm`` -- the entire trajectory is
relaxed in one vectorised matrix product per Picard iteration.
"""

from __future__ import annotations

import numpy as np
import numpy.polynomial.chebyshev as _cheb

SCHEMES = ("trapezoid", "chebyshev")


def cumulative_trapezoid_matrix(n: int, dt: float) -> np.ndarray:
    """Return the ``n x n`` cumulative trapezoidal-integration matrix ``C``.

    For a function sampled as ``f_k = f(k * dt)`` the trapezoidal estimate of
    ``I_k = \\int_0^{t_k} f(s) ds`` is

        I_0 = 0
        I_k = dt * ( f_0/2 + f_1 + ... + f_{k-1} + f_k/2 ),   k >= 1

    which is exactly ``I = C @ f``. ``C`` is lower triangular (causal): row ``k``
    only touches samples ``0..k``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    C = np.zeros((n, n), dtype=np.float64)
    # Row k (k >= 1): weight dt on interior samples, dt/2 on the two endpoints.
    for k in range(1, n):
        C[k, : k + 1] = dt
        C[k, 0] *= 0.5
        C[k, k] *= 0.5
    return C


def build_integral_operators(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the velocity and position Volterra operators for a uniform grid.

    Parameters
    ----------
    t : (n,) ndarray
        Uniformly spaced, strictly increasing time samples starting at ``t[0]``.

    Returns
    -------
    C : (n, n) ndarray
        Single-integration operator, ``v = v0 + C @ a``.
    K2 : (n, n) ndarray
        Double-integration operator, ``r = r0 + v0 * (t - t0) + K2 @ a``.

    Notes
    -----
    ``K2 = C @ C`` is the discrete analogue of the causal Green's kernel
    ``(t - s)``. We verify uniformity because the closed-form trapezoidal
    weights above assume a constant step.
    """
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a 1-D array with at least 2 samples")
    diffs = np.diff(t)
    dt = float(diffs[0])
    if not np.allclose(diffs, dt, rtol=1e-9, atol=1e-12):
        raise ValueError("build_integral_operators requires a uniform time grid")
    C = cumulative_trapezoid_matrix(t.size, dt)
    K2 = C @ C
    return C, K2


def apply_operator(op: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Apply an ``(n, n)`` time-operator to a ``(n, N, 3)`` vector field.

    The operator acts along the time axis for every body/component at once.
    Returns an array shaped like ``field``.
    """
    n = op.shape[0]
    flat = field.reshape(n, -1)
    return (op @ flat).reshape(field.shape)


# --------------------------------------------------------------------------- #
# Spectral scheme: Chebyshev-Gauss-Lobatto nodes + Chebyshev integration matrix
# --------------------------------------------------------------------------- #
def chebyshev_lobatto_nodes(M: int) -> np.ndarray:
    """``M + 1`` CGL nodes on ``[-1, 1]``, ascending (clustered at the ends)."""
    j = np.arange(M + 1)
    return -np.cos(np.pi * j / M)


def _cheb_value_to_coeff(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the CGL value->coefficient Vandermonde ``V[j,k] = T_k(x_j)``."""
    x = chebyshev_lobatto_nodes(M)
    theta = np.arccos(np.clip(x, -1.0, 1.0))
    V = np.cos(np.outer(theta, np.arange(M + 1)))  # T_k(x_j)
    return np.linalg.inv(V), x


def chebyshev_integration_matrix(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodal 'integrate from the left endpoint' operator ``B`` on CGL nodes.

    ``(B @ f)_j`` approximates ``\\int_{-1}^{x_j} f(x) dx`` with spectral accuracy.
    Built column-by-column via numpy's tested Chebyshev integration so it is
    correct by construction (each basis function is integrated analytically).
    """
    Vinv, x = _cheb_value_to_coeff(M)
    Phi = np.zeros((M + 1, M + 1), dtype=np.float64)
    for k in range(M + 1):
        ck = np.zeros(M + 1)
        ck[k] = 1.0
        Ik = _cheb.chebint(ck, m=1, lbnd=-1.0)      # antiderivative, F(-1) = 0
        Phi[:, k] = _cheb.chebval(x, Ik)
    return x, Phi @ Vinv


def chebyshev_operators(t0: float, t1: float, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectral (MCPI) operators on ``[t0, t1]`` with ``M + 1`` CGL nodes."""
    x, B = chebyshev_integration_matrix(M)
    omega = 0.5 * (t1 - t0)                          # half-width (d t / d x)
    t = t0 + omega * (x + 1.0)                        # ascending real-time nodes
    C = omega * B
    K2 = (omega ** 2) * (B @ B)
    return t, C, K2


def build_scheme(scheme: str, t0: float, t1: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified operator builder: returns ``(t_nodes, C, K2)`` for either scheme.

    ``n`` is the number of samples (trapezoid) or ``M + 1`` CGL nodes (chebyshev).
    """
    if scheme == "trapezoid":
        t = np.linspace(t0, t1, n)
        C, K2 = build_integral_operators(t)
        return t, C, K2
    if scheme == "chebyshev":
        return chebyshev_operators(t0, t1, n - 1)
    raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")


# --------------------------------------------------------------------------- #
# Barycentric Lagrange interpolation at CGL nodes (spectral companion)
# --------------------------------------------------------------------------- #
def barycentric_weights_cgl(M: int) -> np.ndarray:
    """Barycentric weights for CGL nodes: ``w_j = (-1)^j`` with half at the ends."""
    w = np.ones(M + 1, dtype=np.float64)
    w[1::2] = -1.0
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def resample_uniform(
    t_nodes: np.ndarray, field: np.ndarray, n_out: int
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample a ``(n, ...)`` field from (possibly CGL) nodes to a
    uniform time grid -- used only for smooth display playback, not the physics."""
    tu = np.linspace(t_nodes[0], t_nodes[-1], n_out)
    flat = field.reshape(field.shape[0], -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(tu, t_nodes, flat[:, c])
    return tu, out.reshape((n_out,) + field.shape[1:])


def barycentric_interpolate(
    t_nodes: np.ndarray, values: np.ndarray, weights: np.ndarray, tq: np.ndarray
) -> np.ndarray:
    """Vectorised barycentric Lagrange interpolation of ``values`` at ``tq``.

    ``values`` may be ``(n,)`` or ``(n, K)`` (several components share the one
    weight matrix -- much cheaper than one call per component). ``tq`` is ``(m,)``,
    clamped to the node span to avoid extrapolation blow-up. Spectrally accurate
    on CGL nodes; exact node hits handled without division by zero.
    """
    tq = np.clip(np.asarray(tq, dtype=np.float64), t_nodes[0], t_nodes[-1])
    vals = np.asarray(values, dtype=np.float64)
    scalar = vals.ndim == 1
    if scalar:
        vals = vals[:, None]
    diff = tq[:, None] - t_nodes[None, :]            # (m, n)
    exact = np.isclose(diff, 0.0)
    safe = np.where(exact, 1.0, diff)
    w_over = weights[None, :] / safe                 # (m, n)
    num = w_over @ vals                              # (m, K)
    den = np.sum(w_over, axis=1)                     # (m,)
    out = num / den[:, None]
    hit_rows = np.any(exact, axis=1)
    if hit_rows.any():
        idx = np.argmax(exact[hit_rows], axis=1)
        out[hit_rows] = vals[idx]
    return out[:, 0] if scalar else out

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

This module builds two dense lower-triangular matrices for a uniform grid of
``n`` samples with spacing ``dt``:

* ``C``   -- single cumulative-integration operator (trapezoidal quadrature),
             so that  ``v = v0 + C @ a``.
* ``K2``  -- the double-integration (position) operator, ``K2 = C @ C``, so that
             ``r = r0 + v0 * t + K2 @ a``.

Applying ``K2 @ a`` is a single BLAS ``gemm`` call, which on Apple-silicon maps
straight onto the Accelerate framework -- the entire trajectory is relaxed in
one vectorised matrix product per Picard iteration.
"""

from __future__ import annotations

import numpy as np


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

"""Validation tests for the Volterra engine (v0.3 advanced core).

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura_engine import presets
from aura_engine.operators import (
    apply_operator,
    barycentric_interpolate,
    barycentric_weights_cgl,
    build_integral_operators,
    chebyshev_operators,
)
from aura_engine.physics import Body, PiecewiseVolterraStream, VolterraSystem


# --------------------------------------------------------------------------- #
# Quadrature operators: trapezoid (2nd order) and Chebyshev (spectral)
# --------------------------------------------------------------------------- #
def test_trapezoid_double_integral_constant():
    t = np.linspace(0.0, 3.0, 400)
    C, K2 = build_integral_operators(t)
    a = np.full_like(t, 2.5)
    assert np.allclose(K2 @ a, 0.5 * 2.5 * t**2, atol=1e-6)
    assert np.allclose(C @ a, 2.5 * t, atol=1e-9)


def test_chebyshev_is_spectral():
    # sin double-integral: spectral scheme must crush trapezoid at equal nodes.
    t0, t1, M = 0.0, 3.0, 40
    t, C, K2 = chebyshev_operators(t0, t1, M)
    a = np.sin(2 * t)
    r_an = t / 2 - np.sin(2 * t) / 4
    v_an = (1 - np.cos(2 * t)) / 2
    assert np.max(np.abs(K2 @ a - r_an)) < 1e-12
    assert np.max(np.abs(C @ a - v_an)) < 1e-12

    tt = np.linspace(t0, t1, M + 1)
    Ct, Kt = build_integral_operators(tt)
    trap_err = np.max(np.abs(Kt @ np.sin(2 * tt) - (tt / 2 - np.sin(2 * tt) / 4)))
    assert trap_err > 1e-4  # trapezoid is orders worse


def test_barycentric_interpolation_spectral():
    t, _, _ = chebyshev_operators(0.0, 3.0, 40)
    w = barycentric_weights_cgl(40)
    f = np.exp(np.sin(2 * t))
    tq = np.linspace(0.0, 3.0, 200)
    err = np.max(np.abs(barycentric_interpolate(t, f, w, tq) - np.exp(np.sin(2 * tq))))
    assert err < 1e-10
    assert np.allclose(barycentric_interpolate(t, f, w, t), f)  # node hits exact


# --------------------------------------------------------------------------- #
# Two-body: closed, conservative orbit (now to spectral precision)
# --------------------------------------------------------------------------- #
def _two_body_circular(n=160, retard=False):
    G, M, m, r = 1.0, 1.0, 1e-6, 1.0
    v = np.sqrt(G * M / r)
    period = 2 * np.pi * r / v
    central = Body("M", M, [0, 0, 0], [0, 0, 0])
    orbiter = Body("m", m, [r, 0, 0], [0, v, 0])
    sysm = VolterraSystem([central, orbiter], G=G, c=1e6, softening=1e-4,
                          t_span=period, n_samples=n, use_retardation=retard,
                          relaxation=0.6)
    return sysm, period


def test_two_body_closed_and_conservative():
    sysm, _ = _two_body_circular()
    res = sysm.relax(80, record_history=False, tol=1e-11)
    assert res.converged
    r = np.linalg.norm(res.trajectory[:, 1, :], axis=1)
    assert np.allclose(r, 1.0, atol=2e-3)
    assert np.linalg.norm(res.trajectory[-1, 1] - res.trajectory[0, 1]) < 2e-2
    assert res.energy_drift < 1e-6
    assert res.momentum_drift < 1e-6
    assert res.angmom_drift < 1e-6


# --------------------------------------------------------------------------- #
# Anderson acceleration: fewer iterations than plain damped Picard
# --------------------------------------------------------------------------- #
def test_anderson_beats_picard():
    picard = _two_body_circular()[0]
    picard.accelerator = "picard"
    rp = picard.relax(200, record_history=False, tol=1e-9)
    anderson = _two_body_circular()[0]      # default accelerator="anderson"
    ra = anderson.relax(200, record_history=False, tol=1e-9)
    assert ra.converged and rp.converged
    assert ra.iterations < rp.iterations


# --------------------------------------------------------------------------- #
# Spectral scheme beats trapezoid on a real preset
# --------------------------------------------------------------------------- #
def test_spectral_beats_trapezoid_on_preset():
    cfg = presets.solar_system()
    cheb = cfg.build(scheme="chebyshev").relax(cfg.iterations, record_history=False)
    trap = cfg.build(scheme="trapezoid").relax(cfg.iterations, record_history=False)
    assert cheb.energy_drift < trap.energy_drift
    assert cheb.energy_drift < 1e-7


# --------------------------------------------------------------------------- #
# Retardation: aberration model conserves far better than naive
# --------------------------------------------------------------------------- #
def test_aberration_conserves_better_than_naive():
    cfg = presets.chaos()
    naive = cfg.build(retardation="naive").propagate(
        cfg.iterations, n_segments=cfg.n_segments, record_history=False)
    aberr = cfg.build(retardation="aberration").propagate(
        cfg.iterations, n_segments=cfg.n_segments, record_history=False)
    # aberration-cancelling retardation conserves energy & angular momentum
    # by ~2 orders of magnitude more than the naive strawman model.
    assert aberr.energy_drift < 0.1 * naive.energy_drift
    assert aberr.angmom_drift < 0.1 * naive.angmom_drift


def test_newtonian_chaos_machine_precision():
    cfg = presets.chaos()
    res = cfg.build(retardation="none").propagate(
        cfg.iterations, n_segments=cfg.n_segments, record_history=False)
    assert res.converged
    assert res.energy_drift < 1e-6          # spectral: essentially exact
    assert res.momentum_drift < 1e-9


# --------------------------------------------------------------------------- #
# EIH 1PN: stable, momentum-conserving, prograde precession
# --------------------------------------------------------------------------- #
def _eih_orbit(pn, e=0.0, c=30.0, n_per=120, norb=1, segs=1):
    G, M, a = 1.0, 1.0, 1.0
    rp = a * (1 - e)
    vp = np.sqrt(G * M * (1 + e) / (a * (1 - e)))
    period = 2 * np.pi * np.sqrt(a**3 / (G * M))
    central = Body("M", M, [0, 0, 0], [0, 0, 0])
    orb = Body("m", 1e-4, [rp, 0, 0], [0, vp, 0])
    sysm = VolterraSystem([central, orb], G=G, c=c, softening=1e-5,
                          t_span=norb * period, n_samples=n_per * norb,
                          use_retardation=False, pn=pn, relaxation=0.6)
    return sysm.propagate(60, n_segments=segs * norb, record_history=False), period


def test_eih_circular_stable():
    res, _ = _eih_orbit("eih", e=0.0, c=30.0, n_per=160)
    assert res.converged
    assert res.energy_drift < 1e-3          # tiny, stable (was a blow-up before the fix)
    assert res.momentum_drift < 1e-3


def test_eih_prograde_precession():
    # measure periapsis longitude drift over 3 orbits; EIH must advance prograde
    res, _ = _eih_orbit("eih", e=0.3, c=25.0, n_per=90, norb=3, segs=12)
    tr = res.trajectory[:, 1, :2]
    rr = np.linalg.norm(tr, axis=1)
    mins = [k for k in range(1, len(rr) - 1) if rr[k] < rr[k - 1] and rr[k] < rr[k + 1]]
    angs = np.array([np.arctan2(tr[k, 1], tr[k, 0]) for k in mins])
    assert len(angs) >= 2
    assert angs[-1] - angs[0] > 0.02        # net prograde advance
    assert res.momentum_drift < 1e-2        # EIH is momentum-conserving


# --------------------------------------------------------------------------- #
# Piecewise streaming: continuity, bounded memory, adaptivity, conservation
# --------------------------------------------------------------------------- #
def _stream(name, **kw):
    cfg = presets.get_preset(name)
    return PiecewiseVolterraStream.from_config(
        cfg, iterations=cfg.iterations, relaxation=cfg.relaxation,
        use_retardation=True, use_pn=False, **kw)


def test_stream_continuous_and_finite():
    s = _stream("Deep Chaos (3-Body)")
    prev = None
    for _ in range(8):
        w = s.step()
        assert np.isfinite(w.trajectory).all()
        if prev is not None:
            assert np.linalg.norm(w.trajectory[0] - prev) < 1e-9
        prev = w.trajectory[-1]


def test_stream_memory_bounded():
    s = _stream("Deep Chaos (3-Body)")
    sizes = []
    for _ in range(25):
        s.step()
        sizes.append(s.buf_R.nbytes)
    assert max(sizes) <= s.max_buffer_samples * s.N * 3 * 8


def test_stream_momentum_conserved():
    s = _stream("Solar System (Inner)")
    for _ in range(20):
        w = s.step()
    assert w.momentum_drift < 1e-6
    assert w.converged


def test_stream_adaptive_stays_finite_when_starved():
    # too few iterations at full span -> adaptive subdivision must keep it finite
    cfg = presets.chaos()
    s = PiecewiseVolterraStream.from_config(
        cfg, iterations=4, relaxation=cfg.relaxation,
        use_retardation=True, use_pn=False)
    prev = None
    for _ in range(6):
        w = s.step()
        assert np.isfinite(w.trajectory).all()
        assert w.span <= s.window_span
        if prev is not None:
            assert np.linalg.norm(w.trajectory[0] - prev) < 1e-9
        prev = w.trajectory[-1]


def test_stream_reset_rewinds():
    s = _stream("Solar System (Inner)")
    for _ in range(5):
        s.step()
    assert s.window_index == 5 and s.t_epoch > 0
    s.reset()
    assert s.window_index == 0 and s.t_epoch == 0.0


# --------------------------------------------------------------------------- #
# Every preset builds, propagates, stays finite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", presets.PRESET_NAMES)
def test_presets_build_and_run(name):
    cfg = presets.sandbox(seed=1) if name == "Custom Sandbox" else presets.get_preset(name)
    res = cfg.build().propagate(cfg.iterations, n_segments=cfg.n_segments,
                                record_history=False)
    assert np.isfinite(res.trajectory).all()
    assert res.trajectory.shape[1] == len(cfg.bodies)

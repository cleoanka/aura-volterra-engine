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
    counts = []
    for _ in range(25):
        s.step()
        counts.append(len(s.history.segments))
    # retained windows are bounded by ceil(buffer / min window) + 1, never grow
    assert max(counts[5:]) <= max(counts[:5]) + 1
    assert s.history.clamped == 0


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


# --------------------------------------------------------------------------- #
# State-dependent-delay (SD-DDE) core: history, breaking points, interpolation
# --------------------------------------------------------------------------- #
def _binary_bodies():
    v = 0.5 * np.sqrt(2.0)
    return [Body("a", 1.0, [0.5, 0, 0], [0, v, 0]), Body("b", 1.0, [-0.5, 0, 0], [0, -v, 0])]


def _sdde_stream(M, nwin, T, track, model="naive"):
    s = PiecewiseVolterraStream(
        _binary_bodies(), G=1.0, c=8.0, softening=0.0, use_retardation=True, use_pn=False,
        relaxation=1.0, window_span=T / nwin, window_samples=M, iterations=200,
        accelerator="picard", retardation=model, pn="none", tol=1e-13, adaptive=False,
        track_breaks=track)
    while s.t_epoch < T - 1e-12:
        s.window_span = min(T / nwin, T - s.t_epoch)
        s.step()
    return s


def test_barycentric_no_snapping_near_nodes():
    """Queries within 1e-8 of a node must not be snapped to the node value."""
    from aura_engine.operators import chebyshev_lobatto_nodes
    M = 32
    t = (chebyshev_lobatto_nodes(M) + 1.0) / 2.0
    w = barycentric_weights_cgl(M)
    tq = t[5] + np.array([0.0, 1e-12, 1e-9, 5e-9])
    err = np.abs(barycentric_interpolate(t, np.sin(3 * t), w, tq) - np.sin(3 * tq))
    assert err.max() < 1e-14


def test_segmented_batch_uses_computed_past():
    """Chained segments must read the *computed* retarded past (not a ballistic
    extrapolation from each segment epoch): result independent of segment count."""
    def run(nseg):
        s = VolterraSystem(_binary_bodies(), G=1.0, c=8.0, softening=0.0, t_span=2.0,
                           n_samples=120 * nseg, relaxation=1.0, accelerator="picard",
                           retardation="naive", pn="none")
        return s.propagate(200, n_segments=nseg, record_history=False, tol=1e-13).trajectory[-1]
    one = run(1)
    for nseg in (2, 8):
        assert np.abs(run(nseg) - one).max() < 1e-6     # was 4.8e-3 with ballistic past


def test_stream_matches_single_window_retarded():
    s = _sdde_stream(60, 8, 2.0, track=0)
    one = VolterraSystem(_binary_bodies(), G=1.0, c=8.0, softening=0.0, t_span=2.0,
                         n_samples=120, relaxation=1.0, accelerator="picard",
                         retardation="naive", pn="none").relax(200, tol=1e-13).trajectory[-1]
    assert np.abs(s.R_epoch - one).max() < 1e-6          # was 4e-6 with the linear buffer


def test_breaking_point_alignment_restores_spectral_convergence():
    """Without alignment the ballistic-history kink at t0 + tau caps accuracy;
    aligned windows converge to ~machine precision (M=32 vs M=48 agree)."""
    T = 2 * (2 * np.pi / np.sqrt(2.0))
    off32, off48 = (_sdde_stream(M, 8, T, track=0).R_epoch for M in (32, 48))
    on32, on48 = (_sdde_stream(M, 8, T, track=2).R_epoch for M in (32, 48))
    assert np.abs(off32 - off48).max() > 1e-7
    assert np.abs(on32 - on48).max() < 1e-12


def test_breaks_are_registered_and_bounded():
    T = 2 * (2 * np.pi / np.sqrt(2.0))
    s = _sdde_stream(32, 8, T, track=2)
    assert s.break_splits >= 2
    assert all(br[1] <= 2 for br in s.breaks)
    assert len(s.breaks) <= 8
    assert s.history.clamped == 0


def test_break_not_duplicated_into_sliver_window():
    """Regression: a truncated re-solve can end a hair before the break; the next
    window must not re-find it, cut a sliver window and duplicate the chain."""
    s = PiecewiseVolterraStream(
        _binary_bodies(), G=1.0, c=8.0, softening=0.0, use_retardation=True, use_pn=False,
        relaxation=1.0, window_span=0.3, window_samples=24, iterations=200,
        accelerator="picard", retardation="naive", pn="none", tol=1e-13, adaptive=False,
        track_breaks=3)
    spans = []
    while s.t_epoch < 1.0:
        spans.append(s.step().span)
    assert min(spans) > 0.05
    assert s.break_splits == 3
    times = sorted({round(br[0], 9) for br in s.breaks})
    assert all(b - a > 0.1 for a, b in zip(times, times[1:]))


@pytest.mark.parametrize("model", ["naive", "aberration"])
def test_vectorized_retarded_field_matches_pair_loop(model):
    cfg = presets.chaos()
    outs = []
    for vec in (True, False):
        s = PiecewiseVolterraStream(
            cfg.bodies, G=cfg.G, c=cfg.c, softening=cfg.softening, use_retardation=True,
            use_pn=False, relaxation=1.0, window_span=0.5, window_samples=24, iterations=100,
            accelerator="picard", retardation=model, pn="none", tol=1e-12, adaptive=False)
        s.vectorized = vec
        for _ in range(3):
            s.step()
        outs.append(s.R_epoch.copy())
    assert np.abs(outs[0] - outs[1]).max() < 1e-13


def test_warm_start_same_answer_fewer_iterations():
    T = 2 * (2 * np.pi / np.sqrt(2.0))
    res = {}
    for warm in (False, True):
        s = PiecewiseVolterraStream(
            _binary_bodies(), G=1.0, c=8.0, softening=0.0, use_retardation=True,
            use_pn=False, relaxation=1.0, window_span=T / 8, window_samples=32,
            iterations=200, accelerator="picard", retardation="naive", pn="none",
            tol=1e-13, adaptive=False, track_breaks=2)
        s.warm_start = warm
        it = 0
        while s.t_epoch < T - 1e-12:
            s.window_span = min(T / 8, T - s.t_epoch)
            it += s.step().iterations
        res[warm] = (s.R_epoch.copy(), it)
    assert np.abs(res[True][0] - res[False][0]).max() < 1e-12
    assert res[True][1] < res[False][1]


def _lw_stream(c, M=24, track=50):
    v = 0.5 * np.sqrt(2.0)
    bodies = [Body("e+", 1.0, [0.5, 0, 0], [0, v, 0]), Body("e-", 1.0, [-0.5, 0, 0], [0, -v, 0])]
    return PiecewiseVolterraStream(
        bodies, G=1.0, c=c, softening=0.0, use_retardation=True, use_pn=False,
        relaxation=1.0, window_span=0.25, window_samples=M, iterations=200,
        accelerator="picard", retardation="lw", pn="none", tol=1e-13, adaptive=False,
        track_breaks=track, charges=[1.0, -1.0])


def test_lienard_wiechert_static_limit_is_coulomb():
    """c -> infinity: LW + relativistic mechanics == Newtonian (G m m = q q = 1)."""
    lw = _lw_stream(1e12, track=0)
    g = PiecewiseVolterraStream(
        _binary_bodies(), G=1.0, c=1e12, softening=0.0, use_retardation=False, use_pn=False,
        relaxation=1.0, window_span=0.25, window_samples=24, iterations=200,
        accelerator="picard", retardation="none", pn="none", tol=1e-13, adaptive=False)
    for _ in range(8):
        lw.step(), g.step()
    assert np.abs(lw.R_epoch - g.R_epoch).max() < 1e-13


def test_lienard_wiechert_neutral_needs_all_levels():
    """Neutral DDE: accuracy is set by the tracked levels, not by M; tracking all
    levels makes M=16 and M=32 agree to round-off."""
    def run(M, track):
        s = _lw_stream(8.0, M=M, track=track)
        while s.t_epoch < 1.0 - 1e-12:
            s.window_span = min(0.25, 1.0 - s.t_epoch)
            s.step()
        return s.R_epoch
    m_effect = np.abs(run(16, 1) - run(32, 1)).max()           # plateau: M barely matters
    level_effect = np.abs(run(32, 1) - run(32, 50)).max()      # the tracked level does
    assert m_effect < 0.5 * level_effect
    assert np.abs(run(16, 50) - run(32, 50)).max() < 1e-14

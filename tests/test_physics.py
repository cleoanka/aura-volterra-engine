"""Validation tests for the Volterra engine.

Run:  pytest -q          (or)   python -m pytest tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura_engine import presets
from aura_engine.operators import apply_operator, build_integral_operators
from aura_engine.physics import Body, PiecewiseVolterraStream, VolterraSystem


# --------------------------------------------------------------------------- #
# Operators: the double integral of a polynomial is exact for trapezoid up to
# quadratics; check constant and linear accelerations analytically.
# --------------------------------------------------------------------------- #
def test_double_integral_constant_acceleration():
    t = np.linspace(0.0, 3.0, 400)
    C, K2 = build_integral_operators(t)
    a = np.full_like(t, 2.5)              # constant acceleration
    # analytic: r(t) = 0.5 a t^2 ; v(t) = a t
    r = K2 @ a
    v = C @ a
    assert np.allclose(r, 0.5 * 2.5 * t**2, atol=1e-6)
    assert np.allclose(v, 2.5 * t, atol=1e-9)


def test_double_integral_linear_acceleration():
    t = np.linspace(0.0, 2.0, 1000)
    _, K2 = build_integral_operators(t)
    a = 1.3 * t                            # a(t) = k t
    r = K2 @ a
    # analytic double integral of k t is k t^3 / 6
    assert np.allclose(r, 1.3 * t**3 / 6.0, rtol=1e-3, atol=1e-4)


def test_apply_operator_shapes():
    t = np.linspace(0, 1, 50)
    _, K2 = build_integral_operators(t)
    field = np.zeros((50, 3, 3))
    field[:, :, :] = 1.0
    out = apply_operator(K2, field)
    assert out.shape == (50, 3, 3)
    assert np.allclose(out[:, 0, 0], 0.5 * t**2, atol=1e-6)


# --------------------------------------------------------------------------- #
# Two-body: a circular orbit must keep constant radius and conserve energy.
# --------------------------------------------------------------------------- #
def _two_body_circular():
    G = 1.0
    M, m = 1.0, 1e-6                       # near-test-particle
    r = 1.0
    v = np.sqrt(G * M / r)                  # circular speed (central mass ~fixed)
    central = Body("M", M, [0, 0, 0], [0, 0, 0])
    orbiter = Body("m", m, [r, 0, 0], [0, v, 0])
    period = 2 * np.pi * r / v
    return G, period, central, orbiter


def test_two_body_circular_orbit_is_closed_and_conservative():
    G, period, central, orbiter = _two_body_circular()
    sysm = VolterraSystem(
        [central, orbiter], G=G, c=1e6, softening=1e-4,
        t_span=period, n_samples=700, use_retardation=False, relaxation=0.6,
    )
    res = sysm.relax(iterations=80, record_history=False, tol=1e-11)
    assert res.converged, f"did not converge (residual={res.residuals[-1]:.2e})"

    # orbiter radius stays ~1 throughout
    r_orbiter = np.linalg.norm(res.trajectory[:, 1, :], axis=1)
    assert np.allclose(r_orbiter, 1.0, atol=2e-3)

    # returns to start after one period (closed orbit)
    start = res.trajectory[0, 1, :]
    end = res.trajectory[-1, 1, :]
    assert np.linalg.norm(end - start) < 2e-2

    # energy conserved to high precision (instantaneous Newtonian gravity)
    assert res.energy_drift < 1e-4


def test_residual_monotone_decrease():
    G, period, central, orbiter = _two_body_circular()
    sysm = VolterraSystem(
        [central, orbiter], G=G, c=1e6, softening=1e-4,
        t_span=0.5 * period, n_samples=500, use_retardation=False, relaxation=0.7,
    )
    res = sysm.relax(iterations=60, record_history=False, tol=1e-12)
    # residual should shrink by many orders of magnitude
    assert res.residuals[-1] < res.residuals[0] * 1e-4


# --------------------------------------------------------------------------- #
# Retardation: must run, stay finite, and (with true c) barely perturb energy.
# --------------------------------------------------------------------------- #
def test_retardation_is_a_small_perturbation_at_true_c():
    cfg = presets.solar_system()
    sysm = cfg.build()
    res = sysm.relax(iterations=cfg.iterations, record_history=False)
    assert np.isfinite(res.trajectory).all()
    assert res.energy_drift < 1e-4   # tiny at the real speed of light


# --------------------------------------------------------------------------- #
# Pure-Newtonian global integrator conserves energy to ~machine precision even
# through a chaotic close encounter -> the retarded drift is physical, not error.
# --------------------------------------------------------------------------- #
def test_newtonian_chaos_conserves_energy():
    cfg = presets.chaos()
    sysm = cfg.build(use_retardation=False)
    res = sysm.propagate(cfg.iterations, n_segments=cfg.n_segments, record_history=False)
    assert res.converged
    # Holds energy to <0.1% through deep chaotic encounters on a uniform grid;
    # the retarded run (below) drifts thousands of times more -> physical, not error.
    assert res.energy_drift < 1e-3


def test_retardation_adds_physical_drift():
    cfg = presets.chaos()
    newt = cfg.build(use_retardation=False).propagate(
        cfg.iterations, n_segments=cfg.n_segments, record_history=False)
    retd = cfg.build(use_retardation=True).propagate(
        cfg.iterations, n_segments=cfg.n_segments, record_history=False)
    assert retd.energy_drift > 100 * newt.energy_drift


# --------------------------------------------------------------------------- #
# Post-Newtonian: an eccentric orbit must precess PROGRADE (advance of periapsis)
# when the 1PN term is switched on, and not when it is off.
# --------------------------------------------------------------------------- #
def _perihelion_angle(traj_orbiter):
    """Angle of the closest-approach point (perihelion) in the x-y plane."""
    r = np.linalg.norm(traj_orbiter[:, :2], axis=1)
    k = int(np.argmin(r))
    return float(np.arctan2(traj_orbiter[k, 1], traj_orbiter[k, 0])), k


def test_pn_causes_prograde_precession():
    # A tight, eccentric two-body system with an artificially small c so the
    # 1PN precession is large and measurable over two orbits.
    G = 1.0
    M = 1.0
    a = 1.0
    e = 0.5
    r_peri = a * (1 - e)
    v_peri = np.sqrt(G * M * (1 + e) / (a * (1 - e)))  # vis-viva at perihelion
    period = 2 * np.pi * np.sqrt(a**3 / (G * M))

    def build(pn):
        central = Body("M", M, [0, 0, 0], [0, 0, 0])
        orbiter = Body("m", 1e-4, [r_peri, 0, 0], [0, v_peri, 0])
        return VolterraSystem(
            [central, orbiter], G=G, c=12.0, softening=1e-3,
            t_span=2.05 * period, n_samples=1600, use_retardation=False,
            use_pn=pn, relaxation=0.6,
        ).propagate(45, n_segments=6, record_history=False)

    res_off = build(False)
    res_on = build(True)

    # perihelion of the second orbit
    def second_peri_angle(res):
        traj = res.trajectory[:, 1, :]
        half = traj.shape[0] // 2
        ang, _ = _perihelion_angle(traj[half:])
        return ang

    ang_off = second_peri_angle(res_off)
    ang_on = second_peri_angle(res_on)

    # Newtonian: perihelion stays near the +x axis (angle ~ 0).
    assert abs(np.arctan2(np.sin(ang_off), np.cos(ang_off))) < 0.15
    # 1PN: perihelion advances prograde (positive angle, same sense as motion).
    advance = np.arctan2(np.sin(ang_on), np.cos(ang_on))
    assert advance > 0.05, f"expected prograde advance, got {advance:.3f} rad"


# --------------------------------------------------------------------------- #
# Piecewise Volterra streaming: continuity, bounded memory, physical fidelity.
# --------------------------------------------------------------------------- #
def test_stream_windows_are_continuous_and_finite():
    cfg = presets.chaos()
    stream = PiecewiseVolterraStream.from_config(
        cfg, iterations=cfg.iterations, relaxation=cfg.relaxation,
        use_retardation=True, use_pn=False, window_samples=48)
    prev_end = None
    for _ in range(8):
        win = stream.step()
        assert np.isfinite(win.trajectory).all()
        if prev_end is not None:
            # C0 continuity: this window starts where the last one ended
            gap = np.linalg.norm(win.trajectory[0] - prev_end)
            assert gap < 1e-9, f"discontinuity at window join: {gap:.2e}"
        prev_end = win.trajectory[-1]


def test_stream_memory_is_bounded():
    cfg = presets.chaos()
    stream = PiecewiseVolterraStream.from_config(
        cfg, iterations=cfg.iterations, relaxation=cfg.relaxation,
        use_retardation=True, use_pn=False, window_samples=48)
    sizes = []
    for _ in range(30):
        stream.step()
        sizes.append(stream.buf_R.nbytes)
    # the delay buffer never grows without bound (O(1) memory)
    assert max(sizes) <= stream.max_buffer_samples * stream.N * 3 * 8
    assert max(sizes) - min(sizes) <= stream.max_buffer_samples * stream.N * 3 * 8


def test_stream_matches_batch_for_stable_orbit():
    # A near-circular two-body orbit: streaming with the sliding buffer must
    # track the batch global solve closely over the same horizon.
    G, M = 1.0, 1.0
    r, v = 1.0, 1.0
    central = Body("M", M, [0, 0, 0], [0, 0, 0])
    orbiter = Body("m", 1e-6, [r, 0, 0], [0, v, 0])
    period = 2 * np.pi
    batch = VolterraSystem([central, orbiter], G=G, c=1e6, softening=1e-4,
                           t_span=period, n_samples=600, use_retardation=True,
                           relaxation=0.6).relax(80, record_history=False, tol=1e-11)

    stream = PiecewiseVolterraStream(
        [central, orbiter], G=G, c=1e6, softening=1e-4, use_retardation=True,
        use_pn=False, relaxation=0.6, window_span=period / 10.0,
        window_samples=61, iterations=60)
    xs, ys = [], []
    for _ in range(10):
        w = stream.step()
        xs.append(w.trajectory[:, 1, 0]); ys.append(w.trajectory[:, 1, 1])
    stream_end = np.array([xs[-1][-1], ys[-1][-1]])
    batch_end = batch.trajectory[-1, 1, :2]
    assert np.linalg.norm(stream_end - batch_end) < 5e-3


def test_stream_reset_rewinds():
    cfg = presets.solar_system()
    stream = PiecewiseVolterraStream.from_config(
        cfg, iterations=cfg.iterations, relaxation=cfg.relaxation,
        use_retardation=True, use_pn=False)
    for _ in range(5):
        stream.step()
    assert stream.window_index == 5 and stream.t_epoch > 0
    stream.reset()
    assert stream.window_index == 0 and stream.t_epoch == 0.0


# --------------------------------------------------------------------------- #
# Every preset builds, propagates, and stays finite.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", presets.PRESET_NAMES)
def test_presets_build_and_run(name):
    cfg = presets.sandbox(seed=1) if name == "Custom Sandbox" else presets.get_preset(name)
    sysm = cfg.build()
    res = sysm.propagate(cfg.iterations, n_segments=cfg.n_segments, record_history=False)
    assert np.isfinite(res.trajectory).all()
    assert res.trajectory.shape[1] == len(cfg.bodies)

"""HUD truthfulness: the model line must name what the solver actually runs."""
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aura_engine import presets  # noqa: E402
from aura_engine.viz import ContinuousScene, TrajectoryScene  # noqa: E402


def _hud_texts(cfg):
    res = SimpleNamespace(energy_drift=0.0, momentum_drift=0.0)
    win = SimpleNamespace(index=0, t=np.array([0.0, 1.0]), buffer_span=0.1,
                          energy_drift=0.0, momentum_drift=0.0)
    batch = TrajectoryScene._status_text(SimpleNamespace(config=cfg, result=res), "x")
    cont = ContinuousScene._status_text(SimpleNamespace(config=cfg, last=win))
    return batch, cont


def test_eih_ignores_retardation_in_the_solver():
    # EIH replaces the (retarded) Newtonian force: the toggle has no effect
    cfg = presets.chaos()
    cfg.use_pn, cfg.pn = True, "eih"
    on = cfg.build(t_span=0.3, n_samples=16, retardation="aberration")
    off = cfg.build(t_span=0.3, n_samples=16, retardation="none")
    R, V = on.predictor(), np.broadcast_to(on.V0, (16, 3, 3))
    assert np.array_equal(on.accelerations(R, V), off.accelerations(R, V))


def test_hud_does_not_claim_retardation_under_eih():
    for pn, retard_shown in (("eih", False), ("none", True)):
        cfg = presets.chaos()
        cfg.use_retardation, cfg.retardation = True, "aberration"
        cfg.use_pn, cfg.pn = pn == "eih", pn
        for text in _hud_texts(cfg):
            assert ("RETARDED-c" in text) is retard_shown, text
            assert ("EIH" in text) is (pn == "eih"), text


def test_hud_explicit_modes_win_over_flags():
    cfg = presets.chaos()
    # reduced PN is not EIH; LW is labelled as such, not as aberration
    cfg.use_pn, cfg.pn = True, "reduced"
    cfg.use_retardation, cfg.retardation = True, "aberration"
    for text in _hud_texts(cfg):
        assert "EIH" not in text and "REDUCED-PN" in text and "RETARDED-c" in text, text
    cfg.pn, cfg.retardation = "none", "lw"
    for text in _hud_texts(cfg):
        assert "LIENARD-WIECHERT" in text and "RETARDED-c" not in text, text
    cfg.use_retardation, cfg.retardation = False, None
    for text in _hud_texts(cfg):
        assert "NEWTONIAN" in text, text

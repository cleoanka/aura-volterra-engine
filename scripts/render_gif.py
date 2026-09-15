"""Render a preset to an animated GIF (headless, Agg backend).

Usage:
    python scripts/render_gif.py "Deep Chaos (3-Body)" assets/chaos.gif
    python scripts/render_gif.py all           # render every preset to assets/

Doubles as an end-to-end smoke test of the engine + visualization pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura_engine import presets                     # noqa: E402
from aura_engine.viz import BG, TrajectoryScene     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SLUGS = {
    "Solar System (Inner)": "solar",
    "Orbital Spacecraft": "spacecraft",
    "Deep Chaos (3-Body)": "chaos",
    "Custom Sandbox": "sandbox",
}


def render(name: str, out: Path, fps: int = 30, seed: int | None = 3) -> None:
    cfg = presets.sandbox(seed=seed) if name == "Custom Sandbox" else presets.get_preset(name)
    system = cfg.build()
    print(f"[{name}] relaxing ({cfg.n_segments} window(s))...", flush=True)
    record = cfg.n_segments == 1  # morph history only meaningful for single window
    result = system.propagate(cfg.iterations, n_segments=cfg.n_segments,
                              record_history=record)
    print(f"[{name}] converged={result.converged}  E-drift={result.energy_drift:.2e}",
          flush=True)

    fig = plt.figure(figsize=(6.0, 6.0), dpi=84, facecolor=BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    scene = TrajectoryScene(fig, ax)
    # Keep GIFs lean: advance propagation enough to cap ~one loop near 160 frames.
    scene.stride = max(3, result.trajectory.shape[0] // 150)
    scene.build(result, cfg)

    frames = scene.total_gif_frames()
    print(f"[{name}] writing {frames} frames -> {out}", flush=True)
    anim = FuncAnimation(fig, scene.render_frame, frames=frames, blit=True, interval=1000 // fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[{name}] done: {out} ({out.stat().st_size//1024} KB)", flush=True)


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "all":
        for name, slug in SLUGS.items():
            render(name, ROOT / "assets" / f"{slug}.gif")
        return
    name = args[0]
    out = Path(args[1]) if len(args) > 1 else ROOT / "assets" / f"{SLUGS.get(name, 'out')}.gif"
    render(name, out)


if __name__ == "__main__":
    main()

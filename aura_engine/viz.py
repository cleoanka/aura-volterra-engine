"""
aura_engine.viz
===============

Shared matplotlib scene for Project AURA, used by both the live customtkinter
app (:mod:`main`) and the headless GIF renderer (``scripts/render_gif.py``).

The scene animates in two phases on one set of blit-friendly artists:

* **RELAXATION** -- the ballistic guess bends into curved orbits. For a single
  global window we replay the *true* Picard iteration history; for a segmented
  (chaotic) solve we progressively reveal the converged spaghetti.
* **PROPAGATION** -- comet heads sweep along the converged trajectory with
  glowing tails, looping forever.

Everything heavy is precomputed; :meth:`TrajectoryScene.step` only mutates line
data, so the Tk event loop never blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- AURA palette ---------------------------------------------------------- #
BG = "#05070f"
PANEL = "#0b0f1a"
GRID = "#152033"
ACCENT = "#38e8ff"
TEXT = "#c7d4e6"
MUTED = "#5b6b82"


def auto_limits(traj_disp: np.ndarray, hint: float | None = None) -> tuple[float, np.ndarray]:
    """Robustly frame a trajectory: half-width and centre in display units.

    Uses the 98th-percentile radius (so a single escaping body cannot blow the
    view out) with a margin, floored by an optional preset ``hint``.
    """
    pts = traj_disp.reshape(-1, 3)[:, :2]
    center = np.median(pts, axis=0)
    rad = np.linalg.norm(pts - center, axis=1)
    half = float(np.percentile(rad, 98)) * 1.18
    if hint is not None:
        half = max(half, 0.6 * hint)
    half = max(half, 1e-6)
    return half, center


def _starfield(ax, half: float, center: np.ndarray, seed: int = 7) -> None:
    """Scatter a static, twinkly star background (captured in the blit bg)."""
    rng = np.random.default_rng(seed)
    n = 220
    xs = center[0] + rng.uniform(-half, half, n) * 1.4
    ys = center[1] + rng.uniform(-half, half, n) * 1.4
    sizes = rng.uniform(0.2, 2.4, n) ** 2
    alphas = rng.uniform(0.15, 0.8, n)
    ax.scatter(xs, ys, s=sizes, c="white", alpha=alphas, linewidths=0, zorder=0)


@dataclass
class _FramePlan:
    n_relax: int
    n_traj: int
    tail: int


class TrajectoryScene:
    """Owns the axes artists and advances the two-phase animation."""

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.playing = True
        self.stride = 2          # propagation frames advanced per tick
        self.interval = 20       # ms between ticks (set by the app's speed slider)
        self._artists: list = []
        self._phase = "relax"
        self._i = 0
        self._g = 0
        self.result = None
        self.config = None

    # ------------------------------------------------------------------ #
    def build(self, result, config) -> list:
        """Rebuild all artists for a freshly solved system. Returns blit list."""
        ax, lu = self.ax, config.length_unit
        self.result = result
        self.config = config

        traj = result.trajectory / lu                      # (n, N, 3)
        self.traj = traj
        self.history = [h / lu for h in result.history]    # possibly []
        self.N = traj.shape[1]

        half, center = auto_limits(traj, config.view_radius)
        self.plan = _FramePlan(
            n_relax=(len(self.history) if self.history else 70),
            n_traj=traj.shape[0],
            tail=max(24, traj.shape[0] // 12),
        )

        ax.clear()
        ax.set_facecolor(BG)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        _starfield(ax, half, center)

        # scale label
        ax.text(0.015, 0.03, f"scale: {half*2:.2f} {config.unit_label} across",
                transform=ax.transAxes, color=MUTED, fontsize=8, family="monospace")

        self.ghost, self.active, self.glow, self.head = [], [], [], []
        for i in range(self.N):
            body = config.bodies[i]
            col = body.color
            # faint full-trajectory "ghost" (static target during relaxation)
            (g,) = ax.plot(traj[:, i, 0], traj[:, i, 1], color=col, lw=1.0,
                           alpha=0.14, zorder=1)
            # active line (morph / reveal / comet tail)
            (a,) = ax.plot([], [], color=col, lw=1.8, alpha=0.95, zorder=3,
                           solid_capstyle="round")
            # glow + bright head markers
            (gl,) = ax.plot([], [], marker="o", ms=body.radius * 1.9, mfc=col,
                            mec="none", alpha=0.20, zorder=4, ls="none")
            (hd,) = ax.plot([], [], marker="o", ms=body.radius, mfc=col,
                            mec="white", mew=0.4, alpha=1.0, zorder=5, ls="none")
            self.ghost.append(g)
            self.active.append(a)
            self.glow.append(gl)
            self.head.append(hd)

        self.status = ax.text(
            0.015, 0.955, "", transform=ax.transAxes, color=ACCENT, fontsize=10,
            family="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=GRID, alpha=0.75),
        )

        # animated artists for blitting
        self._artists = self.active + self.glow + self.head + [self.status]
        self._phase = "relax"
        self._i = 0
        self._g = 0
        self._apply(("relax", 0))
        return self._artists

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._phase = "relax"
        self._i = 0
        self._g = 0

    def total_gif_frames(self) -> int:
        """Frames for one full GIF: relaxation + a single propagation loop."""
        return self.plan.n_relax + self.plan.n_traj // self.stride

    # ------------------------------------------------------------------ #
    def _status_text(self, phase_label: str) -> str:
        cfg, res = self.config, self.result
        flags = []
        if cfg.use_retardation:
            flags.append("RETARDED-c")
        if cfg.use_pn:
            flags.append("1PN")
        flag_s = " ".join(flags) if flags else "NEWTONIAN"
        return (
            f"PROJECT AURA — VOLTERRA ENGINE\n"
            f"phase   : {phase_label}\n"
            f"bodies  : {self.N}    windows: {cfg.n_segments}\n"
            f"model   : {flag_s}\n"
            f"E-drift : {res.energy_drift:.2e}"
        )

    def _apply(self, state) -> list:
        phase, idx = state
        traj = self.traj
        tail = self.plan.tail
        if phase == "relax":
            if self.history:
                cur = self.history[min(idx, len(self.history) - 1)]
                frac = (idx + 1) / len(self.history)
                for i in range(self.N):
                    self.active[i].set_data(cur[:, i, 0], cur[:, i, 1])
                    self.head[i].set_data([cur[-1, i, 0]], [cur[-1, i, 1]])
                    self.glow[i].set_data([cur[-1, i, 0]], [cur[-1, i, 1]])
                self.status.set_text(self._status_text(f"RELAXATION  {frac*100:4.0f}%"))
            else:  # segmented: reveal the converged path progressively
                g = max(1, int((idx + 1) / self.plan.n_relax * traj.shape[0]))
                for i in range(self.N):
                    self.active[i].set_data(traj[:g, i, 0], traj[:g, i, 1])
                    self.head[i].set_data([traj[g - 1, i, 0]], [traj[g - 1, i, 1]])
                    self.glow[i].set_data([traj[g - 1, i, 0]], [traj[g - 1, i, 1]])
                self.status.set_text(
                    self._status_text(f"RELAXATION  {(idx+1)/self.plan.n_relax*100:4.0f}%")
                )
        else:  # propagate
            g = idx
            lo = max(0, g - tail)
            for i in range(self.N):
                self.active[i].set_data(traj[lo:g + 1, i, 0], traj[lo:g + 1, i, 1])
                self.head[i].set_data([traj[g, i, 0]], [traj[g, i, 1]])
                self.glow[i].set_data([traj[g, i, 0]], [traj[g, i, 1]])
            t = self.result.t[g] / self.config.time_unit
            self.status.set_text(
                self._status_text(f"PROPAGATION  t={t:6.2f} {self.config.time_label}")
            )
        return self._artists

    # ------------------------------------------------------------------ #
    def step(self, *_args) -> list:
        """One animation tick (FuncAnimation callback)."""
        if not self.playing or self.result is None:
            return self._artists
        if self._phase == "relax":
            self._i += 1
            if self._i >= self.plan.n_relax:
                self._phase = "prop"
                self._g = 0
                return self._apply(("prop", 0))
            return self._apply(("relax", self._i))
        else:
            self._g = (self._g + self.stride) % self.plan.n_traj
            return self._apply(("prop", self._g))

    def render_frame(self, frame: int) -> list:
        """Deterministic frame for GIF export (frame 0..total_gif_frames)."""
        if frame < self.plan.n_relax:
            return self._apply(("relax", frame))
        g = ((frame - self.plan.n_relax) * self.stride) % self.plan.n_traj
        return self._apply(("prop", g))


class ContinuousScene:
    """Streaming scene for the infinite Piecewise-Volterra run.

    Windows arrive one at a time; their samples are appended to fixed-length
    rolling buffers (so RAM stays bounded), and the view gently follows the
    system. The moving view means the cached blit background is invalid every
    tick, so this scene uses full redraws -- cheap for a handful of poly-lines
    and macOS-safe when driven from a Tk ``after`` loop.
    """

    def __init__(self, fig, ax, maxlen: int = 2600, tail: int = 280):
        self.fig = fig
        self.ax = ax
        self.maxlen = maxlen
        self.tail = tail
        self.last = None
        self.config = None

    def build(self, config) -> None:
        ax = self.ax
        self.config = config
        self.lu = config.length_unit
        self.N = len(config.bodies)
        self.bufx = [np.empty(0) for _ in range(self.N)]
        self.bufy = [np.empty(0) for _ in range(self.N)]
        self._init_view = False
        self.last = None

        ax.clear()
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)

        # star field pinned to the axes (fraction coords) so it never scrolls
        rng = np.random.default_rng(11)
        n = 240
        ax.scatter(rng.uniform(0, 1, n), rng.uniform(0, 1, n),
                   s=rng.uniform(0.2, 2.4, n) ** 2, c="white",
                   alpha=rng.uniform(0.12, 0.8, n), linewidths=0,
                   transform=ax.transAxes, zorder=0)

        self.faint, self.tailln, self.glow, self.head = [], [], [], []
        for body in config.bodies:
            col = body.color
            (f,) = ax.plot([], [], color=col, lw=1.0, alpha=0.18, zorder=1)
            (tl,) = ax.plot([], [], color=col, lw=1.9, alpha=0.95, zorder=3,
                            solid_capstyle="round")
            (gl,) = ax.plot([], [], marker="o", ms=body.radius * 1.9, mfc=col,
                            mec="none", alpha=0.20, zorder=4, ls="none")
            (hd,) = ax.plot([], [], marker="o", ms=body.radius, mfc=col,
                            mec="white", mew=0.4, zorder=5, ls="none")
            self.faint.append(f)
            self.tailln.append(tl)
            self.glow.append(gl)
            self.head.append(hd)

        self.status = ax.text(
            0.015, 0.955, "", transform=ax.transAxes, color=ACCENT, fontsize=10,
            family="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=GRID, alpha=0.78),
        )
        self._half = float(config.view_radius or 1.0)
        self._cx = self._cy = 0.0
        ax.set_xlim(-self._half, self._half)
        ax.set_ylim(-self._half, self._half)
        ax.set_aspect("equal")

    def reset(self) -> None:
        for i in range(self.N):
            self.bufx[i] = np.empty(0)
            self.bufy[i] = np.empty(0)
        self._init_view = False
        self.last = None

    def push(self, win) -> None:
        """Append one window's samples (physical units) to the rolling buffers."""
        R = win.trajectory / self.lu
        for i in range(self.N):
            self.bufx[i] = np.concatenate([self.bufx[i], R[:, i, 0]])[-self.maxlen:]
            self.bufy[i] = np.concatenate([self.bufy[i], R[:, i, 1]])[-self.maxlen:]
        self.last = win

    def _status_text(self) -> str:
        cfg, w = self.config, self.last
        flags = []
        if cfg.use_retardation:
            flags.append("RETARDED-c")
        if cfg.use_pn:
            flags.append("1PN")
        model = " ".join(flags) if flags else "NEWTONIAN"
        t_disp = w.t[-1] / cfg.time_unit
        buf_disp = w.buffer_span / cfg.time_unit
        return (
            f"PROJECT AURA — CONTINUOUS  ∞\n"
            f"window  : {w.index:>6d}\n"
            f"sim t   : {t_disp:>10.2f} {cfg.time_label}\n"
            f"buffer  : {buf_disp:.3g} {cfg.time_label}  (sliding)\n"
            f"model   : {model}\n"
            f"drift/W : {w.energy_drift:.2e}"
        )

    def redraw(self) -> None:
        if self.last is None:
            return
        heads = []
        for i in range(self.N):
            x, y = self.bufx[i], self.bufy[i]
            self.faint[i].set_data(x, y)
            self.tailln[i].set_data(x[-self.tail:], y[-self.tail:])
            if x.size:
                self.head[i].set_data([x[-1]], [y[-1]])
                self.glow[i].set_data([x[-1]], [y[-1]])
                heads.append((x[-1], y[-1]))

        # follow the system: ease centre/zoom toward the bound cluster
        if heads:
            allx = np.concatenate(self.bufx)
            ally = np.concatenate(self.bufy)
            tcx, tcy = float(np.mean([h[0] for h in heads])), float(np.mean([h[1] for h in heads]))
            rad = np.hypot(allx - tcx, ally - tcy)
            thalf = max(float(np.percentile(rad, 96)) * 1.18, 1e-9)
            if not self._init_view:
                self._cx, self._cy, self._half = tcx, tcy, thalf
                self._init_view = True
            else:
                a = 0.10
                self._cx += a * (tcx - self._cx)
                self._cy += a * (tcy - self._cy)
                self._half += a * (thalf - self._half)
            self.ax.set_xlim(self._cx - self._half, self._cx + self._half)
            self.ax.set_ylim(self._cy - self._half, self._cy + self._half)

        self.status.set_text(self._status_text())

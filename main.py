"""
Project AURA -- Volterra N-Body Engine
======================================

Futuristic, dark-mode desktop cockpit for the global waveform-relaxation
orbital engine. A control panel drives an embedded matplotlib canvas that first
animates the *waveform relaxation* (ballistic guess bending into orbits) and
then *propagates* comet heads along the converged trajectory.

Run:
    python main.py

Heavy solves run on a worker thread so the Tk event loop -- and therefore the
whole macOS window -- never freezes. Only line data is mutated on the main
thread each animation tick (blitting), keeping playback smooth on Apple silicon.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation          # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure                    # noqa: E402

try:
    import customtkinter as ctk
except ModuleNotFoundError:  # pragma: no cover - friendly guidance
    sys.stderr.write(
        "\n[Project AURA] customtkinter is not installed.\n"
        "  python -m venv .venv && source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n\n"
    )
    raise

from aura_engine import PiecewiseVolterraStream, presets  # noqa: E402
from aura_engine.viz import (                              # noqa: E402
    ACCENT, BG, GRID, MUTED, PANEL, TEXT, ContinuousScene, TrajectoryScene,
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class AuraApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Project AURA  —  Volterra N-Body Engine")
        self.geometry("1240x820")
        self.minsize(1040, 660)
        self.configure(fg_color=BG)

        self.mono = ctk.CTkFont(family="SF Mono", size=12)
        self.mono_small = ctk.CTkFont(family="SF Mono", size=11)
        self.title_font = ctk.CTkFont(family="SF Mono", size=26, weight="bold")
        self.sub_font = ctk.CTkFont(family="SF Mono", size=12)

        self._solving = False
        self._solve_job = None
        self._pending = False
        self._sandbox_seed = 1
        self.anim: FuncAnimation | None = None
        self.result = None
        self.cfg = None
        self._result_queue: queue.Queue = queue.Queue()

        # --- continuous / infinite-run streaming state --------------------- #
        self._continuous = False
        self._stream = None
        self._stream_thread: threading.Thread | None = None
        self._stream_stop = threading.Event()
        self._stream_pause = threading.Event()
        self._window_queue: queue.Queue = queue.Queue(maxsize=3)
        self._cont_job = None
        self._windows_seen = 0

        self._build_ui()
        self._load_preset_defaults(self.preset_var.get())  # tuned depth/damping
        self.after(60, self._drain_queue)    # main-thread poller for worker results
        self.after(200, self._solve_async)   # first render

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        panel = ctk.CTkFrame(self, width=340, corner_radius=0, fg_color=PANEL)
        panel.grid(row=0, column=0, sticky="nsw")
        panel.grid_propagate(False)

        pad = dict(padx=22, sticky="ew")

        # --- branding -------------------------------------------------- #
        ctk.CTkLabel(panel, text="◤ AURA", font=self.title_font,
                     text_color=ACCENT).grid(row=0, column=0, pady=(26, 0), **pad)
        ctk.CTkLabel(panel, text="VOLTERRA  N-BODY  ENGINE", font=self.sub_font,
                     text_color=MUTED).grid(row=1, column=0, pady=(0, 4), **pad)
        ctk.CTkLabel(panel, text="global waveform relaxation · retarded gravity",
                     font=self.mono_small, text_color=MUTED,
                     wraplength=290, justify="left").grid(row=2, column=0, pady=(0, 14), **pad)
        self._divider(panel, 3)

        # --- preset ---------------------------------------------------- #
        ctk.CTkLabel(panel, text="SCENARIO", font=self.mono_small,
                     text_color=TEXT, anchor="w").grid(row=4, column=0, pady=(12, 4), **pad)
        self.preset_var = ctk.StringVar(value=presets.PRESET_NAMES[0])
        self.preset_menu = ctk.CTkOptionMenu(
            panel, values=presets.PRESET_NAMES, variable=self.preset_var,
            font=self.mono, dropdown_font=self.mono, command=self._on_preset,
            fg_color="#12203a", button_color="#1b3358", button_hover_color="#26507f",
        )
        self.preset_menu.grid(row=5, column=0, pady=(0, 6), **pad)

        self.desc = ctk.CTkLabel(panel, text="", font=self.mono_small,
                                 text_color=MUTED, wraplength=294, justify="left")
        self.desc.grid(row=6, column=0, pady=(0, 8), **pad)

        # --- iteration depth ------------------------------------------ #
        self.iter_label = ctk.CTkLabel(panel, text="RELAXATION DEPTH   45",
                                       font=self.mono_small, text_color=TEXT, anchor="w")
        self.iter_label.grid(row=7, column=0, pady=(6, 2), **pad)
        self.iter_slider = ctk.CTkSlider(panel, from_=10, to=90, number_of_steps=16,
                                         command=self._on_iter, button_color=ACCENT,
                                         progress_color="#1b3358")
        self.iter_slider.set(45)
        self.iter_slider.grid(row=8, column=0, pady=(0, 8), **pad)

        # --- relaxation factor ---------------------------------------- #
        self.relax_label = ctk.CTkLabel(panel, text="DAMPING  w = 0.55",
                                        font=self.mono_small, text_color=TEXT, anchor="w")
        self.relax_label.grid(row=9, column=0, pady=(6, 2), **pad)
        self.relax_slider = ctk.CTkSlider(panel, from_=0.2, to=0.9, number_of_steps=14,
                                          command=self._on_relax, button_color=ACCENT,
                                          progress_color="#1b3358")
        self.relax_slider.set(0.55)
        self.relax_slider.grid(row=10, column=0, pady=(0, 8), **pad)

        # --- physics toggles ------------------------------------------ #
        self.retard_switch = ctk.CTkSwitch(panel, text="Retarded potentials (finite c)",
                                           font=self.mono_small, command=self._schedule_solve,
                                           progress_color=ACCENT)
        self.retard_switch.select()
        self.retard_switch.grid(row=11, column=0, pady=(8, 4), **pad)
        self.pn_switch = ctk.CTkSwitch(panel, text="Post-Newtonian (1PN precession)",
                                       font=self.mono_small, command=self._schedule_solve,
                                       progress_color=ACCENT)
        self.pn_switch.grid(row=12, column=0, pady=(0, 8), **pad)
        self.cont_switch = ctk.CTkSwitch(
            panel, text="Continuous Mode  (∞ infinite run)", font=self.mono_small,
            command=self._on_continuous_toggle, progress_color="#ffd15d")
        self.cont_switch.grid(row=13, column=0, pady=(0, 10), **pad)

        # --- speed ----------------------------------------------------- #
        self.speed_label = ctk.CTkLabel(panel, text="PLAYBACK SPEED   2x",
                                        font=self.mono_small, text_color=TEXT, anchor="w")
        self.speed_label.grid(row=14, column=0, pady=(4, 2), **pad)
        self.speed_slider = ctk.CTkSlider(panel, from_=1, to=8, number_of_steps=7,
                                          command=self._on_speed, button_color=ACCENT,
                                          progress_color="#1b3358")
        self.speed_slider.set(2)
        self.speed_slider.grid(row=15, column=0, pady=(0, 12), **pad)
        self._divider(panel, 16)

        # --- transport buttons ---------------------------------------- #
        btns = ctk.CTkFrame(panel, fg_color="transparent")
        btns.grid(row=17, column=0, pady=(14, 6), **pad)
        btns.grid_columnconfigure((0, 1), weight=1)
        self.play_btn = ctk.CTkButton(btns, text="⏸  Pause", font=self.mono,
                                      command=self._toggle_play, fg_color="#1b3358",
                                      hover_color="#26507f")
        self.play_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        ctk.CTkButton(btns, text="⭮  Restart", font=self.mono, command=self._reset,
                      fg_color="#12203a", hover_color="#1b3358").grid(
            row=0, column=1, padx=(5, 0), sticky="ew")
        self.reroll_btn = ctk.CTkButton(panel, text="🎲  Re-roll sandbox", font=self.mono,
                                        command=self._reroll, fg_color="#12203a",
                                        hover_color="#1b3358")
        self.reroll_btn.grid(row=18, column=0, pady=(2, 10), **pad)

        # --- status readout ------------------------------------------- #
        self._divider(panel, 19)
        self.status = ctk.CTkLabel(panel, text="", font=self.mono_small,
                                   text_color=ACCENT, wraplength=294, justify="left",
                                   anchor="w")
        self.status.grid(row=20, column=0, pady=(14, 20), **pad)

        # --- plot area ------------------------------------------------- #
        plot_frame = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        self.fig = Figure(figsize=(7.4, 7.4), dpi=100, facecolor=BG)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax.set_facecolor(BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.scene = TrajectoryScene(self.fig, self.ax)
        self.scene.stride = 2
        self.cont_scene = ContinuousScene(self.fig, self.ax)

    def _divider(self, parent, row: int) -> None:
        ctk.CTkFrame(parent, height=1, fg_color=GRID).grid(
            row=row, column=0, padx=22, pady=(4, 0), sticky="ew")

    # ------------------------------------------------------------------ #
    # Control callbacks
    # ------------------------------------------------------------------ #
    def _load_preset_defaults(self, name: str) -> None:
        """Sync the depth/damping sliders to a scenario's tuned values."""
        cfg = (presets.sandbox(seed=self._sandbox_seed)
               if name == "Custom Sandbox" else presets.get_preset(name))
        self.iter_slider.set(cfg.iterations)
        self.iter_label.configure(text=f"RELAXATION DEPTH   {cfg.iterations:d}")
        self.relax_slider.set(cfg.relaxation)
        self.relax_label.configure(text=f"DAMPING  w = {cfg.relaxation:.2f}")

    def _on_preset(self, value: str) -> None:
        self._load_preset_defaults(value)
        self.reroll_btn.configure(
            state="normal" if value == "Custom Sandbox" else "disabled")
        self._schedule_solve()

    def _on_iter(self, v: float) -> None:
        self.iter_label.configure(text=f"RELAXATION DEPTH   {int(v):d}")
        self._schedule_solve()

    def _on_relax(self, v: float) -> None:
        self.relax_label.configure(text=f"DAMPING  w = {v:.2f}")
        self._schedule_solve()

    def _on_speed(self, v: float) -> None:
        self.scene.stride = int(v)               # batch: propagation stride
        self.speed_label.configure(text=f"PLAYBACK SPEED   {int(v)}x")

    def _on_continuous_toggle(self) -> None:
        self._continuous = bool(self.cont_switch.get())
        self._schedule_solve()

    def _reroll(self) -> None:
        self._sandbox_seed = (self._sandbox_seed + 1) % 100_000
        self._solve_async()

    def _toggle_play(self) -> None:
        if self._continuous:
            if self._stream is None:
                return
            if self._stream_pause.is_set():      # resume
                self._stream_pause.clear()
                if self._cont_job is None:
                    self._cont_job = self.after(30, self._continuous_tick)
                self.play_btn.configure(text="⏸  Pause")
            else:                                # pause (also idles the worker)
                self._stream_pause.set()
                if self._cont_job is not None:
                    self.after_cancel(self._cont_job)
                    self._cont_job = None
                self.play_btn.configure(text="▶  Play")
            return
        if self.anim is None:
            return
        if self.scene.playing:
            self.scene.playing = False
            self.anim.event_source.stop()
            self.play_btn.configure(text="▶  Play")
        else:
            self.scene.playing = True
            self.anim.event_source.start()
            self.play_btn.configure(text="⏸  Pause")

    def _reset(self) -> None:
        if self._continuous:
            self._start_continuous()             # rewind: restart the stream at t=0
        elif self.result is not None:
            self.scene.reset()

    # ------------------------------------------------------------------ #
    # Async solve (worker thread) + main-thread rebuild
    # ------------------------------------------------------------------ #
    def _schedule_solve(self, *_args) -> None:
        if self._solve_job is not None:
            self.after_cancel(self._solve_job)
        self._solve_job = self.after(320, self._solve_async)

    def _solve_async(self) -> None:
        self._solve_job = None
        if self._continuous:                    # infinite-run streaming path
            self._stop_batch()
            self._start_continuous()
            return
        self._stop_continuous()                 # leaving continuous mode
        if self._solving:                       # coalesce: retry shortly
            self._pending = True
            return
        self._solving = True
        self._pending = False

        name = self.preset_var.get()
        iters = int(self.iter_slider.get())
        relax = float(self.relax_slider.get())
        retard = bool(self.retard_switch.get())
        pn = bool(self.pn_switch.get())
        seed = self._sandbox_seed
        self.status.configure(text="◍ RELAXING WAVEFORM…", text_color="#ffd15d")

        def work() -> None:
            # Runs off the main thread: touches NO Tk objects, only the queue.
            t0 = time.perf_counter()
            try:
                cfg = (presets.sandbox(seed=seed) if name == "Custom Sandbox"
                       else presets.get_preset(name))
                system = cfg.build(relaxation=relax, use_retardation=retard, use_pn=pn)
                # Reflect the actually-used settings so the HUD/status are truthful.
                cfg.use_retardation = retard
                cfg.use_pn = pn
                cfg.relaxation = relax
                cfg.retardation = "aberration" if retard else "none"
                cfg.pn = "eih" if pn else "none"
                record = cfg.n_segments == 1
                res = system.propagate(iters, n_segments=cfg.n_segments,
                                       record_history=record)
                dt = time.perf_counter() - t0
                self._result_queue.put(("ok", cfg, res, dt))
            except Exception as exc:  # surface solver errors to the UI thread
                self._result_queue.put(("err", exc, None, 0.0))

        threading.Thread(target=work, daemon=True).start()

    def _drain_queue(self) -> None:
        """Main-thread poller: apply any finished solve, then reschedule."""
        try:
            while True:
                kind, a, b, c = self._result_queue.get_nowait()
                if kind == "ok":
                    self._on_solved(a, b, c)
                else:
                    self._solving = False
                    self.status.configure(text_color="#ff6b6b",
                                          text=f"✗ SOLVE FAILED\n{a}")
        except queue.Empty:
            pass
        self.after(60, self._drain_queue)

    def _on_solved(self, cfg, res, dt: float) -> None:
        self._solving = False
        self.cfg, self.result = cfg, res
        self.desc.configure(text=cfg.description)

        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

        self.scene.build(res, cfg)
        self.canvas.draw()
        self.anim = FuncAnimation(self.fig, self.scene.step, interval=self.scene.interval,
                                  blit=True, cache_frame_data=False)
        self.scene.playing = True
        self.play_btn.configure(text="⏸  Pause")

        ok = "✓ CONVERGED" if res.converged else "△ PARTIAL"
        # single window: residuals is the iteration history -> report the final
        # one; segmented: one final residual per segment -> report the worst
        if cfg.n_segments <= 1:
            res_label = "final residual"
            res_val = res.residuals[-1] if res.residuals else float("nan")
        else:
            res_label = "worst residual"
            res_val = max(res.residuals) if res.residuals else float("nan")
        self.status.configure(
            text_color=ACCENT,
            text=(f"{ok}  ·  {dt*1000:.0f} ms\n"
                  f"bodies {res.n_bodies}   windows {cfg.n_segments}\n"
                  f"{res_label} {res_val:.1e}\n"
                  f"energy drift  {res.energy_drift:.2e}"),
        )
        if self._pending:                       # a change arrived mid-solve
            self._schedule_solve()

    # ------------------------------------------------------------------ #
    # Continuous / infinite-run streaming (Piecewise Volterra Windowing)
    # ------------------------------------------------------------------ #
    def _stop_batch(self) -> None:
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

    def _stop_continuous(self) -> None:
        """Signal the worker to exit, stop the tick loop, and drain the queue."""
        self._stream_stop.set()
        self._stream_pause.clear()
        if self._cont_job is not None:
            self.after_cancel(self._cont_job)
            self._cont_job = None
        self._stream_thread = None
        self._stream = None
        try:
            while True:
                self._window_queue.get_nowait()
        except queue.Empty:
            pass

    def _start_continuous(self) -> None:
        """Build a fresh Piecewise-Volterra stream and start worker + tick loop."""
        self._stop_continuous()                 # tear down any previous stream

        name = self.preset_var.get()
        iters = int(self.iter_slider.get())
        relax = float(self.relax_slider.get())
        retard = bool(self.retard_switch.get())
        pn = bool(self.pn_switch.get())
        seed = self._sandbox_seed

        cfg = (presets.sandbox(seed=seed) if name == "Custom Sandbox"
               else presets.get_preset(name))
        cfg.use_retardation, cfg.use_pn, cfg.relaxation = retard, pn, relax
        cfg.retardation = "aberration" if retard else "none"
        cfg.pn = "eih" if pn else "none"
        try:
            stream = PiecewiseVolterraStream.from_config(
                cfg, iterations=iters, relaxation=relax,
                use_retardation=retard, use_pn=pn)
        except Exception as exc:  # pragma: no cover - defensive
            self.status.configure(text_color="#ff6b6b", text=f"✗ STREAM FAILED\n{exc}")
            return

        self.cfg = cfg
        self.desc.configure(text=cfg.description)
        self.cont_scene.build(cfg)
        self.canvas.draw()

        # fresh synchronisation primitives for this run
        self._stream = stream
        self._stream_stop = threading.Event()
        self._stream_pause = threading.Event()
        self._window_queue = queue.Queue(maxsize=3)
        self.play_btn.configure(text="⏸  Pause")
        self.status.configure(text_color="#ffd15d", text="◍ STREAMING ∞  (spinning up…)")

        stop, pause, q = self._stream_stop, self._stream_pause, self._window_queue

        def worker() -> None:
            # Off-thread: steps the stream forever, feeding the bounded queue.
            # Queue backpressure paces compute to the display; NO Tk calls here.
            while not stop.is_set():
                if pause.is_set():
                    stop.wait(0.05)
                    continue
                try:
                    win = stream.step()
                except Exception:  # pragma: no cover - defensive
                    return
                while not stop.is_set():
                    try:
                        q.put(win, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        self._stream_thread = threading.Thread(target=worker, daemon=True)
        self._stream_thread.start()
        self._cont_job = self.after(30, self._continuous_tick)

    def _continuous_tick(self) -> None:
        """Main-thread render loop: pull ready windows, plot, follow, reschedule."""
        if not self._continuous or self._stream is None:
            return
        per = max(1, int(self.speed_slider.get()))      # windows consumed per tick
        got = 0
        for _ in range(per):
            try:
                win = self._window_queue.get_nowait()
            except queue.Empty:
                break
            self.cont_scene.push(win)
            self._windows_seen = win.index
            got += 1
        if got:
            self.cont_scene.redraw()
            self.canvas.draw_idle()
            if self._windows_seen % 10 == 0:
                w = self.cont_scene.last
                self.status.configure(
                    text_color=ACCENT,
                    text=(f"∞ CONTINUOUS  ·  window {w.index}\n"
                          f"sim t {w.t[-1]/self.cfg.time_unit:.1f} {self.cfg.time_label}"
                          f"   ({self.cont_scene.N} bodies)\n"
                          f"delay buffer {w.buffer_span/self.cfg.time_unit:.3g} "
                          f"{self.cfg.time_label} (sliding)\n"
                          f"drift/window {w.energy_drift:.2e}"),
                )
        self._cont_job = self.after(34, self._continuous_tick)

    # ------------------------------------------------------------------ #
    def on_close(self) -> None:
        self._stop_continuous()
        if self.anim is not None:
            self.anim.event_source.stop()
        self.destroy()


def main() -> None:
    app = AuraApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()

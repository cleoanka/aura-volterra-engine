"""
aura_engine.presets
====================

Initial conditions and engine settings for the four AURA scenarios:

1. ``solar``       -- Inner Solar System (Sun, Venus, Earth, Mars), SI units.
2. ``spacecraft``  -- Earth, Moon, and a spacecraft on a swing trajectory.
3. ``chaos``       -- A bounded, chaotic three-body tangle (scaled units, lowered
                      ``c`` so retardation is visible).
4. ``sandbox``     -- Randomised bodies about a common barycentre.

Each factory returns a :class:`PresetConfig` bundling the bodies with the solver
knobs and cosmetic hints (axis limits, unit label) the UI needs. All tunables
here were chosen so a *single global window* converges under damped Picard
iteration -- see ``scripts/tune.py`` for the sweep that fixed them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .physics import Body, VolterraSystem

# --- physical constants (SI) --------------------------------------------- #
G_SI = 6.67430e-11
C_SI = 2.99792458e8
AU = 1.495978707e11        # m
DAY = 86400.0              # s
YEAR = 365.25 * DAY        # s
M_SUN = 1.98892e30         # kg
M_EARTH = 5.97219e24
M_VENUS = 4.8675e24
M_MARS = 6.4171e23
M_MOON = 7.342e22
R_EARTH = 6.371e6          # m
R_MOON_ORBIT = 3.844e8     # m


@dataclass
class PresetConfig:
    """A fully specified scenario ready to be handed to :class:`VolterraSystem`."""

    name: str
    description: str
    bodies: list[Body]
    G: float
    c: float
    softening: float
    t_span: float
    n_samples: int
    use_retardation: bool
    use_pn: bool
    relaxation: float
    iterations: int
    n_segments: int = 1               # chained global windows (MCPI-style)
    scheme: str = "chebyshev"         # "chebyshev" (spectral) | "trapezoid"
    retardation: str | None = None    # None -> auto from use_retardation
    pn: str | None = None             # None -> auto from use_pn ("eih")
    window_samples: int = 48          # CGL nodes per streaming window
    length_unit: float = 1.0          # divide positions by this for display
    unit_label: str = "m"
    view_radius: Optional[float] = None  # half-width of the square plot, in units
    time_unit: float = 1.0
    time_label: str = "s"

    def build(self, **overrides) -> VolterraSystem:
        """Construct the engine, allowing UI overrides (iterations handled by caller)."""
        params = dict(
            G=self.G,
            c=self.c,
            softening=self.softening,
            t_span=self.t_span,
            n_samples=self.n_samples,
            use_retardation=self.use_retardation,
            use_pn=self.use_pn,
            relaxation=self.relaxation,
            scheme=self.scheme,
            retardation=self.retardation,
            pn=self.pn,
        )
        params.update(overrides)
        return VolterraSystem(self.bodies, **params)


def _circular_speed(mu: float, r: float) -> float:
    return float(np.sqrt(mu / r))


# --------------------------------------------------------------------------- #
# 1. Inner Solar System
# --------------------------------------------------------------------------- #
def solar_system() -> PresetConfig:
    """Sun + Venus + Earth + Mars over ~0.7 yr, with retardation (tiny) on."""
    mu = G_SI * M_SUN

    def planet(name, mass, a, color, size):
        v = _circular_speed(mu, a)
        return Body(name, mass, [a, 0.0, 0.0], [0.0, v, 0.0], color=color, radius=size)

    venus = planet("Venus", M_VENUS, 0.723 * AU, "#e8c07d", 7.0)
    earth = planet("Earth", M_EARTH, 1.000 * AU, "#5aa9ff", 7.5)
    mars = planet("Mars", M_MARS, 1.524 * AU, "#ff6b57", 6.5)

    bodies = [venus, earth, mars]
    # Recoil the Sun so total momentum is (nearly) zero -> barycentre stays put.
    p = sum(b.mass * b.velocity for b in bodies)
    sun = Body("Sun", M_SUN, [0.0, 0.0, 0.0], -p / M_SUN, color="#ffd76b", radius=16.0)

    return PresetConfig(
        name="Solar System (Inner)",
        description=(
            "Sun, Venus, Earth and Mars. Retardation is on but with the true "
            "speed of light it is a sub-second delay -- a stable, tiny "
            "perturbation. Watch the ballistic guess relax into closed orbits."
        ),
        bodies=[sun, venus, earth, mars],
        G=G_SI,
        c=C_SI,
        softening=0.02 * AU,
        t_span=0.72 * YEAR,
        n_samples=220,
        use_retardation=True,
        use_pn=False,
        relaxation=0.55,
        iterations=60,
        length_unit=AU,
        unit_label="AU",
        view_radius=1.9,
        time_unit=DAY,
        time_label="days",
    )


# --------------------------------------------------------------------------- #
# 2. Earth - Moon - Spacecraft
# --------------------------------------------------------------------------- #
def spacecraft() -> PresetConfig:
    """Earth, Moon and a light spacecraft on an eccentric swing outward."""
    mu_earth = G_SI * M_EARTH

    earth = Body("Earth", M_EARTH, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                 color="#5aa9ff", radius=14.0)
    # Moon on a near-circular orbit.
    v_moon = _circular_speed(mu_earth, R_MOON_ORBIT)
    moon = Body("Moon", M_MOON, [R_MOON_ORBIT, 0.0, 0.0], [0.0, v_moon, 0.0],
                color="#c9d1d9", radius=8.0)

    # Spacecraft: an eccentric (e ~ 0.55) elliptical orbit that swings out to
    # ~0.4 lunar distances and back. Start at perigee (below Earth) moving +x.
    r_p = 3.0 * R_EARTH
    r_a = 0.42 * R_MOON_ORBIT
    a_sc = 0.5 * (r_p + r_a)
    v_p = float(np.sqrt(mu_earth * (2.0 / r_p - 1.0 / a_sc)))  # vis-viva at perigee
    period = 2.0 * np.pi * np.sqrt(a_sc ** 3 / mu_earth)
    craft = Body("AURA-1", 1200.0, [0.0, -r_p, 0.0], [v_p, 0.0, 0.0],
                 color="#7dffc4", radius=5.0)

    bodies = [moon, craft]
    p = sum(b.mass * b.velocity for b in bodies)
    earth.velocity = -p / M_EARTH  # keep barycentre fixed

    return PresetConfig(
        name="Orbital Spacecraft",
        description=(
            "Earth, Moon and the AURA-1 spacecraft on an eccentric orbit that "
            "swings toward the Moon and back. The craft's whole arc is solved on "
            "chained global windows rather than marched step by step."
        ),
        bodies=[earth, moon, craft],
        G=G_SI,
        c=C_SI,
        softening=0.25 * R_EARTH,
        t_span=1.25 * period,          # ~1.25 spacecraft orbits
        n_samples=300,
        use_retardation=True,
        use_pn=False,
        relaxation=0.55,
        iterations=45,
        n_segments=6,
        length_unit=R_MOON_ORBIT,
        unit_label="LD",   # lunar distances
        view_radius=1.15,
        time_unit=DAY,
        time_label="days",
    )


# --------------------------------------------------------------------------- #
# 3. Deep Chaos (bounded 3-body)
# --------------------------------------------------------------------------- #
def chaos() -> PresetConfig:
    """A bounded, chaotic equal-mass triple in scaled units (G = 1)."""
    m = 1.0
    # Start from a Lagrange equilateral relative-equilibrium (side a), but rotate
    # at 0.83x the circular rate so the triangle collapses inward, scatters
    # through the centre and tangles -- bounded (E < 0) but chaotic.
    a = 2.0
    G_chaos = 1.0
    r = a / np.sqrt(3.0)                           # distance from barycentre
    omega = np.sqrt(3.0 * G_chaos * m / a ** 3)    # circular relative-equilibrium rate
    speed = 0.83 * omega * r
    angles = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]
    colors = ["#ff5d8f", "#5df2ff", "#ffd15d"]
    names = ["Alpha", "Beta", "Gamma"]
    bodies = []
    for ang, col, nm in zip(angles, colors, names):
        pos = np.array([r * np.cos(ang), r * np.sin(ang), 0.0])
        tang = np.array([-np.sin(ang), np.cos(ang), 0.0])  # +ve rotation
        bodies.append(Body(nm, m, pos, speed * tang, color=col, radius=9.0))

    # Exact by symmetry, but zero any residual momentum defensively.
    p = sum(bd.mass * bd.velocity for bd in bodies)
    total_m = sum(bd.mass for bd in bodies)
    for bd in bodies:
        bd.velocity = bd.velocity - p / total_m

    return PresetConfig(
        name="Deep Chaos (3-Body)",
        description=(
            "Three equal masses collapsing out of a Lagrange triangle into a "
            "bounded chaotic dance. The speed of gravity is lowered so retarded "
            "potentials smear the interactions -- spaghetti trajectories, each "
            "window solved globally and chained through the scattering."
        ),
        bodies=bodies,
        G=1.0,
        c=18.0,               # lowered 'c' -> visible retardation (artistic)
        softening=0.12,
        t_span=14.0,
        n_samples=520,
        use_retardation=True,
        use_pn=False,
        relaxation=0.5,
        iterations=45,
        n_segments=22,
        length_unit=1.0,
        unit_label="u",
        view_radius=2.6,
        time_unit=1.0,
        time_label="u",
    )


# --------------------------------------------------------------------------- #
# 4. Custom Sandbox (randomised)
# --------------------------------------------------------------------------- #
def sandbox(n_bodies: int = 5, seed: Optional[int] = None) -> PresetConfig:
    """Spawn ``n_bodies`` randomised masses about a common barycentre.

    Positions/velocities are drawn so the cluster is loosely bound (tangential
    bias about the centre of mass) and the momentum is zeroed for framing.
    """
    rng = np.random.default_rng(seed)
    n_bodies = int(np.clip(n_bodies, 2, 8))
    palette = ["#7de3ff", "#ff5d8f", "#ffd15d", "#5dffa0", "#c98bff",
               "#ff9f5d", "#5df2ff", "#ff6b6b"]

    total_mass = 0.0
    raw = []
    for k in range(n_bodies):
        mass = float(rng.uniform(0.6, 2.2))
        pos = rng.uniform(-1.6, 1.6, size=3)
        pos[2] = 0.0
        raw.append([mass, pos])
        total_mass += mass

    bodies: list[Body] = []
    for k, (mass, pos) in enumerate(raw):
        radial_dist = np.linalg.norm(pos) + 1e-6
        radial = pos / radial_dist
        tangential = np.array([-radial[1], radial[0], 0.0])
        # ~circular speed for the enclosed mass, softened, times a random factor
        v_circ = np.sqrt(total_mass / (radial_dist + 0.3))
        speed = float(rng.uniform(0.45, 0.75)) * v_circ
        vel = tangential * speed + rng.normal(0.0, 0.05, size=3)
        vel[2] = 0.0
        bodies.append(
            Body(f"B{k+1}", mass, pos, vel, color=palette[k % len(palette)],
                 radius=5.0 + 2.5 * mass)
        )

    p = sum(b.mass * b.velocity for b in bodies)
    total_m = sum(b.mass for b in bodies)
    for b in bodies:
        b.velocity = b.velocity - p / total_m

    return PresetConfig(
        name="Custom Sandbox",
        description=(
            "Randomised bodies about a shared barycentre, seeded with near-orbital "
            "speeds so the swarm stays loosely bound. Re-roll to spawn a new "
            "system; each is relaxed on chained global windows."
        ),
        bodies=bodies,
        G=1.0,
        c=25.0,
        softening=0.18,
        t_span=7.0,
        n_samples=360,
        use_retardation=True,
        use_pn=False,
        relaxation=0.5,
        iterations=40,
        n_segments=12,
        length_unit=1.0,
        unit_label="u",
        view_radius=3.0,
        time_unit=1.0,
        time_label="u",
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
PRESETS: dict[str, Callable[[], PresetConfig]] = {
    "Solar System (Inner)": solar_system,
    "Orbital Spacecraft": spacecraft,
    "Deep Chaos (3-Body)": chaos,
    "Custom Sandbox": sandbox,
}

PRESET_NAMES: list[str] = list(PRESETS.keys())


def get_preset(name: str) -> PresetConfig:
    """Return a freshly built :class:`PresetConfig` by display name."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {PRESET_NAMES}")
    return PRESETS[name]()

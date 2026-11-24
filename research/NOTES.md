# Paper notes — Windowed Picard–Chebyshev waveform relaxation for state-dependent-delay N-body dynamics

> Status: **research notes, not a paper draft.** Every number here is produced by a
> script in `research/` and stored in `research/results/`. Re-run before quoting.
> Started 2026-11-24 on branch `research/sdde-paper`.

---

## 0. One-paragraph pitch (working)

Retarded (finite-propagation-speed) interactions turn the N-body problem into a
*state-dependent* delay differential equation (SD-DDE): the delay
`τ_ij(t) = |x_i(t) − x_j(t − τ_ij)|/c` depends on the unknown trajectory. We recast it
as a Volterra integral equation and solve it window by window with **undamped Picard
iteration on Chebyshev–Gauss–Lobatto nodes** (MCPI-style waveform relaxation), reading
the converged past from a **spectral segment history**. Three contributions:
(i) because retarded delays keep the Picard operator causal, the quasi-nilpotent bound
`e_n ≤ e_0 x^{2n}/(2n)!` survives the state-dependent delay — no window-length limit
in exact arithmetic, iterations linear in `√L·W`, superlinear convergence that damping
and Anderson acceleration only slow down; (ii) **breaking-point-aligned windows**: the
initial-history discontinuity propagates along `t − τ_ij(t) = d`; locating these roots
on the converged iterate and ending windows there restores spectral convergence
(1e-6 → 1e-15), with a simple order rule (+2 derivatives per level for position
coupling, +1 for velocity coupling); (iii) validation against DelayDiffEq.jl in 128-bit
arithmetic: comparable RHS cost, but machine-precision accuracy that the step-marching
solver does not reach at any tolerance, even with its discontinuity tracking on —
whose API cannot express the implicit light-cone lag.

---

## 1. Literature position (checked 2026-11-24)

Sources: agy survey (`scratchpad`, not kept) + my own WebSearch cross-check.
**Citations marked (✓) were seen in a search result; (?) not yet verified — verify each
before citing.**

| Thread | Closest prior work | What it does NOT do |
|---|---|---|
| MCPI / Picard–Chebyshev in astrodynamics | Bai & Junkins, *Modified Chebyshev–Picard iteration methods for orbit propagation*, J. Astronaut. Sci. 2011 (title ✓ via Rutgers record; authors/year probable); Bai PhD thesis (TAMU) (✓); Woollands & Junkins, *Adaptive Picard–Chebyshev*, JGCD, doi 10.2514/1.G003318 (✓) | ODE only (instantaneous forces) |
| Chebyshev–Picard for DDEs | *Chebyshev–Picard iteration methods for solving delay differential equations*, Math. Comput. Simul. 2023, S0378475423004238 (✓) | **constant** delay; notes ODE-MCPI "cannot be applied directly" to DDEs |
| Spectral collocation for SD-DDEs | N. Hale, arXiv:2402.12952 (✓) — CGL + barycentric resampling, Chebfun, Newton solve, geometric convergence shown incl. state-dependent delays | global Newton (dense Jacobian), not Picard/WR; not N-body; no windowed history |
| WR for functional/delay eqs | Zubik-Kowal & Vandewalle, *WR for functional-differential equations*, SISC 1999 (✓); Bellen–Jackiewicz–Zennaro 1994 SINUM (✓, ODE contractivity) | non-state-dependent delays |
| SD-DDE theory | Hartung, Krisztin, Walther, Wu 2006 handbook chapter (?); Walther, *The solution manifold and C¹-smoothness…*, JDE 2003, doi 10.1016/j.jde.2003.07.001 (✓): `x ↦ x(t−τ(x))` not Lipschitz on C⁰, only on Lipschitz/C¹ sets | no numerics of this type |
| Retarded two-body (physics) | Driver, *A two-body problem of classical electrodynamics: the one-dimensional case*, Ann. Phys. 1963, doi 10.1016/0003-4916(63)90227-6 (✓) — electrodynamic 1D two-body, existence | no MCPI |
| Retarded gravity physics | Laplace (Mécanique céleste); Carlip, *Aberration and the speed of gravity*, arXiv gr-qc/9909087 (✓; journal ref NOT confirmed — agy said Phys. Lett. A 2000, check) | — |
| SD-DDE solver software | Widmann & Rackauckas, *DelayDiffEq: Generating DDE solvers via recursive embedding of ODE solvers*, arXiv:2208.12879 (✓ — agy gave a wrong title); RADAR5 (Guglielmi & Hairer 2001) (?); ddesd (Shampine) (?) | benchmarks are 1–4-dim; no N-body |
| Andò & Sieber arXiv:2410.07375 (✓) | piecewise-polynomial collocation convergence for periodic BVPs of SD-DDEs | BVP, not IVP/WR |

**Gap that survives the check:** Picard/waveform relaxation + Chebyshev for a
*state-dependent* delay IVP, in particular a coupled N-body light-cone system, with
a windowed spectral history and a convergence statement. The constant-delay
Chebyshev–Picard paper (2023) and Hale (2024) are the two papers a referee will
compare against — the paper must say precisely what is new relative to both.

**Honest framing:** the retarded-Newton force law is a **model SD-DDE**, not physics.
Naive retardation is the Laplace-unstable model; the "aberration" variant
`x_j(t−τ) + τ v_j(t−τ)` equals the instantaneous position to `O(τ²a)`, so its good
conservation is largely *by construction* (do not sell it as a physics result).

---

## 2. Problem statement

State `u = (x, v) ∈ R^{6N}`; for `t ≥ t0`

```
x_i'' = Σ_{j≠i} G m_j (s_ij − x_i) / (|s_ij − x_i|² + ε²)^{3/2}
s_ij  = x_j(t − τ_ij)                         (naive)
s_ij  = x_j(t − τ_ij) + τ_ij v_j(t − τ_ij)     (aberration)
τ_ij  = |x_i(t) − x_j(t − τ_ij)| / c           (light cone; unique root if |v_j| < c,
                                                since g'(τ) = 1 − n·v_j/c > 0)
x(t) = φ(t), t ≤ t0                            (initial history — REQUIRED for a DDE)
```

Initial histories implemented (engine `aura_engine/history.py`, reference
`research/reference_dde.jl`, identical definitions):
- **ballistic** `φ = x0 + v0 (t−t0)` → `x''` jumps from 0 to `f(t0)` at `t0`.
- **quadratic** `φ = x0 + v0 s + A0 s²/2`, `A0` = fixed point `A0 = f(φ_{A0})` at `t0`
  (`research/problems.py::consistent_A0`) → solution is C² at `t0`.

Sanity check of `A0` (binary, c = 8): tangential component 0.0884 =
Laplace aberration estimate `a·Ωτ/2` = 0.0884. ✔

## 3. Convergence theory (Volterra / quasi-nilpotent argument)

Window `[t0, t0+W]`, Picard map `G(x)(t) = x0 + v0(t−t0) + ∫_{t0}^{t} (t−s) f[x](s) ds`.

**Key structural fact:** for retarded delays (`τ ≥ 0`) the force at time `s` depends
only on `x` on `(−∞, s]`. So `G` is a *causal (Volterra)* operator even though the
delay is state dependent. (Advanced/Wheeler–Feynman delays would break this →
Fredholm, finite radius.)

Let `E_n(t) = sup_{u≤t} |x_{n}(u) − x*(u)|` (monotone; zero on the history).
On the set `X_M = {x: |x'| ≤ M < c}` (needed because `x ↦ x(t−τ(x))` is only
Lipschitz on Lipschitz functions — Walther 2003), perturbing `x` by `δ` changes
`x_j(t−τ)` by at most `|δ|(1 + 2β/(1−β))`, `β = M/c` (from
`δτ = n·(δx_i − δx_j(t−τ)) / (c(1 − n·v_j/c))`). Hence with `L` the Lipschitz constant of
the instantaneous force, `L_eff = L·(1 + κ(β))`, `κ = O(β)`, and

```
E_{n+1}(t) ≤ L_eff ∫_{t0}^{t} (t−s) E_n(s) ds   ⇒   E_n ≤ E_0 · x^{2n}/(2n)!,   x = √L_eff · W.
```

Consequences (testable):
1. No convergence radius in exact arithmetic; iterations to reach `tol` ≈ smallest `n` with
   `x^{2n}/(2n)! ≤ tol` → grows ~linearly in `x` (≈ `e x / 2`).
2. Asymptotic per-iteration ratio `≈ x²/((2n+1)(2n+2))` — **superlinear**.
3. Damping `w < 1` converts this into linear convergence with rate ≥ `1−w` → harmful.
   Anderson acceleration targets linear convergence and gains nothing here.
4. Delay only enters through `κ(β)`; for physical `β ≪ 1` the SD-DDE costs the same
   number of iterations as the ODE.
5. (Caveat) the bound is global-Lipschitz; near close encounters `L ∝ 1/r³` explodes,
   and iterates must stay in `X_M` (`|v| < c`) — the true limit is nonlinear.

For the binary, `L = 2Ω²` (radial tidal eigenvalue, ∞-norm), so `x = √2 ΩW`.

TODO: make the `X_M` invariance + the aberration model (depends on `v_j(t−τ)`, needs a
bound on `x''`) rigorous; check the constant in `κ(β)`; discrete (CGL) version: the
discrete operator `K2 = C·C` is strictly lower-triangular-like? (No — CGL `B` is dense
with a nilpotent-like spectrum; check `ρ(K2)` numerically, should be ≈0.)

## 4. Results so far

### E2 — single window, iterations vs `x = √2ΩW` (`exp_e2_convergence_theory.py`, `results/e2_convergence.json`)
Binary, c = 8 (β ≈ 0.088), tol 1e-12, CGL nodes `max(24, 14ΩW)`.

| orbits | x | none: Picard / bound / Anderson | naive: Picard / bound / Anderson |
|---|---|---|---|
| 0.5 | 4.44 | 13 / 15 / 16 | 11 / 15 / 12 |
| 1.0 | 8.89 | 19 / 22 / 34 | 13 / 22 / 18 |
| 1.5 | 13.3 | 25 / 29 / 59 | 15 / 29 / 22 |
| 2.0 | 17.8 | 31 / 35 / 96 | 16 / 35 / 25 |

- Bound holds everywhere and is nearly tight for the ODE case (31 vs 35).
- Measured ratios lie below `x²/((2n+1)(2n+2))` and decay superlinearly as predicted.
  Effective `x_eff ≈ 0.83 x` (none), smaller with delay.
- **Retarded converges faster than instantaneous** (16 vs 31 at 2 orbits). Hypothesis:
  the delayed cross-coupling is a shifted Volterra kernel, which adds extra contraction.
  Needs a clean explanation; possible theorem: the cross term contributes
  `(t−s−τ)_+` kernels.
- The predicted transient amplification `e^x/√(2πx)` was **not** observed (error
  decreased monotonically from the Taylor predictor). Bound is worst case; the
  measurement is relative to a large predictor error, so it is not a strong test.

### E2b — practical limits + damping/Anderson ablation (`exp_e2b_limits.py`)
All converged, including a **6-orbit single window** (binary, M = 527, 24 Picard
iterations) and a 4-time-unit chaotic-triple window. Iterations (w=1 Picard / w=0.6
Picard / Anderson w=1): binary 6 orbits 24 / 66 / 62; triple W=4: 12 / 40 / 16.
→ **Undamped Picard is the best solver in every case.** This contradicts the v0.3.0
README ("Anderson 32 vs 54"): that compared Anderson against *damped* Picard.
→ The old lesson "single-window Picard radius ≈ 1 orbit" is wrong for these problems
once enough nodes are used; it was likely a node-starvation/damping effect.
(Converged ≠ accurate: accuracy at long W is in E1.)

### Reference solver (all accuracy numbers below)
`research/reference_dde.jl`: DelayDiffEq.jl `MethodOfSteps(Vern9)`, **128-bit BigFloat,
tol 1e-18** (`bigfloat=True`). Same force law, same light-cone Newton, same initial
history as the engine, but independent code and an independent method (step-marching
with dense history). Float64 DelayDiffEq at tol 1e-13 differs from it by ~1e-12, which
is why the first (Float64) reference showed a fake 1e-12 floor.

### E5 — breaking-point-aligned windows (headline) (`exp_e5_breaks.py`, `results/e5_breaks.json`)
`track_breaks = L`: after a window converges, find the earliest root of
`t − τ_ij(t) = d` for tracked discontinuities `d` (level < L), truncate the window there,
re-solve, register the root as a level+1 discontinuity. Error at T vs BigFloat reference:

| case (history) | W, M | L0 (none) | L1 | L2 | L3 |
|---|---|---|---|---|---|
| binary naive (ballistic) | T/8, 32 | 3.7e-06 | 2.7e-10 | **3.0e-14** | 1.4e-14 |
| binary naive (ballistic) | T/16, 32 | 1.1e-06 | 3.4e-11 | **2.2e-15** | 6.2e-15 |
| binary aberration (ballistic) | T/8, 32 | 3.5e-06 | 2.6e-08 | 1.3e-11 | **6.7e-14** |
| binary aberration (ballistic) | T/16, 64 | 5.9e-08 | 9.2e-10 | 1.7e-13 | **8.5e-15** |
| triple naive (ballistic) | T/8, 32 | 8.0e-08 | 5.6e-13 | **6.2e-15** | 3.4e-15 |
| triple naive (C² history) | T/16, 32 | 6.5e-11 | **2.8e-15** | 2.2e-15 | 1.9e-15 |

Cost: +1–3 windows per tracked level for the binary; the triple has 6 ordered pairs so
L3 multiplies windows (8 → 17) for no gain → **L2 for naive, L3 for aberration** is
the sweet spot. Rule consistent with every column (to be proven):

> jump order at level k = (history jump order) + k·(2 − q), where q is the derivative
> order of the source state the force uses (q=0 naive, q=1 aberration);
> ballistic φ: x″ jumps (order 2); C² φ: x‴ jumps (order 3).

So the aberration model gains one derivative per level, naive gains two — exactly why
aberration needs one more level.

### E1 — spectral convergence without tracking (`exp_e1_accuracy.py`)
Re-run with the BigFloat reference after the barycentric fix (§5.3); numbers in
`results/e1_log.txt`. Qualitative picture (stable across both runs):
- instantaneous gravity (ODE): ~1e-14 for every M ≥ 16 (true spectral MCPI);
- retarded, no tracking: error **non-monotone in M**, stalls at 1e-6…1e-9 —
  algebraic convergence from the breaking point inside the first windows;
- C² initial history: 2–3 orders better, still algebraic for the aberration model.

### E4 — history ablation vs v0.3.0 (`exp_e4_history_ablation.py`, BigFloat ref)
| | v0.3.0 stream | v0.3.0 segmented batch | new (no tracking) | new + L2 (E5) |
|---|---|---|---|---|
| binary naive, W=T/16, M=32 | 6.4e-05 | 1.1e-02 | 1.1e-06 | 2.2e-15 |
| triple naive, W=T/32, M=48 | 6.7e-06 | 3.8e-03 | 3.8e-10 | — |

Segmented batch error *grows* with segment count in v0.3.0 (ballistic past per
segment); now batch ≡ stream to all printed digits.

### E3 — work–precision vs DelayDiffEq.jl (`exp_e3_work_precision.py`, `exp_e3b_dde_tracking.py`)
Cost = RHS evaluations (one full force field at one time point incl. light-cone solves;
DelayDiffEq `nf`, engine `iterations × M`). Wall = warm (JIT excluded), Python vs
Julia → **implementation-dependent**. Cheapest configuration reaching each error:

| case | err ≤ | WR-Chebyshev (L2/L3) rhs / ms | best DelayDiffEq rhs / ms |
|---|---|---|---|
| binary naive | 1e-9 | 828 / 39 | 1162 (Vern7) / 77 |
| binary naive | 1e-13 | 1536 / 61 | 2232 (Vern7) / 78 |
| binary aberration | 1e-9 | 1500 / 65 | 2212 (Vern7) / 91 |
| binary aberration | 1e-13 | 4096 / 92 | not reached |
| triple naive | 1e-6 | 732 / 139 | 462 (Vern7) / 77 |
| triple naive | 1e-11 | 996 / 169 | 1842 (Vern9) / 129 |
| triple naive | 1e-13 | 1464 / 185 | not reached |

Best attainable error — WR: 3.8e-15 / 2.6e-14 / 2.8e-15; DelayDiffEq (any alg/tol):
3.2e-14 / 5.6e-13 / 1.9e-13. **With `dependent_lags` tracking on (E3b)** DelayDiffEq's
best becomes 1.0e-13 / 3.2e-13 / 9.1e-14 — still 1–2 orders short. Structural reason:
its tracking API takes a lag `τ(u, p, t)` of the *current* state; the light-cone lag is
implicit in the history, so only the instantaneous `|x_i−x_j|/c` can be supplied, which
misplaces breaking points by O(β τ). WR locates them on the converged iterate itself.
Without tracking DelayDiffEq is non-monotone in tol (binary naive Vern9: tol 1e-8 →
1.2e-8, tol 1e-10 → 1.6e-8).

**Honest reading:** cost is *comparable* (same order of RHS evaluations; cheaper for
the binary, dearer for the triple at loose tolerance). The clear wins are **attainable
accuracy** (machine precision) and **robustness** (monotone in M). Do not claim speed.

### E6 — physics sanity: Laplace secular torque (naive model)
For a circular equal-mass binary under naive retardation the lagging source gives a
tangential acceleration `a_t = (Gm/d²) sin(θ/2)/cos²(θ/2)`, `θ = Ωτ`,
`τ = d cos(θ/2)/c` (exact for a stationary circular orbit; leading order `Ω/(2c)`),
hence `dL/dt = d·m·a_t`. Engine slope of total L on `[2τ, 2τ+0.2]` (L2 tracking):

| c | β = v/c | analytic dL/dt | engine | rel. diff |
|---|---|---|---|---|
| 8 | 0.088 | 0.088617 | 0.091481 | 3.2 % |
| 16 | 0.044 | 0.044223 | 0.044735 | 1.2 % |
| 32 | 0.022 | 0.022101 | 0.022200 | 0.45 % |
| 64 | 0.011 | 0.011049 | 0.011070 | 0.19 % |

The difference → 0 as ~β^1.3: the formula assumes a *stationary* circle while the real
orbit spirals out at a rate ∝ β. Sanity check only (not a precision benchmark);
also seen at t0: the consistent `A0` tangential component 0.0884 vs `Ωτ/2` = 0.0884.
(Inline script; promote to `research/exp_e6_laplace.py` if it goes in the paper.)

## 5. Bugs found and fixed on this branch (engine)

1. **Segmented batch `propagate(n_segments>1)` used a ballistic past at every segment
   start** instead of the computed trajectory → wrong retarded forces; error grew with
   segment count (binary c=8, T=2: 6.8e-4 @2 seg → 4.8e-3 @8 seg vs 1 window).
   Fixed via `SegmentHistory` (`VolterraSystem(t0=, history=)`). Test:
   `test_segmented_batch_uses_computed_past`.
2. **Stream delay buffer was linearly resampled** (`np.interp`) → retarded lookups
   capped at O(Δt²). Replaced by spectral segment history (each window kept as its own
   CGL data, barycentric interpolation, pruned beyond `η τ_max`). Test:
   `test_stream_matches_single_window_retarded`.
3. **`barycentric_interpolate` snapped queries to nodes** via `np.isclose(diff, 0)`
   (default `atol=1e-8`, absolute time) → O(v·1e-8) error for any query within 1e-8 of a
   node; also 0/0 warnings on short windows. Now exact `diff == 0`. Test:
   `test_barycentric_no_snapping_near_nodes`.
4. **Break re-detection (found by independent review):** a truncated re-solve can
   leave the window end ~1e-6 short of the break; the next window re-found it, cut a
   ~7e-7 sliver window and the 1e-9 dedup tolerance let it register twice, doubling
   the chain. Now roots within `1e-4·W` of a registered break are the same break.
   Test: `test_break_not_duplicated_into_sliver_window`. E3/E5 re-run after the fix.
5. New: breaking-point tracking (`PiecewiseVolterraStream(track_breaks=L)`) and C²
   initial history (`initial_A0=`). Tests:
   `test_breaking_point_alignment_restores_spectral_convergence`,
   `test_breaks_are_registered_and_bounded`.

Findings that contradict the v0.3.0 README (fix README before any public claim):
- "Anderson 32 vs 54 iterations" compared against *damped* Picard; undamped Picard
  beats both (E2/E2b). `test_anderson_beats_picard` encodes that misleading comparison.
- "single-window Picard radius ≈ 1 orbit" — not observed with adequate nodes (6 orbits
  converge in 24 iterations).
- "delay buffer … resampled onto a fixed grid" — no longer true.

## 6. Open items / what a referee will ask

- [ ] Lemma for the breaking-point order rule (E5) + proof; compare with Bellen–Zennaro
      discontinuity propagation for SD-DDEs.
- [ ] Rigor of §3: invariant set `X_M`, aberration model (needs bounded `x″`), the
      discrete CGL operator (`ρ(K2)`), and why delay *speeds up* Picard (E2).
- [ ] Tracking cost for larger N: N(N−1) breaking points per level → windows grow ~N²
      per level. Options: merge breaks closer than a fraction of W; split only the
      affected bodies (multirate WR).
- [ ] Review findings, not yet addressed (design risks): (a) breaks are not tied to a
      body — a jump in body i is tested against every pair, so some splits are
      over-conservative (near-coincident breaks 1.9e-5 apart in the triple);
      (b) with `track_breaks` on, presets whose light delay ≪ window (solar,
      spacecraft) crawl in tiny windows — tracking must stay opt-in / be capped;
      (c) truncated re-solves are not re-checked for convergence (reported in
      `WindowResult.converged`, never silently fixed); (d) a lookup into pruned past
      clamps to the oldest node (2e-4 error with `buffer_safety=1.0`; 0 clamps at the
      default 2.5) — interpolate/extrapolate instead, or raise.
- [ ] Scaling with N (Python loops over pairs today → vectorise over pairs; GPU/Colab
      only if the paper wants N ≥ 100).
- [ ] Long-time runs (many orbits): aberration-model conservation; E6 promoted to a script.
- [ ] Verify every (?) citation in §1; read Hale 2024 and the 2023 constant-delay
      Chebyshev–Picard paper in full and state the delta precisely.
- [ ] RADAR5 / ddesd comparison (optional; DelayDiffEq is the strongest open baseline).
- [ ] A physically meaningful SD-DDE beyond the toy: retarded electrodynamic two-body
      (Driver; Liénard–Wiechert) — same machinery.
- [ ] arXiv: a first submission in math.NA / physics.comp-ph needs an endorser.

## 7. Reproduce

```bash
cd ~/aura-volterra-engine
# Julia env (once): ~/.julia-envs/dde with DelayDiffEq, OrdinaryDiffEq, JSON3
.venv/bin/python -m pytest -q                          # 25 tests
.venv/bin/python research/exp_e2_convergence_theory.py # ~3 s
.venv/bin/python research/exp_e2b_limits.py            # ~7 s
.venv/bin/python research/exp_e5_breaks.py             # ~1 min (+ Julia refs, cached)
.venv/bin/python research/exp_e3_work_precision.py     # ~3 min
.venv/bin/python research/exp_e3b_dde_tracking.py      # ~2 min
.venv/bin/python research/exp_e1_accuracy.py           # ~2 min
git archive 1e66114 aura_engine | tar -x -C /tmp/old_v030
.venv/bin/python research/exp_e4_history_ablation.py /tmp/old_v030
```
References are cached in `research/cache/` (gitignored), keyed by the problem spec.

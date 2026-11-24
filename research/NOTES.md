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
whose API cannot express the implicit light-cone lag; (iv) a physical, **neutral**
SD-DDE — the classical electrodynamic two-body problem (Liénard–Wiechert, relativistic):
solved to 1e-16 (DelayDiffEq best 6e-14), the order rule's neutral prediction (no
smoothing; accuracy set by tracked levels, not M) confirmed, and the 1/c³ mutual
radiation damping reproduced analytically. Honest limits: fixed windows are unsafe near
close encounters (error control required), break tracking scales ∝ N² per level, and
adaptive steppers win in RHS count through close encounters.

---

## 1. Literature position (checked 2026-11-24)

Sources: an LLM-assisted literature survey, cross-checked by web search.
**Citations marked (✓) were seen in a search result; (?) not yet verified — verify each
before citing.**

| Thread | Closest prior work | What it does NOT do |
|---|---|---|
| MCPI / Picard–Chebyshev in astrodynamics | Bai & Junkins, *Modified Chebyshev–Picard iteration methods for orbit propagation*, J. Astronaut. Sci. 2011 (title ✓ via Rutgers record; authors/year probable); Bai PhD thesis (TAMU) (✓); Woollands & Junkins, *Adaptive Picard–Chebyshev*, JGCD, doi 10.2514/1.G003318 (✓) | ODE only (instantaneous forces) |
| Chebyshev–Picard for DDEs | Zhou, Wang, Liu, *Chebyshev–Picard iteration methods for solving delay differential equations*, Math. Comput. Simul. 217 (2024) 1–20, doi 10.1016/j.matcom.2023.09.023 (✓; multi-interval variant included) | **constant** delay; notes ODE-MCPI "cannot be applied directly" to DDEs |
| Spectral collocation for SD-DDEs | N. Hale, arXiv:2402.12952 (✓) — CGL + barycentric resampling, Chebfun, Newton solve, geometric convergence shown incl. state-dependent delays | global Newton (dense Jacobian), not Picard/WR; not N-body; no windowed history |
| WR for functional/delay eqs | Zubik-Kowal & Vandewalle, *WR for functional-differential equations*, SISC 1999 (✓); Bellen–Jackiewicz–Zennaro 1994 SINUM (✓, ODE contractivity) | non-state-dependent delays |
| SD-DDE theory | Hartung, Krisztin, Walther, Wu, *Functional differential equations with state-dependent delays: theory and applications*, Handbook of DEs: ODEs vol. 3 (2006) 435–545 (✓); Walther, *The solution manifold and C¹-smoothness…*, JDE 2003, doi 10.1016/j.jde.2003.07.001 (✓): `x ↦ x(t−τ(x))` not Lipschitz on C⁰, only on Lipschitz/C¹ sets | no numerics of this type |
| Retarded two-body (physics) | Driver, *A two-body problem of classical electrodynamics: the one-dimensional case*, Ann. Phys. 1963, doi 10.1016/0003-4916(63)90227-6 (✓) — electrodynamic 1D two-body, existence | no MCPI |
| Retarded gravity physics | Laplace (Mécanique céleste); Carlip, *Aberration and the speed of gravity*, arXiv gr-qc/9909087 (✓; journal ref NOT confirmed — agy said Phys. Lett. A 2000, check) | — |
| SD-DDE solver software | Widmann & Rackauckas, *DelayDiffEq: Generating DDE solvers via recursive embedding of ODE solvers*, arXiv:2208.12879 (✓ — agy gave a wrong title); RADAR5: Guglielmi & Hairer, Computing 67 (2001) 1–12, doi 10.1007/s006070170013 (✓); ddesd (Shampine) (?) | benchmarks are 1–4-dim; no N-body |
| **Breaking points for implicit/state-dependent DDEs** | Guglielmi & Hairer, *Computing breaking points in implicit delay differential equations* (✓ title; preprint page returned HTTP 403 — full text NOT read yet) | step-marching (RADAR5); the closest prior work to our break alignment — the delta is doing it inside windowed Picard–Chebyshev with one-sided join lookups and Newton landing on the converged iterate |
| Recent survey | Humphries, Eremin, Wang, *Practicalities of State-Dependent and Threshold Delay Differential Equations*, arXiv:2510.17126 (✓, partly read) | breaking point of order k (Bellen–Zennaro def.); propagation ξ_j − τ(ξ_j, u(ξ_j)) = ξ_i raises the order by **at least one** (1st-order DDEs); only finitely many need tracking; location accuracy ξ(h) − ξ_j = O(h^{p/(k+1)}) (Thm 4.2); "it is ξ_j^h, and not ξ_j, that should be in the mesh". Methods: FCRK Runge–Kutta; no spectral/WR; no neutral discussion seen |
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

**Why retardation speeds Picard up (E2, measured + explained).** Binary, plain Picard,
tol 1e-12, iterations vs c at fixed W (inline sweep, 2026-11-24):

| W | c = 1e9 | 1000 | 100 | 32 | 16 | 8 | 4 |
|---|---|---|---|---|---|---|---|
| 0.5 orbit | 13 | 13 | 13 | 12 | 12 | 11 | 10 |
| 2 orbits | 31 | 31 | 28 | 24 | 20 | 16 | 12 |

Monotone in τ and → the ODE count as c → ∞. The naive "t < nτ support" argument is
irrelevant here (W/τ = 71). Refined bound: split the Lipschitz constant into the
instantaneous self part `L_s` (dependence on `x_i(t)`) and the delayed cross part `L_c`
(on `x_j(t−τ)`). Each Picard step applies `K2` with either part; a term with m cross
applications has kernel `(t − s − mτ)_+^{2n−1}/(2n−1)!`, so

```
E_n ≤ E_0 Σ_m C(n,m) L_s^{n−m} L_c^m (W − mτ)_+^{2n} / (2n)!
```

The shift enters with power 2n, so the bound does decrease with τ. **Evaluated
numerically** (`L_s = L_c = 2`, ratio 1e-12) — bound / observed iterations:

| W | c = 1e9 | 100 | 32 | 16 | 8 | 4 |
|---|---|---|---|---|---|---|
| 0.5 orbit | 15 / 13 | 15 / 13 | 14 / 12 | 14 / 12 | 13 / 11 | 13 / 10 |
| 2 orbits | 35 / 31 | 35 / 28 | 34 / 24 | 32 / 20 | 31 / 16 | 29 / 12 |

Verdict: right direction, tight for short windows (constant offset 2–3), but it explains
only a fraction of the speed-up on long windows (35→29 predicted vs 31→12 observed). My
first heuristic `exp(−n²τ/W)` overstated it. **Unexplained remainder.** Candidate (not
verified): phase — in the rotating binary the delay rotates the cross error mode by
`Ωτ`, and a magnitude-only Lipschitz bound cannot see that cancellation. Tried: radial
infall (W = 0.6) and head-on approach (W = 1.5), c = 1e9…4 → 8→7 and 7→6 iterations,
i.e. little effect — but **inconclusive**: non-rotating configurations only allow short
windows before collision, where even the rotating binary shows a small effect (13→10).
Fair test done: softened (ε = 0.5) radial pair oscillating through the centre — long
windows without rotation. Same `x = ωW`: 0.5 period 7→6; 2 periods 13→12; x-matched
to the binary (x ≈ 17.8, W = 4.44, M = 248) **16 → 14** for c = 1e9 → 4, versus
**31 → 12** for the rotating binary. → The large extra speed-up needs rotation;
**supports the phase hypothesis** (delay rotates the cross error by Ωτ, partial
cancellation), the non-rotating residue (16→14) is the size the shifted-kernel bound
predicts. Not a proof: next step is the linearised error operator in the co-rotating
frame (symbol `(J_s + J_c R(Ωτ) e^{−sτ})/s²`).


**Discrete operator (checked).** On `[−1, 1]` (length 2) the CGL double-integration
matrix `K2 = B·B` satisfies `‖K2⁸‖_∞ = 3.13e-9` for M = 16…128 — identical to the
continuous Volterra bound `2^16/16! = 3.14e-9` (M = 8: 3.42e-9). So the discrete Picard
iteration inherits the factorial bound. `ρ(K2)` is not exactly 0 (3.7e-2 at M = 8,
1.1e-2 at 16, ≈2.7e-3 for M ≥ 32 — eigenvalues of this highly non-normal, nearly
nilpotent matrix are ill-conditioned, so the floor is probably round-off); it would
only matter once `(LW²)^n/(2n)!` falls below `ρ^n`, i.e. far below tol.

TODO: make the `X_M` invariance + the aberration model (depends on `v_j(t−τ)`, needs a
bound on `x''`) rigorous; check the constant in `κ(β)`.

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
- **Retarded converges faster than instantaneous** (16 vs 31 at 2 orbits) — explained in
  §3: shifted-kernel bound gets the direction, only part of the size; remainder open.
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
| binary naive | 1e-9 | 576 / 32 | 1162 (Vern7) / 77 |
| binary naive | 1e-13 | 1752 / 51 | 2232 (Vern7) / 78 |
| binary aberration | 1e-9 | 1500 / 68 | 2212 (Vern7) / 91 |
| binary aberration | 1e-13 | 3584 / 91 | not reached |
| triple naive | 1e-6 | 624 / 81 | 462 (Vern7) / 77 |
| triple naive | 1e-11 | 852 / 94 | 1842 (Vern9) / 129 |
| triple naive | 1e-13 | 1008 / 106 | not reached |

(Re-run 2026-11-24 after vectorised field + warm start; earlier run: triple 1e-13 at
1464 rhs / 185 ms.)

Best attainable error — WR: 6.0e-15 / 2.6e-14 / 1.9e-15; DelayDiffEq (any alg/tol):
3.2e-14 / 5.6e-13 / 1.9e-13. **With `dependent_lags` tracking on (E3b)** DelayDiffEq's
best becomes 1.0e-13 / 3.2e-13 / 9.1e-14 — still 1–2 orders short. Structural reason:
its tracking API takes a lag `τ(u, p, t)` of the *current* state; the light-cone lag is
implicit in the history, so only the instantaneous `|x_i−x_j|/c` can be supplied, which
misplaces breaking points by O(β τ). WR locates them on the converged iterate itself.
Without tracking DelayDiffEq is non-monotone in tol (binary naive Vern9: tol 1e-8 →
1.2e-8, tol 1e-10 → 1.6e-8).

**Honest reading:** cost is *comparable to better* in RHS count (fewer evaluations at
every threshold ≤ 1e-9; dearer for the triple only at 1e-6), and warm wall time is now
lower too — but that last part is Python+NumPy vs Julia on tiny N, so do not lead with
speed. The clear wins are **attainable accuracy** (machine precision) and
**robustness** (monotone in M).

### E8 — breaking-point order rule, measured (`exp_e8_break_order.py`)
Single window [0, 0.2] containing exactly one breaking point (ξ₁ ≈ τ = 0.1245, binary
c = 8), no tracking, M = 8…96, error at W vs 128-bit reference (tol 1e-20):

| model + φ | predicted jump order k | fitted slope of log err vs log M | k − 1 |
|---|---|---|---|
| aberration + ballistic | 3 | −2.21 | 2 |
| naive + ballistic | 4 | −3.17 | 3 |
| aberration + C² | 4 | −3.50 | 3 |
| naive + C² | 5 | −4.13 | 4 |

Ordering exactly as predicted (including the predicted tie naive+ballistic ≈
aberration+C²), and every slope ≈ **−(k − 1)**. Errors are non-monotone in M (typical
for a kink at a non-node point), so slopes are least-squares fits.

**Positioning (important).** For first-order DDEs the standard result (Bellen–Zennaro;
see Humphries–Eremin–Wang, arXiv:2510.17126) is that each propagation raises the
breaking-point order by *at least one*. The rule below is that result specialised to
second-order systems `x'' = f(x, x_delayed, x'_delayed, x''_delayed)` written in
Volterra form — +2 per level through positions, +1 through velocities, 0 through
accelerations (neutral) — plus the quadrature rate. Present it as a corollary with the
neutral case and the measured M-rate as the new content, not as a new theorem.

**Lemma (draft, to prove).** Let `φ` have a jump in the `k_φ`-th derivative at `t0`
(`k_φ = 2` ballistic, `3` C²) and let the force depend on the source's `q`-th
derivative (`q = 0` naive, `1` aberration). Then the level-ℓ breaking point carries a
jump in `x^{(k_ℓ)}`, `k_ℓ = k_φ + ℓ(2 − q)`, and an un-split Chebyshev–Picard window
containing it has end-point error `O(M^{−(k_ℓ − 1)})`.
Proof sketch: (i) propagation — `x_j(t − τ(t))` inherits the jump of `x_j^{(k)}` at the
root of `t − τ(t) = d` (transversal, since `1 − τ' > 0`); a q-th derivative lowers it by
q; double integration raises it by 2. (ii) rate — the window error is the
Clenshaw–Curtis-type quadrature error of `∫∫ f` with `f = x''`; `f^{(k−2)}` is of
bounded variation, and CC quadrature with ν-th derivative BV converges as
`O(M^{−ν−1})` (Trefethen, *ATAP*, Ch. 19) ⇒ `O(M^{−(k−1)})`.
Consequence: aligning windows with levels ℓ < L leaves only jumps of order
`k_L`, i.e. error `O(M^{−(k_L−1)})` — with L = 2 naive (k=6 → M^{-5}) / L = 3 aberration
(k=5 → M^{-4}), matching where E5 reaches round-off.

### E7 — scaling with N, error control (`exp_e7_scaling.py`, `results/e7_log.txt`)
Bound random clusters (`problems.cluster`, N equal masses, softening 0.05, c = 10,
T = 1, naive retardation). Reference: DelayDiffEq Vern9 tol 1e-14 + `dependent_lags`
(Float64 — BigFloat is too slow at N = 16). Competitor: same at tol 1e-10.

**Finding 1 — fixed windows are not safe.** Without error control the engine reports
`converged=True` and is wrong by up to 1.5e-1 (N=3: a close encounter). Picard
convergence is convergence of the *discrete* problem; it says nothing about resolution.
→ added a-posteriori control `accuracy_tol` (trailing Chebyshev coefficients of the
**velocities**; halve the window; next window starts at 1.5× the last accepted span).
Positions under-estimate the error by 2–3 orders near encounters (measured: pos-based
tol 1e-12 → err 1e-8; vel-based tol 1e-12 → err 3e-12).

| N | DDE tol 1e-10: err / nf / wall | WR-B (C² φ + acc 1e-10): err / rhs / wall | WR-C (+L1): err / rhs / windows / wall |
|---|---|---|---|
| 3 | 2.3e-11 / 3250 / 0.27 s | 8.9e-11 / 4496 / 0.44 s | 6.4e-13 / 5488 / 38 / 0.53 s |
| 4 | 1.7e-11 / 1602 / 0.27 s | 3.3e-11 / 2048 / 0.24 s | 4.7e-13 / 4800 / 25 / 0.69 s |
| 6 | 8.6e-11 / 738 / 0.25 s | 1.4e-10 / 720 / 0.09 s | 1.8e-14 / 4528 / 35 / 1.6 s |
| 8 | 3.7e-11 / 3074 / 1.17 s | 6.1e-11 / 3744 / 1.14 s | 1.2e-13 / 11728 / 81 / 7.2 s |
| 12 | 1.8e-09 / 2130 / 2.09 s | 6.8e-11 / 2608 / 1.19 s | 1.9e-13 / 29936 / 148 / 53 s |
| 16 | 1.3e-09 / 2786 / 5.03 s | 3.6e-10 / 2608 / 1.91 s | 1.0e-12 / 52032 / 257 / 2e+02 s |

**Finding 2 — modes.** WR-B tracks its tolerance at every N with RHS cost 1.0–1.5× of
DelayDiffEq, and is *faster in wall time from N ≈ 12* (N = 16: 2.2 s vs 5.0 s) (all-pairs field vectorised over
nodes × pairs; per-call cost amortises). WR-C (break tracking) buys ~1e-13 but the
number of level-1 breaks is N(N−1) → windows ∝ N² → unusable beyond N ≈ 6 (re-run after review fixes: N = 16 → 257 windows, 200 s). Practical
recommendation: **C² initial history + error control** (B) in general; **tracking (C)**
only for few bodies when machine precision is the goal. (Close encounters favour
adaptive steppers in RHS count — MCPI literature addresses them with Sundman/KS
regularisation; out of scope here.)

Engine changes for E7: vectorised retarded field `_retarded_field` (all pairs × nodes,
one interpolation call per *source* body; equals the per-pair loop to 1e-16, test
`test_vectorized_retarded_field_matches_pair_loop`); body-tagged breaks
`(time, level, body)` (correctness: a jump of body j only propagates via pairs (i, j);
no measurable change on E5 problems — symmetric configurations); `accuracy_tol`.

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
Script: `research/exp_e6_laplace.py` (`results/e6_laplace.json`).

**Aberration model, derived (E6 script, second part).** Same stationary-circle argument
with `s = x_j(t−τ) + τ v_j(t−τ)`: tangential offset `r(sin θ − θ cos θ) ≈ rθ³/3`, so
`dL/dt ≈ G m² θ³ / (6d)` — **β³**, not β. Exact (light-cone θ, full |s − x_i|³) vs
engine early slope on `[3τ, 3τ + 0.2]`:

| c | exact | θ³/6 | engine | rel. diff |
|---|---|---|---|---|
| 8 | 8.866e-4 | 9.101e-4 | 8.753e-4 | 1.27 % |
| 16 | 1.140e-4 | 1.148e-4 | 1.137e-4 | 0.25 % |
| 32 | 1.435e-5 | 1.438e-5 | 1.434e-5 | 0.06 % |
| 64 | 1.797e-6 | 1.798e-6 | 1.797e-6 | 0.01 % |

Difference → 0 like β² (orbit evolution). This also explains E9's c^−2.8 scaling.


### E9 — long-time behaviour, 20 orbits (`exp_e9_longtime.py`, `results/e9_longtime.json`)
Binary, ballistic φ, tracking (L2 naive / L3 aberration) + `accuracy_tol=1e-12`, window
P/4, M = 24. Secular rates = linear fit over the run; accuracy vs DelayDiffEq (Vern9 +
`dependent_lags`, tol 1e-14) at t = 20 P.

| c | model | ΔE/|E₀| | ΔL/L₀ | dL/dt | |engine − DDE| at 20 P |
|---|---|---|---|---|---|
| 8 | naive | +0.86 | +1.54 | 1.2e-2 | 3.6e-12 |
| 8 | aberration | +0.15 | +0.086 | 7.0e-4 | 6.9e-11 |
| 32 | naive | +0.71 | +0.86 | 6.7e-3 | 2.0e-11 |
| 32 | aberration | +3.6e-3 | +1.8e-3 | 1.4e-5 | 4.2e-11 |

- Long-time agreement with the independent solver: 4e-12…7e-11 after 20 orbits.
- Aberration model: residual secular torque drops 49× for 4× larger c, i.e. ∝ c^−2.8 ≈
  **β³** — consistent with the expansion `x_j(t−τ) + τ v_j(t−τ) = x_j(t) − τ²a_j/2 +
  τ³ȧ_j/3 + …`: the τ² term is radial (no torque), the first tangential term is τ³ȧ.
  (Coefficient derived and verified in E6: `dL/dt ≈ G m² θ³/(6d)`.)
- Naive model: run-averaged rates are not representative (the orbit changes by O(1));
  its initial rate ∝ 1/c is verified in E6.
- Framing consequence for the paper: the "aberration" model is *not* conservative,
  only higher order (β³) — state it that way; never call it physical gravity.

### E10 — a physical SD-DDE: classical electrodynamic two-body (`exp_e10_lienard_wiechert.py`)
Answer to "retarded Newton is a toy": `retardation="lw"` = Liénard–Wiechert fields
(Gaussian units, Coulomb constant 1) + relativistic mechanics `a = (F − (v·F)v/c²)/(γm)`;
"classical positronium" (charges ±1, unit masses, Coulomb-circular start, c = 8 → β ≈ 0.09).
The radiation field `n×((n−β)×β̇)/(cκ³R)` needs the source **acceleration** at the
retarded time → a **neutral** SD-DDE (Driver's problem class). Reference: DelayDiffEq
`neutral=true`, source acceleration from the history derivative `h(p, t, Val{1})`, 128-bit.
c → ∞ check: identical to the Newtonian binary to 1e-15.

(a) engine error vs tracking level (W = T/4, T = 1 ≈ 8τ):

| L | windows | M=16 | M=24 | M=32 | M=48 |
|---|---|---|---|---|---|
| 0 | 4 | 6.9e-6 | 2.7e-6 | 1.3e-6 | 9.5e-7 |
| 1 | 5 | 2.2e-8 | 1.9e-8 | 1.7e-8 | 1.7e-8 |
| 2 | 6 | 1.6e-10 | 1.6e-10 | 1.7e-10 | 1.7e-10 |
| 4 | 7 | 1.6e-14 | 5.7e-15 | 9.3e-15 | 1.1e-14 |
| 6 | 8 | 1.1e-16 | 2.2e-16 | 2.2e-16 | 1.4e-16 |
| ≥8 | 9 | 1.7e-16 | 2.8e-16 | 1.4e-16 | 9.7e-17 |

Exactly the neutral prediction of the order rule (q = 2 ⇒ `k_ℓ = k_φ`, no smoothing):
**M does not help at fixed L** (plateaus), and each tracked level gains a constant
factor ≈ 60–100 — the neutral jump amplitude decays per level by
`|∂f/∂a_j| ≈ q²/(m c² R) = 1/64`. Once every break inside T is tracked: round-off.

(b) DelayDiffEq neutral, Float64, best over Vern9/Vern7 × lags/no-lags × tol 1e-8…1e-14:
**6.3e-14** (Vern9, tol 1e-14, nf = 4674). WR: **~1e-16** at M = 16 (≈ 9 windows × ~10
iterations × 16 nodes ≈ 1.5k RHS).

Engineering needed for the neutral case (both are generic, also used by gravity runs):
- **One-sided source lookups at joins**: the acceleration jumps *in value* at a break, so
  a window's last node must read the left limit and its first node the right limit
  (`SegmentHistory.acc(side=)`). Without it: ~M⁻² error from the end-node weights
  (observed 1.8e-7 / 7.8e-8 at M = 32 / 48 before the fix).
- **Root refinement**: the bisection root comes from a window that straddles the kink
  (only ~6e-8 accurate); a Newton step on `g(t_end) = t_end − τ(t_end) − d` with the
  truncated window's Chebyshev τ lands the window end on the root to ~1e-15.

### E11 — LW physics check: mutual radiation damping (`exp_e11_lw_radiation.py`)
To O(1/c²) the LW two-body problem is conservative Darwin dynamics; the first
non-conservative term is the 1/c³ cross term `F_i = (2/3c³) q_i q_j ȧ_j` (the model has no
self-force). Circular ±1 binary: `ȧ = −Ω²v` ⇒ `dE_Darwin/dt = −(4/3)Ω²v²/c³ = −1.333/c³`
— exactly half the Larmor dipole power (the other half would be the self-force).
Engine (track all levels, M = 24, slope of the Darwin energy on t ∈ [1, 3]):

| c | engine | predicted | ratio |
|---|---|---|---|
| 8 | −2.508e-3 | −2.604e-3 | 0.963 |
| 16 | −3.188e-4 | −3.255e-4 | 0.979 |
| 32 | −4.033e-5 | −4.069e-5 | 0.991 |
| 64 | −5.067e-6 | −5.086e-6 | 0.996 |

Ratio → 1 with a ~1/c correction (next order). Confirms the LW implementation down to
the radiation-reaction order — it is real classical electrodynamics, not a toy.

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
5. **Misleading clamp counter:** pruned-history lookups were counted during Picard
   iterations too (wandering iterates), so a warning/counter fired although the
   converged solution never left the retained past (cluster N=8: 67 → 0). Now counted
   only on the final evaluation at the fixed point.
6. New: warm-start predictor (4th-order Taylor from the previous window's Chebyshev
   polynomial): ~10 % fewer iterations (81→73 binary, 90→82 triple), same answer
   (`test_warm_start_same_answer_fewer_iterations`). Truncated re-solves now fall back
   to halving if they do not converge.
7. New: breaking-point tracking (`PiecewiseVolterraStream(track_breaks=L)`) and C²
   initial history (`initial_A0=`). Tests:
   `test_breaking_point_alignment_restores_spectral_convergence`,
   `test_breaks_are_registered_and_bounded`.

8. **Second independent review (2026-11-24) — confirmed and fixed:**
   - *Near-coincident break roots were merged or dropped* (time-based dedup 1e-4·W,
     start margin 1e-6·span). In the neutral LW model this silently cost ~5e-7 for nearly
     symmetric systems (masses 1 and 1+ε). Now breaks are identified by their **origin**
     `(receiver i, source j, parent d)` — the same root re-found after a re-solve is not
     new, two different roots are never merged by time; roots within 1e-9·span of a
     window start are *registered* (descendants tracked) instead of dropped; roots closer
     than 1e-9·W are aligned together but each keeps its own origin. Root landing uses the
     closed-form light-cone derivative `τ' = n·(v_i − v_j)/(c − n·v_j)` (well conditioned on
     sliver windows; the Chebyshev derivative was not). Warm start / Newton landing are
     skipped on windows < 1e-6·W.
     Near-symmetric LW (c = 8, T = 1), max |M16 − M32| before → after:
     ε = 1e-3: 5.4e-7 → 1.4e-11; 1e-4: 9e-9 → 5e-11; 1e-5: 5.4e-7 → 2e-9;
     1e-6: 1e-10 → 2e-13; 1e-7: 8e-15 → 7e-15; ε = 0, 0.1: round-off.
     **Open:** ε = 1e-3…1e-5 is not yet at round-off. Against the 128-bit reference M = 16
     is at 1e-15 but M = 32/48 at 1e-11…2e-9; the deviation starts in the first full window
     after the second sliver; it is not the one-sided-lookup tolerance (varying it does
     not help, shrinking it breaks M = 16). Root cause not found.
   - *Error control halved before break alignment* (kink inflated the tail → needless
     halvings, slivers). Now alignment first, control second.
   - *`_trunc_err` not Galilean-invariant* (scaled by max|V| incl. the centre-of-mass
     velocity). Now relative to the spread about the centre of mass.
   - *Clamp counter included rejected trial solves*: now only the accepted solve counts,
     and the warning is issued once per accepted window.
   - **Regression caught while fixing:** an earlier slice edit (closed-form τ') had
     deleted `_trunc_err`, so `accuracy_tol` raised AttributeError and no test covered it.
     E7 numbers predate that edit and were re-run afterwards. New test
     `test_accuracy_control_resolves_close_encounter_and_is_boost_invariant`.
   - Clean per review: LW formulas (checked against finite differences of the LW
     potentials, 8e-9 = FD error; radiation term 12 % of the force), vectorised field vs
     loop (N = 4, unequal masses, softening, both schemes), `_A_cur` handling, one-sided
     history logic, `_end_derivatives` scaling, `_span_hint` bounds.
   - Noted, not fixed: Anderson needs ~45 % more iterations than Picard on LW (it does not
     see the lagged source acceleration); LW with N = 3 and full tracking cascades into
     ~5e-5-wide windows (correct but slow).

Findings that contradict the v0.3.0 README (fix README before any public claim):
- "Anderson 32 vs 54 iterations" compared against *damped* Picard; undamped Picard
  beats both (E2/E2b). `test_anderson_beats_picard` encodes that misleading comparison.
- "single-window Picard radius ≈ 1 orbit" — not observed with adequate nodes (6 orbits
  converge in 24 iterations).
- "delay buffer … resampled onto a fixed grid" — no longer true.

## 6. Open items / what a referee will ask

- [~] Lemma for the breaking-point order rule: measured (E8), proof sketch written;
      make rigorous + compare with Bellen–Zennaro discontinuity propagation.
- [ ] Rigor of §3: invariant set `X_M`, aberration model (needs bounded `x″`), the
      discrete CGL operator (`ρ(K2)`), and evaluate the shifted-kernel bound of §3 against E2.
- [ ] Tracking cost for larger N (E7: windows ∝ N²). Options: split only the affected
      body (multirate WR: windows per body), or skip breaks whose jump order exceeds
      what M can resolve (order ≥ ~5 with C² φ is already below 1e-10 at M = 16).
- [ ] Error control wastes rejected attempts; better predictor for re-solves / new
      windows (higher-order Taylor from the previous window's end) should cut the
      ~7 iterations per window.
- [ ] Review findings, not yet addressed (design risks): (a) [done: body-tagged breaks];
      (b) with `track_breaks` on, presets whose light delay ≪ window (solar,
      spacecraft) crawl in tiny windows — tracking must stay opt-in / be capped;
      (c) truncated re-solves are not re-checked for convergence (reported in
      `WindowResult.converged`, never silently fixed); (d) a lookup into pruned past
      clamps to the oldest node (2e-4 error with `buffer_safety=1.0`; 0 clamps at the
      default 2.5) — interpolate/extrapolate instead, or raise.
- [x] Scaling with N: vectorised (E7). GPU/Colab only if the paper wants N ≥ 100.
- [x] Long-time runs (E9); β³ coefficient derived + verified (E6).
- [ ] Verify every (?) citation in §1; read Hale 2024 and the 2023 constant-delay
      Chebyshev–Picard paper in full and state the delta precisely.
- [ ] RADAR5 / ddesd comparison (optional; DelayDiffEq is the strongest open baseline).
- [x] Physical SD-DDE: Liénard–Wiechert two-body, neutral (E10); radiation damping
      verified (E11). [ ] N > 2 charges; relate to Driver's existence/uniqueness results.
- [ ] arXiv: a first submission in math.NA / physics.comp-ph needs an endorser.

## 7. Reproduce

```bash
cd ~/aura-volterra-engine
# Julia env (once): ~/.julia-envs/dde with DelayDiffEq, OrdinaryDiffEq, JSON3
.venv/bin/python -m pytest -q                          # 31 tests
.venv/bin/python research/exp_e7_scaling.py 3 4 6 8 12 16  # ~6 min (N=16 WR-C dominates)
.venv/bin/python research/exp_e6_laplace.py            # ~10 s
.venv/bin/python research/exp_e8_break_order.py        # ~30 s
.venv/bin/python research/exp_e9_longtime.py           # ~30 s
.venv/bin/python research/exp_e10_lienard_wiechert.py  # ~1 min
.venv/bin/python research/exp_e11_lw_radiation.py      # ~5 s
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

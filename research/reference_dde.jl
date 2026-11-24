# Independent reference for the retarded (state-dependent-delay) N-body problem.
# Solves u = [x; v],  x' = v,  v_i' = sum_j G m_j (s_ij - x_i)/(|s_ij - x_i|^2 + eps^2)^{3/2}
#   s_ij = x_j(t - tau_ij)              ("naive")
#   s_ij = x_j(t - tau_ij) + tau_ij v_j(t - tau_ij)   ("aberration")
#   tau_ij solves tau = |x_i(t) - x_j(t - tau)| / c   (Newton, as in the Python engine)
# History phi(t), t < t0: ballistic  x0 + v0 (t - t0)  (same as the Python engine).
# Usage: julia --project=~/.julia-envs/dde reference_dde.jl problem.json out.json
using DelayDiffEq, OrdinaryDiffEq, JSON3, LinearAlgebra

prob_in = JSON3.read(read(ARGS[1], String))
N  = length(prob_in.masses)
m  = Float64.(collect(prob_in.masses))
G  = Float64(prob_in.G); c = Float64(prob_in.c); eps2 = Float64(prob_in.softening)^2
model = String(prob_in.retardation)
X0 = reduce(vcat, [Float64.(collect(r)) for r in prob_in.R0])
V0 = reduce(vcat, [Float64.(collect(v)) for v in prob_in.V0])
t0 = Float64(prob_in.t0); T = Float64(prob_in.T)
tol = Float64(get(prob_in, :tol, 1e-12))
alg_name = String(get(prob_in, :alg, "Vern9"))
tout = Float64.(collect(prob_in.tout))

bigf = Bool(get(prob_in, :bigfloat, false))
FT = bigf ? BigFloat : Float64
bigf && setprecision(BigFloat, 128)
X0 = FT.(X0); V0 = FT.(V0)
G = FT(G); c = FT(c); eps2 = FT(eps2); m = FT.(m); t0 = FT(t0); T = FT(T)
u0 = vcat(X0, V0)
hist_kind = String(get(prob_in, :history, "ballistic"))
A0 = FT.(haskey(prob_in, :A0) ? reduce(vcat, [Float64.(collect(a)) for a in prob_in.A0]) : zeros(3N))
# ballistic: x0 + v0 s ; quadratic: x0 + v0 s + a0 s^2/2 (a0 = force at t0 -> C^2 start)
hist(p, t; idxs=nothing) = hist_kind == "quadratic" ?
    vcat(X0 .+ V0 .* (t - t0) .+ FT(0.5) .* A0 .* (t - t0)^2, V0 .+ A0 .* (t - t0)) :
    vcat(X0 .+ V0 .* (t - t0), V0)
# derivative of the history (needed by the neutral Lienard-Wiechert model)
hist(p, t, ::Type{Val{1}}; idxs=nothing) = hist_kind == "quadratic" ?
    vcat(V0 .+ A0 .* (t - t0), A0) : vcat(V0, zero(A0))
charges = haskey(prob_in, :charges) ? FT.(collect(prob_in.charges)) : zeros(FT, N)
cross3(a, b) = [a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1]]

function lw_rhs!(du, u, h, p, t)
    # Lienard-Wiechert fields (Gaussian units, Coulomb constant 1), relativistic
    # mechanics; source x, v, a taken at the retarded time (neutral SD-DDE)
    x = @view u[1:3N]; v = @view u[3N+1:6N]
    du[1:3N] .= v
    for i in 1:N
        xi = x[3i-2:3i]; vi = v[3i-2:3i]
        F = zeros(eltype(u), 3)
        for j in 1:N
            i == j && continue
            tau = norm(x[3j-2:3j] .- xi) / c
            for _ in 1:6
                uj = h(p, t - tau)
                xj = uj[3j-2:3j]; vj = uj[3N+3j-2:3N+3j]
                rel = xi .- xj; dist = norm(rel) + eps(eltype(u))^4
                g = tau - dist / c
                dg = 1 - dot(rel ./ dist, vj) / c
                tau = max(tau - g / dg, zero(tau))
            end
            uj = h(p, t - tau); duj = h(p, t - tau, Val{1})
            xj = uj[3j-2:3j]; vj = uj[3N+3j-2:3N+3j]; aj = duj[3N+3j-2:3N+3j]
            rel = xi .- xj; R2 = dot(rel, rel) + eps2; R = sqrt(R2)
            n = rel ./ R; b = vj ./ c; bd = Bool(get(prob_in, :no_radiation, false)) ? zero(aj) : aj ./ c
            kappa = 1 - dot(n, b); nb = n .- b
            E = charges[j] .* (nb .* ((1 - dot(b, b)) / (kappa^3 * R2)) .+
                               cross3(n, cross3(nb, bd)) ./ (c * kappa^3 * R))
            B = cross3(n, E)
            F .+= charges[i] .* (E .+ cross3(vi ./ c, B))
        end
        gam = 1 / sqrt(1 - dot(vi, vi) / c^2)
        du[3N+3i-2:3N+3i] .= (F .- (dot(vi, F) / c^2) .* vi) ./ (gam * m[i])
    end
    nothing
end

function rhs!(du, u, h, p, t)
    model == "lw" && return lw_rhs!(du, u, h, p, t)
    x = @view u[1:3N]; v = @view u[3N+1:6N]
    du[1:3N] .= v
    a = zeros(eltype(u), 3N)
    for i in 1:N
        xi = x[3i-2:3i]
        for j in 1:N
            i == j && continue
            if model == "none"
                s = x[3j-2:3j]
            else
                d0 = x[3j-2:3j] .- xi
                tau = norm(d0) / c
                for _ in 1:6
                    uj = h(p, t - tau)
                    xj = uj[3j-2:3j]; vj = uj[3N+3j-2:3N+3j]
                    rel = xi .- xj; dist = norm(rel) + eps(eltype(u))^4
                    g = tau - dist / c
                    dg = 1.0 - dot(rel ./ dist, vj) / c
                    tau = max(tau - g / dg, 0.0)
                end
                uj = h(p, t - tau)
                s = uj[3j-2:3j]
                if model == "aberration"
                    s = s .+ tau .* uj[3N+3j-2:3N+3j]
                end
            end
            d = s .- xi
            r2 = dot(d, d) + eps2
            a[3i-2:3i] .+= (G * m[j]) .* d ./ r2^1.5
        end
    end
    du[3N+1:6N] .= a
    nothing
end

track = Bool(get(prob_in, :dependent_lags, false))
if track && model != "none"
    # DelayDiffEq tracks discontinuities through lag functions of the CURRENT
    # state only; the true lag is implicit (light cone), so give the
    # instantaneous approximation |x_i - x_j| / c (best this API allows).
    lags = Tuple((u, p, t) -> norm(u[3i-2:3i] .- u[3j-2:3j]) / c for i in 1:N for j in 1:N if i != j)
    prob = DDEProblem(rhs!, u0, hist, (t0, T); dependent_lags = lags, neutral = model == "lw")
else
    prob = DDEProblem(rhs!, u0, hist, (t0, T); neutral = model == "lw")
end
tol = FT(tol)
alg = alg_name == "Vern9" ? MethodOfSteps(Vern9()) :
      alg_name == "Tsit5" ? MethodOfSteps(Tsit5()) :
      alg_name == "RK4"   ? MethodOfSteps(RK4()) : MethodOfSteps(Vern7())
# warm-up solve on a short span so `wall` excludes JIT compilation
solve(remake(prob; tspan=(t0, t0 + (T - t0) / 50)), alg; abstol=tol, reltol=tol)
t_start = time()
sol = solve(prob, alg; abstol=tol, reltol=tol, maxiters=10^8, dense=true)
wall = time() - t_start
U = [Float64.(sol(FT(t))) for t in tout]
out = Dict(
    "retcode" => string(sol.retcode),
    "nf" => sol.stats.nf, "naccept" => sol.stats.naccept, "nreject" => sol.stats.nreject,
    "wall" => wall,
    "R" => [[u[1:3N]...] for u in U],
    "V" => [[u[3N+1:6N]...] for u in U],
)
open(ARGS[2], "w") do io
    JSON3.write(io, out)
end
println("retcode=", sol.retcode, " nf=", sol.stats.nf, " wall=", round(wall, digits=3))

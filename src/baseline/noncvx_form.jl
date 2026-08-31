# Nonconvex global baseline for the data-driven Wasserstein DRO
# reformulations (1.3) and (1.6) of
#   Zhang & Zhong (2025), "Moment Relaxations for Data-Driven Wasserstein
#   Distributionally Robust Optimization".
#
# (1.3) is the standard finite-dimensional reformulation of (1.1) obtained
# from Slater's strong duality (with λ the dual to the Wasserstein-radius
# constraint):
#
#   min_{x ∈ X, λ ≥ 0}  f(x) + r^p λ
#       + (1/N) Σ_{i=1}^N  sup_{ξ ∈ Ξ}  [ F(x, ξ) - λ ‖ξ - ξ̂^(i)‖^p ].
#
# The pointwise supremum is a (generally nonconvex) polynomial optimization
# problem in ξ. When F arises from a linear-recourse value function, (1.6)
# gives the LP-dual representation
#
#   F(x, ξ) = max_{u ∈ ℝ^{n₂}}  uᵀ B(ξ) x + b(ξ)ᵀ u + d(ξ)
#             s.t.              c(ξ) - Aᵀ u ≥ 0,
#
# so the inner sup in (1.3) becomes a joint nonconvex polynomial
# maximization over (ξ, u). In `src/types.jl` this is packaged as a
# `SampleLinearRecourse`, which stores the same data through a polynomial
# matrix C(ξ), a polynomial matrix A(ξ), and a polynomial vector b(ξ) with
#
#   F(x, ξ) = max_y  (1, x)ᵀ C(ξ) (1, y)    s.t. A(ξ) y - b(ξ) ≥ 0,
#
# the recourse variable `y ∈ ℝ^{m₂}` (mapped to u in (1.6)) optionally
# bounded by |y_k| ≤ B_k (see `SampleLinearRecourse.B`; B_k ≤ 0 encodes ∞).
#
# `eval_noncvx_Wass` mirrors the I/O contract of `eval_moment_Wass` in
# `src/moment_relax.jl`: it consumes the same subproblem object, the
# augmented state augstate = (x̄, w̄) (where w̄ plays the role of the dual
# λ used in `solve_main_level`), a vector of i.i.d. samples ξ̂^(1),…,ξ̂^(N),
# and a `WassInfo` (Wasserstein radius r and order p). It returns the same
# sample-averaged subgradient cut for `MainProblem.ϕ` — the first entry is
# the constant, the next `length(x̄)` entries are the gradient in x, and the
# last entry is the subgradient in w — built from the per-sample
# subgradients (2.8) / (2.9) of the paper. The only methodological
# difference vs. `eval_moment_Wass` is that the inner supremum is solved
# directly by handing the polynomial program to `PolyJuMP.QCQP.Optimizer`,
# which automatically lifts every degree-≥3 monomial to an auxiliary JuMP
# variable with a quadratic equality and forwards the resulting QCQP to a
# nonconvex global solver (e.g., `Gurobi.Optimizer` with `NonConvex = 2`).
# The inner solver is supplied via the `noncvx_solver` kwarg as a callable
# that returns a fresh MOI optimizer instance (matching the
# `() -> Gurobi.Optimizer(GRB_ENV)` style used in `example/`).


# Internal helper: substitute the symbolic variables `sym_vars` (from
# DynamicPolynomials) in the polynomial `poly` with the JuMP decision
# variables `jump_vars`, producing a JuMP expression. High-degree monomials
# stay as `NonlinearExpr` and are later reformulated to QCQP by the
# `PolyJuMP.QCQP.Optimizer` bridges.
function _noncvx_subs_jump(poly, sym_vars::Vector, jump_vars::Vector{VariableRef})
    # guard against the (rare) case in which `subs` collapsed `poly` to a number
    poly isa Real && return Float64(poly)
    # qualify the polynomial accessors — `coefficient` / `monomial` / `terms`
    # are also exported by JuMP, so use the `MultivariatePolynomials` versions.
    expr = zero(AffExpr)
    for t in MultivariatePolynomials.terms(poly)
        c   = Float64(MultivariatePolynomials.coefficient(t))
        mon = MultivariatePolynomials.monomial(t)
        # build c · ∏_k jump_vars[k]^deg(mon, sym_vars[k])
        term_val = c
        for (k, sv) in enumerate(sym_vars)
            e = Int(MultivariatePolynomials.degree(mon, sv))
            e == 0 && continue
            term_val = term_val * jump_vars[k]^e
        end
        expr = expr + term_val
    end
    return expr
end


# Per-sample helper for the SamplePolynomialLoss case with F = max_k F[k]:
# for the sample i and each polynomial F_k, solve the nonconvex inner
# maximisation `max_ξ F_k(x̄, ξ) - w̄·‖ξ-ξ̂‖^p` on Ξ, then return the cut
# from the argmax-k branch. Uses the constant-polynomial shortcut for F_k
# that no longer depends on ξ after the x=>x̄ substitution. Called via
# `pmap` from `eval_noncvx_Wass`; for true parallel execution the caller
# must have added workers (e.g. `addprocs(...)`) and loaded MoWDRO on them
# (`@everywhere using MoWDRO`). Returns `nothing` if any k-th QCQP fails
# — same conservative semantics as before.
function _gen_noncvx_cut_polynomial_loss(
        i::Int,
        loss::SamplePolynomialLoss,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo,
        fs::Vector,
        ∇fs::Vector,
        x̄::Vector{Float64},
        w̄::Float64,
        print::Int,
        noncvx_solver,
    )
    ξ̂ = samples[i]
    d = length(ξ̂)
    dim_x = length(x̄)
    K = length(fs)
    # shared PolyJuMP.QCQP.Optimizer factory (silent when print <= 1)
    inner_factory = if print <= 1
        () -> begin
            inner = noncvx_solver()
            MOI.set(inner, MOI.Silent(), true)
            inner
        end
    else
        noncvx_solver
    end
    # track the argmax-k inner-sup value and the associated cut
    best_val = -Inf
    best_cut = nothing
    for k = 1:K
        f_k = fs[k]
        # Constant-polynomial shortcut: sup_{ξ∈Ξ}[c - w̄·p(ξ)] = c at ξ=ξ̂,
        # so the exact cut is [c; 0; r^p]; skip the QCQP solve entirely.
        if f_k isa Real || maxdegree(f_k) <= 0
            c = convert(Float64, f_k)
            if c > best_val
                best_val = c
                best_cut = [c; zeros(dim_x); wassinfo.r^wassinfo.p]
            end
            continue
        end
        # nonconvex inner maximisation of the k-th branch
        model = Model(() -> PolyJuMP.QCQP.Optimizer(inner_factory()))
        @variable(model, ξ[1:d])
        # encode Ξ = { ξ : h_j(ξ) ≥ 0 } and any equality part (variety) of Ξ
        for h in loss.Ξ.p
            @constraint(model, _noncvx_subs_jump(h, loss.ξ, ξ) >= 0)
        end
        if loss.Ξ.V != FullSpace()
            for h in loss.Ξ.V.I.p
                @constraint(model, _noncvx_subs_jump(h, loss.ξ, ξ) == 0)
            end
        end
        # objective F_k(x̄, ξ) - w̄ ‖ξ - ξ̂‖^p
        F_expr  = _noncvx_subs_jump(f_k, loss.ξ, ξ)
        pen_sym = sum((loss.ξ[j] - ξ̂[j])^wassinfo.p for j = 1:d)
        P_expr  = _noncvx_subs_jump(pen_sym, loss.ξ, ξ)
        @objective(model, Max, F_expr - w̄ * P_expr)
        optimize!(model)
        if !is_solved_and_feasible(model, allow_almost=true)
            if print >= 0
                println("DEBUG: nonconvex polynomial-loss subproblem $i (branch k=$k) did not solve, status: ",
                        termination_status(model))
                println("DEBUG: the current main problem solution is\n", x̄)
                println("DEBUG: the current Wasserstein auxiliary variable is ", w̄)
            end
            return nothing
        end
        ξ_star = value.(ξ)
        v̂ = convert(Float64, subs(f_k, loss.ξ => ξ_star))
        ĝ = [convert(Float64, subs(g, loss.ξ => ξ_star)) for g in ∇fs[k]]
        p̂ = sum((ξ_star[j] - ξ̂[j])^wassinfo.p for j = 1:d)
        val_k = v̂ - w̄ * p̂
        if val_k > best_val
            best_val = val_k
            best_cut = [v̂ - ĝ' * x̄; ĝ; wassinfo.r^wassinfo.p - p̂]
        end
    end
    return best_cut
end


# ---------------------------------------------------------------------------
# eval_noncvx_Wass — single-stage polynomial loss (paper (1.3) with F a
# polynomial in (x, ξ)). Same input / output contract as `eval_moment_Wass`:
# returns the aggregate cut for `MainProblem.ϕ` at augstate = (x̄, w̄). The
# i-th inner supremum in (1.3) is solved directly as a (generally nonconvex)
# polynomial maximization in ξ ∈ Ξ via `PolyJuMP.QCQP.Optimizer`.
function eval_noncvx_Wass(
        loss::SamplePolynomialLoss,
        augstate::Vector{Float64},
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo;
        print::Int = 0,
        noncvx_solver = DEFAULT_NCVX
    )
    N = length(samples)
    # alias the augmented state — same convention as `eval_moment_Wass` and
    # the `solve_main_level` driver in `src/level_bundle.jl`.
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    K = length(loss.F)
    # specialize each F[k] and ∇ₓF[k] at x = x̄ (polynomials in ξ alone)
    fs  = [subs(loss.F[k], loss.x => x̄) for k in 1:K]
    ∇fs = [[subs(g, loss.x => x̄) for g in loss.∇ₓF[k]] for k in 1:K]
    # parallelise the per-sample nonconvex solves; pmap preserves the input
    # index order, so cuts[i] is always the cut for samples[i].
    cuts = pmap(
        i -> _gen_noncvx_cut_polynomial_loss(
                i, loss, samples, wassinfo, fs, ∇fs, x̄, w̄, print, noncvx_solver),
        1:N,
    )
    # if any sample's QCQP solve failed, propagate the same `nothing` that
    # the original sequential implementation would have returned.
    any(isnothing, cuts) && return nothing
    return combine_linear_cuts(Vector{Vector{Float64}}(cuts))
end


# Per-sample helper for the SampleLinearRecourse case; same calling
# convention as `_gen_noncvx_cut_polynomial_loss`.
function _gen_noncvx_cut_linear_recourse(
        i::Int,
        recourse::SampleLinearRecourse,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo,
        x̄::Vector{Float64},
        w̄::Float64,
        n_y::Int,
        d::Int,
        print::Int,
        noncvx_solver,
        val_add_bound::Float64,
    )
    ξ̂ = samples[i]
    # i-th inner supremum in (1.3) with F as in (1.6):
    #   max_{ξ, y}  (1, x̄)ᵀ C(ξ) (1, y) - w̄ · Σ_j (ξ_j - ξ̂^(i)_j)^p
    #   s.t.        ξ ∈ Ξ,  A(ξ) y - b(ξ) ≥ 0,  |y_k| ≤ B_k  (B_k > 0)
    # `PolyJuMP.QCQP.Optimizer` does not forward `MOI.Silent` to the
    # inner solver, so we cannot use `set_silent(model)` after the fact.
    # Instead, set the silent flag on the inner solver *before* wrapping
    # it with `PolyJuMP.QCQP`.
    inner_factory = if print <= 1
        () -> begin
            inner = noncvx_solver()
            MOI.set(inner, MOI.Silent(), true)
            inner
        end
    else
        noncvx_solver
    end
    model = Model(() -> PolyJuMP.QCQP.Optimizer(inner_factory()))
    @variable(model, ξ[1:d])
    @variable(model, y[1:n_y])
    # recourse-variable box bound  |y_k| ≤ B_k
    # (per `src/types.jl`, B_k ≤ 0 encodes "unbounded")
    for k = 1:n_y
        if recourse.B[k] > 0.0
            set_lower_bound(y[k], -recourse.B[k])
            set_upper_bound(y[k],  recourse.B[k])
        end
    end
    # optional additional explicit box (parity with `eval_moment_Wass`)
    if val_add_bound > 0.0
        for k = 1:n_y
            set_lower_bound(y[k], -val_add_bound)
            set_upper_bound(y[k],  val_add_bound)
        end
    end
    # joint substitution lists for the helper
    sym_vars  = [recourse.ξ; recourse.y]
    jump_vars = [ξ; y]
    # support set Ξ (inequalities, plus any equality variety part)
    for h in recourse.Ξ.p
        @constraint(model, _noncvx_subs_jump(h, sym_vars, jump_vars) >= 0)
    end
    if recourse.Ξ.V != FullSpace()
        for h in recourse.Ξ.V.I.p
            @constraint(model, _noncvx_subs_jump(h, sym_vars, jump_vars) == 0)
        end
    end
    # recourse-feasibility constraints A(ξ) y - b(ξ) ≥ 0
    for r in recourse.A * recourse.y - recourse.b
        @constraint(model, _noncvx_subs_jump(r, sym_vars, jump_vars) >= 0)
    end
    # full polynomial objective in (ξ, y) — F_sym is a scalar polynomial
    F_sym   = ([1.0; x̄])' * recourse.C * [1; recourse.y]
    pen_sym = sum((recourse.ξ[j] - ξ̂[j])^wassinfo.p for j = 1:d)
    F_expr  = _noncvx_subs_jump(F_sym,   sym_vars, jump_vars)
    P_expr  = _noncvx_subs_jump(pen_sym, sym_vars, jump_vars)
    @objective(model, Max, F_expr - w̄ * P_expr)
    optimize!(model)
    if !is_solved_and_feasible(model, allow_almost=true) 
        if print >= 0
            println("DEBUG: nonconvex linear-recourse subproblem $i did not solve, status: ",
                    termination_status(model))
            println("DEBUG: the current main problem solution is\n", x̄)
            println("DEBUG: the current Wasserstein auxiliary variable is ", w̄)
        end
        error("Fail to solve the nonconvex baseline approach.") # prevent stagnation from slow nonconvex baseline approaches
        return nothing
    end
    ξ_star = value.(ξ)
    y_star = value.(y)
    # ĉ = C(ξ*) · [1; y*] — same shape as in `eval_moment_Wass` (cf. (2.9))
    C_at_xi = convert.(Float64, subs.(recourse.C, recourse.ξ => ξ_star))
    ĉ = C_at_xi * [1.0; y_star]
    p̂ = sum((ξ_star[j] - ξ̂[j])^wassinfo.p for j = 1:d)
    # cut'·[1; x; w] = ĉ[1] + ĉ[2:n_x+1]' x + (r^p - p̂) w
    return [ĉ; wassinfo.r^wassinfo.p - p̂]
end


# ---------------------------------------------------------------------------
# eval_noncvx_Wass — two-stage linear recourse (paper (1.3) combined with F
# as in (1.6)). The inner sup is treated jointly in (ξ, y), where y is the
# recourse / dual variable (denoted u in (1.6) and `SampleLinearRecourse.y`
# in `src/types.jl`). The returned cut aggregates per-sample subgradients
# constructed exactly as in (2.9) of the paper.
function eval_noncvx_Wass(
        recourse::SampleLinearRecourse,
        augstate::Vector{Float64},
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo;
        print::Int = 0,
        noncvx_solver = DEFAULT_NCVX,
        val_add_bound::Float64 = -1.0
    )
    N = length(samples)
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    n_y = length(recourse.y)
    d   = length(recourse.ξ)
    # parallelise the per-sample nonconvex solves; pmap preserves the input
    # index order, so cuts[i] is always the cut for samples[i].
    cuts = pmap(
        i -> _gen_noncvx_cut_linear_recourse(
                i, recourse, samples, wassinfo, x̄, w̄,
                n_y, d, print, noncvx_solver, val_add_bound),
        1:N,
    )
    # if any sample's QCQP solve failed, propagate the same `nothing` that
    # the original sequential implementation would have returned.
    any(isnothing, cuts) && return nothing
    return combine_linear_cuts(Vector{Vector{Float64}}(cuts))
end

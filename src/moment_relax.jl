# Moment relaxation for the loss and recourse functions

# Per-sample helper that builds and solves the SOS moment-relaxation models
# for the SamplePolynomialLoss case. When the loss F = max_k F[k] holds K
# polynomials, this loops over k = 1..K, solves the k-th relaxation, and
# returns the argmax-k cut. Returns `nothing` on unrecoverable infeasibility
# for any sub-problem (matching the single-polynomial semantics). Called via
# `pmap` from `eval_moment_Wass`; for true parallel execution the caller
# must have added workers (e.g. `addprocs(...)`) and loaded MoWDRO on them
# (`@everywhere using MoWDRO`).
function _gen_moment_cut_polynomial_loss(
        i::Int,
        loss::SamplePolynomialLoss,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo,
        fs::Vector,
        ∇fs::Vector,
        x̄::Vector{Float64},
        w̄::Float64,
        relaxdeg::Int,
        mom_solver,
        print::Int,
        val_relax_tol::Float64,
    )
    ξ̂ = samples[i]
    d = length(ξ̂)
    dim_x = length(x̄)
    K = length(fs)
    # Wasserstein penalty polynomial (shared across all F_k)
    p = sum((loss.ξ[j]-ξ̂[j])^wassinfo.p for j=1:d)
    # track the argmax-k relaxation value and cut
    best_val = -Inf
    best_cut = nothing
    for k = 1:K
        f_k = fs[k]
        # Constant-polynomial shortcut: after substituting x = x̄, f_k has
        # no ξ dependence. Then sup_{ξ∈Ξ} [c - w̄·p(ξ)] = c (attained at
        # ξ = ξ̂ where p = 0), so the exact per-sample DRO cut is
        #   cut = [c; 0; r^p],
        # equal to c + w̄·r^p at (x̄, w̄) — matching the true contribution
        # w̄·r^p + sup_ξ (c - w̄·p) identically at every (x̄, w̄).
        if f_k isa Real || maxdegree(f_k) <= 0
            c = convert(Float64, f_k)
            if c > best_val
                best_val = c
                best_cut = [c; zeros(dim_x); wassinfo.r^wassinfo.p]
            end
            continue
        end
        # SOS moment relaxation for the k-th polynomial
        model = SOSModel(mom_solver)
        if print <= 1
            set_silent(model)
        end
        @variable(model, optval)
        @objective(model, Min, optval)
        @constraint(model, constr, f_k-w̄*p <= optval, domain=loss.Ξ, maxdegree=relaxdeg)
        optimize!(model)
        val_k = nothing
        cut_k = nothing
        if is_solved_and_feasible(model, allow_almost=true)
            μ̄ = moments(constr)
            v̂ = expectation(μ̄, f_k)
            p̂ = expectation(μ̄, p)
            ĝ = map(m -> expectation(μ̄, m), ∇fs[k])
            val_k = objective_value(model)
            cut_k = [v̂-ĝ'*x̄; ĝ; wassinfo.r^wassinfo.p-p̂]
        elseif termination_status(model) == SLOW_PROGRESS
            if print >= 0
                println("DEBUG: slow progress reported by the solver...")
            end
            μ̄ = moments(constr)
            v̂ = expectation(μ̄, f_k)
            p̂ = expectation(μ̄, p)
            ĝ = map(m -> expectation(μ̄, m), ∇fs[k])
            v̄ = objective_value(model)
            v̂_check = v̂ - w̄*p̂
            if abs(v̄-v̂_check) / (1.0+max(abs(v̄),abs(v̂_check))) > val_relax_tol
                if print >= 1
                    println("DEBUG: The loss function evaluation error is ", v̄-v̂_check)
                    println("DEBUG: the current Wasserstein auxiliary variable is ", w̄)
                    println("DEBUG: the moment relaxation model is\n", model)
                end
            end
            val_k = v̄
            cut_k = [v̂-ĝ'*x̄; ĝ; wassinfo.r^wassinfo.p-p̂]
        else
            if print >= 1
                println("DEBUG: the moment relaxation degree is ", relaxdeg)
                println("DEBUG: the moment relaxation domain is\n", loss.Ξ)
                println("DEBUG: the moment relaxation objective is\n", f_k-w̄*p)
                println("DEBUG: the current main problem solution is\n", x̄)
                println("DEBUG: the current Wasserstein auxiliary variable is ", w̄)
                println("DEBUG: the moment relaxation model is\n", model)
                println("The moment relaxation for polynomial k=", k,
                        " has failed with status: ", termination_status(model))
            end
            return nothing
        end
        if val_k > best_val
            best_val = val_k
            best_cut = cut_k
        end
    end
    return best_cut
end

# evaluate the moment relaxation for the Wasserstein distributionally
# robust polynomial loss function with a given augmented state (x,w)
function eval_moment_Wass(
        loss::SamplePolynomialLoss,
        augstate::Vector{Float64},
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo;
        print::Int = 0,
        relaxdeg::Int = 0,
        mom_solver = DEFAULT_SDP,
        val_relax_tol::Float64 = VAL_TOL
    )
    N = length(samples)
    # alias the augmented state
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    K = length(loss.F)
    # specialise each F[k] and ∇ₓF[k] at x = x̄ (polynomials in ξ alone)
    fs  = [subs(loss.F[k],  loss.x=>x̄) for k in 1:K]
    ∇fs = [subs.(loss.∇ₓF[k], loss.x=>x̄) for k in 1:K]
    # set the default relaxation degree — largest across all F_k in ξ
    if relaxdeg <= 0
        deg_max = 0
        for f in fs
            if !(f isa Real)
                deg_max = max(deg_max, maxdegree(f))
            end
        end
        relaxdeg = max(deg_max, wassinfo.p)
    end
    # parallelise the per-sample SOS solves; pmap preserves the input
    # index order, so cuts[i] is always the cut for samples[i].
    cuts = pmap(
        i -> _gen_moment_cut_polynomial_loss(
                i, loss, samples, wassinfo, fs, ∇fs, x̄, w̄,
                relaxdeg, mom_solver, print, val_relax_tol),
        1:N,
    )
    # if any sample's SOS solve failed, propagate the same `nothing` that
    # the original sequential implementation would have returned.
    any(isnothing, cuts) && return nothing
    # return the aggregate cut
    return combine_linear_cuts(Vector{Vector{Float64}}(cuts))
end

# Per-sample helper for the SampleLinearRecourse case; same calling
# convention as `_gen_moment_cut_polynomial_loss`.
function _gen_moment_cut_linear_recourse(
        i::Int,
        recourse::SampleLinearRecourse,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo,
        x̄::Vector{Float64},
        w̄::Float64,
        relaxdeg::Int,
        mom_solver,
        flag_rad_prod::Bool,
        flag_lin_prod::Bool,
        val_add_bound::Float64,
        print::Int,
        val_relax_tol::Float64,
        N::Int,
    )
    ξ̂ = samples[i]
    d = length(ξ̂)
    # define the polynomial objective
    p = sum((recourse.ξ[j]-ξ̂[j])^wassinfo.p for j=1:d)
    f = [1;x̄]'*recourse.C*[1;recourse.y] - w̄*p
    # define the semi-algebraic set
    Y = basicsemialgebraicset(FullSpace(), recourse.A*recourse.y-recourse.b)
    if minimum(recourse.B) > 0.0
        m = length(recourse.y)
        Y = intersect(Y, basicsemialgebraicset(FullSpace(), [recourse.B[i]^2-recourse.y[i]^2 for i in 1:m]))
    end
    S = intersect(recourse.Ξ, Y)
    if flag_rad_prod
        R2 = (wassinfo.r^wassinfo.p*N)^(2/wassinfo.p)
        p2 = (recourse.ξ-ξ̂)'*(recourse.ξ-ξ̂)
        m = length(recourse.b)
        S = intersect(S, basicsemialgebraicset(FullSpace(), [(recourse.A*recourse.y-recourse.b)[i]*(R2-p2) for i in 1:m]))
    end
    if flag_lin_prod
        m = length(recourse.b)
        Y = recourse.A*recourse.y-recourse.b
        S = intersect(S, basicsemialgebraicset(FullSpace(), [Y[i]*Y[j] for i in 1:m for j in i:m]))
    end
    if val_add_bound > 0.0
        B = val_add_bound
        n = length(recourse.y)
        S = intersect(S, basicsemialgebraicset(FullSpace(), [B^2-recourse.y[i]^2 for i in 1:n]))
    end
    # define the SOS optimization model
    model = SOSModel(mom_solver)
    if print <= 1
        set_silent(model)
    end
    @variable(model, optval)
    @objective(model, Min, optval)
    @constraint(model, constr, optval >= f, domain=S, maxdegree=relaxdeg)
    # solve the SOS model and extract the (pseudo-)moments/measure
    optimize!(model)
    if is_solved_and_feasible(model, allow_almost=true)
        μ̄ = moments(constr)
        # retrieve the pseudo-expectations for the polynomials
        ĉ = map(m->expectation(μ̄,m), recourse.C*[1;recourse.y])
        p̂ = expectation(μ̄,p)
        return [ĉ;wassinfo.r^wassinfo.p-p̂]
    elseif termination_status(model) == SLOW_PROGRESS
        if print >= 0
            println("DEBUG: slow progress reported by the solver...")
        end
        μ̄ = moments(constr)
        # retrieve the pseudo-expectations for the polynomials
        ĉ = map(m->expectation(μ̄,m), recourse.C*[1;recourse.y])
        p̂ = expectation(μ̄,p)
        # check if the objective values agree
        v̄ = objective_value(model)
        v̂ = [1;x̄]'*ĉ-w̄*p̂
        if abs(v̄-v̂) / (1.0+max(abs(v̄),abs(v̂))) > val_relax_tol
            if print >= 1
                println("DEBUG: The recourse evaluation error is ", v̄-v̂)
            end
        end
        return [ĉ;wassinfo.r^wassinfo.p-p̂]
    else
        if print >= 1
            println("DEBUG: the moment relaxation degree is ", relaxdeg)
            println("DEBUG: the moment relaxation domain is\n", S)
            println("DEBUG: the moment relaxation objective is\n", f)
            println("DEBUG: the current main problem solution is\n", x̄)
            println("DEBUG: the current Wasserstein auxiliary variable is ", w̄)
            println("DEBUG: the moment relaxation model is\n", model)
            println("The moment relaxation has failed with status: ", termination_status(model))
        end
        return nothing
    end
end

# evaluate the moment relaxation for the Wasserstein distributionally
# robust linear recourse problem with a given augmented state (x,w)
function eval_moment_Wass(
        recourse::SampleLinearRecourse,
        augstate::Vector{Float64},
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo;
        print::Int = 0,
        relaxdeg::Int = 0,
        flag_rad_prod::Bool = false,
        flag_lin_prod::Bool = false,
        val_add_bound::Float64 = -1.0,
        mom_solver = DEFAULT_SDP,
        val_relax_tol::Float64 = VAL_TOL
    )
    N = length(samples)
    # set the default relaxation degree
    if relaxdeg <= 0
        deg_A = maximum(maxdegree.(recourse.A))
        deg_C = maximum(maxdegree.(recourse.C))
        deg_b = maximum(maxdegree.(recourse.b))
        deg_Ξ = 0
        if recourse.Ξ.V != FullSpace()
            deg_Ξ = maximum([maxdegree.(recourse.Ξ.p);maxdegree(recourse.Ξ.V.I.p)])
        else
            deg_Ξ = maximum(maxdegree.(recourse.Ξ.p))
        end
        relaxdeg = maximum([deg_A,deg_b,deg_C,deg_Ξ,wassinfo.p])
    end
    # alias the augmented state
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    # parallelise the per-sample SOS solves; pmap preserves the input
    # index order, so cuts[i] is always the cut for samples[i].
    cuts = pmap(
        i -> _gen_moment_cut_linear_recourse(
                i, recourse, samples, wassinfo, x̄, w̄,
                relaxdeg, mom_solver, flag_rad_prod, flag_lin_prod,
                val_add_bound, print, val_relax_tol, N),
        1:N,
    )
    # if any sample's SOS solve failed, propagate the same `nothing` that
    # the original sequential implementation would have returned.
    any(isnothing, cuts) && return nothing
    # return the aggregate cut to the main problem
    return combine_linear_cuts(Vector{Vector{Float64}}(cuts))
end


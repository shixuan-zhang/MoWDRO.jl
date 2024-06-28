# Moment relaxation for the recourse function

# build semidefinite relaxation for the recourse problem
function build_semidefinite_relax(
        sample::Vector{Float64},
        recourse::RecourseData;
        solve::Module = DEFAULT_SDP
    )::RecourseProblem
    # declare the JuMP model
    model = Model(solver.Optimizer)
    # define the pseudo second moment matrix M of ξ and η
    dim_M = 1 + recourse.dim_ξ + recourse.dim_η
    M = @variable(model, [1:dim_M,1:dim_M], PSD, base_name="M")
    # fix the degree 0 (constant) part of M
    fix_value(M[1,1], 1.0)
    # alias the degree 1 part of M to the variables ξ and η
    ξ = M[1,2:(recourse.dim_ξ+1)]
    η = M[1,(recourse_ξ+2):end]
    # add bounds on ξ and η
    for i in 1:recourse.dim_ξ
        if recourse.min_ξ[i] > -Inf
            set_lower_bound(ξ[i], recourse.min_ξ[i])
        end
        if recourse.max_ξ[i] < Inf
            set_upper_bound(ξ[i], recourse.max_ξ[i])
        end
    end
    for i in 1:recourse.dim_η
        if recourse.min_η[i] > -Inf
            set_lower_bound(η[i], recourse.min_η[i])
        end
        if recourse.max_η[i] < Inf
            set_upper_bound(η[i], recourse.max_η[i])
        end
    end
    # impose other constraints on the pseudo moment M
    for G in recourse.G
        @constraint(model, M*G >= 0)
    end
    for H in recourse.H
        @constraint(model, M*H == 0)
    end
    # obtain the objective expressions Ψ(M)
    Ψ = AffExpr[]
    for F in recourse.F
        push!(Ψ, M*F)
    end
    # add the augmented term -|ξ - ξ̄|² = -|ξ|² + 2ξ'⋅ξ̄ - |ξ̄|²
    norm2_ξ = sum([M[i,i] for i in 2:(recourse.dim_η+1)])
    push!(Ψ, -norm2_ξ + 2*ξ'*sample - sample'*sample)
    return RecourseProblem(model, Ψ)
end










# build linear relaxation for the recourse problem
function build_linear_relax(
        recourse::RecourseData;
        solver::Module = DEFAULT_LP
    )::RecourseProblem
    # declare the JuMP model
    model = Model(solver.Optimizer)
    # define the translated and scaled variables
    # ξ = min_ξ + (max_ξ-min_ξ)⋅z, z ∈ [0,1], if it is bounded
    # ξ = min_ξ + z, or ξ = max_ξ - z, u ≥ 0, if only lower- or upper-bounded
    u = @variable(model, [1:recourse.dim_ξ], base_name="z")
    # define the matrix variable W corresponding to second moments Ξ of ξ
    Z = @variable(model, [1:recourse.dim_ξ,1:recourse.dim_ξ], Symmetric, base_name="Z")
    # define a vector indicating whether and how each component of ξ is bounded
    b = zeros(Int, recourse.dim_ξ)
    # impose the bounds on z and set recovery expressions for ξ
    expr_ξ = Vector{AffExpr}(undef, recourse.dim_ξ)
    for i in 1:recourse.dim_ξ
        if recourse.min_ξ[i] > -Inf && recourse.max_ξ[i] < Inf
            b[i] = 0 # bounded below and above
            set_lower_bound(z[i], 0.0)
            set_upper_bound(z[i], 1.0)
            expr_ξ[i] = recourse.min_ξ[i]*(1-z[i]) + recourse.max_ξ*z[i]
        elseif recourse.min_ξ[i] > -Inf
            b[i] = 1
            expr_ξ[i] = recourse.min_ξ[i] + z[i]
            set_lower_bound(u[i], 0.0)
        elseif recourse.max_ξ[i] < Inf
            b[i] = -1
            expr_ξ[i] = recourse.max_ξ[i] - z[i]
        else # no upper or lower bounds
            # FIXME: choose how to represent this case
        end
    end
    # impose McCormick constraints on Z and set recovery expressions for Ξ
    expr_Ξ = Matrix{AffExpr}(undef, recourse.dim_ξ, recourse.dim_ξ)
    # TODO: complete the function
end

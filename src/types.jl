## Data structures for single-stage and two-stage Wasserstein 
## Distributionally Robust Optimization (WDRO) problems

# structure for linear cuts to add to the master (first-stage) problem
# which are affine linear functions of the form a'⋅x + b
struct LinearCut
    # coefficient vector of the subgradient
    a::Vector{Float64}
    # constant term in the linear cut
    b::Float64
end

# structure for the master (first-stage) problem model
struct MasterProblem
    # master problem JuMP model
    model::Model
    # state, control, and Wasserstein auxiliary variable references
    x::Vector{VariableRef}
    y::Vector{VariableRef}
    w::VariableRef
    # variable reference for recourse/loss function
    ϕ::VariableRef
    # linear objective coefficient vectors
    f_x::Vector{Float64}
    f_y::Vector{Float64}
end

# structure for the master (first-stage) solution
struct MasterSolution
    # master problem state and control solutions
    x::Vector{Float64}
    y::Vector{Float64}
    # master problem objective and recourse value
    f::Float64
    ϕ::Float64
end


# structure for (second-stage) recourse/loss functions of the form:
# g(x,ξ) := max  (1,x)'⋅[(1,ξ,η)'⋅F₀⋅(1,ξ,η),…,(1,ξ,η)'⋅Fₙ⋅(1,ξ,η)]
#           s.t. (1,ξ,η)'⋅Gⱼ⋅(1,ξ,η) ≥ 0, ∀ j,
#                (1,ξ,η)'⋅Hₖ⋅(1,ξ,η) = 0, ∀ k,
#                ξ₋ ≤ ξ ≤ ξ₊, η₋ ≤ η ≤ η₊ (possibly ±∞),
# where each Fᵢ,Gⱼ,Hₖ are (1+m+l)×(1+m+l) real symmetric matrices,
# and x ∈ Rⁿ (master state), ξ ∈ Rᵐ (uncertainty), and η ∈ Rˡ (recourse).
struct RecourseData
    # dimensions of the variables
    dim_x::Int
    dim_ξ::Int
    dim_η::Int
    # array of objective and constraint matrices
    F::Vector{Matrix{Float64}}
    G::Vector{Matrix{Float64}}
    H::Vector{Matrix{Float64}}
    # bounds for the variables
    min_ξ::Vector{Float64}
    max_ξ::Vector{Float64}
end

# structure for the Wasserstein (second-stage) recourse optimization model
# max  (1,x)'⋅(Ψ₀(M),Ψ₁(M),…,Ψₙ(M)) - w⋅Ψₙ₊₁(M)
# s.t. Gⱼ⋅M ≥ 0, ∀ j,
#      Hₖ⋅M = 0, ∀ k,
#      (ξ,η,M) ∈ (Relaxed) Second Moment Set.
# Here, ξ̄ is the given sample of ξ, Ψ₀,…,Ψₙ,Ψₙ₊₁ are affine expressions in M, 
# where Ψₙ₊₁(M) correspond to the 2-norm term |ξ - ξ̄|² in Wasserstein DRO.
struct RecourseProblem
    # recourse problem JuMP model
    model::Model
    # objective coefficients of the augmented state vector (1,x,w)
    Ψ::Vector{AffExpr}
end


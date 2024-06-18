## Data structures for single-stage and two-stage Wasserstein 
## Distributionally Robust Optimization (WDRO) problems

# structure for linear cuts to add to the master (first-stage) problem
struct LinearCut
    # coefficient vector of the subgradient
    vec::Vector{Float64}
    # constant term in the linear cut
    val::Float64
end

# structure for the master (first-stage) problem model
struct MasterProblem
    # master problem JuMP model
    model::JuMP.Model
    # state and control variable references
    var_stat::Vector{JuMP.VariableRef}
    var_ctrl::Vector{JuMP.VariableRef}
    # variable reference for recourse/loss function
    var_loss::JuMP.VariableRef
    # linear objective coefficient vectors
    obj_stat::Vector{Float64}
    obj_ctrl::Vector{Float64}
end

# structure for the master (first-stage) solution
struct MasterSolution
    # master problem state and control solutions
    sol_stat::Vector{Float64}
    sol_ctrl::Vector{Float64}
    # master problem objective and recourse value
    val_obj::Float64
    val_rec::Float64
end


# structure for (second-stage) recourse/loss functions of the form:
# g(x,ξ) := min  (1,x)'⋅[(1,ξ,η)'⋅F₀⋅(1,ξ,η),…,(1,ξ,η)'⋅Fₙ⋅(1,ξ,η)]
#           s.t. (1,ξ,η)'⋅Gⱼ⋅(1,ξ,η) ≥ 0, ∀ j,
#                (1,ξ,η)'⋅Hₖ⋅(1,ξ,η) = 0, ∀ k,
#                ξ₋ ≤ ξ ≤ ξ₊, η₋ ≤ η ≤ η₊,
# where each Fᵢ,Gⱼ,Hₖ are (1+m+l)×(1+m+l) real symmetric matrices,
# and x ∈ Rⁿ (state), ξ ∈ Rᵐ (uncertainty), η ∈ Rˡ (auxiliary/recourse)
struct RecourseData
    # dimensions of the variables
    dim_x::Int
    dim_ξ::Int
    dim_η::Int
    # array of objective and constraint matrices
    mat_obj::Vector{Matrix{Float64}}
    mat_con_i::Vector{Matrix{Float64}}
    mat_con_e::Vector{Matrix{Float64}}
    # bounds for the variables
    val_min_ξ::Vector{Float64}
    val_max_ξ::Vector{Float64}
    val_min_η::Vector{Float64}
    val_max_η::Vector{Float64}
end


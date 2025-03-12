## Data structures for single-stage and two-stage Wasserstein 
## Distributionally Robust Optimization (WDRO) problems

# structure for the main (first-stage) problem model
struct MainProblem
    # main problem JuMP model
    model::Model
    # state, control, and Wasserstein auxiliary variable references
    x::Vector{VariableRef}
    u::Vector{VariableRef}
    w::VariableRef
    # variable reference for recourse/loss function
    ϕ::VariableRef
    # linear objective coefficient vectors
    f_x::Vector{Float64}
    f_u::Vector{Float64}
end

# structure for the main (first-stage) solution
struct MainSolution
    # main problem state and control solutions
    x::Vector{Float64}
    u::Vector{Float64}
    # main problem objective and recourse value
    f::Float64
    ϕ::Float64
end

# structure for Wasserstein robustness info
struct WassInfo
    # Wasserstein radius 
    r::Float64
    # norm choice (p-norm)
    p::Int
end


# abstract type of sample subproblems 
abstract type SampleSubproblem end

# structure for (second-stage) recourse linear optimization subproblem
# F(x,ξ) := max  (1,x₁,…,xₙ)ᵀ⋅C(ξ)⋅(1,y₁,…,yₘ)
#           s.t. A(ξ)y-b(ξ) ≥ 0,
# ξ ∈ Ξ.
# Here, ξ is the uncertainty vector, y is the recourse decision,
# A, b, and C are vectors and matrices with polynomial entries in ξ.
struct SampleLinearRecourse <: SampleSubproblem
    # PolyJuMP/DynamicPolynomials (Symbolic) Variables
    x::Vector
    ξ::Vector
    y::Vector
    # Polynomial Recourse Coefficients
    C::Matrix 
    A::Matrix
    b::Vector
    # Semi-algebraic Uncertainty Set
    Ξ::BasicSemialgebraicSet
    # Recourse Variable Bounds
    B::Vector
end

# structure for (first-stage) polynomial loss function F(x,ξ)
struct SamplePolynomialLoss <: SampleSubproblem 
    # PolyJuMP/DynamicPolynomials (Symbolic) Variables
    x::Vector
    ξ::Vector
    # Polynomial Loss Function and Its Gradient in x
    F::Polynomial
    ∇ₓF::Vector
    # Semi-algebraic Uncertainty Set
    Ξ::BasicSemialgebraicSet
end

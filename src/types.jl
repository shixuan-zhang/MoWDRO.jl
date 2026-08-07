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
#           s.t. A(ξ)y-b(ξ) ≥ 0, |y| ≤ B,
# ξ ∈ Ξ.
# Here, ξ is the uncertainty vector, y is the recourse decision,
# A, b, and C are vectors and matrices with polynomial entries in ξ,
# B is the bound vector on the absolute values of y (negative means ∞).
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
    # Recourse Variable Absolute Bounds
    B::Vector
end

# structure for (first-stage) polynomial loss function
#   F(x,ξ) := max_{k=1,…,K} F[k](x,ξ),
# where each F[k] is a polynomial in (x,ξ). A plain polynomial loss is the
# special case K = 1, handled transparently by the outer constructor below.
struct SamplePolynomialLoss <: SampleSubproblem
    # PolyJuMP/DynamicPolynomials (Symbolic) Variables
    x::Vector
    ξ::Vector
    # Polynomial Loss Functions and Their Gradients in x
    # F[k] is a Polynomial in (x,ξ); ∇ₓF[k] is the Vector of ∂F[k]/∂x_i.
    F::Vector
    ∇ₓF::Vector
    # Semi-algebraic Uncertainty Set
    Ξ::BasicSemialgebraicSet
end

# backward-compat outer constructor for the single-polynomial case
SamplePolynomialLoss(x::Vector, ξ::Vector, F::Polynomial, ∇ₓF::Vector,
                     Ξ::BasicSemialgebraicSet) =
    SamplePolynomialLoss(x, ξ, [F], [∇ₓF], Ξ)

# numerical example for a portfolio management problem where
# F(x,ξ) := C₁(ξᵀx) + C₂(ξᵀx)² + ⋯ + Cₖ(ξᵀx)ᵏ, 
# x ∈ [0,1]ⁿ, ∑ᵢxᵢ = 1, ξ = Proj(D⋅η,[0,1]ⁿ), 
# η ∼ Uniform(0,1)ᵐ, D ∈ Mat(n,m) with normalized columns,
# and C₂,…,Cₖ are nonnegative (C₁ may be negative) so F is convex in x for any ξ.


using JuMP, HiGHS
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_SAMPLE = 100
const WASS_INFO = [WassInfo(1e-2,2),
                   WassInfo(1e-1,2),
                   WassInfo(1.0,2)]
const LOSS_COEF = [-1.0,1.0]

# function that conducts the experiment on the portfolio examples
function experiment_portfolio(
        n::Int,             # number of decisions
        m::Int,             # number of factors
        C::Vector{Float64}, # loss function coefficients
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used
        N::Int = NUM_SAMPLE; # number of samples to take
        D::Matrix{Float64} = zeros(0,0), # dependence matrix in the factor model
        f_x::Vector{Float64} = zeros(0)
    )
    # check if the orthogonal matrix is supplied
    if size(D) != (n,m)
        D = rand(n,m)
        # normalize the columns
        for i = 1:m
            D[:,i] ./= norm(D[:,i])
        end
    end
    # take the samples of the uncertainty
    samples = map(η->min.(max.(D*η,0),1), [rand(m) for _ in 1:N])
    # randomly generate the linear objective function if not supplied
    if length(f_x) != n
        f_x = rand(n)
    end
    # save the degree of the loss function
    k = length(C)
    # define the loss function
    @polyvar x[1:n] ξ[1:n]
    F = sum(C[i]*(x'*ξ)^i for i in 1:k)
    ∇ₓF = differentiate(F,x)
    Ξ = basic_semialgebraic_set(FullSpace(), [[ξ[i] for i in 1:n]; [1-ξ[i] for i in 1:n]])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    # print the problem information
    println("Start the experiment on the portfolio management problem...")
    println("The number of decisions is ", n)
    println("The number of factors is ", m)
    println("The number of samples is ", N)
    println("The loss function is ", F)
    println("The static cost function is ", f_x'*x)
    # loop over all Wasserstein robustness settings
    for wassinfo in W
        # define the main linear optimization problem 
        model = Model(HiGHS.Optimizer)
        set_silent(model)
        x = @variable(model, 0 <= x[1:n] <= 1, base_name="x")
        w = @variable(model, w >= 0, base_name="w")
        ϕ = @variable(model, ϕ >= 0, base_name="ϕ")
        # add the linear constraint
        @constraint(model, ones(n)'*x == 1)
        main = MainProblem(model, x, VariableRef[], w, ϕ, f_x, Float64[])
        # solve the problem
        sol = solve_main_level(main, loss, samples, wassinfo, print=true)
        println("The master problem is solved successfully!")
        println("x = ", sol.x)
        println("f = ", sol.f)
        println("ϕ = ", sol.ϕ)
    end
end


# run the experiment
experiment_portfolio(2, 3, LOSS_COEF, WASS_INFO, 5)

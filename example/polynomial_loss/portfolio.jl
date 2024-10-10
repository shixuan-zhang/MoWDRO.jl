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
const NUM_VAR = 2
const NUM_FAC = 3
const DEG_LOSS = 2
const NUM_SAMPLE = 5
const WASS_INFO = [WassInfo(0.0,2),
                   WassInfo(1e-2,2),
                   WassInfo(2e-2,2),
                   WassInfo(5e-2,2),
                   WassInfo(1e-1,2),
                   WassInfo(2e-1,2),
                   WassInfo(5e-1,2),
                   WassInfo(1.0,2)]


# function that conducts the experiment on the portfolio examples
function experiment_portfolio(
        n::Int,             # number of decisions
        m::Int,             # number of factors
        k::Int,             # degree of the loss function
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used
        N::Int = NUM_SAMPLE; # number of samples to take
        C::Vector{Float64} = zeros(0), # loss function coefficients
        D::Matrix{Float64} = zeros(0,0), # dependence matrix in the factor model
        f_x::Vector{Float64} = zeros(0)
    )
    # check if the loss function coefficients are supplied
    if length(C) != k
        C = rand(k)
    end
    C[1] = max(-0.1, -0.5*C[2]) # TODO: switch to a more reasonable choice
    # check if the orthogonal matrix is supplied
    if size(D) != (n,m)
        D = rand(n,m) .* 2 .- 1
        # normalize the columns
        for i = 1:m
            D[:,i] ./= norm(D[:,i])
        end
    end
    # take the samples of the uncertainty
    samples = map(η->min.(max.(D*η,0),1), [rand(m) for _ in 1:N])
    # randomly generate the linear objective function if not supplied
    if length(f_x) != n
        f_x = rand(n) .* 2 .- 1
    end
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
    println()
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
        println("The main problem is solved successfully for Wasserstein radius = ", wassinfo.r)
        println("x = ", sol.x)
        println("f = ", sol.f)
        println("ϕ = ", sol.ϕ)
        println()
    end
end


# run the experiment
experiment_portfolio(NUM_VAR, NUM_FAC, DEG_LOSS, WASS_INFO)

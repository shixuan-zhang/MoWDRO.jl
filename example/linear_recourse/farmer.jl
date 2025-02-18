# numerical example for a two-stage farmer crop planning problem 
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,1]ⁿ, ∑ᵢ xᵢ = 1, where fₓ ∈ [0,1]ⁿ, and 
# F(x,ξ) := min  -∑ᵢ pᵢ⋅zᵢ
#           s.t. zᵢ ≤ ξᵢxᵢ, ∀ i,
#                rⱼᵀ z ≤ sⱼ,   ∀ j,
#                zᵢ ≥ 0,       ∀ i.
# Here, pᵢ ≥ 0 are the selling prices for the crops,
# ξᵢ ∼ Uniform(0,qᵢ) is the random crop growing factors,
# rⱼ and sⱼ are additional technical (e.g., storage) constraints.
# By linear duality, we can write F alternatively as
# F(x,ξ) = max  -(1,x)ᵀ⋅[0 sᵀ; 0 diag(q)]⋅(1,y)
#          s.t. [I Rᵀ; I] y ≥ [p; 0].


using JuMP, HiGHS
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_CROP = 2
const NUM_TECH = 1 
const MAX_CROP_YIELD = [20.0,30.0]
const SELL_PRICE = [15.0,18.0]
const COST_SEED = [3.0,4.0]
const TECH_CONS = [0.5 0.6] # weighted sum <= 1.0
const NUM_SAMPLE = 20 
const MIN_AUX = 1.0e-2
const DEG_WASS = 4
const WASS_INFO = [WassInfo(0.0,DEG_WASS),
                   WassInfo(1.0e-4,DEG_WASS),
                   WassInfo(1.0e-3,DEG_WASS),
                   WassInfo(1.0e-2,DEG_WASS),
                   WassInfo(1.0e-1,DEG_WASS),
                   WassInfo(1.0e0,DEG_WASS)]

# function that conducts experiments on the farmer's planning problem
function experiment_farmer(
        n::Int,              # number of crops
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used 
        N::Int = NUM_SAMPLE; # number of samples
        p::Vector{Float64} = SELL_PRICE,     # vector of crop selling prices
        q::Vector{Float64} = MAX_CROP_YIELD, # vector of crop growing factors
        R::Matrix{Float64} = TECH_CONS,      # matrix of technical constraints
        s::Vector{Float64} = ones(NUM_TECH), # vector of technical constraints
        f_x::Vector{Float64} = COST_SEED,    # vector of part costs
    )
    # take the samples of salvage prices and demands
    samples = [rand(n) .* q for _ in 1:N]
    # alias the number of technical constraints
    m = NUM_TECH
    # define the two-stage linear recourse function
    @polyvar x[1:n] ξ[1:n] y[1:n+m]
    C = -[0 zeros(n)'; zeros(n) diagm(ξ); s zeros(m,n)]'
    A = [I R'; I] .+ 0.0*sum(ξ) # to promote the type
    b = [p; zeros(n+m)]
    Ξ = basic_semialgebraic_set(FullSpace(), 
                                [[ξ[i] for i in 1:n];
                                 [q[i]-ξ[i] for i in 1:n]
                                ])
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ)
    # print the problem information
    println("Start the experiment on the two-stage farmer's planning problem...")
    println("The number of crops is ", n)
    println("The number of samples is ", N)
    println("The first-stage cost function is ", f_x'*x)
    println("The second-stage cost function is ", [1;x]'*C*[1;y])
    println("The second-stage constraints are ", A*y - b)
    println()
    # loop over all Wasserstein robustness settings
    for wassinfo in W
        # define the main linear optimization problem 
        model = Model(HiGHS.Optimizer)
        set_silent(model)
        x = @variable(model, 0 <= x[1:n] <= 1, base_name="x")
        w = @variable(model, w >= 0, base_name="w")
        ϕ = @variable(model, ϕ, base_name="ϕ")
        main = MainProblem(model, x, VariableRef[], w, ϕ, f_x, Float64[])
        # solve the problem
        sol = solve_main_level(main, recourse, samples, wassinfo, min_aux=MIN_AUX, print=true)
        println("The main problem is solved successfully for Wasserstein radius = ", wassinfo.r)
        println("x = ", sol.x)
        println("f = ", sol.f)
        println("ϕ = ", sol.ϕ)
        println()
    end
end

# run the experiment
experiment_farmer(NUM_CROP, WASS_INFO)

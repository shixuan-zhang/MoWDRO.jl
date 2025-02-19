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


using TSSOS
using JuMP, HiGHS
using LinearAlgebra, DynamicPolynomials
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
    @polyvar ξ[1:n] y[1:n+m]
    C = -[0 zeros(n)'; zeros(n) diagm(ξ); s zeros(m,n)]'
    A = [I R'; I] .+ 0.0*sum(ξ) # to promote the type
    b = [p; zeros(n+m)]
    D = [[ξ[i] for i in 1:n]; [q[i]-ξ[i] for i in 1:n]]
    x̄ = zeros(n)
    w̄ = 100.0
    relaxdeg=4
    # print the problem information
    println("Start the experiment on the two-stage farmer's planning problem...")
    println("The number of crops is ", n)
    println("The number of samples is ", N)
    println()
    # loop over all Wasserstein robustness settings
    for wassinfo in W
        # loop over samples to see if the moment relaxations can be solved successfully
        for i = 1:N # TODO: parallelize this for-loop
            ξ̂ = samples[i]
            # define the polynomial objective 
            π = ((ξ-ξ̂)'*(ξ-ξ̂))^2
            f = [1;x̄]'*C*[1;y] - w̄*π
            # define the semi-algebraic set
            S = [-f; D; A*y-b]
            m = 0
            R2 = (wassinfo.r^wassinfo.p*N)^(2/wassinfo.p)
            p2 = (ξ-ξ̂)'*(ξ-ξ̂)
            k = length(b)
            for j = 1:k
                append!(S, [(A*y-b)[j]*(R2-p2)])
            end
            B = 1.0e2
            for j = 1:n
                append!(S, [B+y[j]])
                append!(S, [B-y[j]])
            end
            for l = 1:div(wassinfo.p,2,RoundDown)
                append!(S, [B^(2*l)-y[j]^(2*l) for j in 1:n])
            end
            # check the moment relaxation model
            println("The moment relaxation is defined by polynomial inequalities:")
            for s in S
                println(s)
            end
            # solve the moment relaxation
            opt,sol,data = tssos_first(S, [y; ξ], relaxdeg, numeq=m)
            println("The moment relaxation for the recourse problem is solved with optimal value = ", opt)
        end
    end
end

# run the experiment
experiment_farmer(NUM_CROP, WASS_INFO)

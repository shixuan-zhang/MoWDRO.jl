# numerical example for a two-stage inventory (newsvendor) problem
# (see Chapter 1.2.1 in Shapiro-Dentcheva-Ruszczyński(2009)), defined by
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,1]ⁿ, where fₓ = c ∈ [0,1]ⁿ, and 
# F(x,ξ) := min  ∑ᵢ zᵢ
#           s.t. zᵢ ≥ bᵢ(dᵢ - xᵢ), ∀ i,
#                zᵢ ≥ hᵢ(xᵢ - dᵢ), ∀ i,
# where bᵢ ∼ Unif(cᵢ,B) is the backorder cost,
# hᵢ ∼ Unif(0,H) is the holding cost,
# dᵢ ∼ Unif(0,D) is the demand, and 
# ξ = (bᵢ,hᵢ,dᵢ)ᵢ includes all the uncertainties.
# By linear program duality, we can rewrite F as
# F(x,ξ) = max  ∑ᵢ bᵢ(dᵢ-xᵢ)sᵢ + hᵢ(xᵢ-dᵢ)tᵢ
#          s.t. sᵢ + tᵢ = 1, ∀ i,
#               sᵢ, tᵢ ≥ 0, ∀ i,
# or using the notation y = (s₁,…,sₙ,t₁,…,tₙ)
# F(x,ξ) = max  (1,x)ᵀ⋅[0 [bd;-hd]ᵀ; 0 (diag(-b),diag(h))]⋅(1,y)
#          s.t. [(eᵢ+eᵢ₊ₙ)ᵢ; -(eᵢ+eᵢ₊ₙ)ᵢ; I]⋅y ≥ [1;…;1;-1;…,-1;0;…;0]


using JuMP, HiGHS
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_PROD = 1 
const DEMAND_MAX = 10.0
const BACKLOG_MAX = 2.0
const HOLDING_MAX = 0.5
const NUM_SAMPLE = 20
const DEG_WASS = 2
const WASS_INFO = [WassInfo(0.0,DEG_WASS),
                   WassInfo(1.0e-2,DEG_WASS),
                   WassInfo(1.0e-1,DEG_WASS),
                   WassInfo(1.0,DEG_WASS)]

# function that conducts experiments on the inventory (newsvendor) problem
function experiment_newsvendor(
        n::Int,              # number of parts
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used 
        N::Int = NUM_SAMPLE; # number of samples
        f_x::Vector{Float64} = zeros(0), # vector of purchase costs
        D::Float64 = DEMAND_MAX,         # bound on the demands
        H::Float64 = HOLDING_MAX,        # bound on the holding costs
        B::Float64 = BACKLOG_MAX,        # bound on the backlogging costs
        flag_random::Bool = false
    )
    # randomly generate the linear objective function if not supplied
    if length(f_x) != n
        f_x = ones(n) ./ collect(2:n+1)
        if flag_random
            f_x = rand(n)
        end
    end
    # take the samples of backlogging costs, holding costs, and demands
    samples = [[rand(n).*(B .- f_x) .+ f_x;
                rand(n).*H;
                rand(n).*D] for _ in 1:N]
    # define the two-stage linear recourse function
    @polyvar x[1:n] ξ[1:3*n] y[1:2*n]
    C = [0        [ξ[1:n].*ξ[2*n+1:3*n];-ξ[n+1:2*n].*ξ[2*n+1:3*n]]'; 
         zeros(n) hcat(diagm(-ξ[1:n]), diagm(ξ[n+1:2*n]))          ]
    A = [zeros(2*n,2*n); I] .+ 0.0*sum(ξ) # to promote type
    for i in 1:n
        A[i,i] += 1
        A[i,i+n] += 1
        A[n+i,i] -= 1
        A[n+i,n+i] -= 1
    end
    b = [ones(n); -ones(n); zeros(2*n)] .+ 0.0*sum(ξ) # to promote type
    Ξ = basic_semialgebraic_set(FullSpace(), 
                                [[ξ[i] - f_x[i] for i in 1:n];
                                 ξ[n+1:3*n];
                                 [B-ξ[i] for i in 1:n];
                                 [H-ξ[i] for i in n+1:2*n];
                                 [D-ξ[i] for i in 2*n+1:3*n]
                                ])
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ)
    # print the problem information
    println("Start the experiment on the two-stage newsvendor problem...")
    println("The number of products is ", n)
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
        sol = solve_main_level(main, recourse, samples, wassinfo, print=true)
        println("The main problem is solved successfully for Wasserstein radius = ", wassinfo.r)
        println("x = ", sol.x)
        println("f = ", sol.f)
        println("ϕ = ", sol.ϕ)
        println()
    end
end

# run the experiment
experiment_newsvendor(NUM_PROD, WASS_INFO)

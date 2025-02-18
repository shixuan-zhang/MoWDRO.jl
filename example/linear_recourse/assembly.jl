# numerical example for a two-stage multiproduct assembly problem 
# (see Chapter 1.3.1 in Shapiro-Dentcheva-Ruszczyński(2009)), defined by
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,1]ⁿ, where fₓ ∈ [0,1]ⁿ, and 
# F(x,ξ) := min  -∑ᵢ rᵢ⋅zᵢ - ∑ⱼ sᵢ(ξ)⋅wⱼ
#           s.t. wⱼ = xⱼ - pⱼᵀz, ∀ j,
#                0 ≤ zᵢ ≤ qᵢ(ξ), ∀ i,
#                wⱼ ≥ 0,         ∀ j.
# Here, rᵢ > 0 is the regular price, 
# sᵢ(ξ) ∼ Uniform(0,S) is the random salvage price,
# pⱼ ≥ 0 are the assembly coefficients, P = [p₁,…,pₘ], 
# qᵢ(ξ) ∼ Uniform(0,D) is the random demand with an 
# upper bound D > 0.
# By linear duality, we can write F alternatively as
# F(x,ξ) = max  -(xᵀ, q(ξ)ᵀ)⋅y 
#               = (1,x)ᵀ⋅[0  0 -q(ξ)ᵀ; 0 -I 0]⋅(1,y)
#          s.t. [I 0; Pᵀ I; 0 I] y ≥ [s(ξ); r; 0]


using JuMP, HiGHS
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_PART = 3
const NUM_PROD = 2 
const REG_PRICE_MIN = 50.0
const REG_PRICE_MAX = 200.0
const PART_COST_MAX = 5.0
const PART_COST_MIN = 2.0
const SALVAGE_MAX = 2.0
const DEMAND_MAX = 1.0
const NUM_SAMPLE = 20 
const DEG_WASS = 4
const WASS_INFO = [WassInfo(0.0,DEG_WASS),
                   WassInfo(1.0e-2,DEG_WASS),
                   WassInfo(1.0e-1,DEG_WASS),
                   WassInfo(1.0,DEG_WASS)]

# function that conducts experiments on the multiproduct assembly problem
function experiment_assembly(
        n::Int,              # number of parts
        m::Int,              # number of products
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used 
        N::Int = NUM_SAMPLE; # number of samples
        P::Matrix{Float64} = zeros(0,0), # matrix of assembly coefficients 
        r::Vector{Float64} = zeros(0),   # vector of regular prices
        f_x::Vector{Float64} = zeros(0), # vector of part costs
        D::Float64 = DEMAND_MAX,         # bound on the demands
        S::Float64 = SALVAGE_MAX,        # bound on the salvage prices
        flag_random::Bool = false
    )
    # check if the assembly coefficients are supplied
    if size(P) != (n,m)
        P = ones(n,m) ./ n
        if flag_random
            P = rand(n,m) .+ 0.1
            for i = 1:m
                P[i,:] ./= sum(P[i,:])
            end
        end
    end
    # check if the regular prices are supplied
    if length(r) != m
        v = ones(m) ./ collect(2:m+1)
        if flag_random
            v = rand(m)
        end
        r = v .* (REG_PRICE_MAX-REG_PRICE_MIN) .+ REG_PRICE_MIN
    end
    # randomly generate the linear objective function if not supplied
    if length(f_x) != n
        v = ones(n) ./ collect(2:n+1)
        if flag_random
            v = rand(n)
        end
        f_x = v .* (PART_COST_MAX-PART_COST_MIN) .+ PART_COST_MIN
    end
    # take the samples of salvage prices and demands
    samples = [[rand(m)*D;rand(n)*S] for _ in 1:N]
    # define the two-stage linear recourse function
    # ξᵢ represents qᵢ, i = 1,…,m, and ξⱼ₊ₘ represents sⱼ, j = 1,…,n
    @polyvar x[1:n] ξ[1:m+n] y[1:n+m]
    C = [zeros(n+1)' -ξ[1:m]'; zeros(n) -I zeros(n,m)]
    A = [I zeros(n,m); P' I; zeros(m,n) I] .+ 0.0*sum(ξ) # to promote the type
    b = [ξ[m+1:m+n]; r; zeros(m)]
    Ξ = basic_semialgebraic_set(FullSpace(), 
                                [[ξ[i] for i in 1:m+n];
                                 [D-ξ[i] for i in 1:m];
                                 [S-ξ[i+m] for i in 1:n]
                                ])
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ)
    # print the problem information
    println("Start the experiment on the two-stage product assembly problem...")
    println("The number of assembly parts is ", n)
    println("The number of products is ", m)
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
experiment_assembly(NUM_PART, NUM_PROD, WASS_INFO)

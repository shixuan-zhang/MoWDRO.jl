# numerical example for a two-stage multiproduct assembly problem 
# (see Chapter 1.3.1 in Shapiro-Dentcheva-Ruszczyński(2009)), defined by
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,1]ⁿ, where fₓ ∈ [0,1]ⁿ, and 
# F(x,ξ) := min  -∑ᵢ rᵢ⋅zᵢ - ∑ⱼ sᵢ(ξ)⋅wⱼ + ∑ⱼ gⱼ⋅uⱼ
#           s.t. wⱼ - uⱼ = xⱼ - ∑ᵢ pᵢⱼ⋅zᵢ, ∀ j,
#                0 ≤ zᵢ ≤ qᵢ(ξ),           ∀ i,
#                wⱼ, uⱼ ≥ 0,               ∀ j.
# Here, rᵢ > 0 is the regular price, 
# sᵢ is the salvage price,
# gⱼ is the late price for part purchasing,
# pᵢⱼ is the percentage of part j in product i,
# qᵢ(ξ) ∼ Uniform(0,Dᵢ) is the random demand with an 
# upper bound Dᵢ > 0.
# By linear duality, we can write F alternatively as
# F(x,ξ) = max  -(xᵀ, q(ξ)ᵀ)⋅y 
#               = (1,x)ᵀ⋅[0  0 -q(ξ)ᵀ; 0 -I 0]⋅(1,y)
#          s.t. [I 0; -I 0; Pᵀ I; 0 -I; 0 I] y ≥ [s; -g; r; -r; 0]


using JuMP, HiGHS
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_PART = 5
const NUM_PROD = 5
const PRICE_MIN = 2.0
const PRICE_MAX = 8.0
const COST_MAX = 0.8
const COST_MIN = 0.2
const LATE_RATIO = 2.0
const SALVAGE_RATIO = 0.8
const DEMAND_MAX = 2.0
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
        r::Vector{Float64} = zeros(0),   # vector of regular product prices
        f_x::Vector{Float64} = zeros(0), # vector of part costs
        g::Vector{Float64} = zeros(0),   # vector of late part costs
        s::Vector{Float64} = zeros(0),   # vector of part salvage prices
        D::Float64 = DEMAND_MAX,         # bound on the demands
    )
    # check if the assembly coefficients are supplied
    if size(P) != (n,m)
        P = zeros(n,m)
        for j = 1:m
            for i = j:n
                P[i,j] = 1/(n+1-j)
            end
        end
    end
    # check if the regular prices are supplied
    if length(r) != m
        r = zeros(m)
        for j = 1:m
            r[j] = PRICE_MIN + (PRICE_MAX-PRICE_MIN)*(j-1)/(m-1)
        end
    end
    # check if the part prices are supplied
    if length(f_x) != n
        f_x = zeros(n)
        for i = 1:n
            f_x[i] = COST_MIN + (COST_MAX-COST_MIN)*(i-1)/(n-1)
        end
    end
    if length(g) != n
        g = LATE_RATIO * f_x
    end
    if length(s) != n
        s = SALVAGE_RATIO * f_x
    end
    # take the samples of salvage prices and demands
    samples = [rand(m) for _ in 1:N]
    # define the two-stage linear recourse function, where ξᵢ represents qᵢ, i = 1,…,m
    @polyvar x[1:n] ξ[1:m] y[1:n+m]
    C = [zeros(n+1)' -D*ξ'; zeros(n) -I zeros(n,m)]
    A = [I zeros(n,m); -I zeros(n,m); P' I; zeros(m,n) -I; zeros(m,n) I] .+ 0.0*sum(ξ) # to promote the type
    b = [s; -g; r; -2*r; zeros(m)] .+ 0.0*sum(ξ) # to promote the type
    Ξ = basic_semialgebraic_set(FullSpace(), 
                                [[ξ[i] for i in 1:m];
                                 [1-ξ[i] for i in 1:m]
                                ])
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ)
    # print the problem information
    println("Start the experiment on the two-stage product assembly problem...")
    println("The number of assembly parts is ", n)
    println("The number of products is ", m)
    println("The assembly coefficient matrix is\n", P)
    println("The regular prices are ", r)
    println("The salvage prices are ", s)
    println("The late purchase prices are ", g)
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
        time_start = time()
        sol = solve_main_level(main, recourse, samples, wassinfo, print=true)
        println("The main problem is solved successfully for Wasserstein radius = ", wassinfo.r)
        println("x = ", sol.x)
        println("f = ", sol.f)
        println("ϕ = ", sol.ϕ)
        println("The total computation time is ", time()-time_start)
        println()
    end
end

# run the experiment
experiment_assembly(NUM_PART, NUM_PROD, WASS_INFO)

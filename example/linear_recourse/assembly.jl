# numerical example for a two-stage multiproduct assembly problem 
# (adapted from Chapter 1.3.1 in Shapiro-Dentcheva-Ruszczyński(2009)):
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


using JuMP, HiGHS, Mosek, MosekTools
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets, Statistics
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_PART = 20 
const NUM_PROD = 20
const PRICE_MIN = 1.0
const PRICE_MAX = 5.0
const COST_MAX = 1.0
const COST_MIN = 0.8
const LATE_RATIO = 3.0
const SALV_RATIO = 0.5
const DEMAND_MAX = 2.0
const DEMAND_DIFF = 0.4
const NUM_TRAIN = 10 
const NUM_TEST = 1000
const MOM_SOLVER = Mosek.Optimizer
const DEG_WASS = 2
const WASS_INFO = [WassInfo(i*1.0e-2,DEG_WASS) for i in 0:20]

# function to generate correlated demand data (scaled to be between [0,1])
function generate_data(
        m::Int;                   # number of products
        δ::Float64 = DEMAND_DIFF, # range of adjacent entries
        k::Int = 0,               # starting index for the correlation
    )
    if k <= 0
        k = ceil(Int,m/2)
    end
    d = zeros(m)
    d[k] = rand()
    for i = 1:(m-1)
        i1 = 1+(k+i-2)%m
        i2 = 1+(k+i-1)%m
        lb = 0.0
        ub = 1.0
        if d[i1] >= 0.5
            lb = max(d[i1]-δ, 0.0)
        else
            ub = min(d[i1]+δ, 1.0)
        end
        d[i2] = lb + (ub-lb)*rand()
    end
    return d
end

# function that conducts experiments on the multiproduct assembly problem
function experiment_assembly(
        n::Int,              # number of parts
        m::Int,              # number of products
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used 
        N::Int = NUM_TRAIN,  # number of training samples
        M::Int = NUM_TEST;   # number of testing samples
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
            for i =1:(j-1)
                P[i,j] = 1/(j-1) * 0.2
            end
            for i = j:n
                P[i,j] = 1/(n+1-j) * 0.8
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
            f_x[i] = COST_MAX - (COST_MAX-COST_MIN)*(i-1)/(n-1)
        end
    end
    if length(g) != n
        g = LATE_RATIO * f_x
    end
    if length(s) != n
        s = SALV_RATIO * f_x
    end
    # take the samples of salvage prices and demands
    sample_train = [generate_data(m) for _ in 1:N]
    sample_test = [generate_data(m) for _ in 1:M]
    # define the two-stage linear recourse function, where ξᵢ represents qᵢ, i = 1,…,m
    @polyvar x[1:n] ξ[1:m] y[1:n+m]
    C = [zeros(n+1)' -D*ξ'; zeros(n) -I zeros(n,m)]
    A = [I zeros(n,m); -I zeros(n,m); P' I; zeros(m,n) -I; zeros(m,n) I] .+ 0.0*sum(ξ) # to promote the type
    b = [s; -g; r; -r; zeros(m)] .+ 0.0*sum(ξ) # to promote the type
    Ξ = basicsemialgebraicset(FullSpace(), 
                              [[ξ[i] for i in 1:m];
                               [1-ξ[i] for i in 1:m];
                               [ξ[i]*(1-ξ[i]) for i in 1:m]
                              ])
    B = [g; r]
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ, B)
    # print the problem information
    println("Start the experiment on the two-stage product assembly problem...")
    println("The number of assembly parts is ", n)
    println("The number of products is ", m)
    println("The regular prices are ", r)
    println("The salvage prices are ", s)
    println("The late purchase prices are ", g)
    println("The number of training samples is ", N)
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
        sol = solve_main_level(main, recourse, sample_train, wassinfo, print=1, mom_solver=MOM_SOLVER)
        println("The main problem is solved successfully for Wasserstein radius = ", wassinfo.r)
        println("x = ", sol.x)
        println("f = ", sol.f)
        println("ϕ = ", sol.ϕ)
        println("The training sample objective = ", sol.f+sol.ϕ)
        println("The total computation time is ", time()-time_start)
        println("Start the out-of-sample test for the solution...")
        # evaluate the out-of-sample performance
        _, vals = eval_nominal(recourse, sol.x, sample_test, details=true)
        println("The testing sample mean = ", mean(vals)+sol.f)
        println("The testing sample standard deviation = ", std(vals))
        println("\n\n")
    end
end

# run the experiment
experiment_assembly(NUM_PART, NUM_PROD, WASS_INFO)

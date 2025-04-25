# numerical example for a two-stage mixing problem 
# (adapted from Chapter 1.3.1 in Shapiro-Dentcheva-Ruszczyński(2009)):
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,1]ⁿ, where fₓ ∈ [0,1]ⁿ, and 
# F(x,ξ) := min  -∑ᵢ rᵢ⋅zᵢ - ∑ⱼ sᵢ(ξ)⋅wⱼ + ∑ⱼ gⱼ⋅uⱼ
#           s.t. wⱼ - uⱼ = tⱼ(ξ)⋅xⱼ - ∑ᵢ pᵢⱼ⋅zᵢ, ∀ j,
#                0 ≤ zᵢ ≤ qᵢ(ξ),                 ∀ i,
#                wⱼ, uⱼ ≥ 0,                     ∀ j.
# Here, rᵢ > 0 is the product price, 
# gⱼ is the late price for ingredient purchasing,
# pᵢⱼ is the percentage of ingredient j in product i,
# sᵢ(ξ) ∼ Uniform(0,1) is the random salvage price 
# factor for ingredient i,
# tᵢ(ξ) ∼ Uniform(0,1) is the random spoilage 
# percentage for ingredient i, 
# qᵢ(ξ) ∼ Uniform(0,1) is the random demand factor, so 
# that the total demand is qᵢ⋅D for some D > 0.
# We assume an ingredient is either perishable, or has a 
# discounted salvage price.
# By linear duality, we can write F alternatively as
# F(x,ξ) = max  -(t(ξ)⋅xᵀ, q(ξ)ᵀ)⋅y 
#               = (1,x)ᵀ⋅[0 0 -q(ξ)ᵀ; 0 -diag(t(ξ)) 0]⋅(1,y)
#          s.t. [I 0; -I 0; Pᵀ I; 0 -I; 0 I] y ≥ [s(ξ); -g; r; -r; 0]


using JuMP
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets, Statistics
using DataFrames, CSV
# use commercial solvers for efficiency and numerical stability
using Gurobi, Mosek, MosekTools 
const GRB_ENV = Gurobi.Env()
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_PART = 20 
const NUM_PROD = 20
const PRICE_MIN = 1.0
const PRICE_MAX = 4.0
const COST_MAX = 1.0
const COST_MIN = 0.5
const LATE_RATIO = 3.0
const SALVAGE_MAX = 1.0
const DEMAND_MAX = 1.0

const MIN_AUX = 1.0e-1
const MAX_AUX = 1.0e3
const MIN_PHI = -1.0e2
const OPT_GAP = 1.0e-2
const NUM_TRAIN = 5 
const NUM_TEST = 10000
const DEG_WASS = 2
const NUM_DIG = 3
const WASS_INFO = [WassInfo(round(i*2.0e-2,digits=NUM_DIG),DEG_WASS) for i in 0:25]

OUTPUT_FILE = "../result_mixing_$(NUM_PART)_$(NUM_PROD).csv"
if length(ARGS) > 0
    OUTPUT_FILE = ARGS[1]
end

# function that conducts experiments on the multiproduct mixing problem
function experiment_mixing(
        n::Int,              # number of ingredients
        m::Int,              # number of products
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used 
        N::Int = NUM_TRAIN,  # number of training samples
        M::Int = NUM_TEST;   # number of testing samples
        P::Matrix{Float64} = zeros(0,0), # matrix of mixing coefficients 
        r::Vector{Float64} = zeros(0),   # vector of regular product prices
        f_x::Vector{Float64} = zeros(0), # vector of ingredient costs
        g::Vector{Float64} = zeros(0),   # vector of late ingredient costs
        s::Vector{Float64} = zeros(0),   # vector of maximum salvage prices
        t::Vector{Int} = zeros(Int,0),   # vector of minimum unspoiled percentages
        D::Float64 = DEMAND_MAX,         # bound on the demands
    )
    # check if the mixing coefficients are supplied
    if size(P) != (n,m)
        P = zeros(n,m)
        for j = 1:m
            for i =1:(j-1)
                P[i,j] = round(0.1/(j-1),digits=NUM_DIG)
            end
            for i = j:n
                P[i,j] = round(0.9/(n+1-j),digits=NUM_DIG)
            end
        end
    end
    # check if the product prices are supplied
    if length(r) != m
        r = zeros(m)
        for j = 1:m
            r[j] = round(PRICE_MAX - (PRICE_MAX-PRICE_MIN)*(j-1)/(m-1), digits=NUM_DIG)
        end
    end
    # check if the ingredient prices are supplied
    if length(f_x) != n
        f_x = zeros(n)
        for i = 1:n
            f_x[i] = round(COST_MIN + (COST_MAX-COST_MIN)*(i-1)/(n-1), digits=NUM_DIG)
        end
    end
    if length(g) != n
        g = round.(LATE_RATIO * f_x, digits=NUM_DIG)
    end
    if length(s) != n
        s = round.(SALVAGE_MAX * f_x, digits=NUM_DIG)
    end
    if length(t) != n
        t = zeros(Int,n)
        for i = 1:n
            if iseven(i)
                t[i] = 1
            end
        end
    end
    # take the samples of salvage prices and demands
    sample_train = [round.(rand(m+n),digits=NUM_DIG) for _ in 1:N]
    sample_test = [round.(rand(m+n),digits=NUM_DIG) for _ in 1:M]
    # declare the recourse variables
    # where ξᵢ stands for qᵢ, i = 1,…,m, ξⱼ for tⱼ or sⱼ, j = m+1,…,m+n.
    @polyvar x[1:n] ξ[1:m+n] y[1:n+m]
    # distinguish the perishable vs nonperishable ingredients
    C_t = ones(n) .+ 0.0*sum(ξ)
    b_s = s .+ 0.0*sum(ξ)
    for i = 1:n
        if t[i] == 1 # nonperishable ingredient
            b_s[i] = s[i]*ξ[m+i]
        else         # perishable ingredient
            C_t[i] = ξ[m+i]
        end
    end
    # define the two-stage linear recourse function, 
    C = [zeros(n+1)' -D*ξ[1:m]'; zeros(n) -Diagonal(C_t) zeros(n,m)]
    A = [I zeros(n,m); -I zeros(n,m); P' I; zeros(m,n) -I; zeros(m,n) I] .+ 0.0*sum(ξ) # to promote the type
    b = [b_s; -g; r; -r; zeros(m)]
    Ξ = basicsemialgebraicset(FullSpace(), 
                              [[ξ[i] for i in 1:m+n];
                               [1-ξ[i] for i in 1:m+n];
                               [ξ[i]*(1-ξ[i]) for i in 1:m+n]
                              ])
    B = [g; r]
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ, B)
    # print the problem information
    println("Start the experiment on the two-stage product mixing problem...")
    println("The number of ingredient is ", n)
    println("The number of products is ", m)
    println("The product prices are ", r)
    println("The ingredient prices are ", f_x)
    println("The ingredient maximum salvage prices are ", s)
    println("The late ingredient prices are ", g)
    println("The number of training samples is ", N)
    println("The first-stage cost function is ", f_x'*x)
    println("The second-stage cost function is ", [1;x]'*C*[1;y])
    println("The second-stage constraints are ", A*y - b)
    println()
    # prepare the table for output
    WASS_RAD   = Float64[]
    WASS_DEG   = Int[]
    TRAIN_OBJ  = Float64[]
    TRAIN_TIME = Float64[]
    TEST_MEAN  = Float64[]
    TEST_STD   = Float64[]
    TEST_MED   = Float64[]
    TEST_Q90   = Float64[]
    TEST_Q10   = Float64[]
    # loop over all Wasserstein robustness settings
    for wassinfo in W
        # define the main linear/quadratic optimization problem 
        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
        set_attribute(model, "OutputFlag", 0)
        x = @variable(model, 0 <= x[1:n] <= 1, base_name="x")
        w = @variable(model, w >= 0, base_name="w")
        ϕ = @variable(model, ϕ, base_name="ϕ")
        main = MainProblem(model, x, VariableRef[], w, ϕ, f_x, Float64[])
        # solve the problem
        time_start = time()
        sol = solve_main_level(main, 
                               recourse, 
                               sample_train, 
                               wassinfo, 
                               print=1, 
                               opt_gap=OPT_GAP,
                               max_aux=MAX_AUX,
                               min_aux=MIN_AUX,
                               min_phi=MIN_PHI,
                               mom_solver=Mosek.Optimizer)
        time_finish = time()
        println("The main problem is solved successfully for Wasserstein radius = ", wassinfo.r)
        println("x = ", sol.x)
        println("f = ", sol.f)
        println("ϕ = ", sol.ϕ)
        println("The training sample objective = ", sol.f+sol.ϕ)
        println("The total computation time is ", time_finish-time_start)
        println("Start the out-of-sample test for the solution...")
        # evaluate the out-of-sample performance
        _, vals = eval_nominal(recourse, sol.x, sample_test, details=true)
        println("The testing sample mean = ", mean(vals)+sol.f)
        println("The testing sample standard deviation = ", std(vals))
        # update the output file
        append!(WASS_DEG, wassinfo.p)
        append!(WASS_RAD, wassinfo.r)
        append!(TRAIN_TIME, time_finish-time_start)
        append!(TRAIN_OBJ, sol.f+sol.ϕ)
        append!(TEST_MEAN, mean(vals)+sol.f)
        append!(TEST_STD, std(vals))
        vec_quant = quantile(vals, [0.1,0.5,0.9])
        append!(TEST_Q10, vec_quant[1])
        append!(TEST_MED, vec_quant[2])
        append!(TEST_Q90, vec_quant[3])
        output = DataFrame(:WASS_DEG   => WASS_DEG,
                           :WASS_RAD   => WASS_RAD,
                           :TRAIN_TIME => TRAIN_TIME,
                           :TRAIN_OBJ  => TRAIN_OBJ,
                           :TEST_MEAN  => TEST_MEAN,
                           :TEST_STD   => TEST_STD,
                           :TEST_Q10   => TEST_Q10,
                           :TEST_MED   => TEST_MED,
                           :TEST_Q90   => TEST_Q90)
        CSV.write(OUTPUT_FILE, output)
        println("Update the result in ", OUTPUT_FILE)
        println("\n\n")
    end
end

# run the experiment
experiment_mixing(NUM_PART, NUM_PROD, WASS_INFO)

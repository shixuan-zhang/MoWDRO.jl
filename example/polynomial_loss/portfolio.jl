# numerical example for a portfolio management problem defined by
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,1]ⁿ, ∑ᵢxᵢ = 1, where fₓ ∈ [-1,1]ⁿ, 
# and F(x,ξ) := C₁(ξᵀx) + C₂(ξᵀx)² + ⋯ + Cₖ(ξᵀx)ᵏ, ξ = Proj(D⋅η,[0,1]ⁿ), 
# η ∼ Uniform(0,1)ᵐ, D ∈ Mat(n,m) with normalized columns,
# and C₂,…,Cₖ are nonnegative (C₁ may be negative) so F is convex in x for any ξ.


using JuMP, HiGHS
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets, Statistics
using DataFrames, CSV
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_VAR = 5
const NUM_FAC = 10
const DEG_LOSS = 2

const MIN_AUX = 1.0e-1
const MAX_AUX = 1.0e3
const MIN_PHI = -1.0e2
const OPT_GAP = 1.0e-2
const NUM_TRAIN = 20 
const NUM_TEST = 10000 
const WASS_INFO = [WassInfo(0.0,DEG_LOSS),
                   WassInfo(1e-2,DEG_LOSS),
                   WassInfo(2e-2,DEG_LOSS),
                   WassInfo(5e-2,DEG_LOSS),
                   WassInfo(1e-1,DEG_LOSS),
                   WassInfo(2e-1,DEG_LOSS),
                   WassInfo(5e-1,DEG_LOSS),
                   WassInfo(1.0,DEG_LOSS)]

OUTPUT_FILE = "../result_portfolio_$(NUM_VAR)_$(NUM_FAC).csv"
if length(ARGS) > 0
    OUTPUT_FILE = ARGS[1]
end


# function that conducts the experiment on the portfolio examples
function experiment_portfolio(
        n::Int,             # number of decisions
        m::Int,             # number of factors
        k::Int,             # degree of the loss function
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used
        N::Int = NUM_TRAIN,  # number of training samples
        M::Int = NUM_TEST;   # number of testing samples
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
    sample_train = map(η->min.(max.(D*η,0),1), [rand(m) for _ in 1:N])
    sample_test = map(η->min.(max.(D*η,0),1), [rand(m) for _ in 1:N])
    # randomly generate the linear objective function if not supplied
    if length(f_x) != n
        f_x = rand(n)
    end
    # define the loss function
    @polyvar x[1:n] ξ[1:n]
    F = sum(C[i]*(x'*ξ)^i for i in 1:k)
    ∇ₓF = differentiate(F,x)
    Ξ = basicsemialgebraicset(FullSpace(), [[ξ[i]*(1-ξ[i]) for i in 1:n];
                                            [ξ[i] for i in 1:n];
                                            [1-ξ[i] for i in 1:n]])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    # print the problem information
    println("Start the experiment on the portfolio management problem...")
    println("The number of decisions is ", n)
    println("The number of factors is ", m)
    println("The number of samples is ", N)
    println("The loss function is ", F)
    println("The static cost function is ", f_x'*x)
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
        time_start = time()
        sol = solve_main_level(main, 
                               loss, 
                               sample_train, 
                               wassinfo, 
                               print=1, 
                               opt_gap=OPT_GAP,
                               max_aux=MAX_AUX,
                               min_aux=MIN_AUX,
                               min_phi=MIN_PHI)
        time_finish = time()
        println("The main problem is solved successfully for Wasserstein radius = ", wassinfo.r)
        println("x = ", sol.x)
        println("f = ", sol.f)
        println("ϕ = ", sol.ϕ)
        println("The training sample objective = ", sol.f+sol.ϕ)
        println("The total computation time is ", time_finish-time_start)
        println("Start the out-of-sample test for the solution...")
        # evaluate the out-of-sample performance
        _, vals = eval_nominal(loss, sol.x, sample_test, details=true)
        println("The testing sample mean = ", mean(vals)+sol.f)
        println("The testing sample standard deviation = ", std(vals))
        # update the output file
        append!(WASS_DEG, wassinfo.p)
        append!(WASS_RAD, wassinfo.r)
        append!(TRAIN_TIME, time_finish-time_start)
        append!(TRAIN_OBJ, sol.f+sol.ϕ)
        append!(TEST_MEAN, mean(vals)+sol.f)
        append!(TEST_STD, std(vals))
        vec_quant = quantile(vals.+sol.f, [0.1,0.5,0.9])
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
experiment_portfolio(NUM_VAR, NUM_FAC, DEG_LOSS, WASS_INFO)

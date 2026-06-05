# numerical example for a portfolio management problem defined by
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,1]ⁿ, ∑ᵢxᵢ = 1, where fₓ ∈ [-1,1]ⁿ,
# and F(x,ξ) := C₁(ξᵀx) + C₂(ξᵀx)² + ⋯ + Cₖ(ξᵀx)ᵏ, ξ = Proj(D⋅η,[0,1]ⁿ),
# η ∼ Uniform(0,1)ᵐ, D ∈ Mat(n,m) with normalized columns,
# and C₂,…,Cₖ are chosen such that Φ(t) := C₁t + C₂t² + ⋯ + Cₖtᵏ is
# a convex univariate polynomial.


using JuMP, TOML
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets, Statistics
using DataFrames, CSV
using Gurobi, Mosek, MosekTools
const GRB_ENV = Gurobi.Env()
include("../../../src/MoWDRO.jl")
using .MoWDRO

# Resolve config path: explicit ARGS[1] wins; otherwise look for a sibling
# TOML with the same base name as the script.
CONFIG_PATH = if length(ARGS) >= 1
    ARGS[1]
else
    sibling = joinpath(@__DIR__, splitext(basename(@__FILE__))[1] * ".toml")
    isfile(sibling) || error(
        "no config supplied and no default sibling TOML at $sibling; " *
        "usage: julia $(@__FILE__) [<config.toml>]"
    )
    sibling
end
CONFIG = TOML.parsefile(CONFIG_PATH)

# bind experiment-wide settings from [experiment]
const EXP_CFG = CONFIG["experiment"]
TRAIN_SIZES = Vector{Int}(EXP_CFG["training sample sizes"])
TEST_SIZE   = Int(EXP_CFG["testing sample size"])
OPT_GAP     = Float64(EXP_CFG["target optimality gap"])
MIN_AUX     = Float64(EXP_CFG["Wasserstein dual min"])
MAX_AUX     = Float64(EXP_CFG["Wasserstein dual max"])
MIN_PHI     = Float64(EXP_CFG["loss lower bound"])
BASELINE    = String(get(EXP_CFG, "baseline method", "none"))
RADIUS_SCALING = Int(get(EXP_CFG, "radius scaling", 0))
# Wasserstein radii from explicit list and/or {start, stop, step} sweeps.
WASS_ORDER = Int(EXP_CFG["Wasserstein order"])
WASS_RADII = Float64[]
if haskey(EXP_CFG, "Wasserstein radii")
    append!(WASS_RADII, Float64.(EXP_CFG["Wasserstein radii"]))
end
if haskey(EXP_CFG, "Wasserstein sweeps")
    for sw in EXP_CFG["Wasserstein sweeps"]
        append!(WASS_RADII, collect(Float64(sw["start"]):Float64(sw["step"]):Float64(sw["stop"])))
    end
end

# bind problem-specific settings from [problem]
const PROB_CFG = CONFIG["problem"]
NUM_VAR  = Int(PROB_CFG["number of variables"])
NUM_FAC  = Int(PROB_CFG["number of factors"])
DEG_LOSS = Int(PROB_CFG["loss polynomial degree"])

# OUTPUT_FILE: derived from script name + problem params, placed in the
# directory where `julia` was invoked; allow ARGS[2] override.
OUTPUT_FILE = if length(ARGS) >= 2
    ARGS[2]
else
    joinpath(pwd(), "result_portfolio_$(NUM_VAR)_$(NUM_FAC).csv")
end


# function that conducts the experiment on the portfolio examples
function experiment_portfolio(
        n::Int,                                     # number of decisions
        m::Int,                                     # number of factors
        k::Int,                                     # degree of the loss function
        wass_radii::Vector{Float64},                # Wasserstein radii to sweep
        wass_order::Int,                            # shared Wasserstein order (p)
        train_sizes::Vector{Int} = TRAIN_SIZES,     # list of training-sample sizes to sweep
        test_size::Int = TEST_SIZE;                 # number of testing samples
        C::Vector{Float64} = zeros(0),              # loss polynomial coefficients
        D::Matrix{Float64} = zeros(0,0),            # factor-model dependence matrix
        f_x::Vector{Float64} = zeros(0),            # linear-cost coefficients
        baseline::String = BASELINE,                # baseline method to compare against
        radius_scaling::Int = RADIUS_SCALING,       # s in r/(N/N_min)^(1/s); s ≤ 0 disables scaling
    )
    baseline in ("none", "noncvx") || error(
        "baseline must be one of \"none\", \"noncvx\"; got \"$baseline\""
    )
    isempty(train_sizes) && error("training sample sizes array must be non-empty")
    # Sample Φ''(t) = p₁(t)² + p₂(t)² with p₁, p₂ random polynomials of
    # degree ≤ ⌊(k-2)/2⌋, then integrate twice to obtain C₂,…,Cₖ. By
    # Hilbert's theorem two squares already span every univariate
    # nonnegative polynomial, so this guarantees Φ is globally convex.
    # Pick C₁ ∈ (-Φ'(1), 0) so that Φ has its minimizer in (0, 1).
    # For odd k, Cₖ stays 0 (an odd-degree Φ'' cannot be globally ≥ 0).
    if length(C) != k
        C = zeros(k)
        l = (k-2) ÷ 2
        for _ = 1:2
            p = rand(l+1) .* 2 .- 1
            for i = 2:(2* + 2)
                for a = max(0, i-2-l):min(l, i-2)
                    C[i] += p[a+1] * p[(i-2-a)+1]
                end
            end
        end
        for i = 2:(2*l + 2)
            C[i] /= i * (i - 1)
        end
        C[1] = -rand() * sum(i * C[i] for i = 2:k)
    end
    # check if the orthogonal matrix is supplied
    if size(D) != (n,m)
        D = rand(n,m) .* 2 .- 1
        # normalize the columns
        for i = 1:m
            D[:,i] ./= norm(D[:,i])
        end
    end
    # take the samples of the uncertainty (draw the largest training set once,
    # then later iterations reuse a strict prefix of it)
    N_max = maximum(train_sizes)
    sample_train_full = map(η->min.(max.(D*η,0),1), [rand(m) for _ in 1:N_max])
    sample_test       = map(η->min.(max.(D*η,0),1), [rand(m) for _ in 1:test_size])
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
    println("Training sample sizes to sweep: ", train_sizes)
    println("Number of testing samples: ", test_size)
    println("The loss function is ", F)
    println("The static cost function is ", f_x'*x)
    println()
    # prepare the table for output
    WASS_RAD   = Float64[]
    WASS_DEG   = Int[]
    TRAIN_SIZE = Int[]
    TRAIN_OBJ  = Float64[]
    TRAIN_TIME = Float64[]
    TEST_MEAN  = Float64[]
    TEST_STD   = Float64[]
    TEST_MED   = Float64[]
    TEST_Q90   = Float64[]
    TEST_Q10   = Float64[]
    # nonconvex-baseline outputs (defined only if baseline == "noncvx")
    if baseline == "noncvx"
        NCVX_OBJ   = Float64[]
        NCVX_TIME  = Float64[]
        NCVX_MEAN  = Float64[]
        NCVX_STD   = Float64[]
        NCVX_MED   = Float64[]
        NCVX_Q90   = Float64[]
        NCVX_Q10   = Float64[]
    end
    # loop over all (training-sample size, Wasserstein radius) combinations
    N_min = minimum(train_sizes)
    for N in train_sizes
        sample_train = sample_train_full[1:N]
        for r in wass_radii
            # auto-scale the Wasserstein radius by (N/N_min)^(1/s), where
            # s = radius_scaling; s ≤ 0 disables scaling.
            scaled_r = radius_scaling > 0 ? r / (N / N_min)^(1.0 / radius_scaling) : r
            wassinfo = WassInfo(scaled_r, wass_order)
            # define the main linear optimization problem
            model = Model(() -> Gurobi.Optimizer(GRB_ENV))
            set_attribute(model, "OutputFlag", 0)
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
                                   min_phi=MIN_PHI,
                                   mom_solver=Mosek.Optimizer)
            time_finish = time()
            println("The main problem is solved for Wasserstein radius = ", wassinfo.r,
                    ", training size = ", N)
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
            append!(TRAIN_SIZE, N)
            append!(TRAIN_TIME, time_finish-time_start)
            append!(TRAIN_OBJ, sol.f+sol.ϕ)
            append!(TEST_MEAN, mean(vals)+sol.f)
            append!(TEST_STD, std(vals))
            vec_quant = quantile(vals.+sol.f, [0.1,0.5,0.9])
            append!(TEST_Q10, vec_quant[1])
            append!(TEST_MED, vec_quant[2])
            append!(TEST_Q90, vec_quant[3])
            # ---------------------------------------------------------------
            # Optional: solve the same instance with the nonconvex global
            # baseline (level bundle with `eval_noncvx_Wass`) at the same
            # Wasserstein radius and record `NCVX_*`.
            if baseline == "noncvx"
                println("Solve the same instance with the nonconvex global baseline...")
                model_NC = Model(() -> Gurobi.Optimizer(GRB_ENV))
                set_silent(model_NC)
                x_NC = @variable(model_NC, 0 <= x_NC[1:n] <= 1, base_name="x_NC")
                w_NC = @variable(model_NC, w_NC >= 0, base_name="w_NC")
                ϕ_NC = @variable(model_NC, ϕ_NC >= 0, base_name="ϕ_NC")
                @constraint(model_NC, ones(n)'*x_NC == 1)
                main_NC = MainProblem(model_NC, x_NC, VariableRef[], w_NC, ϕ_NC, f_x, Float64[])
                noncvx_solver = () -> begin
                    opt = Gurobi.Optimizer(GRB_ENV)
                    MOI.set(opt, MOI.RawOptimizerAttribute("NonConvex"), 2)
                    opt
                end
                eval_noncvx_cut = (subproblem, augstate, samples, wassinfo; print=0) ->
                    eval_noncvx_Wass(subproblem, augstate, samples, wassinfo;
                                     noncvx_solver=noncvx_solver, print=print)
                time_start_NC = time()
                sol_NC = solve_main_level(main_NC,
                                          loss,
                                          sample_train,
                                          wassinfo,
                                          print=1,
                                          opt_gap=OPT_GAP,
                                          max_aux=MAX_AUX,
                                          min_aux=MIN_AUX,
                                          min_phi=MIN_PHI,
                                          cut_evaluator=eval_noncvx_cut)
                time_finish_NC = time()
                println("  Nonconvex baseline x          = ", sol_NC.x)
                println("  Nonconvex baseline objective  = ", sol_NC.f + sol_NC.ϕ)
                println("  Nonconvex baseline time       = ", time_finish_NC - time_start_NC)
                _, vals_NC = eval_nominal(loss, sol_NC.x, sample_test, details=true)
                append!(NCVX_OBJ,  sol_NC.f + sol_NC.ϕ)
                append!(NCVX_TIME, time_finish_NC - time_start_NC)
                append!(NCVX_MEAN, mean(vals_NC) + sol_NC.f)
                append!(NCVX_STD,  std(vals_NC))
                vec_quant_NC = quantile(vals_NC .+ sol_NC.f, [0.1, 0.5, 0.9])
                append!(NCVX_Q10, vec_quant_NC[1])
                append!(NCVX_MED, vec_quant_NC[2])
                append!(NCVX_Q90, vec_quant_NC[3])
                println("  Nonconvex baseline test mean  = ", mean(vals_NC) + sol_NC.f)
                println("  Nonconvex baseline test std   = ", std(vals_NC))
            end
            output = DataFrame(:WASS_DEG   => WASS_DEG,
                               :WASS_RAD   => WASS_RAD,
                               :TRAIN_SIZE => TRAIN_SIZE,
                               :TRAIN_TIME => TRAIN_TIME,
                               :TRAIN_OBJ  => TRAIN_OBJ,
                               :TEST_MEAN  => TEST_MEAN,
                               :TEST_STD   => TEST_STD,
                               :TEST_Q10   => TEST_Q10,
                               :TEST_MED   => TEST_MED,
                               :TEST_Q90   => TEST_Q90)
            if baseline == "noncvx"
                output.NCVX_OBJ  = NCVX_OBJ
                output.NCVX_TIME = NCVX_TIME
                output.NCVX_MEAN = NCVX_MEAN
                output.NCVX_STD  = NCVX_STD
                output.NCVX_Q10  = NCVX_Q10
                output.NCVX_MED  = NCVX_MED
                output.NCVX_Q90  = NCVX_Q90
            end
            CSV.write(OUTPUT_FILE, output)
            println("Update the result in ", OUTPUT_FILE)
            println("\n\n")
        end
    end
end


# run the experiment
experiment_portfolio(NUM_VAR, NUM_FAC, DEG_LOSS,
                     WASS_RADII, WASS_ORDER,
                     TRAIN_SIZES, TEST_SIZE;
                     baseline       = BASELINE,
                     radius_scaling = RADIUS_SCALING)

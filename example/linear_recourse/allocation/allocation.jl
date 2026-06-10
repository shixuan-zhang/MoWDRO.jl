# numerical example for a two-stage commodity allocation problem
# (adapted from Duque, Mehrotra, and Morton (2022)):
# min E[F(x,ξ)], x ∈ [0,1]ⁿ, where
# F(x,ξ) := max  ∑ᵢ xᵢ⋅uᵢ + ∑ⱼ ξⱼ⋅vⱼ
#           s.t. uᵢ + vⱼ ≤ dᵢⱼ,     ∀ i = 1,…,n, j = 1,…,m,
#                -s ≤ uᵢ ≤ h,       ∀ i = 1,…,n,
#                0 ≤ vⱼ ≤ s,        ∀ j = 1,…,m.
# Here, xᵢ ∈ [0,Lᵢ] is the supply allocated to location i,
# ξⱼ ∼ LogNormal(1,1) is the random demand at site j,
# dᵢⱼ > 0 is the Euclidean distance between locations i and j,
# where the locations are randomly distributed on [0,1]²;
# h > 0 is the unit cost for holding inventory,
# s > 0 is the unit cost for subcontracted demand.
# Let P be the pairing matrix with the form (e the all-1 vector)
# P = [e 0 0 I;
#      0 e 0 I;
#      0 0 e I]
# for n = 3, any m > 1 in this example.
# We can then write F in the matrix form as
# F(x,ξ) = max  (1,x)ᵀ⋅[0 ξᵀ; 0 I 0]⋅(1,y)
#          s.t. -P⋅y ≥ -d,
#               [I 0; -I 0; 0 I; 0 -I] y ≥ [-s; -h; 0; -s]

using Distributed
# load modules on the main process
using JuMP, TOML
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets, Statistics
using DataFrames, CSV
# use commercial solvers for efficiency and numerical stability
using Gurobi, Mosek, MosekTools
using MoWDRO
const GRB_ENV = Gurobi.Env()
# load modules on the worker processes
let
    setup_expr = quote
        using JuMP
        using LinearAlgebra, DynamicPolynomials, SemialgebraicSets
        using Gurobi, Mosek, MosekTools
        using MoWDRO
        const GRB_ENV = Gurobi.Env()
    end
    for w in workers()
        w == myid() && continue
        remotecall_wait(w) do
            Base.eval(Main, setup_expr)
        end
    end
end

include(joinpath(@__DIR__, "..", "..", "experiment_common.jl"))
CONFIG_PATH = resolve_config_path(@__FILE__)
CONFIG      = TOML.parsefile(CONFIG_PATH)

# bind problem-specific settings from [problem]
const PROB_CFG = CONFIG["problem"]
NUM_FACILITY     = Int(PROB_CFG["number of facilities"])
NUM_SITE         = Int(PROB_CFG["number of sites"])
COST_HOLDING     = Float64(PROB_CFG["holding cost"])
COST_SUBCONTRACT = Float64(PROB_CFG["subcontract cost"])
MEAN_DEMAND      = Float64(PROB_CFG["mean demand"])
VAR_DEMAND       = Float64(PROB_CFG["demand variance"])
MAX_CAPACITY     = Float64(PROB_CFG["maximum capacity"])
NUM_DIG          = Int(PROB_CFG["number of digits"])

# bind experiment-wide settings from [experiment]
const EXP_CFG = CONFIG["experiment"]
SEED           = apply_random_seed!(EXP_CFG)
TRAIN_SIZES    = parse_train_sizes(EXP_CFG)
TEST_SIZE      = Int(EXP_CFG["testing sample size"])
OPT_GAP        = Float64(EXP_CFG["target optimality gap"])
MIN_AUX        = Float64(EXP_CFG["Wasserstein dual min"])
MAX_AUX        = Float64(EXP_CFG["Wasserstein dual max"])
MIN_PHI        = Float64(EXP_CFG["loss lower bound"])
BASELINE       = String(get(EXP_CFG, "baseline method", "none"))
RADIUS_SCALING = Int(get(EXP_CFG, "radius scaling", 0))
WASS_ORDER     = Int(EXP_CFG["Wasserstein order"])
# round the radii to NUM_DIG digits to match the rest of the allocation data
WASS_RADII     = parse_wass_radii(EXP_CFG)

OUTPUT_FILE = resolve_output_file("result_allocation_$(NUM_FACILITY)_$(NUM_SITE).csv")

# Build the data tuple for the Hanasusanto-Kuhn (2018) copositive baseline
# `solve_two_stage_copos` from the allocation-problem parameters. The
# returned NamedTuple has fields (c, X, Q, q, T, h, W, S, t) matching the
# positional arguments expected by `solve_two_stage_copos`.
function build_copos_baseline_data(
        n::Int, m::Int,
        d::Vector{Float64}, P::Matrix{Float64},
        D::Float64, s::Float64, h::Float64,
    )
    # The allocation primal recourse is a MAX-LP in y = (u; v) ∈ ℝ^{n+m}:
    #   F(x, ξ) = max  x^T u + ξ^T v
    #     s.t.   P [u; v] ≤ d                 (n·m distance constraints)
    #           -s·1_n ≤ u ≤ h·1_n            (n inventory bounds)
    #            0·1_m ≤ v ≤ s·1_m            (m subcontract bounds)
    # Stack the inequalities as  K y ≤ ℓ  with
    #   K = [ P        ;        ℓ = [ d      ;
    #        -I_n  0   ;               s·1_n ;
    #         I_n  0   ;               h·1_n ;
    #         0   -I_m ;               0·1_m ;
    #         0    I_m ]                s·1_m ]
    # of row dimension N_μ = n·m + 2n + 2m.  By LP duality (primal bounded
    # ⇒ strong duality)
    #   F(x, ξ) = min ℓ^T μ   s.t.  K^T μ = [x; ξ],  μ ≥ 0,
    # which is exactly paper Eq. (3) form
    #   Z(x, ξ) = inf (Qξ + q)^T y  s.t.  T(x)ξ + h(x) ≤ W y
    # with paper's y ≡ μ, Q = 0 (the dual cost has no ξ-dependence), q = ℓ,
    # the (n+m) equalities K^T μ = [x; ξ] split into 2(n+m) inequalities,
    # and the N_μ sign constraints μ ≥ 0 folded into W.  This yields
    # M = 2(n+m) + N_μ = 4n + 4m + n·m rows before the support-set
    # extension.  The "sufficient expensive recourse" assumption of the
    # paper holds because every component of the dual cost q = ℓ ≥ 0.
    N_y_HK = n*m + 2n + 2m              # dim of paper's y = dual variable μ
    K_HK   = m                          # dim(ξ) — only m random demand factors
    Mh_HK  = 2*(n + m) + N_y_HK         # row dim of W before support extension
    #         = 4n + 4m + n·m
    # Assemble K and ℓ from the primal allocation constraints (K y ≤ ℓ).
    K_mat = [P;
             -Matrix{Float64}(I, n, n)  zeros(n, m);
              Matrix{Float64}(I, n, n)  zeros(n, m);
              zeros(m, n)              -Matrix{Float64}(I, m, m);
              zeros(m, n)               Matrix{Float64}(I, m, m)]
    ℓ_vec = [d; s*ones(n); h*ones(n); zeros(m); s*ones(m)]
    # Paper cost (Qξ + q)^T y on y = μ equals ℓ^T μ:
    #   no ξ-dependence  →  Q_HK = 0,  q_HK = ℓ.
    Q_HK = zeros(N_y_HK, K_HK)
    q_HK = copy(ℓ_vec)
    # Paper constraint matrix W (4n+4m+n·m rows × N_μ cols).
    # The equality K^T μ = [x; ξ] splits into K^T μ ≥ [x; ξ] and K^T μ ≤ [x; ξ].
    # Rewriting each in the paper's "T(x)ξ + h(x) ≤ W μ" sense gives:
    #   • upper-split rows  [1 .. n+m]            :  [x; ξ] ≤  K^T μ   →  W = +K^T, RHS = [x; ξ]
    #   • lower-split rows  [n+m+1 .. 2(n+m)]     : -[x; ξ] ≤ -K^T μ   →  W = -K^T, RHS = -[x; ξ]
    #   • sign-constraint rows  [2(n+m)+1 .. end]:        0 ≤  μ      →  W = I_{N_μ}, RHS = 0.
    W_HK = zeros(Mh_HK, N_y_HK)
    W_HK[1:(n+m),                    :] .=  K_mat'                       # upper-split
    W_HK[(n+m+1):(2*(n+m)),          :] .= -K_mat'                       # lower-split
    W_HK[(2*(n+m)+1):Mh_HK,          :] .=  Matrix{Float64}(I, N_y_HK, N_y_HK)
    # Affine decomposition T(x) = T_0 + Σ_l x_l T_l, h(x) = h_0 + Σ_l x_l h_l
    # stored 1-indexed: T_HK[l+1] == T_l, h_HK[l+1] == h_l  (length n+1).
    # The RHS T(x)ξ + h(x) at the various row blocks is:
    #   block 1 (upper-split rows 1..n+m), RHS = +[x; ξ]:
    #     row i ≤ n           :  +x_i        →  h_HK[i+1][i]                 = +1
    #     row n+j  (1 ≤ j ≤ m):  +ξ_j        →  T_HK[1][n+j, j]              = +1
    #   block 2 (lower-split rows (n+m)+1..2(n+m)), RHS = -[x; ξ]:
    #     row (n+m)+i (i ≤ n) :  -x_i        →  h_HK[i+1][(n+m)+i]           = -1
    #     row (n+m)+(n+j)     :  -ξ_j        →  T_HK[1][(n+m)+(n+j), j]      = -1
    #   block 3 (sign rows)   :   0          →  no entries (T = 0, h = 0).
    T_HK = [zeros(Mh_HK, K_HK) for _ in 0:n]
    h_HK = [zeros(Mh_HK)       for _ in 0:n]
    # block 1: upper-split
    for i = 1:n
        h_HK[i + 1][i] = 1.0
    end
    for j = 1:m
        T_HK[1][n + j, j] = 1.0
    end
    # block 2: lower-split
    for i = 1:n
        h_HK[i + 1][(n + m) + i] = -1.0
    end
    for j = 1:m
        T_HK[1][(n + m) + (n + j), j] = -1.0
    end
    # Support set Ξ = ℝ^m_+ — only ξ_j ≥ 0 (LogNormal demand samples have
    # no natural upper bound), so S has zero rows.
    S_HK = zeros(0, K_HK)
    t_HK = zeros(0)
    # First-stage polyhedron X = [0, D]^n encoded row-by-row as [a_i; b_i]:
    X_HK = Vector{Vector{Float64}}()
    for i = 1:n
        a_lo = zeros(n); a_lo[i] = -1.0;  push!(X_HK, [a_lo; 0.0])     # −x_i ≤ 0
        a_hi = zeros(n); a_hi[i] =  1.0;  push!(X_HK, [a_hi; D])       #  x_i ≤ D
    end
    # First-stage cost c = 0 — the allocation `MainProblem` is built with a
    # zero linear cost (no f_x^T x term).
    c_HK = zeros(n)
    return (c = c_HK, X = X_HK,
            Q = Q_HK, q = q_HK,
            T = T_HK, h = h_HK, W = W_HK,
            S = S_HK, t = t_HK)
end

# function that generates the Euclidean distances between facilities and demand sites
function generate_distance_pairs(
        n::Int,
        m::Int;
        print::Int = 0
    )
    # generate the locations
    coord_facility = [rand(2) for _ in 1:n]
    coord_site = [rand(2) for _ in 1:m]
    if print > 0
        println("The facility location coordinates:")
        for coord in coord_facility
            println(round.(coord,digits=NUM_DIG))
        end
        println("The demand site coordinates:")
        for coord in coord_site
            println(round.(coord,digits=NUM_DIG))
        end
    end
    # prepare the distance vector
    vec_dist = zeros(n*m)
    for i = 1:n, j = 1:m
        vec_dist[m*(i-1)+j] = round.(norm(coord_facility[i]-coord_site[j]),digits=NUM_DIG)
    end
    return vec_dist
end

# function that conducts experiments on the facility allocation problem
function experiment_allocation(
        n::Int,                                     # number of facilities
        m::Int,                                     # number of demand sites
        wass_radii::Vector{Float64},                # Wasserstein radii to sweep
        wass_order::Int,                            # shared Wasserstein order (p)
        train_sizes::Vector{Int} = TRAIN_SIZES,     # list of training-sample sizes to sweep
        test_size::Int = TEST_SIZE;                 # number of testing samples
        D::Float64 = MAX_CAPACITY,                  # maximum facility capacity
        h::Float64 = COST_HOLDING,                  # cost for holding inventory
        s::Float64 = COST_SUBCONTRACT,              # cost for subcontracted demand
        μ::Float64 = MEAN_DEMAND,                   # mean factor for the demand
        σ::Float64 = VAR_DEMAND,                    # variance factor for the demand
        d::Vector{Float64} = zeros(0),              # distance vector
        baseline::String = BASELINE,                # baseline method for comparison:
                                                    #   "none"   — none
                                                    #   "copos"  — Hanasusanto-Kuhn (2018) copositive formulation
                                                    #   "noncvx" — nonconvex global optimization formulation
                                                    #   "all"    — both baselines
        radius_scaling::Int = RADIUS_SCALING,       # s in r/(N/N_min)^(1/s); s ≤ 0 disables scaling
    )
    baseline in ("none", "copos", "noncvx", "all") || error(
        "baseline must be one of \"none\", \"copos\", \"noncvx\", \"all\"; got \"$baseline\""
    )
    isempty(train_sizes) && error("training sample sizes array must be non-empty")
    # take the samples of random demands (draw the largest training set once,
    # then later iterations reuse a strict prefix of it)
    N_max = maximum(train_sizes)
    sample_train_full = [round.(μ*exp.(randn(m)*σ),digits=NUM_DIG) for _ in 1:N_max]
    sample_test       = [round.(μ*exp.(randn(m)*σ),digits=NUM_DIG) for _ in 1:test_size]
    # construct the pairwise indicator matrix
    P = zeros(m*n, m+n)
    for i = 1:n
        P[m*(i-1)+1:m*i,i] = ones(m)
        for j = 1:m
            P[m*(i-1)+j,n+j] = 1.0
        end
    end
    # check if the distance vector is supplied
    if length(d) != m*n
        d = generate_distance_pairs(n,m,print=1)
    end
    # declare the recourse variables (y = [u,v])
    @polyvar x[1:n] ξ[1:m] y[1:n+m]
    # define the two-stage linear recourse function,
    C = [zeros(n+1)' ξ'; zeros(n) I zeros(n,m)]
    A = [-P; I zeros(n,m); -I zeros(n,m); zeros(m,n) I; zeros(m,n) -I] .+ 0.0*sum(ξ) # to promote the type
    b = [-d; -s*ones(n); -h*ones(n); zeros(m); -s*ones(m)] .+ 0.0*sum(ξ) # to promote the type
    Ξ = basicsemialgebraicset(FullSpace(),
                              [ξ[i] for i in 1:m]
                              )
    B = s * ones(n+m)
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ, B)
    # print the problem information
    println("Start the experiment on the two-stage facility allocation problem...")
    println("The number of facility locations is ", n)
    println("The number of demand sites is ", m)
    println("The cost of subcontracted demand is ", s)
    println("The cost of holding inventory is ", h)
    println("The maximum facility capacity is ", D)
    println("Training sample sizes to sweep: ", train_sizes)
    println("Number of testing samples: ", test_size)
    println("The second-stage cost function is ", [1;x]'*C*[1;y])
    println("The second-stage constraints are ", A*y - b)
    println()
    # prepare the table for output
    WASS_RAD   = Float64[]
    WASS_DEG   = Int[]
    WASS_IDX   = Int[]
    TRAIN_SIZE = Int[]
    TRAIN_OBJ  = Float64[]
    TRAIN_TIME = Float64[]
    TEST_MEAN  = Float64[]
    TEST_STD   = Float64[]
    TEST_MED   = Float64[]
    TEST_Q90   = Float64[]
    TEST_Q10   = Float64[]
    # copositive-baseline outputs (defined only if baseline ∈ ("copos","all"))
    if baseline in ("copos", "all")
        CPOS_OBJ   = Float64[]
        CPOS_TIME  = Float64[]
        CPOS_MEAN  = Float64[]
        CPOS_STD   = Float64[]
        CPOS_MED   = Float64[]
        CPOS_Q90   = Float64[]
        CPOS_Q10   = Float64[]
    end
    # nonconvex-baseline outputs (defined only if baseline ∈ ("noncvx","all"))
    if baseline in ("noncvx", "all")
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
        for (radius_idx, wass_r) in enumerate(wass_radii)
            # auto-scale the Wasserstein radius by (N/N_min)^(1/s), where
            # s = radius_scaling; s ≤ 0 disables scaling. `radius_idx` is
            # the 1-based position of `wass_r` in the original `wass_radii`
            # list and is preserved as the `WASS_IDX` CSV column so that
            # rows sharing a configured radius can be matched across
            # training-sample sizes even when the actual radius is scaled.
            scaled_r = radius_scaling > 0 ? wass_r / (N / N_min)^(1.0 / radius_scaling) : wass_r
            wassinfo = WassInfo(scaled_r, wass_order)
            # define the main linear/quadratic optimization problem
            model = Model(() -> Gurobi.Optimizer(GRB_ENV))
            set_attribute(model, "OutputFlag", 0)
            x = @variable(model, 0 <= x[1:n] <= D, base_name="x")
            w = @variable(model, w >= 0, base_name="w")
            ϕ = @variable(model, ϕ, base_name="ϕ")
            main = MainProblem(model, x, VariableRef[], w, ϕ, zeros(n), Float64[])
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
            println("The main problem is solved for Wasserstein radius = ", wassinfo.r,
                    ", training size = ", N)
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
            append!(WASS_IDX, radius_idx)
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
            # Optional: solve the same instance with the H-K (2018) copositive
            # baseline at the same Wasserstein radius and record `CPOS_*`.
            if baseline in ("copos", "all")
                # Build paper Eq.(1)+Eq.(3) data for the Hanasusanto-Kuhn (2018)
                # copositive baseline; see `build_copos_baseline_data` for details.
                data_HK = build_copos_baseline_data(n, m, d, P, D, s, h)
                println("Built copositive-baseline data: matrix size = ",
                        m + (4n + 4m + n*m) + 1,
                        " per sample (", N, " samples).")
                println("Solve the same instance with the Hanasusanto-Kuhn copositive baseline...")
                time_start_HK = time()
                res_HK = solve_two_stage_copos(data_HK.c, data_HK.X,
                                               data_HK.Q, data_HK.q,
                                               data_HK.T, data_HK.h, data_HK.W,
                                               data_HK.S, data_HK.t,
                                               sample_train, wassinfo.r;
                                               solver = Mosek.Optimizer,
                                               silent = true,
                                               δ = 0.0)
                time_finish_HK = time()
                println("  Copositive baseline status    = ", res_HK.status)
                println("  Copositive baseline x         = ", res_HK.x)
                println("  Copositive baseline objective = ", res_HK.objective_value)
                println("  Copositive baseline time      = ", time_finish_HK - time_start_HK)
                # out-of-sample test on the H-K solution using the same test samples
                _, vals_HK = eval_nominal(recourse, res_HK.x, sample_test, details=true)
                f_HK = 0.0    # allocation main problem has no first-stage cost
                append!(CPOS_OBJ,  res_HK.objective_value)
                append!(CPOS_TIME, time_finish_HK - time_start_HK)
                append!(CPOS_MEAN, mean(vals_HK) + f_HK)
                append!(CPOS_STD,  std(vals_HK))
                vec_quant_HK = quantile(vals_HK .+ f_HK, [0.1, 0.5, 0.9])
                append!(CPOS_Q10, vec_quant_HK[1])
                append!(CPOS_MED, vec_quant_HK[2])
                append!(CPOS_Q90, vec_quant_HK[3])
                println("  Copositive baseline test mean = ", mean(vals_HK) + f_HK)
                println("  Copositive baseline test std  = ", std(vals_HK))
            end
            # ---------------------------------------------------------------
            # Optional: solve the same instance with the nonconvex global
            # baseline (level bundle with `eval_noncvx_Wass`) at the same
            # Wasserstein radius and record `NCVX_*`. The inner polynomial
            # supremum is routed to Gurobi with `NonConvex=2`.
            if baseline in ("noncvx", "all")
                println("Solve the same instance with the nonconvex global baseline...")
                model_NC = Model(() -> Gurobi.Optimizer(GRB_ENV))
                set_attribute(model_NC, "OutputFlag", 0)
                x_NC = @variable(model_NC, 0 <= x_NC[1:n] <= D, base_name="x_NC")
                w_NC = @variable(model_NC, w_NC >= 0, base_name="w_NC")
                ϕ_NC = @variable(model_NC, ϕ_NC, base_name="ϕ_NC")
                main_NC = MainProblem(model_NC, x_NC, VariableRef[], w_NC, ϕ_NC, zeros(n), Float64[])
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
                                          recourse,
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
                # out-of-sample test on the nonconvex-baseline solution
                _, vals_NC = eval_nominal(recourse, sol_NC.x, sample_test, details=true)
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
            # write the (possibly augmented) result file
            output = DataFrame(:WASS_DEG   => WASS_DEG,
                               :WASS_RAD   => WASS_RAD,
                               :WASS_IDX   => WASS_IDX,
                               :TRAIN_SIZE => TRAIN_SIZE,
                               :TRAIN_TIME => TRAIN_TIME,
                               :TRAIN_OBJ  => TRAIN_OBJ,
                               :TEST_MEAN  => TEST_MEAN,
                               :TEST_STD   => TEST_STD,
                               :TEST_Q10   => TEST_Q10,
                               :TEST_MED   => TEST_MED,
                               :TEST_Q90   => TEST_Q90)
            if baseline in ("copos", "all")
                output.CPOS_OBJ  = CPOS_OBJ
                output.CPOS_TIME = CPOS_TIME
                output.CPOS_MEAN = CPOS_MEAN
                output.CPOS_STD  = CPOS_STD
                output.CPOS_Q10  = CPOS_Q10
                output.CPOS_MED  = CPOS_MED
                output.CPOS_Q90  = CPOS_Q90
            end
            if baseline in ("noncvx", "all")
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
            # ensure per-iteration logs reach the terminal in real time —
            # Distributed workers leave the main process with a block-
            # buffered stdout when output is piped or redirected.
            flush(stdout)
        end
    end
end

# run the experiment
experiment_allocation(NUM_FACILITY, NUM_SITE,
                      WASS_RADII, WASS_ORDER,
                      TRAIN_SIZES, TEST_SIZE;
                      baseline       = BASELINE,
                      radius_scaling = RADIUS_SCALING)

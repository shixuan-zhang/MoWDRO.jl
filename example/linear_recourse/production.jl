# numerical example for a two-stage production problem 
# (adapted from Chapter 1.3.1 in Shapiro-Dentcheva-Ruszczyński(2009)):
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,D]ⁿ, where fₓ ∈ [0,1]ⁿ, and 
# F(x,ξ) := min  -∑ᵢ rᵢ⋅zᵢ - ∑ⱼ sᵢ(ξ)⋅wⱼ + ∑ⱼ gⱼ⋅uⱼ
#           s.t. wⱼ - uⱼ = tⱼ(ξ)⋅xⱼ - ∑ᵢ pᵢⱼ⋅zᵢ, ∀ j = 1,…,n,
#                0 ≤ zᵢ ≤ qᵢ(ξ),                 ∀ i = 1,…,m,
#                wⱼ, uⱼ ≥ 0,                     ∀ j = 1,…,n.
# Here, rᵢ > 0 is the product price, 
# gⱼ is the late price for ingredient purchasing,
# pᵢⱼ is the percentage of ingredient j in product i,
# sᵢ(ξ) ∼ Uniform(0,1) is the random salvage price 
# factor for ingredient i,
# tᵢ(ξ) ∼ Uniform(0,1) is the random spoilage 
# percentage for ingredient i, 
# qᵢ(ξ) ∼ LogNormal(0,σ) is the random demand factor, so 
# that the total demand is qᵢ⋅dᵢ for some dᵢ > 0.
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
const NUM_PART = 5#20 
const NUM_PROD = 5#20
const PRICE_MIN = 2.0
const PRICE_MAX = 5.0
const COST_MAX = 2.0
const COST_MIN = 1.0
const LATE_RATIO = 3.0
const SALVAGE_MAX = 1.0
const DEMAND_MAX = 2.0
const DEMAND_MIN = 1.0
const DEMAND_VAR = 0.1
const STORAGE_MAX = 5.0

const MIN_AUX = 1.0e-1
const MAX_AUX = 1.0e3
const MIN_PHI = -1.0e2
const OPT_GAP = 1.0e-2
const NUM_TRAIN = 10 
const NUM_TEST = 10000
const DEG_WASS = 2
const NUM_DIG = 3
const WASS_INFO = [[WassInfo(round(i*1.0e-2,digits=NUM_DIG),DEG_WASS) for i in 0:9];
                  [WassInfo(round(i*1.0e-1,digits=NUM_DIG),DEG_WASS) for i in 1:9];
                  [WassInfo(round(i*1.0e0,digits=NUM_DIG),DEG_WASS) for i in 1:10]]

OUTPUT_FILE = "../result_production_$(NUM_PART)_$(NUM_PROD).csv"
if length(ARGS) > 0
    OUTPUT_FILE = ARGS[1]
end

# Build the data tuple for the Hanasusanto-Kuhn (2018) copositive baseline
# `solve_two_stage_copos` from the production-problem parameters. The
# returned NamedTuple has fields (c, X, Q, q, T, h, W, S, t) matching the
# positional arguments expected by `solve_two_stage_copos`.
function build_copos_baseline_data(
        n::Int, m::Int,
        r::Vector{Float64}, s::Vector{Float64}, t::Vector{Int},
        g::Vector{Float64}, d::Vector{Float64}, P::Matrix{Float64},
        D::Float64, f_x::Vector{Float64},
    )
    # Map the production primal-min recourse  (variables z ∈ ℝ^m, w, u ∈ ℝ^n,
    # cost  −r^T z − s(ξ)^T w + g^T u,  with n equalities, m inequalities,
    # and sign constraints z, w, u ≥ 0)  directly into paper Eq. (3) form
    #   Z(x,ξ) = inf_{y free} (Qξ+q)^T y  s.t.  T(x)ξ + h(x) ≤ W y.
    # We set y = (z; w; u) ∈ ℝ^{m+2n}, split the n equalities into
    # 2n inequalities, and fold the m bound constraints and the
    # (m+2n) sign constraints into W.  This yields M = 4n+2m rows
    # before the support-set extension.  This mapping satisfies the
    # paper's "sufficient expensive recourse" assumption because the
    # production data has g_j > s_j(ξ) ≥ 0 for every ingredient j.
    N_y_HK = m + 2n                     # dim of paper's y = (z; w; u)
    K_HK   = m + n                      # dim(ξ)
    Mh_HK  = 4n + 2m                    # row dim of W before support extension
    # Cost vector  Q ξ + q  on y = (z; w; u):
    #   z-block (length m):  -r              (constant)
    #   w-block (length n):  -s(ξ)           (nonperishable: -s_j ξ_{m+j};  perishable: -s_j)
    #   u-block (length n):  +g              (constant)
    Q_HK = zeros(N_y_HK, K_HK)
    q_HK = zeros(N_y_HK)
    for i = 1:m;  q_HK[i]            = -r[i];  end                # z block
    for j = 1:n
        if t[j] == 1                                              # nonperishable
            Q_HK[m + j, m + j] = -s[j]
        else                                                      # perishable
            q_HK[m + j] = -s[j]
        end
    end
    for j = 1:n;  q_HK[m + n + j]    =  g[j];  end                # u block
    # Constraint matrix W (4n+2m rows × m+2n cols).
    # Block rows:
    #   [1   .. n  ]   equality upper-split :   [ P_{j,:}  e_j  -e_j ]
    #   [n+1 .. 2n ]   equality lower-split :   [-P_{j,:} -e_j   e_j ]
    #   [2n+1.. 2n+m]  z ≤ q  →  -z ≥ -q     :   [-e_i      0     0  ]
    #   [2n+m+1..2n+2m] z ≥ 0                :   [ e_i      0     0  ]
    #   [2n+2m+1..3n+2m] w ≥ 0               :   [ 0        e_j   0  ]
    #   [3n+2m+1..4n+2m] u ≥ 0               :   [ 0        0     e_j]
    W_HK = zeros(Mh_HK, N_y_HK)
    for j = 1:n
        W_HK[j,         1:m]          .=  P[j, :]            # P here is n×m (ingredient × product)
        W_HK[j,         m + j]         =  1.0
        W_HK[j,         m + n + j]     = -1.0
        W_HK[n + j,     1:m]          .= -P[j, :]
        W_HK[n + j,     m + j]         = -1.0
        W_HK[n + j,     m + n + j]     =  1.0
    end
    for i = 1:m
        W_HK[2n + i,         i] = -1.0   # z ≤ q
        W_HK[2n + m + i,     i] =  1.0   # z ≥ 0
    end
    for j = 1:n
        W_HK[2n + 2m + j, m + j]      = 1.0   # w ≥ 0
        W_HK[3n + 2m + j, m + n + j]  = 1.0   # u ≥ 0
    end
    # Affine decomposition T(x) = T_0 + Σ_l x_l T_l, h(x) = h_0 + Σ_l x_l h_l
    # stored 1-indexed: T_HK[l+1] == T_l, h_HK[l+1] == h_l.
    # The RHS T(x)ξ + h(x) at the various row blocks is:
    #   row j      (equality upper):  +t_j(ξ) x_j
    #   row n+j    (equality lower):  -t_j(ξ) x_j
    #   row 2n+i   (z ≤ q):           -q_i(ξ) = -d_i ξ_i
    #   other rows (sign constraints): 0
    # with t_j(ξ) = 1 (nonperishable, t[j]==1) or ξ_{m+j} (perishable).
    T_HK = [zeros(Mh_HK, K_HK) for _ in 0:n]
    h_HK = [zeros(Mh_HK)       for _ in 0:n]
    for j = 1:n
        if t[j] == 1            # nonperishable: t_j(ξ) x_j = x_j  →  only h_j is nonzero
            h_HK[j + 1][j]      =  1.0   # equality upper row j
            h_HK[j + 1][n + j]  = -1.0   # equality lower row n+j
        else                    # perishable: t_j(ξ) x_j = x_j ξ_{m+j}  →  only T_j is nonzero
            T_HK[j + 1][j,         m + j] =  1.0
            T_HK[j + 1][n + j,     m + j] = -1.0
        end
    end
    for i = 1:m
        T_HK[1][2n + i, i] = -d[i]
    end
    # Support set Ξ = { ξ ∈ ℝ^K_+ : ξ_{m+i} ≤ 1, i = 1,…,n }.
    # The demand factors ξ_1..ξ_m have no upper bound (LogNormal samples).
    S_HK = zeros(n, K_HK)
    for i = 1:n;  S_HK[i, m + i] = 1.0;  end
    t_HK = ones(n)
    # First-stage polyhedron X = [0, D]^n  encoded row-by-row as [a_i; b_i]:
    X_HK = Vector{Vector{Float64}}()
    for i = 1:n
        a_lo = zeros(n); a_lo[i] = -1.0;  push!(X_HK, [a_lo; 0.0])     # −x_i ≤ 0
        a_hi = zeros(n); a_hi[i] =  1.0;  push!(X_HK, [a_hi; D])       #  x_i ≤ D
    end
    c_HK = collect(f_x)
    return (c = c_HK, X = X_HK,
            Q = Q_HK, q = q_HK,
            T = T_HK, h = h_HK, W = W_HK,
            S = S_HK, t = t_HK)
end

# function that conducts experiments on the multiproduct production problem
function experiment_production(
        n::Int,              # number of ingredients
        m::Int,              # number of products
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used
        N::Int = NUM_TRAIN,  # number of training samples
        M::Int = NUM_TEST;   # number of testing samples
        D::Float64 = STORAGE_MAX,        # maximum ingredient storage capacity
        f_x::Vector{Float64} = zeros(0), # vector of ingredient costs
        P::Matrix{Float64} = zeros(0,0), # matrix of production coefficients
        r::Vector{Float64} = zeros(0),   # vector of regular product prices
        d::Vector{Float64} = zeros(0),   # vector of standard demands
        σ::Vector{Float64} = zeros(0),   # vector of demand logarithmic variance
        g::Vector{Float64} = zeros(0),   # vector of late ingredient costs
        s::Vector{Float64} = zeros(0),   # vector of maximum salvage prices
        t::Vector{Int} = zeros(Int,0),   # vector of minimum unspoiled percentages
        baseline::String = "none",       # string for baseline methods for comparison:
                                         #   "none"   — none
                                         #   "copos"  — Hanasusanto-Kuhn (2018) copositive formulation
                                         #   "noncvx" — nonconvex global optimization formulation
                                         #   "all"    — both baselines
    )
    baseline in ("none", "copos", "noncvx", "all") || error(
        "baseline must be one of \"none\", \"copos\", \"noncvx\", \"all\"; got \"$baseline\""
    )
    # check if the production coefficients are supplied
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
    # check if the standard demands are supplied
    if length(d) != m
        d = zeros(m)
        for j = 1:m
            d[j] = round(DEMAND_MAX - (DEMAND_MAX-DEMAND_MIN)*(j-1)/(m-1), digits=NUM_DIG)
        end
    end
    # check if the demand logarithmic variances are supplied
    if length(σ) != m
        σ = DEMAND_VAR * ones(m)
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
    sample_train = [round.([exp.(randn(m).*σ);rand(n)],digits=NUM_DIG) for _ in 1:N]
    sample_test = [round.([exp.(randn(m).*σ);rand(n)],digits=NUM_DIG) for _ in 1:M]
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
    C = [zeros(n+1)' -(d.*ξ[1:m])'; zeros(n) -Diagonal(C_t) zeros(n,m)]
    A = [I zeros(n,m); -I zeros(n,m); P' I; zeros(m,n) -I; zeros(m,n) I] .+ 0.0*sum(ξ) # to promote the type
    b = [b_s; -g; r; -r; zeros(m)]
    Ξ = basicsemialgebraicset(FullSpace(), 
                              [[ξ[i] for i in 1:m+n];
                               [1-ξ[i] for i in m+1:m+n];
                               [ξ[i]*(1-ξ[i]) for i in m+1:m+n]
                              ])
    B = [g; r]
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ, B)
    # print the problem information
    println("Start the experiment on the two-stage production problem...")
    println("The number of ingredient is ", n)
    println("The number of products is ", m)
    println("The product prices are ", r)
    println("The ingredient prices are ", f_x)
    println("The ingredient maximum salvage prices are ", s)
    println("The late ingredient prices are ", g)
    println("The number of training samples is ", N)
    println("The number of testing samples is ", M)
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
    # copositive-baseline outputs (defined only if baseline ∈ ("copos","all"))
    if baseline in ("copos", "all")
        COPS_OBJ   = Float64[]
        COPS_TIME  = Float64[]
        COPS_MEAN  = Float64[]
        COPS_STD   = Float64[]
        COPS_MED   = Float64[]
        COPS_Q90   = Float64[]
        COPS_Q10   = Float64[]
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
    # loop over all Wasserstein robustness settings
    for wassinfo in W
        # define the main linear/quadratic optimization problem 
        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
        set_attribute(model, "OutputFlag", 0)
        x = @variable(model, 0 <= x[1:n] <= D, base_name="x")
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
        vec_quant = quantile(vals.+sol.f, [0.1,0.5,0.9])
        append!(TEST_Q10, vec_quant[1])
        append!(TEST_MED, vec_quant[2])
        append!(TEST_Q90, vec_quant[3])
        # ---------------------------------------------------------------
        # Optional: solve the same instance with the H-K (2018) copositive
        # baseline at the same Wasserstein radius and record `COPS_*`.
        if baseline in ("copos", "all")
            # Build paper Eq.(1)+Eq.(3) data for the Hanasusanto-Kuhn (2018)
            # copositive baseline; see `build_copos_baseline_data` for details.
            data_HK = build_copos_baseline_data(n, m, r, s, t, g, d, P, D, f_x)
            println("Built copositive-baseline data: matrix size = ",
                    (m + n) + (4n + 2m + n) + 1,
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
                                           δ = 0.1)
            time_finish_HK = time()
            println("  Copositive baseline status    = ", res_HK.status)
            println("  Copositive baseline x         = ", res_HK.x)
            println("  Copositive baseline objective = ", res_HK.objective_value)
            println("  Copositive baseline time      = ", time_finish_HK - time_start_HK)
            # out-of-sample test on the H-K solution using the same test samples
            _, vals_HK = eval_nominal(recourse, res_HK.x, sample_test, details=true)
            f_HK = f_x' * res_HK.x
            append!(COPS_OBJ,  res_HK.objective_value)
            append!(COPS_TIME, time_finish_HK - time_start_HK)
            append!(COPS_MEAN, mean(vals_HK) + f_HK)
            append!(COPS_STD,  std(vals_HK))
            vec_quant_HK = quantile(vals_HK .+ f_HK, [0.1, 0.5, 0.9])
            append!(COPS_Q10, vec_quant_HK[1])
            append!(COPS_MED, vec_quant_HK[2])
            append!(COPS_Q90, vec_quant_HK[3])
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
                           :TRAIN_TIME => TRAIN_TIME,
                           :TRAIN_OBJ  => TRAIN_OBJ,
                           :TEST_MEAN  => TEST_MEAN,
                           :TEST_STD   => TEST_STD,
                           :TEST_Q10   => TEST_Q10,
                           :TEST_MED   => TEST_MED,
                           :TEST_Q90   => TEST_Q90)
        if baseline in ("copos", "all")
            output.COPS_OBJ  = COPS_OBJ
            output.COPS_TIME = COPS_TIME
            output.COPS_MEAN = COPS_MEAN
            output.COPS_STD  = COPS_STD
            output.COPS_Q10  = COPS_Q10
            output.COPS_MED  = COPS_MED
            output.COPS_Q90  = COPS_Q90
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
    end
end

# run the experiment
experiment_production(NUM_PART, NUM_PROD, WASS_INFO, baseline="all")

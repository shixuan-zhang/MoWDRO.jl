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

using JuMP
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets, Statistics
using DataFrames, CSV
# use commercial solvers for efficiency and numerical stability
using Gurobi, Mosek, MosekTools 
const GRB_ENV = Gurobi.Env()
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_FACILITY = 10
const NUM_SITE = 20
const COST_HOLDING = 1.0
const COST_SUBCONTRACT = 10.0
const MAX_CAPACITY = 100.0

const MIN_AUX = 1.0e-1
const MAX_AUX = 1.0e3
const MIN_PHI = 0.0
const OPT_GAP = 1.0e-2
const NUM_TRAIN = 10 
const NUM_TEST = 10000
const DEG_WASS = 2
const NUM_DIG = 3
const WASS_INFO = [[WassInfo(round(i*5.0e-2,digits=NUM_DIG),DEG_WASS) for i in 0:4];
                  [WassInfo(round(i*1.0e-1,digits=NUM_DIG),DEG_WASS) for i in 3:10]]

OUTPUT_FILE = "../result_allocation_$(NUM_FACILITY)_$(NUM_SITE).csv"
if length(ARGS) > 0
    OUTPUT_FILE = ARGS[1]
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
        n::Int,              # number of facilities
        m::Int,              # number of demand sites
        W::Vector{WassInfo}, # list of Wasserstein robustness settings to be used 
        N::Int = NUM_TRAIN,  # number of training samples
        M::Int = NUM_TEST;   # number of testing samples
        D::Float64 = MAX_CAPACITY,     # maximum facility capacity
        h::Float64 = COST_HOLDING,     # cost for holding inventory
        s::Float64 = COST_SUBCONTRACT, # cost for subcontracted demand
        d::Vector{Float64} = zeros(0)  # distance vector
    )
    # take the samples of random demands
    sample_train = [round.(exp.(randn(m).+1),digits=NUM_DIG) for _ in 1:N]
    sample_test = [round.(exp.(randn(m).+1),digits=NUM_DIG) for _ in 1:M]
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
    println("The number of training samples is ", N)
    println("The number of testing samples is ", M)
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
experiment_allocation(NUM_FACILITY,NUM_SITE,WASS_INFO)

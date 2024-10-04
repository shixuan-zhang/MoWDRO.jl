# Moment relaxations for Wasserstein Distributionally Robust Optimization
module MoWDRO

# import basic mathematics and statistics modules
using LinearAlgebra, Random, Statistics
using DynamicPolynomials, SumOfSquares
using JuMP
using Formatting
# set default solvers
import HiGHS, SDPA, ECOS
const DEFAULT_LP = HiGHS
const DEFAULT_QCP = ECOS
const DEFAULT_SDP = SDPA

# export types and methods for application programming interface
export MainProblem, MainSolution
export SampleSubproblem, SampleLinearRecourse, SamplePolynomialLoss
export solve_master_level, eval_Wass_recourse, eval_nom_recourse

# define module-wide shared parameters
const NUM_DIG = 8
const VAL_TOL = 1.0e-6
const VAL_INF = 1.0e9
const NUM_MAX_ITER = 1000

# define basic types
include("types.jl")
# define the main methods
include("methods.jl")


# include algorithms and relaxations
include("moment_relax.jl")
include("level_bundle.jl")


end

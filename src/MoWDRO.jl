# Moment relaxations for Wasserstein Distributionally Robust Optimization
module MoWDRO

# import basic mathematics and statistics modules
using LinearAlgebra, Random, Statistics
using DynamicPolynomials, MultivariateBases
using SumOfSquares, MomentOpt
using JuMP
using Format
# set default solvers
import HiGHS, CSDP, COSMO
const DEFAULT_LP = optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true)
const DEFAULT_SDP = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => true, "affine" => 1)
#const DEFAULT_SDP = optimizer_with_attributes(COSMO.Optimizer, MOI.Silent() => false)

# export types and methods for application programming interface
export MainProblem, MainSolution, WassInfo
export SampleSubproblem, SampleLinearRecourse, SamplePolynomialLoss
export solve_main_level, eval_nominal, eval_moment_Wass

# define module-wide shared parameters
const NUM_DIG = 8
const VAL_TOL = 1.0e-6
const VAL_INF = 1.0e9
const VAL_INIT_AUX = 1.0e2
const NUM_MAX_ITER = 1000
const MAX_DEG_RELAX = 6

# define basic types
include("types.jl")
# define the main methods
include("methods.jl")

# import alternative certificates
include("certificates.jl")

# include algorithms and relaxations
include("moment_relax.jl")
include("level_bundle.jl")


end

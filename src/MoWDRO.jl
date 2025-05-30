# Moment relaxations for Wasserstein Distributionally Robust Optimization
module MoWDRO

# import basic mathematics and statistics modules
using LinearAlgebra, DynamicPolynomials, SumOfSquares, SemialgebraicSets, MultivariateMoments
using JuMP, PolyJuMP
using Format
# set default solvers
import HiGHS, CSDP
const DEFAULT_LP = HiGHS.Optimizer
const DEFAULT_SDP = CSDP.Optimizer

# export types and methods for application programming interface
export MainProblem, MainSolution, WassInfo
export SampleSubproblem, SampleLinearRecourse, SamplePolynomialLoss
export solve_main_level, eval_nominal, eval_moment_Wass

# define module-wide shared parameters
const NUM_DIG = 6
const VAL_TOL = 1.0e-6
const VAL_INF = 1.0e8
const NUM_MAX_ITER = 1000

# define basic types
include("types.jl")
# define the main methods
include("methods.jl")

# include algorithms and relaxations
include("moment_relax.jl")
include("level_bundle.jl")


end

# Moment relaxations for Wasserstein Distributionally Robust Optimization
module MoWDRO

# import basic mathematics and statistics modules
import LinearAlgebra, Random, Statistics
# import optimization modules
import JuMP, Clp, CSDP

# define module-wide shared parameters
const NUM_DIG = 8
const VAL_TOL = 1.0e-8
const VAL_INF = 1.0e9
const NUM_MAX_ITER = 1000

# define basic types
include("types.jl")

# include algorithms and relaxations
include("linear_relax.jl")
include("level_bundle.jl")

# define the main methods
include("methods.jl")


end

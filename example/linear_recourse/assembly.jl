# numerical example for a two-stage multiproduct assembly problem 
# (see Chapter 1.3.1 in Shapiro-Dentcheva-Ruszczyński(2009)), defined by
# min fₓᵀx + E[F(x,ξ)], x ∈ [0,1]ⁿ, where fₓ ∈ [0,1]ⁿ, and 
# F(x,ξ) := min  -∑ᵢ rᵢ⋅zᵢ - ∑ⱼ sᵢ(ξ)⋅wⱼ
#           s.t. wⱼ = xⱼ - aⱼᵀz, ∀ j,
#                0 ≤ zᵢ ≤ bᵢ(ξ), ∀ i,
#                wⱼ ≥ 0,         ∀ j.
# Here, rᵢ > 0 is the regular price, 
# sᵢ(ξ) ∼ Uniform(0,rᵢ/2) is the random salvage price,
# aⱼ ≥ 0 are the assembly coefficients, a = [a₁,…,aₘ], 
# bᵢ(ξ) ∼ Uniform(0,B) is the random demand.
# By linear duality, we can write F alternatively as
# F(x,ξ) = max  -(xᵀ, b(ξ)ᵀ)⋅y
#          s.t. [I 0; aᵀ I; 0 I] y ≥ [s(ξ); r; 0]


using JuMP, HiGHS
using LinearAlgebra, DynamicPolynomials, SemialgebraicSets
include("../../src/MoWDRO.jl")
using .MoWDRO

# experiment parameters
const NUM_VAR = 5
const NUM_FAC = 10
const DEG_LOSS = 2
const NUM_SAMPLE = 20 
const WASS_INFO = [WassInfo(0.0,2),
                   WassInfo(1e-2,2),
                   WassInfo(2e-2,2),
                   WassInfo(5e-2,2),
                   WassInfo(1e-1,2),
                   WassInfo(2e-1,2),
                   WassInfo(5e-1,2),
                   WassInfo(1.0,2)]

# function that conducts experiments on the multiproduct assembly problem
function experiment_assembly(
        n::Int,
        m::Int,
    )

    # TODO: finish the function implementation
end

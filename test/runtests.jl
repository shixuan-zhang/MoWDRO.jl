include("../src/MoWDRO.jl")
using .MoWDRO
using Test
include("./bundle.jl")

@testset "MoWDRO.jl" begin
    test_level_quadratic()
end

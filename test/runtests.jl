include("../src/MoWDRO.jl")
using .MoWDRO
using Test
include("./bundle.jl")
include("./moment.jl")

@testset "MoWDRO.jl" begin
    #test_level_quadratic()
    test_moment_relax()
end

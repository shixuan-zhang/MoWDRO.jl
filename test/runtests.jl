include("../src/MoWDRO.jl")
using .MoWDRO
using Test
include("./bundle.jl")
include("./moment.jl")
include("./noncvx.jl")

@testset "MoWDRO.jl" begin
    test_level_quadratic()
    test_proximal_quadratic()
    test_moment_polynomial_loss_1()
    test_moment_polynomial_loss_2()
    test_moment_polynomial_loss_3()
    test_moment_polynomial_loss_4()
    test_moment_linear_recourse()
    test_noncvx_polynomial_loss_1()
    test_noncvx_polynomial_loss_2()
    test_noncvx_polynomial_loss_3()
    test_noncvx_polynomial_loss_4()
    test_noncvx_linear_recourse()
end

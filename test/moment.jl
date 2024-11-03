# test the moment relaxation (of Schmüdgen type)

using SumOfSquares, DynamicPolynomials, CSDP
include("../src/certificates.jl")

# define the test functions 

function test_moment_relax()
    @polyvar x[1:2]
    f = -x[1] + x[1]^2
    X = @set x[1] >= 0 && x[2] >= 0 && x[1] <= 1 && x[2] <= 1
    model = SOSModel(CSDP.Optimizer)
    set_silent(model)
    @variable(model, v)
    @objective(model, Min, v)
    ideal_certificate = SOSC.Newton(SOSCone(), MB.MonomialBasis, tuple())
    certificate = Schmüdgen(ideal_certificate, SOSCone(), MB.MonomialBasis, 2)
    @constraint(model, c, f <= v, domain=X, certificate=certificate)
    optimize!(model)
    @test is_solved_and_feasible(model) 
end

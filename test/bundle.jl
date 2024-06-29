# test the bundle methods for the master problem

using Clp

# define the test functions

norm2 = (x) -> (x'*x, 2*x) 

function test_level_quadratic(
        dim::Int = 10
    )
    # define the master model
    model = Model(Clp.Optimizer)
    # define the variables
    x = @variable(model, [1:dim], lower_bound=-1.0, upper_bound=1.0, base_name="x")
    w = @variable(model, lower_bound=0.0, base_name="w")
    ϕ = @variable(model, lower_bound=0.0, base_name="ϕ")
    # define the master linear objective coefficient
    f_x = zeros(dim)
    # run the test
    master = MasterProblem(model, x, VariableRef[], w, ϕ, f_x, Float64[])
    @test MoWDRO.solve_master_level(master, norm2, print=true) ≈ 0.0
end

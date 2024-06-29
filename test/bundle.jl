# test the bundle methods for the master problem

using JuMP, HiGHS

# define the test functions

# function f(x,w) = |x|² - 2⋅w
function test_norm2(z::Vector{Float64})
    x = z[1:end-1]
    w = z[end]
    return x'*x-2*w, [2*x; -2]
end

function test_level_quadratic(
        dim::Int = 10
    )
    # define the master model without solver output
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    # define the variables
    x = @variable(model, [1:dim], lower_bound=-1.0, upper_bound=1.0, base_name="x")
    w = @variable(model, lower_bound=0.0, base_name="w")
    ϕ = @variable(model, lower_bound=0.0, base_name="ϕ")
    # define the master linear objective coefficient
    f_x = zeros(dim)
    # run the test
    master = MasterProblem(model, x, VariableRef[], w, ϕ, f_x, Float64[], 2.0)
    sol = solve_master_level(master, test_norm2)
    @test sol.ϕ ≈ 0.0
end

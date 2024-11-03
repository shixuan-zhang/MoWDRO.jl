# test the bundle methods for the main problem

using DynamicPolynomials, SumOfSquares, SemialgebraicSets
using JuMP, HiGHS

# define the test functions

# define deterministic quadratic loss function F(x,ξ) = |x|²
function define_norm2_loss(n::Int)
    @polyvar x[1:n] ξ[1:n]
    F = x'*x
    ∇ₓF = 2*x
    Ξ = basic_semialgebraic_set(FullSpace(), [1 - ξ'*ξ])
    return SamplePolynomialLoss(x,ξ,F,∇ₓF,Ξ)
end

function test_level_quadratic(
        dim::Int = 10
    )
    # define the main model without solver output
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    # define the variables
    x = @variable(model, [1:dim], lower_bound=-1.0, upper_bound=1.0, base_name="x")
    w = @variable(model, lower_bound=0.0, base_name="w")
    ϕ = @variable(model, lower_bound=0.0, base_name="ϕ")
    # define the main linear objective coefficient and the main problem
    f_x = zeros(dim)
    main = MainProblem(model, x, VariableRef[], w, ϕ, f_x, Float64[])
    # define the polynomial loss function
    loss = define_norm2_loss(dim)
    sample = zeros(dim)
    # run the test
    sol = solve_main_level(main, loss, [sample]) 
    @test sol.ϕ > 0.0 && sol.ϕ < MoWDRO.VAL_TOL
end

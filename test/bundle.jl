# test the bundle methods for the main problem

using DynamicPolynomials, SumOfSquares, SemialgebraicSets
using JuMP, ECOS

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
    model = Model(ECOS.Optimizer)
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
    @test isapprox(sol.ϕ, 0.0, atol = MoWDRO.VAL_TOL)
end

# check the proximal bundle method against the level bundle method by solving
# the same instance with both and comparing the optimal objectives.
function test_proximal_quadratic(
        dim::Int = 10
    )
    # build two identical main models (cuts are added in place, so each method
    # needs its own fresh model)
    function build_main()
        model = Model(ECOS.Optimizer)
        set_silent(model)
        x = @variable(model, [1:dim], lower_bound=-1.0, upper_bound=1.0, base_name="x")
        w = @variable(model, lower_bound=0.0, base_name="w")
        ϕ = @variable(model, lower_bound=0.0, base_name="ϕ")
        return MainProblem(model, x, VariableRef[], w, ϕ, zeros(dim), Float64[])
    end
    loss = define_norm2_loss(dim)
    sample = zeros(dim)
    # baseline: level bundle method
    sol_level = solve_main_level(build_main(), loss, [sample]; print=-1)
    # proximal bundle method
    sol_prox  = solve_main_proximal(build_main(), loss, [sample]; print=-1)
    # the proximal bundle method should reach the same optimum (ϕ ≈ 0)
    @test isapprox(sol_prox.ϕ, 0.0, atol = MoWDRO.VAL_TOL)
    # ... and agree with the level bundle method on both objective components
    @test isapprox(sol_prox.f, sol_level.f, atol = MoWDRO.VAL_TOL)
    @test isapprox(sol_prox.ϕ, sol_level.ϕ, atol = MoWDRO.VAL_TOL)
end

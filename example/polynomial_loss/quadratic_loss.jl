# numerical example for nonconvex quadratic loss functions of the form
# g(x,ξ) := x₁⋅ξ₁ξ₂ + ⋯ + xₙ⋅ξₙξ₁, ξᵢ ∈ [0,1] with uniform distributions



# function that conducts the experiment on the quadratic loss functions
function experiment_quad_loss(
        n::Int;
        num_rep::Int = 1,
        Wass_rad::Float64 = 0.0
    )
    # define the master linear optimization problem
    model = Model(HiGHS.Optimizer)
    x = @variable(model, [1:n]
end

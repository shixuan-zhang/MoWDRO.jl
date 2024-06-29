# numerical example for nonconvex quadratic loss functions of the form
# g(x,ξ) := x₁⋅ξ₁ξ₂ + ⋯ + xₙ⋅ξₙξ₁



# function that conducts the experiment on the quadratic loss functions
function experiment_quad_loss(
        dim::Int;
        num_rep::Int = 1,
        vec_rad::Vector{Float64} = 0.1*collect(1:10)
    )
    # define the master linear optimization problem
    model = Model(HiGHS.Optimizer)
    # TODO: add the variables and bounds
end

## Basic methods for data types conversion and mathematical functions

# build a linear cut from the current state variable and solution information
function build_linear_cut(
        x::Vector{Float64},
        ∇ϕ::Vector{Float64},
        ϕ::Float64
    )
    return LinearCut(∇ϕ, ϕ-∇ϕ'*x)
end

# evaluate the Wasserstein DRO recourse problem with given augmented state (x,w)
function eval_Wass_recourse(
        recourse::RecourseProblem,
        x_w::Vector{Float64}
    )
    # set the linear objective expression
    @objective(recourse.model, Max, [1,x_w]'*recourse.Ψ)
    # solve the problem
    optimize!(recourse.model)
    # check if the solution exists
    if !is_solved_and_feasible(recourse.model)
        println(" The recourse evaluation failed!")
    end
    ϕ = objective_value(recourse.model)
    ∇ϕ = [1,x_w]'*value.(recourse.Ψ)
    # return the cut and the recourse model
    return build_linear_cut(x_w, ∇ϕ, ϕ), recourse
end


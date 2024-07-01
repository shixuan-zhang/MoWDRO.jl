## Basic methods for data types conversion and mathematical functions

# build a linear cut from the current state variable and solution information
function build_linear_cut(
        x::Vector{Float64},
        ∇ϕ::Vector{Float64},
        ϕ::Float64
    )
    return LinearCut(∇ϕ, ϕ-∇ϕ'*x)
end

# combine linear cuts from different uncertainty outcomes/realizations
function combine_linear_cuts(
        cuts::Vector{LinearCut};
        prob::Vector{Float64} = 0.0
    )
    # check if the probability vector is properly supplied
    N = length(cuts)
    if length(prob) != N
        prob = ones(N) ./ N
    end
    # aggregate the cuts
    a = sum([cuts[i].a .* prob[i] for i in 1:N]) ./ N
    b = sum([cuts[i].b * prob[i] for i in 1:N]) ./ N
    return LinearCut(a, b)
end

# evaluate the Wasserstein DRO recourse problems with given augmented state (x,w)
function eval_Wass_recourse(
        recourse::Vector{RecourseProblem},
        x_w::Vector{Float64}
    )
    # prepare the array of linear cuts
    N = length(recourse)
    cuts = LinearCut[]
    # loop over the outcomes/realizations of ξ
    for i in 1:N
        # set the linear objective expression
        @objective(recourse[i].model, Max, [1,x_w]'*recourse[i].Ψ)
        # solve the problem
        optimize!(recourse[i].model)
        # check if the solution exists
        if !is_solved_and_feasible(recourse[i].model)
            println(" The recourse evaluation failed!")
        end
        ϕ = objective_value(recourse[i].model)
        ∇ϕ = [1,x_w]'*value.(recourse[i].Ψ)
        # store the generated linear cut
        push!(cuts, build_linear_cut(x_w, ∇ϕ, ϕ))
    end
    # return the aggregate cut to the master problem
    return combine_linear_cuts(cuts)
end

# evaluate the nominal SO recourse problems with given state x
function eval_nom_recourse(
        recourse::RecourseProblem,
        x::Vector{Float64}
    )
    
end

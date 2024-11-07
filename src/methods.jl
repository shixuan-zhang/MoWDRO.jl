## Basic methods for data types conversion and mathematical functions

# combine linear cuts of the form:
# cut'*[1;x] = cut[1] + cut[2:end]'*x
# from different uncertainty outcomes/realizations
function combine_linear_cuts(
        cuts::Vector{Vector{Float64}};
        prob::Vector{Float64} = Float64[]
    )
    # check if the probability vector is properly supplied
    N = length(cuts)
    if length(prob) != N
        prob = ones(N) ./ N
    end
    # aggregate the cuts
    return sum([cuts[i] .* prob[i] for i in 1:N]) ./ N
end


# evaluate the nominal polynomial loss function with the given state
function eval_nominal(
        loss::SamplePolynomialLoss,
        state::Vector{Float64},
        samples::Vector{Vector{Float64}}
    )
    N = length(samples)
    cuts = Vector{Float64}[]
    # alias the state vector
    x̄ = state
    # loop over all samples for the loss function evaluation
    for i = 1:N # TODO: parallelize this for-loop
        ξ̂ = samples[i]
        v̂ = convert(Float64,subs(loss.F,loss.x=>x̄,loss.ξ=>ξ̂))
        ĝ = convert.(Float64,subs.(loss.∇ₓF,loss.x=>x̄,loss.ξ=>ξ̂))
        # store the cut
        push!(cuts, [v̂-ĝ'*x̄;ĝ])
    end
    # return the aggregate cut
    return combine_linear_cuts(cuts)
end

# evaluate the nominal stochastic linear recourse problems with given state x
function eval_nominal(
        recourse::SampleLinearRecourse,
        state::Vector{Float64},
        samples::Vector{Vector{Float64}}
    )
    N = length(samples)
    cuts = Vector{Float64}[]
    # loop over all samples for the linear recourse problem
    for i = 1:N # TODO: parallelize this for-loop
        ξ̂ = samples[i]
        # retrieve the linear objective function at the sample
        C = convert.(Float64,subs.(recourse.C,recourse.ξ=>ξ̂))
        c = C'*[1;state]
        # retrieve the linear constraints at the sample
        A = convert.(Float64,subs.(recourse.A,recourse.ξ=>ξ̂))
        b = convert.(Float64,subs.(recourse.b,recourse.ξ=>ξ̂))
        # formulate a JuMP linear optimization model
        M = JuMP.Model(DEFAULT_LP)
        # define the recourse variables and constraints
        @variable(M, y[1:length(c)-1])
        @constraint(M, A*y - b >= 0)
        @objective(M, Max, c'*[1;y])
        # solve the recourse model
        optimize!(M)
        if is_solved_and_feasible(M)
            # retrieve the recourse solutions
            ȳ = value.(y)
            # store the cut
            push!(cuts, C*[1;ȳ])
        else
            error("The nominal evaluation has failed with status: ", termination_status(M))
        end
    end
    # return the aggregate cut
    return combine_linear_cuts(cuts)
end

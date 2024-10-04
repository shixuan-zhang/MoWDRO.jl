# Moment relaxation for the loss and recourse functions

# evaluate the moment relaxation for Wasserstein distributionally
# robust linear recourse problem with given augmented state (x,w)
function eval_moment_Wass(
        recourse::SampleLinearRecourse,
        samples::Vector{Vector{Float64}},
        augstate::Vector{Float64} 
    )
    N = length(recourse)
    cuts = Vector{Float64}[]
    # alias the augmented state
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    # loop over all samples for the linear recourse moment relaxation
    for i in 1:N
        # TODO: complete the SOSModel here
    end
    # return the aggregate cut to the main problem
    return combine_linear_cuts(cuts)
end


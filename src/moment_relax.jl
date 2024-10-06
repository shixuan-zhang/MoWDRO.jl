# Moment relaxation for the loss and recourse functions

# evaluate the moment relaxation for the Wasserstein distributionally
# robust polynomial loss function with a given augmented state (x,w)
function eval_moment_Wass(
        loss::SamplePolynomialLoss,
        samples::Vector{Vector{Float64}},
        augstate::Vector{Float64},
        wassinfo::WassInfo;
        relaxdeg::Int = 0
    )
    N = length(recourse)
    cuts = Vector{Float64}[]
    # set the loss function at the given state
    f = subs(loss.F,loss.x=>x̄)
    # set the default relaxation degree
    if relaxdeg <= 0
        relax_deg = max(maxdegree(f),wassinfo.p)
    end
    # alias the augmented state
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    # loop over all samples for the linear recourse moment relaxation
    for i = 1:N # TODO: parallelize this for-loop
        ξ̂ = samples[i]
        d = length(ξ̂)
        # define the polynomial objective
        p = sum((loss.ξ[j]-ξ̂[j])^wassinfo.p for j=1:d)
        # define the SOS optimization model
        model = SOSModel(DEFAULT_SDP.Optimizer)
        @variable(model, optval)
        @objective(model, Min, optval)
        @constraint(model, constr, f-w̄*p <= optval, domain=loss.Ξ)
        # solve the SOS model and extract the (pseudo-)moments/measure
        optimize!(model)
        μ = moments(constr)
        # retrieve the pseudo-expectations for the polynomials
        v̂ = expectation(μ,f)
        p̂ = expectation(μ,p)
        ĝ = map(m->expectation(μ,m), loss.∇ₓF)
        # store the cut
        push!(cuts, [v̂-ĝ'*x̄;ĝ;wassinfo.r-p̂])
    end
    # return the aggregate cut
    return combine_linear_cuts(cuts)
end

# evaluate the moment relaxation for the Wasserstein distributionally
# robust linear recourse problem with a given augmented state (x,w)
function eval_moment_Wass(
        recourse::SampleLinearRecourse,
        samples::Vector{Vector{Float64}},
        augstate::Vector{Float64},
        wassinfo::WassInfo;
        relaxdeg::Int = 0
    )
    N = length(recourse)
    cuts = Vector{Float64}[]
    # set the default relaxation degree
    if relaxdeg <= 0
        relax_deg = 4 # TODO: change it to be consistent with degrees of A, b, C, and Ξ
    end
    # alias the augmented state
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    # loop over all samples for the linear recourse moment relaxation
    for i = 1:N # TODO: parallelize this for-loop
        ξ̂ = samples[i]
        d = length(ξ̂)
        # define the polynomial objective 
        p = sum((recourse.ξ[j]-ξ̂[j])^wassinfo.p for j=1:d)
        f = [1;x̄]'*recourse.C*[1;recourse.y] - w̄*p
        # define the semi-algebraic set
        S = intersection(recourse.Ξ, basic_semialgebraic_set(FullSpace(), recourse.A*recourse.y-recourse.b))
        # define the SOS optimization model
        model = SOSModel(DEFAULT_SDP.Optimizer)
        @variable(model, optval)
        @objective(model, Min, optval)
        @constraint(model, constr, optval >= f, domain=S, maxdegree=relaxdeg)
        # solve the SOS model and extract the (pseudo-)moments/measure
        optimize!(model)
        μ = moments(constr)
        # retrieve the pseudo-expectations for the polynomials
        Ĉ = map(m->expectation(μ,m), recourse.C)
        ŷ = map(m->expectation(μ,m), recourse.y)
        p̂ = map(m->expectation(μ,m), p)
        # store the cut
        push!(cuts, [Ĉ*[1;ŷ];wassinfo.r-p̂])
    end
    # return the aggregate cut to the main problem
    return combine_linear_cuts(cuts)
end


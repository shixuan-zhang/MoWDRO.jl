# Moment relaxation for the loss and recourse functions

# evaluate the moment relaxation for the Wasserstein distributionally
# robust polynomial loss function with a given augmented state (x,w)
function eval_moment_Wass(
        loss::SamplePolynomialLoss,
        augstate::Vector{Float64},
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo;
        relaxdeg::Int = 0
    )
    N = length(samples)
    cuts = Vector{Float64}[]
    # alias the augmented state
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    # set the loss function at the given state
    f = subs(loss.F, loss.x=>x̄)
    # set the default relaxation degree
    if relaxdeg <= 0
        relaxdeg = max(maxdegree(f),wassinfo.p)
    end
    # loop over all samples for the loss function moment relaxation
    for i = 1:N # TODO: parallelize this for-loop
        ξ̂ = samples[i]
        d = length(ξ̂)
        # define the polynomial objective
        p = sum((loss.ξ[j]-ξ̂[j])^wassinfo.p for j=1:d)
        # define the moment optimization model
        model = GMPModel(DEFAULT_SDP)
        @variable(model, μ, Meas(loss.ξ, support=loss.Ξ))
        @objective(model, Max, Mom(f-w̄*p,μ))
        @constraint(model, Mom(1,μ) == 1)
        set_approximation_mode(model, PRIMAL_RELAXATION_MODE())
        set_approximation_degree(model, relaxdeg)
        # solve the moment model and extract the (pseudo-)moments/measure
        optimize!(model)
        if termination_status(model) ∈ [OPTIMAL, ALMOST_OPTIMAL]
            μ̄ = measure(μ)
            # retrieve the pseudo-expectations for the polynomials
            v̂ = expectation(μ̄,f)
            p̂ = expectation(μ̄,p)
            ĝ = map(m->expectation(μ̄,m), subs.(loss.∇ₓF,loss.x=>x̄))
            # store the cut
            push!(cuts, [v̂-ĝ'*x̄;ĝ;wassinfo.r-p̂])
        else
            error("The moment relaxation has failed with status: ", termination_status(model))
        end
    end
    # return the aggregate cut
    return combine_linear_cuts(cuts)
end

# evaluate the moment relaxation for the Wasserstein distributionally
# robust linear recourse problem with a given augmented state (x,w)
function eval_moment_Wass(
        recourse::SampleLinearRecourse,
        augstate::Vector{Float64},
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo;
        relaxdeg::Int = 0,
        flag_rad_prod = false,
        val_add_bound = -1.0
    )
    N = length(samples)
    cuts = Vector{Float64}[]
    # set the default relaxation degree
    if relaxdeg <= 0
        deg_A = maximum(maxdegree.(recourse.A))
        deg_C = maximum(maxdegree.(recourse.C)) + 1
        deg_b = maximum(maxdegree.(recourse.b))
        deg_Ξ = 0
        if recourse.Ξ.V != FullSpace()
            deg_Ξ = maximum([maxdegree.(recourse.Ξ.p);maxdegree(recourse.Ξ.V.I.p)])
        else
            deg_Ξ = maximum(maxdegree.(recourse.Ξ.p))
        end
        relaxdeg = maximum([deg_A+2, deg_b+2, deg_C, deg_Ξ, wassinfo.p])
    end
    # alias the augmented state
    x̄ = augstate[1:end-1]
    w̄ = augstate[end]
    # loop over all samples for the linear recourse moment relaxation
    for i = 1:N # TODO: parallelize this for-loop
        ξ̂ = samples[i]
        d = length(ξ̂)
        # define the polynomial objective 
        p = ((recourse.ξ-ξ̂)'*(recourse.ξ-ξ̂))^ceil(Int,wassinfo.p/2)
        f = [1;x̄]'*recourse.C*[1;recourse.y] - w̄*p
        # define the semi-algebraic set
        S = intersect(recourse.Ξ, basic_semialgebraic_set(FullSpace(), recourse.A*recourse.y-recourse.b))
        if flag_rad_prod
            R2 = (wassinfo.r^wassinfo.p*N)^(2/wassinfo.p)
            p2 = (recourse.ξ-ξ̂)'*(recourse.ξ-ξ̂)
            m = length(recourse.b)
            S = intersect(S, basic_semialgebraic_set(FullSpace(), [(recourse.A*recourse.y-recourse.b)[i]*(R2-p2) for i in 1:m]))
        end
        if val_add_bound > 0.0
            B = val_add_bound
            n = length(recourse.y)
            S = intersect(S, basic_semialgebraic_set(FullSpace(), [B^2-recourse.y[i]^2 for i in 1:n]))
        end
        # define the moment optimization model
        model = GMPModel(DEFAULT_SDP)
        @variable(model, μ, Meas([recourse.ξ; recourse.y], support=S))
        @objective(model, Max, Mom(f,μ))
        @constraint(model, Mom(1,μ) == 1)
        set_approximation_mode(model, PRIMAL_RELAXATION_MODE())
        set_approximation_degree(model, relaxdeg)
        # solve the moment model and extract the (pseudo-)moments/measure
        optimize!(model)
        if termination_status(model) ∈ [OPTIMAL, ALMOST_OPTIMAL]
            μ̄ = measure(μ)
            # retrieve the pseudo-expectations for the polynomials
            Ĉ = map(m->expectation(μ̄,m), recourse.C)
            ŷ = map(m->expectation(μ̄,m), recourse.y)
            p̂ = expectation(μ̄,p)
            # store the cut
            push!(cuts, [Ĉ*[1;ŷ];wassinfo.r-p̂])
        else
            println("DEBUG: the moment relaxation degree is\n", relaxdeg)
            println("DEBUG: the moment relaxation domain is\n", S)
            println("DEBUG: the moment relaxation objective is\n", f)
            println("DEBUG: the moment relaxation model is\n", model)
            error("The moment relaxation has failed with status: ", termination_status(model))
        end
    end
    # return the aggregate cut to the main problem
    return combine_linear_cuts(cuts)
end


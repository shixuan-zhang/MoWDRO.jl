# Moment relaxation for the loss and recourse functions

# evaluate the moment relaxation for the Wasserstein distributionally
# robust polynomial loss function with a given augmented state (x,w)
function eval_moment_Wass(
        loss::SamplePolynomialLoss,
        augstate::Vector{Float64},
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo;
        print::Int = 0,
        relaxdeg::Int = 0,
        mom_solver = DEFAULT_SDP,
        val_relax_tol::Float64 = VAL_TOL
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
        # define the SOS optimization model
        model = SOSModel(mom_solver)
        if print < 1
            set_silent(model)
        end
        @variable(model, optval)
        @objective(model, Min, optval)
        @constraint(model, constr, f-w̄*p <= optval, domain=loss.Ξ, maxdegree=relaxdeg)
        # solve the SOS model and extract the (pseudo-)moments/measure
        optimize!(model)
        # retrieve the pseudo-expectations for the polynomials
        if is_solved_and_feasible(model, allow_almost=true)
            μ̄ = moments(constr)
            v̂ = expectation(μ̄,f)
            p̂ = expectation(μ̄,p)
            ĝ = map(m->expectation(μ̄,m), subs.(loss.∇ₓF,loss.x=>x̄))
            # store the cut
            push!(cuts, [v̂-ĝ'*x̄;ĝ;wassinfo.r^wassinfo.p-p̂])
        elseif termination_status(model) == SLOW_PROGRESS
            if print > 0
                println("DEBUG: slow progress reported by the solver...")
            end
            μ̄ = moments(constr)
            v̂ = expectation(μ̄,f)
            p̂ = expectation(μ̄,p)
            ĝ = map(m->expectation(μ̄,m), subs.(loss.∇ₓF,loss.x=>x̄))
            v̄ = objective_value(model)
            v̂ = v̂-w̄*p̂
            if abs(v̄-v̂) / (1.0+max(abs(v̄),abs(v̂))) > val_relax_tol
                if print > 0
                    println("DEBUG: The loss function evaluation error is ", v̄-v̂)
                end
            end
            push!(cuts, [v̂-ĝ'*x̄;ĝ;wassinfo.r^wassinfo.p-p̂])
        else 
            println("DEBUG: the moment relaxation degree is ", relaxdeg)
            println("DEBUG: the moment relaxation domain is\n", loss.Ξ)
            println("DEBUG: the moment relaxation objective is\n", f-w̄*p)
            println("DEBUG: the current main problem solution is\n", x̄)
            println("DEBUG: the current Wasserstein auxiliary variable is ", w̄)
            println("DEBUG: the moment relaxation model is\n", model)
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
        print::Int = 0,
        relaxdeg::Int = 0,
        flag_rad_prod::Bool = false,
        flag_lin_prod::Bool = false,
        val_add_bound::Float64 = -1.0,
        mom_solver = DEFAULT_SDP,
        val_relax_tol::Float64 = VAL_TOL
    )
    N = length(samples)
    cuts = Vector{Float64}[]
    # set the default relaxation degree
    if relaxdeg <= 0
        deg_A = maximum(maxdegree.(recourse.A))
        deg_C = maximum(maxdegree.(recourse.C))
        deg_b = maximum(maxdegree.(recourse.b))
        deg_Ξ = 0
        if recourse.Ξ.V != FullSpace()
            deg_Ξ = maximum([maxdegree.(recourse.Ξ.p);maxdegree(recourse.Ξ.V.I.p)])
        else
            deg_Ξ = maximum(maxdegree.(recourse.Ξ.p))
        end
        relaxdeg = maximum([deg_A,deg_b,deg_C,deg_Ξ,wassinfo.p])
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
        Y = basicsemialgebraicset(FullSpace(), recourse.A*recourse.y-recourse.b)
        if minimum(recourse.B) > 0.0
            m = length(recourse.y)
            Y = intersect(Y, basicsemialgebraicset(FullSpace(), [recourse.B[i]^2-recourse.y[i]^2 for i in 1:m]))
        end
        S = intersect(recourse.Ξ, Y)
        if flag_rad_prod
            R2 = (wassinfo.r^wassinfo.p*N)^(2/wassinfo.p)
            p2 = (recourse.ξ-ξ̂)'*(recourse.ξ-ξ̂)
            m = length(recourse.b)
            S = intersect(S, basicsemialgebraicset(FullSpace(), [(recourse.A*recourse.y-recourse.b)[i]*(R2-p2) for i in 1:m]))
        end
        if flag_lin_prod
            m = length(recourse.b)
            Y = recourse.A*recourse.y-recourse.b
            S = intersect(S, basicsemialgebraicset(FullSpace(), [Y[i]*Y[j] for i in 1:m for j in i:m]))
        end
        if val_add_bound > 0.0
            B = val_add_bound
            n = length(recourse.y)
            S = intersect(S, basicsemialgebraicset(FullSpace(), [B^2-recourse.y[i]^2 for i in 1:n]))
        end
        # define the SOS optimization model
        model = SOSModel(mom_solver)
        if print < 1
            set_silent(model)
        end
        @variable(model, optval)
        @objective(model, Min, optval)
        @constraint(model, constr, optval >= f, domain=S, maxdegree=relaxdeg)
        # solve the SOS model and extract the (pseudo-)moments/measure
        optimize!(model)
        if is_solved_and_feasible(model, allow_almost=true)
            μ̄ = moments(constr)
            # retrieve the pseudo-expectations for the polynomials
            ĉ = map(m->expectation(μ̄,m), recourse.C*[1;recourse.y])
            p̂ = expectation(μ̄,p)
            # store the cut
            push!(cuts, [ĉ;wassinfo.r^wassinfo.p-p̂])
        elseif termination_status(model) == SLOW_PROGRESS
            if print > 0
                println("DEBUG: slow progress reported by the solver...")
            end
            μ̄ = moments(constr)
            # retrieve the pseudo-expectations for the polynomials
            ĉ = map(m->expectation(μ̄,m), recourse.C*[1;recourse.y])
            p̂ = expectation(μ̄,p)
            # check if the objective values agree
            v̄ = objective_value(model)
            v̂ = [1;x̄]'*ĉ-w̄*p̂
            if abs(v̄-v̂) / (1.0+max(abs(v̄),abs(v̂))) > val_relax_tol
                if print > 0
                    println("DEBUG: The recourse evaluation error is ", v̄-v̂)
                end
            end
            push!(cuts, [ĉ;wassinfo.r^wassinfo.p-p̂])
        else
            println("DEBUG: the moment relaxation degree is ", relaxdeg)
            println("DEBUG: the moment relaxation domain is\n", S)
            println("DEBUG: the moment relaxation objective is\n", f)
            println("DEBUG: the current main problem solution is\n", x̄)
            println("DEBUG: the current Wasserstein auxiliary variable is ", w̄)
            println("DEBUG: the moment relaxation model is\n", model)
            error("The moment relaxation has failed with status: ", termination_status(model))
        end
    end
    # return the aggregate cut to the main problem
    return combine_linear_cuts(cuts)
end


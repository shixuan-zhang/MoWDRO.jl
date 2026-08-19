# Bundle methods for solving the main problem of the form
# min  f_x'⋅x + f_u'⋅u + ϕ(x,w)
# s.t. (x,u) ∈ Feasible Set, w ≥ 0.

# default parameters for the level bundle method
const DEFAULT_LEVEL = 1/(2+sqrt(2))

# default parameters for the proximal bundle method (Kiwiel, Math. Prog. 1990)
const DEFAULT_INIT_WEIGHT   = 1.0
const DEFAULT_MIN_WEIGHT    = 1.0e-8
const DEFAULT_SERIOUS_RATIO = 0.1
const DEFAULT_TIGHT_RATIO   = 0.5
const DEFAULT_WEIGHT_UPDATE = 2
const WEIGHT_UPDATE_FEAS    = 10
const WEIGHT_INCREASE_THRES = 10

# helper function for the level bundle method which finds a feasible w 
# through bisection and returns the cut together with the updated w
function bisection_feas_cut(
        subproblem::T,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo,
        sol_x::Vector{Float64},
        sol_w::Float64,
        eval_cut::Function;
        max_aux::Float64 = VAL_INF,
        min_aux::Float64 = 0.0,
        coef_max::Float64 = VAL_INF,
        feas_tol::Float64 = VAL_TOL,
        flag_safe::Bool = true,
        print::Int = 0
    ) where T <: SampleSubproblem
    if feas_tol <= 0.0
        error("Invalid bisection tolerance supplied in Wasserstein dual variable adjustment.")
    end
    # initialization
    w_lb = max(sol_w, min_aux)
    w_ub = max_aux
    w_temp = (w_lb + w_ub) / 2
    w_best = w_temp
    cut_temp = eval_cut(subproblem, [sol_x;w_temp], samples, wassinfo, print=print-1)
    cut_best = cut_temp
    # bisection in w
    while w_ub - w_lb > feas_tol
        if isnothing(cut_temp) || maximum(abs.(cut_temp)) > coef_max
            w_lb = w_temp
        else
            cut_best = cut_temp
            w_best = w_temp
            w_ub = w_temp
        end
        w_temp = (w_lb + w_ub) / 2
        cut_temp = eval_cut(subproblem, [sol_x;w_temp], samples, wassinfo, print=print-1)
    end
    if flag_safe
        w_best = min(w_best+feas_tol, max_aux)
        cut_best = eval_cut(subproblem, [sol_x;w_best], samples, wassinfo, print=print-1)
    end
    if isnothing(cut_best) || maximum(abs.(cut_best)) > coef_max
        error("The moment relaxation is numerically infeasible for any Wasserstein dual.")
    end
    if print >= 0
        println("  Adjusting Wasserstein auxiliary variable from ", sol_w, " to ", w_best, " for cut generation.")
    end
    return cut_best, w_best
end

# function implementing a level bundle method to solve the main problem
function solve_main_level(
        main::MainProblem,
        subproblem::T,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo = WassInfo(.0,2);
        max_iter::Int = NUM_MAX_ITER,
        opt_gap::Float64 = VAL_TOL,
        max_aux::Float64 = VAL_INF,
        min_aux::Float64 = 0.0,
        min_phi::Float64 = -VAL_INF,
        max_cut_coef::Float64 = VAL_INF, # for numerical stability
        tol_aux_feas::Float64 = VAL_TOL, # for numerical stability
        level::Float64 = DEFAULT_LEVEL,
        mom_solver = DEFAULT_SDP,
        cut_evaluator = nothing,
        print::Int = 1
    )::MainSolution where T <: SampleSubproblem
    # `cut_evaluator` lets the caller swap the inner-supremum solver used to
    # generate cuts for `MainProblem.ϕ`. When `nothing`, fall back to the
    # moment-relaxation evaluator with the supplied `mom_solver` (default).
    eval_cut = isnothing(cut_evaluator) ?
        (subproblem, augstate, samples, wassinfo; print=0) ->
            eval_moment_Wass(subproblem, augstate, samples, wassinfo;
                             mom_solver=mom_solver, print=print) :
        cut_evaluator
    # check if Wasserstein ambiguity is needed
    flag_Wass = false
    if wassinfo.r > VAL_TOL
        flag_Wass = true
    end
    # get the state dimension
    dim_x = length(main.x)
    # set the objective expression
    obj = main.f_x'*main.x + main.f_u'*main.u + main.ϕ 
    # add the artificial bound on the Wasserstein auxiliary variable w 
    set_upper_bound(main.w, max_aux)
    if min_aux > 0.0
        set_lower_bound(main.w, min_aux)
    end
    # add the artificial bound on the recourse auxiliary variable ϕ
    set_lower_bound(main.ϕ, min_phi)
    # get the initial solution and lower bound
    @objective(main.model, Min, obj)
    optimize!(main.model)
    sol_x = round.(value.(main.x),digits=NUM_DIG)
    sol_u = round.(value.(main.u),digits=NUM_DIG)
    min_obj = objective_value(main.model)
    val_f = round(main.f_x'*sol_x + main.f_u'*sol_u,digits=NUM_DIG)
    # retrieve the Wasserstein dual variable if present
    sol_w = 0.0
    if flag_Wass
        sol_w = round(value(main.w),digits=NUM_DIG)
    end
    # print the starting message
    if print >= 0
        println(" Start the level bundle method for the main problem...")
        println(" The initial Wasserstein dual variable = ", sol_w)
    end
    # get the initial upper bound
    cut = zeros(dim_x+2)
    if flag_Wass
        cut = eval_cut(subproblem, [sol_x;sol_w], samples, wassinfo, print=print-1)
        if isnothing(cut) || maximum(abs.(cut)) > max_cut_coef # the moment relaxation is unbounded/infeasible
            cut, sol_w = bisection_feas_cut(subproblem, samples, wassinfo, sol_x, sol_w, eval_cut, 
                                            max_aux=max_aux, min_aux=min_aux, feas_tol=tol_aux_feas, 
                                            coef_max=max_cut_coef, print=print)
        end
    else
        cut[1:dim_x+1] = eval_nominal(subproblem, sol_x, samples)
    end
    # round the cut coefficient to avoid numerical issues
    cut = round.(cut,digits=NUM_DIG)
    val_ϕ = cut'*[1;sol_x;sol_w]
    max_obj = val_ϕ + val_f
    # initialize the return values
    opt_x = sol_x
    opt_u = sol_u
    opt_f = val_f
    opt_ϕ = val_ϕ
    # print the initial bounds
    if print >= 0
        println(" The initial lower bound = ", min_obj)
        println(" The initial upper bound = ", max_obj)
    end
    iter = 1
    # loop until the bounds are close (in either the absolute or the relative sense)
    while (max_obj - min_obj) / max(1, abs(min_obj)) > opt_gap
        # update the loss/recourse approximation
        @constraint(main.model, main.ϕ >= cut'*[1;main.x;main.w])
        # get an updated lower bound
        optimize!(main.model)
        if termination_status(main.model) != OPTIMAL && !has_values(main.model)
            if print >= 0
                println("DEBUG: the level bounding step runs into issues...\n", 
                        solution_summary(main.model,verbose=true))
                println("DEBUG: the current level bounding step problem x = ", sol_x)
                if flag_Wass
                    println("DEBUG: the current level bounding step problem w = ", sol_w)
                end
                println("DEBUG: the current level bounding step model is \n", main.model)
            end
            error("The level method bounding step has failed with status: ", termination_status(main.model))
        end
        min_obj = objective_value(main.model)
        if (max_obj - min_obj) / max(1, abs(min_obj)) <= opt_gap
            if print >= 0
                printfmtln(" The level method has converged with the updated lower bound {}.", min_obj)
            end
            break
        end
        # calculate the level
        val_lev = round(level*max_obj + (1-level)*min_obj,digits=NUM_DIG)
        # build the projection model
        con_proj = @constraint(main.model, obj <= val_lev)
        obj_proj = (main.x-sol_x)'*(main.x-sol_x)
        if flag_Wass
            obj_proj += (main.w-sol_w)^2 / sqrt(max_aux)
        end
        @objective(main.model, Min, obj_proj)
        # find the next iterate
        optimize!(main.model)
        if termination_status(main.model) != OPTIMAL && !has_values(main.model)
            if print >= 0
                println("DEBUG: the level projection step runs into issues...\n", 
                        solution_summary(main.model,verbose=true))
                println("DEBUG: the current level projection step problem x = ", sol_x)
                if flag_Wass
                    println("DEBUG: the current level projection step problem w = ", sol_w)
                end
                println("DEBUG: the current level projection step problem model is \n", main.model)
            end
            error("The level method projection step has failed with status: ", termination_status(main.model))
        end
        sol_x = round.(value.(main.x),digits=NUM_DIG)
        sol_u = round.(value.(main.u),digits=NUM_DIG)
        sol_w = 0.0
        if flag_Wass
            sol_w = round(value(main.w),digits=NUM_DIG)
        end
        val_f = main.f_x'*sol_x + main.f_u'*sol_u
        # get an updated upper bound
        cut = zeros(dim_x+2)
        if flag_Wass
            cut = eval_cut(subproblem, [sol_x;sol_w], samples, wassinfo, print=print-1)
            if isnothing(cut) || maximum(abs.(cut)) > max_cut_coef # the moment relaxation is unbounded/infeasible
                cut, sol_w = bisection_feas_cut(subproblem, samples, wassinfo, sol_x, sol_w, eval_cut, 
                                                max_aux=max_aux, min_aux=min_aux, feas_tol=tol_aux_feas,
                                                coef_max=max_cut_coef, print=print)
            end
        else
            cut[1:dim_x+1] = eval_nominal(subproblem, sol_x, samples)
        end
        # round the cut coefficient to avoid numerical issues
        cut = round.(cut, digits=NUM_DIG)
        # check if a better solution is encountered
        val_ϕ = cut'*[1;sol_x;sol_w]
        if val_ϕ + val_f < max_obj
            max_obj = val_ϕ + val_f
            opt_x = sol_x
            opt_u = sol_u
            opt_f = val_f
            opt_ϕ = val_ϕ
        end
        # restore the optimization model
        delete(main.model, con_proj)
        @objective(main.model, Min, obj)
        # print the update if needed
        if print >= 0
            printfmtln(" Iteration {}: current objective = {:<6.2e}, upper bound = {:<6.2e}, lower bound = {:<6.2e}",
                       iter, val_ϕ+val_f, max_obj, min_obj)
            if print >= 1
                println("  The current feasible x = ", sol_x)
                println("  The current Wasserstein dual variable = ", sol_w)
            end
        end
        iter += 1
        # check if maximum number of iteration is reached
        if iter > max_iter
            if print >= 0
                printfmtln(" The level bundle method does not converge within {} iterations", max_iter)
            end
            return MainSolution(opt_x, opt_u, opt_f, opt_ϕ)
        end
    end
    if max_obj - min_obj < -opt_gap
        if print >= 0
            println("DEBUG: the last added cut is\n", cut)
        end
        println("DEBUG: the lower bound or the upper bound returned by the level method may be invalid!")
    end
    if print >= 0
        printfmtln(" The level bundle method has converged within {} iteration(s)", iter)
    end
    return MainSolution(opt_x, opt_u, opt_f, opt_ϕ)
end

# function implementing a proximal bundle method (Kiwiel, Math. Prog. 1990, Algorithm 2.1)
# to solve the main problem. The stability center (ctr_x, ctr_w) is updated by
# either a "serious step" or kept by a "null step", based on the descent test
# f(y^{k+1}) ≤ f(x^k) + m_L⋅v^k, where y^{k+1} is the proximal QP solution
# and v^k = f̂(y^{k+1}) - f(x^k) is the predicted descent of the polyhedral model.
function solve_main_proximal(
        main::MainProblem,
        subproblem::T,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo = WassInfo(.0,2);
        max_iter::Int = NUM_MAX_ITER,
        opt_gap::Float64 = VAL_TOL,
        max_aux::Float64 = VAL_INF, # maximum value for feasibility search of initial Wasserstein dual 
        min_aux::Float64 = 0.0,     # minimum value for feasibility search of initial Wasserstein dual 
        min_phi::Float64 = -VAL_INF,
        max_cut_coef::Float64  = VAL_INF, # for numerical stability
        tol_aux_feas::Float64  = VAL_TOL, # for numerical stability 
        init_weight::Float64   = DEFAULT_INIT_WEIGHT,
        min_weight::Float64    = DEFAULT_MIN_WEIGHT,
        serious_ratio::Float64 = DEFAULT_SERIOUS_RATIO,
        tight_ratio::Float64   = DEFAULT_TIGHT_RATIO,
        mom_solver = DEFAULT_SDP,
        cut_evaluator = nothing,
        print::Int = 1
    )::MainSolution where T <: SampleSubproblem
    # validate proximal bundle parameters
    if !(0.0 < serious_ratio < 0.5) || !(serious_ratio < tight_ratio < 1.0) ||
            init_weight <= 0.0 || min_weight <= 0.0
        error("Invalid proximal bundle parameters: require 0 < serious_ratio < 0.5 < tight_ratio < 1, init_weight > 0, min_weight > 0.")
    end
    # `cut_evaluator` lets the caller swap the inner-supremum solver used to
    # generate cuts for `MainProblem.ϕ`. When `nothing`, fall back to the
    # moment-relaxation evaluator with the supplied `mom_solver` (default).
    eval_cut = isnothing(cut_evaluator) ?
        (subproblem, augstate, samples, wassinfo; print=0) ->
            eval_moment_Wass(subproblem, augstate, samples, wassinfo;
                             mom_solver=mom_solver, print=print) :
        cut_evaluator
    # check if Wasserstein ambiguity is needed
    flag_Wass = wassinfo.r > VAL_TOL
    # get the state dimension and set up the linear objective expression
    dim_x = length(main.x)
    obj = main.f_x'*main.x + main.f_u'*main.u + main.ϕ
    set_lower_bound(main.ϕ, min_phi)
    # find the initial stability center by solving the model without cuts
    @objective(main.model, Min, obj)
    optimize!(main.model)
    ctr_x = value.(main.x)
    ctr_u = value.(main.u)
    ctr_w = flag_Wass ? value(main.w) : 0.0
    # generate the initial cut at the stability center
    cut = zeros(dim_x + 2)
    if flag_Wass
        cut = eval_cut(subproblem, [ctr_x;ctr_w], samples, wassinfo, print=print-1)
        if isnothing(cut) || maximum(abs.(cut)) > max_cut_coef
            cut, ctr_w = bisection_feas_cut(subproblem, samples, wassinfo, ctr_x, ctr_w, eval_cut, 
                                            max_aux=max_aux, min_aux=min_aux, feas_tol=tol_aux_feas, 
                                            coef_max=max_cut_coef, print=print)
        end
    else
        cut[1:dim_x+1] = eval_nominal(subproblem, ctr_x, samples)
    end
    ctr_val_f = main.f_x'*ctr_x + main.f_u'*ctr_u
    ctr_val_ϕ = cut'*[1;ctr_x;ctr_w]
    ctr_obj = ctr_val_f + ctr_val_ϕ
    # add the initial cut to the polyhedral approximation
    @constraint(main.model, main.ϕ >= cut'*[1;main.x;main.w])
    # initialize the best solution and the proximal weight
    opt_x = ctr_x
    opt_u = ctr_u
    opt_f = ctr_val_f
    opt_ϕ = ctr_val_ϕ
    weight = init_weight
    # print the starting message
    if print >= 0
        println(" Start the proximal bundle method for the main problem...")
        printfmtln(" The initial center objective = {:<6.4e}", ctr_obj)
        if flag_Wass
            println(" The initial Wasserstein dual variable = ", ctr_w)
        end
    end
    iter = 1
    while iter <= max_iter
        # Step 1 (Direction finding): solve the proximal QP to obtain y^{k+1}
        prox_obj = obj + (weight/2) * sum((main.x[i] - ctr_x[i])^2 for i in 1:dim_x)
        if flag_Wass
            prox_obj += (weight/2) * (main.w - ctr_w)^2
        end
        @objective(main.model, Min, prox_obj)
        optimize!(main.model)
        if termination_status(main.model) != OPTIMAL || !has_values(main.model)
            if print >= 0
                println("DEBUG: the proximal bundle direction step runs into issues...\n",
                        solution_summary(main.model,verbose=true))
                println("DEBUG: the current stability center x = ", ctr_x)
                if flag_Wass
                    println("DEBUG: the current stability center w = ", ctr_w)
                end
            end
            error("The proximal bundle direction step has failed with status: ", termination_status(main.model))
        end
        sol_x = value.(main.x)
        sol_u = value.(main.u)
        sol_w = flag_Wass ? value(main.w) : 0.0
        sol_phi_model = value(main.ϕ)
        sol_val_f = main.f_x'*sol_x + main.f_u'*sol_u
        f_hat_trial = sol_val_f + sol_phi_model
        # the predicted descent v^k = f̂(y^{k+1}) - f(x^k) is non-positive
        v_k = f_hat_trial - ctr_obj
        if v_k > VAL_TOL
            printfmtln("DEBUG: the proximal bundle predicted descent v^k = {:<6.2e} is positive", v_k)
        end
        # Step 2 (Stopping criterion): estimated gap <= opt_gap (scaled by |ctr_obj|) / [(max_aux-min_aux)^2 ⋅ sol_u]
        est_gap = abs(v_k) + (max_aux-min_aux)*sqrt(weight*abs(v_k))
        if est_gap <= opt_gap * max(1,abs(ctr_obj))
            if print >= 0
                printfmtln(" The proximal bundle method has converged at iteration {} with estimated gap = {:<6.2e}",
                           iter, est_gap)
            end
            break
        end
        # Step 4 (Linearization updating, partial): evaluate the true f at y^{k+1}
        # and generate a new cut at that point
        cut = zeros(dim_x + 2)
        cut_valid = true
        if flag_Wass
            cut = eval_cut(subproblem, [sol_x;sol_w], samples, wassinfo, print=print-1)
            if isnothing(cut) || maximum(abs.(cut)) > max_cut_coef
                cut_valid = false
            end
        else
            cut[1:dim_x+1] = eval_nominal(subproblem, sol_x, samples)
        end
        if !cut_valid
            # cut generation failed: treat as null step and raise the weight to
            # pull the next trial closer to the (feasible) stability center
            weight = min(weight * WEIGHT_UPDATE_FEAS, VAL_INF)
            if print >= 0
                printfmtln(" Iteration {} (null step, cut failed): raising weight to {:<6.2e}",
                           iter, weight)
            end
            iter += 1
            continue
        end
        val_ϕ_trial = cut'*[1;sol_x;sol_w]
        f_trial = sol_val_f + val_ϕ_trial
        # add the new cut to the polyhedral approximation of ϕ
        @constraint(main.model, main.ϕ >= cut'*[1;main.x;main.w])
        # track the best solution encountered so far
        if f_trial < opt_f + opt_ϕ
            opt_x = sol_x
            opt_u = sol_u
            opt_f = sol_val_f
            opt_ϕ = val_ϕ_trial
        end
        # Step 3 (Descent test): serious step if (2.3) holds, null step otherwise
        if f_trial <= ctr_obj + serious_ratio * v_k
            # Step 5 (Weight updating, serious step): decrease the weight if (2.12) holds
            if f_trial <= ctr_obj + tight_ratio * v_k
                weight = max(weight / DEFAULT_WEIGHT_UPDATE, min_weight)
            end
            # update the stability center
            ctr_x = sol_x
            ctr_u = sol_u
            ctr_w = sol_w
            ctr_obj = f_trial
            ctr_val_f = sol_val_f
            ctr_val_ϕ = val_ϕ_trial
            if print >= 0
                printfmtln(" Iteration {} (serious step): center value = {:<6.4e}, predicted descent = {:<6.2e}, weight = {:<6.2e}",
                           iter, ctr_obj, v_k, weight)
                if print >= 1
                    println("  The current feasible x = ", ctr_x)
                    println("  The current Wasserstein dual variable = ", ctr_w)
                end
            end
        else
            # Step 5 (Weight updating, null step): increase the weight if (2.17) holds.
            # The linearization error of ϕ at the center, using the new cut, is
            # α = ϕ(center) - (cut at trial)(center), which is ≥ 0 by convexity.
            alpha_ϕ = ctr_val_ϕ - cut'*[1;ctr_x;ctr_w]
            if alpha_ϕ > -WEIGHT_INCREASE_THRES * v_k
                weight = min(weight * DEFAULT_WEIGHT_UPDATE, VAL_INF)
            end
            if print >= 0
                printfmtln(" Iteration {} (null step): trial value = {:<6.4e}, predicted descent = {:<6.2e}, weight = {:<6.2e}",
                           iter, f_trial, v_k, weight)
            end
        end
        iter += 1
    end
    if iter > max_iter && print >= 0
        printfmtln(" The proximal bundle method does not converge within {} iterations", max_iter)
    end
    return MainSolution(opt_x, opt_u, opt_f, opt_ϕ)
end

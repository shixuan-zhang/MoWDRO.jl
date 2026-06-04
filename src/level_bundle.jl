# Level bundle method for solving the main problem of the form
# min  f_x'⋅x + f_u'⋅u + ϕ(x,w)
# s.t. (x,u) ∈ Feasible Set, w ≥ 0.

# default parameters for the level bundle method 
const DEFAULT_LEVEL = 1/(2+sqrt(2))
const BISECTION_TOL = 1e-3

# helper function that finds feasible w through bisection and 
# returns the cut together with the updated w
function bisection_feas_cut(
        subproblem::T,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo,
        sol_x::Vector{Float64},
        sol_w::Float64,
        eval_cut::Function;
        max_aux::Float64 = VAL_INF,
        min_aux::Float64 = 0.0,
        feas_tol::Float64 = -1.0,
        print::Int = 0
    ) where T <: SampleSubproblem
    # set the tolerance if not supplied
    if feas_tol < 0.0
        feas_tol = (max_aux-min_aux) * BISECTION_TOL
    end
    # initialization
    w_lb = max(sol_w, min_aux)
    w_ub = max_aux
    w_temp = (w_lb + w_ub) / 2
    w_best = w_temp
    cut_temp = eval_cut(subproblem, [sol_x;w_temp], samples, wassinfo, print=print)
    cut_best = cut_temp
    # bisection in w
    while w_ub - w_lb > feas_tol
        if isnothing(cut_temp)
            w_lb = w_temp
        else
            cut_best = cut_temp
            w_best = w_temp
            w_ub = w_temp
        end
        w_temp = (w_lb + w_ub) / 2
        cut_temp = eval_cut(subproblem, [sol_x;w_temp], samples, wassinfo, print=print)
    end
    if isnothing(cut_best)
        error("The moment relaxation is infeasible or unbounded.")
    end
    if print > 0
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
    # retrieve the Wasserstein auxiliary variable if present
    sol_w = 0.0
    if flag_Wass
        sol_w = round(value(main.w),digits=NUM_DIG)
    end
    # get the initial upper bound
    cut = zeros(dim_x+2)
    if flag_Wass
        cut = eval_cut(subproblem, [sol_x;sol_w], samples, wassinfo, print=print-1)
        if isnothing(cut) # the moment relaxation is unbounded/infeasible
            cut, sol_w = bisection_feas_cut(subproblem, samples, wassinfo, sol_x, sol_w, eval_cut, 
                                            max_aux=max_aux, min_aux=min_aux, print=print-1)
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
    # print the starting message
    if print > 0
        println(" Start the level bundle method for the main problem...")
        println(" The initial lower bound = ", min_obj)
        println(" The initial upper bound = ", max_obj)
    end
    iter = 1
    # loop until the bounds are close
    while max_obj - min_obj > opt_gap
        # update the loss/recourse approximation
        @constraint(main.model, main.ϕ >= cut'*[1;main.x;main.w])
        # get an updated lower bound
        optimize!(main.model)
        if termination_status(main.model) != OPTIMAL && !has_values(main.model)
            println("DEBUG: the level bounding step runs into issues...\n", 
                    solution_summary(main.model,verbose=true))
            println("DEBUG: the current level bounding step problem x = ", sol_x)
            if flag_Wass
                println("DEBUG: the current level bounding step problem w = ", sol_w)
            end
            println("DEBUG: the current level bounding step model is \n", main.model)
            error("The level method bounding step has failed with status: ", termination_status(main.model))
        end
        min_obj = objective_value(main.model)
        if max_obj - min_obj <= opt_gap
            if print > 0
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
            println("DEBUG: the level projection step runs into issues...\n", 
                    solution_summary(main.model,verbose=true))
            println("DEBUG: the current level projection step problem x = ", sol_x)
            if flag_Wass
                println("DEBUG: the current level projection step problem w = ", sol_w)
            end
            println("DEBUG: the current level projection step problem model is \n", main.model)
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
            if isnothing(cut) # the moment relaxation is unbounded/infeasible
                cut, sol_w = bisection_feas_cut(subproblem, samples, wassinfo, sol_x, sol_w, eval_cut,
                                                max_aux=max_aux, min_aux=min_aux, print=print-1)
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
        if print > 0
            printfmtln(" Iteration {}: current objective = {:<6.2e}, upper bound = {:<6.2e}, lower bound = {:<6.2e}",
                       iter, val_ϕ+val_f, max_obj, min_obj)
        end
        iter += 1
        # check if maximum number of iteration is reached
        if iter > max_iter
            if print > 0
                printfmtln(" The level bundle method does not converge within {} iterations", max_iter)
            end
            return MainSolution(opt_x, opt_u, opt_f, opt_ϕ)
        end
    end
    if max_obj - min_obj < -opt_gap
        error("Invalid upper or lower bound in the level method!")
    end
    if print > 0
        printfmtln(" The level bundle method has converged within {} iteration(s)", iter)
    end
    return MainSolution(opt_x, opt_u, opt_f, opt_ϕ)
end

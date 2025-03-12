# Level bundle method for solving the main problem of the form
# min  f_x'⋅x + f_u'⋅u + ϕ(x,w)
# s.t. (x,u) ∈ Feasible Set, w ≥ 0.

# Artificial upper bound on w is needed for convergence.
const VAL_GAP = 1.0e-2
const VAL_BOUND = 1.0e6
const VAL_LEVEL = 0.5
const VAL_POS_DUAL = 1.0e-2

function solve_main_level(
        main::MainProblem,
        subproblem::T,
        samples::Vector{Vector{Float64}},
        wassinfo::WassInfo = WassInfo(.0,2);
        max_iter::Int = NUM_MAX_ITER,
        opt_gap::Float64 = VAL_GAP,
        max_aux::Float64 = VAL_BOUND,
        min_aux::Float64 = VAL_POS_DUAL,
        level::Float64 = VAL_LEVEL,
        mom_solver = DEFAULT_SDP,
        print::Int = 1
    )::MainSolution where T <: SampleSubproblem
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
    set_lower_bound(main.ϕ, -max_aux)
    # get the initial solution and lower bound
    @objective(main.model, Min, obj)
    optimize!(main.model)
    sol_x = value.(main.x)
    sol_u = value.(main.u)
    min_obj = objective_value(main.model)
    val_f = main.f_x'*sol_x + main.f_u'*sol_u
    # retrieve the Wasserstein auxiliary variable if present
    sol_w = 0.0
    if flag_Wass
        sol_w = value(main.w)
    end
    # get the initial upper bound
    cut = zeros(dim_x+2)
    if flag_Wass
        cut = eval_moment_Wass(subproblem, [sol_x;sol_w], samples, wassinfo, mom_solver=mom_solver, print=print-1)
    else
        cut[1:dim_x+1] = eval_nominal(subproblem, sol_x, samples)
    end
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
        # calculate the level
        val_lev = level*max_obj + (1-level)*min_obj
        # build the projection model
        con_proj = @constraint(main.model, obj <= val_lev)
        obj_proj = (main.x-sol_x)'*(main.x-sol_x)
        if flag_Wass
            obj_proj += (main.w-sol_w)^2
        end
        @objective(main.model, Min, obj_proj)
        # find the next iterate
        optimize!(main.model)
        sol_x = value.(main.x)
        sol_u = value.(main.u)
        sol_w = 0.0
        if flag_Wass
            sol_w = value(main.w)
        end
        val_f = main.f_x'*sol_x + main.f_u'*sol_u
        # get an updated upper bound
        cut = zeros(dim_x+2)
        if flag_Wass
            cut = eval_moment_Wass(subproblem, [sol_x;sol_w], samples, wassinfo, mom_solver=mom_solver, print=print-1)
        else
            cut[1:dim_x+1] = eval_nominal(subproblem, sol_x, samples)
        end
        val_ϕ = cut'*[1;sol_x;sol_w]
        # check if a better solution is encountered
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
        # get an updated lower bound
        optimize!(main.model)
        min_obj = objective_value(main.model)
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
        if max_obj - min_obj < -opt_gap
            error("Invalid upper or lower bound in the level method!")
        end
    end
    if print > 0
        printfmtln(" The level bundle method has converged within {} iteration(s)", iter)
    end
    return MainSolution(opt_x, opt_u, opt_f, opt_ϕ)
end

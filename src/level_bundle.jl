# Level bundle method for solving the master problem of the form
# min  f_x'⋅x + f_y'⋅y + ϕ(x,w)
# s.t. (x,y) ∈ Feasible Set, w ≥ 0.
# Artificial upper bound on w is needed for convergence.

function solve_master_level(
        master::MasterProblem,
        recourse::Function;
        max_iter::Int = NUM_MAX_ITER,
        opt_gap::Float64 = VAL_TOL,
        max_aux::Float64 = VAL_INF,
        level::Float64 = 0.5,
        print::Bool = false
    )::MasterSolution
    # set the objective expression
    obj = master.f_x'*master.x + master.f_y'*master.y + master.ϕ + master.r*master.w
    # add the artificial bound on the Wasserstein auxiliary variable w 
    set_upper_bound(master.w, max_aux)
    # get the initial solution and lower bound
    @objective(master.model, Min, obj)
    optimize!(master.model)
    sol_x = value.(master.x)
    sol_y = value.(master.y)
    sol_w = value(master.w)
    min_obj = objective_value(master.model)
    val_f = master.f_x'*sol_x + master.f_y'*sol_y
    # get the initial upper bound
    val_ϕ, ∇ϕ = recourse([sol_x; sol_w])
    cut = build_linear_cut([sol_x; sol_w], ∇ϕ, val_ϕ)
    max_obj = val_f + val_ϕ + master.r*sol_w
    # print the starting message
    if print
        println(" Start the level bundle method for the master problem...")
    end
    iter = 1
    # loop until the bounds are close
    while max_obj - min_obj > opt_gap
        # update the loss/recourse approximation
        @constraint(master.model, master.ϕ >= cut.b + cut.a'*[master.x; master.w])
        # calculate the level
        val_lev = level*max_obj + (1-level)*min_obj
        # build the projection model
        con_proj = @constraint(master.model, obj <= val_lev)
        @objective(master.model, Min, (master.x - sol_x)'*(master.x - sol_x) + (master.w - sol_w)^2)
        # find the next iterate
        optimize!(master.model)
        sol_x = value.(master.x)
        sol_y = value.(master.y)
        sol_w = value(master.w)
        val_f = master.f_x'*sol_x + master.f_y'*sol_y
        # get an updated upper bound
        val_ϕ, ∇ϕ = recourse([sol_x; sol_w])
        cut = build_linear_cut([sol_x; sol_w], ∇ϕ, val_ϕ)
        max_obj = min(max_obj, val_ϕ + val_f + master.r*sol_w)
        # restore the optimization model
        delete(master.model, con_proj)
        @objective(master.model, Min, obj)
        # get an updated lower bound
        optimize!(master.model)
        min_obj = objective_value(master.model)
        # print the update if needed
        if print
            printfmtln(" Iteration {}: current objective = {:<6.2e}, upper bound = {:<6.2e}, lower bound = {:<6.2e}",
                       iter, val_ϕ+val_f+master.r*sol_w, max_obj, min_obj)
            println("DEBUG: w = ", sol_w, ", x = ", sol_x)
        end
        iter += 1
        # check if maximum number of iteration is reached
        if iter > max_iter
            if print
                printfmtln(" The level bundle method does not converge within {} iterations", max_iter)
            end
            return MasterSolution(sol_x, sol_y, max_obj, val_f)
        end
    end
    if print
        printfmtln(" The level bundle method has converged within {} iteration(s)", iter)
    end
    return MasterSolution(sol_x, sol_y, max_obj, val_f)
end

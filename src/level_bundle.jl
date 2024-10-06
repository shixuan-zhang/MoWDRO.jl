# Level bundle method for solving the main problem of the form
# min  f_x'⋅x + f_u'⋅u + ϕ(x,w)
# s.t. (x,u) ∈ Feasible Set, w ≥ 0.
# Artificial upper bound on w is needed for convergence.

function solve_main_level(
        main::MainProblem,
        subproblem::T,
        wassinfo::WassInfo = WassInfo(.0,2);
        max_iter::Int = NUM_MAX_ITER,
        opt_gap::Float64 = VAL_TOL,
        max_aux::Float64 = VAL_INF,
        level::Float64 = 0.5,
        print::Bool = false
    )::MainSolution where T <: SampleSubproblem
    # set the objective expression
    obj = main.f_x'*main.x + main.f_y'*main.y + main.ϕ + main.r*main.w
    # add the artificial bound on the Wasserstein auxiliary variable w 
    set_upper_bound(main.w, max_aux)
    # get the initial solution and lower bound
    @objective(main.model, Min, obj)
    optimize!(main.model)
    sol_x = value.(main.x)
    sol_y = value.(main.y)
    sol_w = value(main.w)
    min_obj = objective_value(main.model)
    val_f = main.f_x'*sol_x + main.f_y'*sol_y
    # get the initial upper bound
    cut = LinearCut(Float64[], 0.0)
    if T == WassersteinRecourseProblem # FIXME: to be fixed here
        cut = eval_Wass_recourse(recourse, [sol_x;sol_w])
    elseif T == NominalRecourseProblem
        cut = eval_nom_recourse(recourse, sol_x)
    else
        error("Unsupported recourse problem type!")
    end
    val_ϕ = cut.a'*[sol_x; sol_w] + cut.b
    max_obj = val_ϕ + val_f + main.r*sol_w
    # print the starting message
    if print
        println(" Start the level bundle method for the main problem...")
    end
    iter = 1
    # loop until the bounds are close
    while max_obj - min_obj > opt_gap
        # update the loss/recourse approximation
        @constraint(main.model, main.ϕ >= cut.b + cut.a'*[main.x; main.w])
        # calculate the level
        val_lev = level*max_obj + (1-level)*min_obj
        # build the projection model
        con_proj = @constraint(main.model, obj <= val_lev)
        @objective(main.model, Min, (main.x - sol_x)'*(main.x - sol_x) + (main.w - sol_w)^2)
        # find the next iterate
        optimize!(main.model)
        sol_x = value.(main.x)
        sol_y = value.(main.y)
        sol_w = value(main.w)
        val_f = main.f_x'*sol_x + main.f_y'*sol_y
        # get an updated upper bound
        cut = LinearCut(Float64[], 0.0)
        if T == WassersteinRecourseProblem
            cut = eval_Wass_recourse(recourse, [sol_x;sol_w])
        elseif T == NominalRecourseProblem
            cut = eval_nom_recourse(recourse, sol_x)
        else
            error("Unsupported recourse problem type!")
        end
        val_ϕ = cut.a'*[sol_x; sol_w] + cut.b
        max_obj = min(max_obj, val_f + val_ϕ + main.r*sol_w)
        # restore the optimization model
        delete(main.model, con_proj)
        @objective(main.model, Min, obj)
        # get an updated lower bound
        optimize!(main.model)
        min_obj = objective_value(main.model)
        # print the update if needed
        if print
            printfmtln(" Iteration {}: current objective = {:<6.2e}, upper bound = {:<6.2e}, lower bound = {:<6.2e}",
                       iter, val_ϕ+val_f+main.r*sol_w, max_obj, min_obj)
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

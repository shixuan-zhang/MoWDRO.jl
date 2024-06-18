# Level bundle method for solving the master problem

function solve_master_level(
        master::MasterProblem,
        recourse::Function;
        max_iter::Int = NUM_MAX_ITER,
        opt_gap::Float64 = VAL_TOL,
        level::Float64 = 0.5,
        print::Bool = false
    )::MasterSolution
    # get the initial solution and lower bound
    expr_obj = master.obj_stat'*master.var_stat + master.obj_ctrl'*master.var_ctrl + master.var_loss
    JuMP.@objective(master.model, Min, expr_obj)
    JuMP.optimize!(master.model)
    sol_stat = JuMP.value.(master.var_stat)
    sol_ctrl = JuMP.value.(master.var_ctrl)
    val_lobd = JuMP.objective_value(master.model)
    val_mobj = master.obj_stat'*sol_stat + master.obj_ctrl'*sol_ctrl
    # get the initial upper bound
    loss, cut = recourse(sol_stat)
    val_upbd = loss + val_mobj
    # loop until the bounds are close
    while val_upbd - val_lobd > opt_gap
        # update the loss/recourse approximation
        JuMP.@constraint(master.model, master.var_loss >= cut.val + cut.vec'*master.var_stat)
        # calculate the level
        val_lev = level*val_upbd + (1-level)*val_lobd
        # build the projection model
        con_proj = JuMP.@constraint(master.model, expr_obj <= val_lev)
        JuMP.@objective(master.model, Min, norm(master.var_stat - sol_stat)^2)
        # find the next iterate
        JuMP.optimize!(master.model)
        sol_stat = JuMP.value.(master.var_stat)
        sol_ctrl = JuMP.value.(master.var_ctrl)
        val_mobj = master.obj_stat'*sol_stat + master.obj_ctrl'*sol_ctrl
        # get an updated upper bound
        loss, cut = recourse(sol_stat)
        val_upbd = min(val_upbd, loss + val_mobj)
        # restore the optimization model
        JuMP.delete(master.model, con_proj)
        JuMP.@objective(master.model, Min, expr_obj)
        # get an updated lower bound
        JuMP.optimize!(master.model)
        val_lobd = JuMP.objective_value(master.model)
    end
    return MasterSolution(sol_stat, sol_ctrl, val_upbd, val_mobj)
end

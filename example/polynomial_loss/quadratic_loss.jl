# numerical example for nonconvex quadratic loss functions of the form
# g(x,ξ) := x₁⋅ξ₁ξ₂ + ⋯ + xₙ⋅ξₙξ₁, ξᵢ ∈ [-1,1] with uniform distributions

using JuMP, HiGHS
include("../../src/MoWDRO.jl")
using .MoWDRO


# function that conducts the experiment on the quadratic loss functions
function experiment_quad_loss(
        n::Int, # variable dimension
        N::Int; # number of samples
        rad_Wass::Float64 = 1.0
    )
    # define the master linear optimization problem with randomly generated linear objective
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    x = @variable(model, 0 <= x[1:n] <= 1, base_name="x")
    w = @variable(model, w >= 0, base_name="w")
    ϕ = @variable(model, ϕ >= 0, base_name="ϕ")
    f_x = rand(n)
    master = MasterProblem(model, x, VariableRef[], w, ϕ, f_x, Float64[], rad_Wass)
    # generate the samples for the recourse problems
    sample = [rand(n).*2 .- 1 for _ in 1:N]
    # define the recourse data and build the recourse problems
    F = Matrix{Float64}[]
    push!(F, zeros(n+1,n+1))
    for i = 1:n
        Fᵢ = zeros(n+1,n+1)
        if i < n
            Fᵢ[i+1,i+2] = 1.0
        else
            Fᵢ[2,n+1] = 1.0
        end
        push!(F, Fᵢ) 
    end
    data = RecourseData(n,n,0,F,Matrix{Float64}[],Matrix{Float64}[],zeros(n),ones(n),Float64[],Float64[])
    recourse = WassersteinRecourseProblem[]
    for i = 1:N
        push!(recourse, build_semidefinite_relax(sample[i], data)) #TODO: check that the SDP recourse is set up correctly
    end
    # solve the problem
    sol = solve_master_level(master, recourse, print=true)
    println("The master problem is solved successfully!")
    println("x = ", sol.x)
    println("f = ", sol.f)
    println("ϕ = ", sol.ϕ)
end

experiment_quad_loss(10, 100)

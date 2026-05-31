# test the moment-relaxation cut evaluator `eval_moment_Wass`
# for both the polynomial-loss and linear-recourse dispatches.
# Uses the same problem instances as `test/noncvx.jl` so the two
# evaluators are checked against the same closed-form cut.

using DynamicPolynomials, SemialgebraicSets
using CSDP

# ---------------------------------------------------------------------------
# `SamplePolynomialLoss` dispatch — same instance as `test_noncvx_polynomial_loss`.
function test_moment_polynomial_loss()
    @polyvar x[1:1] ξ[1:1]
    F   = x[1] * ξ[1] + 0
    ∇ₓF = [ξ[1] + 0]
    Ξ   = basicsemialgebraicset(FullSpace(), [ξ[1], 1 - ξ[1]])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    augstate = [1.0, 2.0]
    samples  = [[0.5]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_moment_Wass(loss, augstate, samples, wassinfo;
                           mom_solver = CSDP.Optimizer)
    @test isapprox(cut, [0.0, 0.75, 0.9375], atol = 1e-4)
end

# ---------------------------------------------------------------------------
# `SampleLinearRecourse` dispatch — same instance as `test_noncvx_linear_recourse`.
function test_moment_linear_recourse()
    @polyvar x[1:1] ξ[1:1] y[1:1]
    C = [0.0  0.0;
         0.0  ξ[1]]                          .+ 0.0 * sum(ξ)
    A = reshape([1.0; -1.0], 2, 1)           .+ 0.0 * sum(ξ)
    b = [0.0; -1.0]                          .+ 0.0 * sum(ξ)
    Ξ = basicsemialgebraicset(FullSpace(), [ξ[1], 1 - ξ[1]])
    B = [1.0]
    recourse = SampleLinearRecourse(x, ξ, y, C, A, b, Ξ, B)
    augstate = [1.0, 2.0]
    samples  = [[0.5]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_moment_Wass(recourse, augstate, samples, wassinfo;
                           mom_solver = CSDP.Optimizer)
    @test isapprox(cut, [0.0, 0.75, 0.9375], atol = 1e-4)
end

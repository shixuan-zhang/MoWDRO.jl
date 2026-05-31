# test the nonconvex global cut evaluator `eval_noncvx_Wass`
# for both the polynomial-loss and linear-recourse dispatches.

using DynamicPolynomials, SemialgebraicSets
using SCIP

# ---------------------------------------------------------------------------
# Closed-form check for the `SamplePolynomialLoss` dispatch.
# F(x, ξ) = x · ξ on Ξ = [0, 1], one sample ξ̂ = 0.5, (r, p) = (1, 2),
# augstate = (x̄ = 1.0, w̄ = 2.0). The inner sup
#   max_{ξ ∈ [0,1]}  ξ - 2(ξ - 0.5)^2
# is concave with interior maximizer ξ* = 0.75, so the analytic
# cut [v̂ - ĝ x̄; ĝ; r^p - p̂] = [0; 0.75; 0.9375].
function test_noncvx_polynomial_loss()
    @polyvar x[1:1] ξ[1:1]
    F   = x[1] * ξ[1] + 0
    ∇ₓF = [ξ[1] + 0]
    Ξ   = basicsemialgebraicset(FullSpace(), [ξ[1], 1 - ξ[1]])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    augstate = [1.0, 2.0]
    samples  = [[0.5]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_noncvx_Wass(loss, augstate, samples, wassinfo;
                           noncvx_solver = SCIP.Optimizer)
    @test isapprox(cut, [0.0, 0.75, 0.9375], atol = 1e-4)
end

# ---------------------------------------------------------------------------
# Closed-form check for the `SampleLinearRecourse` dispatch.
# Encodes F(x, ξ) = max_y x·ξ·y with y ∈ [0, 1] via
#   (1, x)ᵀ [0 0; 0 ξ] (1, y) = x · ξ · y,
# and A = [1; -1], b = [0; -1] so A y - b ≥ 0 ⇔ y ∈ [0, 1]. The joint
# inner sup over (ξ, y) ∈ [0, 1]² is bilinear-minus-quadratic (nonconvex),
# with global maximizer (ξ*, y*) = (0.75, 1) — the same ξ* as test 1 by
# elimination — giving ĉ = C(ξ*)·[1; y*] = [0; 0.75] and cut [0; 0.75; 0.9375].
function test_noncvx_linear_recourse()
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
    cut = eval_noncvx_Wass(recourse, augstate, samples, wassinfo;
                           noncvx_solver = SCIP.Optimizer)
    @test isapprox(cut, [0.0, 0.75, 0.9375], atol = 1e-4)
end

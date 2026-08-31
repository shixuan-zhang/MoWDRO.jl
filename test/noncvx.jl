# test the nonconvex global cut evaluator `eval_noncvx_Wass`
# for both the polynomial-loss and linear-recourse dispatches.
# Uses the same problem instances as `test/moment.jl` so the two
# evaluators are checked against the same closed-form cut.

using DynamicPolynomials, SemialgebraicSets

# ---------------------------------------------------------------------------
# `SamplePolynomialLoss` dispatch — Test 1: baseline bilinear F on bounded Ξ.
# F(x, ξ) = x · ξ on Ξ = [0, 1], one sample ξ̂ = 0.5, (r, p) = (1, 2),
# augstate = (x̄ = 1.0, w̄ = 2.0). The inner sup
#   max_{ξ ∈ [0,1]}  ξ − 2(ξ − 0.5)²
# is concave with interior maximizer ξ* = 0.75, so the analytic
# cut [v̂ − ĝ x̄; ĝ; r^p − p̂] = [0; 0.75; 0.9375].
function test_noncvx_polynomial_loss_1()
    @polyvar x[1:1] ξ[1:1]
    F   = x[1] * ξ[1] + 0
    ∇ₓF = [ξ[1] + 0]
    Ξ   = basicsemialgebraicset(FullSpace(), [ξ[1], 1 - ξ[1]])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    augstate = [1.0, 2.0]
    samples  = [[0.5]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_noncvx_Wass(loss, augstate, samples, wassinfo)
    @test isapprox(cut, [0.0, 0.75, 0.9375], atol = 1e-4)
end

# ---------------------------------------------------------------------------
# `SamplePolynomialLoss` dispatch — Test 2: nonconcave inner max, multivariate ξ.
# F = x·(ξ_1² + ξ_2²) on Ξ = [-1, 1]². Inner objective is convex in ξ so a
# local-only solver would get stuck at the saddle ξ = (0, 0) instead of the
# boundary maximizer ξ* = (±1, ±1) — this is the stress test that the QCQP
# path is genuinely nonconvex-global. Analytic cut = [0; 2; -1].
function test_noncvx_polynomial_loss_2()
    @polyvar x[1:1] ξ[1:2]
    F   = x[1] * (ξ[1]^2 + ξ[2]^2) + 0
    ∇ₓF = [(ξ[1]^2 + ξ[2]^2) + 0]
    Ξ   = basicsemialgebraicset(FullSpace(), [1 - ξ[1]^2, 1 - ξ[2]^2])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    augstate = [1.0, 0.5]
    samples  = [[0.0, 0.0]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_noncvx_Wass(loss, augstate, samples, wassinfo)
    @test isapprox(cut, [0.0, 2.0, -1.0], atol = 1e-4)
end

# ---------------------------------------------------------------------------
# `SamplePolynomialLoss` dispatch — Test 3: max-of-two-polynomials loss,
# multivariate ξ. Exercises the per-branch QCQP + argmax-k loop in
# `_gen_noncvx_cut_polynomial_loss`. F_1 strictly dominates F_2 at
# (x̄, w̄) = (1, 2) on Ξ = [0, 1]². Analytic cut = [0; 1.5; 0.875].
function test_noncvx_polynomial_loss_3()
    @polyvar x[1:1] ξ[1:2]
    F1  = x[1] * (ξ[1] + ξ[2]) + 0
    F2  = -x[1] * (ξ[1] + ξ[2]) - 1.0
    ∇F1 = differentiate(F1, x)
    ∇F2 = differentiate(F2, x)
    Ξ   = basicsemialgebraicset(FullSpace(), [ξ[1], 1 - ξ[1], ξ[2], 1 - ξ[2]])
    loss = SamplePolynomialLoss(x, ξ, [F1, F2], [∇F1, ∇F2], Ξ)
    augstate = [1.0, 2.0]
    samples  = [[0.5, 0.5]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_noncvx_Wass(loss, augstate, samples, wassinfo)
    @test isapprox(cut, [0.0, 1.5, 0.875], atol = 1e-4)
end

# ---------------------------------------------------------------------------
# `SamplePolynomialLoss` dispatch — Test 4: nonneg-orthant Ξ, bilinear
# (indefinite-quadratic) F. F = x·ξ_1·ξ_2 is quadratic and non-concave;
# combined with w̄ = 2 and ξ̂ = (0.5, 0.5) the inner obj is globally concave
# with an interior maximizer ξ* = (2/3, 2/3) in the orthant. Analytic
# cut = [0; 4/9; 17/18]. Exercises the QCQP path with the mixed bilinear
# term ξ_1·ξ_2 (lifted by PolyJuMP.QCQP.Optimizer) and orthant constraints.
function test_noncvx_polynomial_loss_4()
    @polyvar x[1:1] ξ[1:2]
    F   = x[1] * ξ[1] * ξ[2] + 0
    ∇ₓF = [ξ[1] * ξ[2] + 0]
    Ξ   = basicsemialgebraicset(FullSpace(), [ξ[1] + 0, ξ[2] + 0])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    augstate = [1.0, 2.0]
    samples  = [[0.5, 0.5]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_noncvx_Wass(loss, augstate, samples, wassinfo)
    # SCIP's spatial branch-and-bound has looser precision on this nonconvex
    # bilinear QCQP than SCIP's LP-relaxation path on the affine tests above,
    # so widen the tolerance a bit.
    @test isapprox(cut, [0.0, 4/9, 17/18], atol = 1e-3)
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
    cut = eval_noncvx_Wass(recourse, augstate, samples, wassinfo)
    @test isapprox(cut, [0.0, 0.75, 0.9375], atol = 1e-4)
end

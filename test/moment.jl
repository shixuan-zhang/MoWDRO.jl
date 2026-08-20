# test the moment-relaxation cut evaluator `eval_moment_Wass`
# for both the polynomial-loss and linear-recourse dispatches.
# Uses the same problem instances as `test/noncvx.jl` so the two
# evaluators are checked against the same closed-form cut.

using DynamicPolynomials, SemialgebraicSets
using CSDP

# ---------------------------------------------------------------------------
# `SamplePolynomialLoss` dispatch — Test 1: baseline bilinear F on bounded Ξ,
# concave inner max. Same instance as `test_noncvx_polynomial_loss_1`.
function test_moment_polynomial_loss_1()
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
# `SamplePolynomialLoss` dispatch — Test 2: nonconcave inner max, multivariate ξ.
# F = x·(ξ_1² + ξ_2²) on Ξ = [-1, 1]². At (x̄, w̄) = (1, 0.5) the inner objective
# is `(ξ_1²+ξ_2²) − 0.5·(ξ_1²+ξ_2²) = 0.5·(ξ_1²+ξ_2²)`, convex in ξ on the box,
# so each maximizer sits at ±1 (any vertex) rather than at ξ̂ = (0, 0).
# Analytic cut = [0; 2; -1].
function test_moment_polynomial_loss_2()
    @polyvar x[1:1] ξ[1:2]
    F   = x[1] * (ξ[1]^2 + ξ[2]^2) + 0
    ∇ₓF = [(ξ[1]^2 + ξ[2]^2) + 0]
    Ξ   = basicsemialgebraicset(FullSpace(), [1 - ξ[1]^2, 1 - ξ[2]^2])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    augstate = [1.0, 0.5]
    samples  = [[0.0, 0.0]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_moment_Wass(loss, augstate, samples, wassinfo;
                           mom_solver = CSDP.Optimizer)
    @test isapprox(cut, [0.0, 2.0, -1.0], atol = 1e-4)
end

# ---------------------------------------------------------------------------
# `SamplePolynomialLoss` dispatch — Test 3: max-of-two-polynomials loss,
# multivariate ξ. F = max(F_1, F_2) with F_1 = x·(ξ_1+ξ_2) and
# F_2 = −x·(ξ_1+ξ_2) − 1. At (x̄, w̄) = (1, 2) on Ξ = [0, 1]² the inner sups
# are separable in each ξ_j: F_1 branch = 2·0.625 = 1.25 at ξ* = (0.75, 0.75);
# F_2 branch = 2·(−0.375) − 1 = −1.75 at ξ* = (0.25, 0.25). Argmax = F_1,
# analytic cut = [0; 1.5; 0.875].
function test_moment_polynomial_loss_3()
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
    cut = eval_moment_Wass(loss, augstate, samples, wassinfo;
                           mom_solver = CSDP.Optimizer)
    @test isapprox(cut, [0.0, 1.5, 0.875], atol = 1e-4)
end

# ---------------------------------------------------------------------------
# `SamplePolynomialLoss` dispatch — Test 4: nonneg-orthant Ξ, bilinear
# (indefinite-quadratic) F. F = x·ξ_1·ξ_2 is quadratic and non-concave (its
# bare Hessian is [[0,1],[1,0]] with eigenvalues ±1). With w̄ = 2 and
# ξ̂ = (0.5, 0.5) the inner obj `ξ_1ξ_2 − 2·((ξ_1−0.5)² + (ξ_2−0.5)²)` has
# Hessian [[-4,1],[1,-4]] (globally concave), so there is a unique interior
# maximizer ξ* = (2/3, 2/3) inside the orthant. Analytic cut = [0; 4/9; 17/18].
function test_moment_polynomial_loss_4()
    @polyvar x[1:1] ξ[1:2]
    F   = x[1] * ξ[1] * ξ[2] + 0
    ∇ₓF = [ξ[1] * ξ[2] + 0]
    Ξ   = basicsemialgebraicset(FullSpace(), [ξ[1] + 0, ξ[2] + 0])
    loss = SamplePolynomialLoss(x, ξ, F, ∇ₓF, Ξ)
    augstate = [1.0, 2.0]
    samples  = [[0.5, 0.5]]
    wassinfo = WassInfo(1.0, 2)
    cut = eval_moment_Wass(loss, augstate, samples, wassinfo;
                           mom_solver = CSDP.Optimizer)
    @test isapprox(cut, [0.0, 4/9, 17/18], atol = 1e-4)
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

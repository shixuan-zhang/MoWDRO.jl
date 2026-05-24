# Copositive reformulation of the two-stage Wasserstein distributionally
# robust linear program. Implements Theorem 5 (Eq. 28) of
#   Hanasusanto & Kuhn (2018), "Conic Programming Reformulations of Two-Stage
#   Distributionally Robust Linear Programs over Wasserstein Balls".
#
# `solve_two_stage_copos` builds the entire program in one piece — no
# scenario decomposition, no level-bundle subgradient evaluation — and
# replaces the copositive cone constraint by its standard inner approximation
#   𝒞⁰ = { P + N : P ⪰ 0, N ≥ 0 entry-wise }
# so the program can be solved by an ordinary SDP solver.
#
# Notation matches the paper exactly: c, x, X, Q, q, T(x), h(x), W (Eq. 3),
# S, t, ε, ξ̂_i, λ, ψ_i, φ_i, s_i. As in Eq. (11), the symbols Q, q, T(x),
# h(x), W are overwritten by their support-set-augmented values inside the
# function so that the assembly of Eq. (28) reads exactly like the paper.

# input format follows Eqs. (1) and (3) of the paper:
#   (1) min  c^T x + L(x)          s.t.  x ∈ X
#   (3) Z(x,ξ) = inf (Qξ+q)^T y    s.t.  T(x)ξ + h(x) ≤ Wy,  y ∈ ℝ^{N_2}
# and Ξ = { ξ ∈ ℝ^K_+ : Sξ ≤ t }.
#
# The first-stage feasible set X is taken to be a polyhedron
#   X = { x ∈ ℝ^{N_1} : a_i^T x ≤ b_i, i = 1,…,|X| }
# and passed as a vector whose i-th entry is the augmented coefficient
# vector `[a_i; b_i] ∈ ℝ^{N_1+1}`.
#
# T(x) and h(x) are passed via their affine-in-x decomposition (the paper
# states them explicitly as affine):
#   T(x) = T_0 + Σ_{l=1}^{N_1} x_l T_l,   with T_l ∈ ℝ^{M × K},
#   h(x) = h_0 + Σ_{l=1}^{N_1} x_l h_l,   with h_l ∈ ℝ^M.
# These are stored 1-indexed as `T[l+1] == T_l` and `h[l+1] == h_l`, so
# both `T` and `h` have length `N₁ + 1`.
function solve_two_stage_copos(
        c::AbstractVector{<:Real},
        X::AbstractVector{<:AbstractVector{<:Real}},
        Q::AbstractMatrix{<:Real},
        q::AbstractVector{<:Real},
        T::AbstractVector{<:AbstractMatrix{<:Real}},
        h::AbstractVector{<:AbstractVector{<:Real}},
        W::AbstractMatrix{<:Real},
        S::AbstractMatrix{<:Real},
        t::AbstractVector{<:Real},
        samples::Vector{<:AbstractVector{<:Real}},
        ε::Real;
        solver = DEFAULT_SDP,
        δ::Real = 0.0,
        silent::Bool = true
    )
    # dimensions (the paper's sample index set is [I]; here `nsmpl` plays
    # the role of I because the symbol I is reserved for the identity).
    N₁ = length(c)
    N₂ = size(Q, 1)
    K  = size(Q, 2)
    M  = size(W, 1)
    J  = length(t)
    nsmpl = length(samples)
    @assert size(Q) == (N₂, K)             "Q must be N_2 × K"
    @assert length(q) == N₂                "q must have length N_2"
    @assert size(W) == (M, N₂)             "W must be M × N_2"
    @assert size(S) == (J, K)              "S must be J × K"
    @assert all(length(ξ̂) == K for ξ̂ in samples) "each sample ξ̂_i must lie in ℝ^K"
    @assert length(T) == N₁ + 1            "T must have length N_1 + 1 (T[l+1] = T_l for l = 0,…,N_1)"
    @assert length(h) == N₁ + 1            "h must have length N_1 + 1 (h[l+1] = h_l for l = 0,…,N_1)"
    @assert all(size(Tl) == (M, K) for Tl in T) "every T_l must be M × K"
    @assert all(length(hl) == M for hl in h)    "every h_l must have length M"
    @assert all(length(row) == N₁ + 1 for row in X) "each row of X must have length N_1 + 1, encoding [a_i; b_i]"
    # build the JuMP optimization model
    model = Model(solver)
    silent && set_silent(model)
    # first-stage decision and feasible set: x ∈ X   (paper Eq. 1)
    # X is a polyhedron given by the linear inequalities a_i^T x ≤ b_i.
    @variable(model, x[1:N₁])
    for row in X
        a = @view row[1:N₁]
        b = row[N₁+1]
        @constraint(model, dot(a, x) <= b)
    end
    # assemble T(x), h(x) from the affine decomposition: T_0 + Σ x_l T_l, h_0 + Σ x_l h_l
    Tx = N₁ == 0 ? Matrix{Float64}(T[1]) : T[1] .+ sum(x[l] .* T[l+1] for l = 1:N₁)
    hx = N₁ == 0 ? Vector{Float64}(h[1]) : h[1] .+ sum(x[l] .* h[l+1] for l = 1:N₁)
    # extend (Q, q, T(x), h(x), W) per paper Eq. (11); the same letters
    # are reused so that the SDP construction below reads like Eq. (28).
    Q  = vcat(Q, S)                                                          # (N₂+J) × K
    q  = vcat(q, -t)                                                         # (N₂+J)
    W  = [W            zeros(M, J);
          zeros(J, N₂) -Matrix{Float64}(I, J, J)]                            # (M+J) × (N₂+J)
    Tx = vcat(Tx, zeros(J, K))                                               # (M+J) × K
    hx = vcat(hx, zeros(J))                                                  # (M+J)
    Nψ = N₂ + J     # post-extension dimension of ψ_i, φ_i (the [N_2+J] in Eq. 28)
    Mh = M  + J     # post-extension row dim of T(x), h(x), W
    nmat = K + Mh + 1
    # Lagrange / dual variables of the copositive program (paper Eq. 28)
    @variable(model, λ ≥ 0)
    @variable(model, s[1:nsmpl])
    @variable(model, ψ[1:nsmpl, 1:Nψ])
    @variable(model, φ[1:nsmpl, 1:Nψ])
    # one copositive (relaxed to 𝒞⁰) matrix constraint per sample i ∈ [I]
    for i = 1:nsmpl
        ψᵢ = ψ[i, :]
        φᵢ = φ[i, :]
        sᵢ = s[i]
        ξ̂  = samples[i]
        # blocks of the symmetric matrix in Eq. (28) (δ from Eq. 24 set by kwarg)
        # (1,1): λ I_K + Q^T diag(φ_i) Q                  [K × K]
        QtΦQ = Q' * (φᵢ .* Q)
        # (1,2): −½ T(x)^T − Q^T diag(φ_i) W^T            [K × (M+J)]
        b12  = (-0.5) .* Tx' .- Q' * (φᵢ .* W')
        # (1,3): −λ ξ̂_i − ½ Q^T ψ_i                       [K]
        b13  = (-λ) .* ξ̂ .- 0.5 .* (Q' * ψᵢ)
        # (2,2): W diag(φ_i) W^T + δ I_{M+J}              [(M+J) × (M+J)]
        WΦWt = W * (φᵢ .* W')
        # (2,3): ½ (W ψ_i − h(x))                         [M+J]
        b23  = 0.5 .* (W * ψᵢ .- hx)
        # (3,3): s_i                                       [scalar]
        # assemble the symmetric block matrix Mᵢ of Eq. (28)
        Mᵢ = Matrix{AffExpr}(undef, nmat, nmat)
        for r = 1:K, cc = 1:K
            Mᵢ[r, cc] = QtΦQ[r, cc] + (r == cc ? convert(AffExpr, λ) : AffExpr(0.0))
        end
        for r = 1:K, cc = 1:Mh
            Mᵢ[r, K+cc] = b12[r, cc]
            Mᵢ[K+cc, r] = b12[r, cc]
        end
        for r = 1:K
            Mᵢ[r, nmat] = b13[r]
            Mᵢ[nmat, r] = b13[r]
        end
        for r = 1:Mh, cc = 1:Mh
            Mᵢ[K+r, K+cc] = WΦWt[r, cc] + (r == cc ? AffExpr(δ) : AffExpr(0.0))
        end
        for r = 1:Mh
            Mᵢ[K+r, nmat] = b23[r]
            Mᵢ[nmat, K+r] = b23[r]
        end
        Mᵢ[nmat, nmat] = convert(AffExpr, sᵢ)
        # 𝒞⁰ inner approximation: Mᵢ = Pᵢ + Nᵢ with Pᵢ ⪰ 0 and Nᵢ ≥ 0 element-wise.
        # Mᵢ is symmetric by construction; symmetry of Nᵢ follows automatically.
        Pᵢ = @variable(model, [1:nmat, 1:nmat] in PSDCone())
        Nᵢ = @variable(model, [1:nmat, 1:nmat], lower_bound = 0.0)
        @constraint(model, [r=1:nmat, cc=1:nmat], Mᵢ[r, cc] == Pᵢ[r, cc] + Nᵢ[r, cc])
    end
    # objective from paper Eq. (28) / Theorem 5
    @objective(model, Min,
        dot(c, x) + ε^2 * λ
        + (1/nsmpl) * sum(
            s[i] + dot(q, ψ[i, :]) - λ * dot(samples[i], samples[i])
            + sum(φ[i, j] * q[j]^2 for j = 1:Nψ)
            for i = 1:nsmpl
        )
    )
    optimize!(model)
    return (
        objective_value = objective_value(model),
        x = value.(x),
        λ = value(λ),
        status = termination_status(model),
        model = model
    )
end

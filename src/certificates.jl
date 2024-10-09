# SumOfSquares Certificate for Schmüdgen Positivstellensatz
# this implementation is contributed by Benoît Legat on
# [Certificate Page](https://jump.dev/SumOfSquares.jl/stable/generated/Extension/certificate/)

import MultivariateBases as MB
const SOS = SumOfSquares
const SOSC = SOS.Certificate
struct Schmüdgen{IC <: SOSC.AbstractIdealCertificate, CT <: SOS.SOSLikeCone, BT <: SOS.AbstractPolynomialBasis} <: SOSC.AbstractPreorderCertificate
    ideal_certificate::IC
    cone::CT
    basis::Type{BT}
    maxdegree::Int
end

SOSC.cone(certificate::Schmüdgen) = certificate.cone

function SOSC.preprocessed_domain(::Schmüdgen, domain::BasicSemialgebraicSet, p)
    return SOSC.with_variables(domain, p)
end

function SOSC.preorder_indices(::Schmüdgen, domain::SOSC.WithVariables)
    n = length(domain.inner.p)
    if n >= Sys.WORD_SIZE
        error("There are $(2^n - 1) products in Schmüdgen's certificate, they cannot even be indexed with `$Int`.")
    end
    return map(SOSC.PreorderIndex, 1:(2^n-1))
end

function SOSC.multiplier_basis(certificate::Schmüdgen, index::SOSC.PreorderIndex, domain::SOSC.WithVariables)
    q = SOSC.generator(certificate, index, domain)
    return SOSC.maxdegree_gram_basis(certificate.basis, variables(domain), SOSC.multiplier_maxdegree(certificate.maxdegree, q))
end
function SOSC.multiplier_basis_type(::Type{Schmüdgen{IC, CT, BT}}) where {IC, CT, BT}
    return BT
end

function SOSC.generator(::Schmüdgen, index::SOSC.PreorderIndex, domain::SOSC.WithVariables)
    I = [i for i in eachindex(domain.inner.p) if !iszero(index.value & (1 << (i - 1)))]
    return prod([domain.inner.p[i] for i in eachindex(domain.inner.p) if !iszero(index.value & (1 << (i - 1)))])
end

SOSC.ideal_certificate(certificate::Schmüdgen) = certificate.ideal_certificate
SOSC.ideal_certificate(::Type{<:Schmüdgen{IC}}) where {IC} = IC

SOS.matrix_cone_type(::Type{<:Schmüdgen{IC, CT}}) where {IC, CT} = SOS.matrix_cone_type(CT)

using Infinity, DimensionalData

export depends, ∞

abstract type Dimension{T <: Real} <: Operator{T} end
SymbolicUtils.istree(::Dimension) = false
getops(::Dimension) = ()
# v4: symbolic time-independent solving
depends(op::Operator, x::Dimension) = x ∈ filter_type(Dimension, op)
"""Size as a map from an infinite set of ℂ^ℝ functions to ℂ^ℝ functions."""
Base.size(::Dimension) = (ℶ₂, ℶ₂) # Map from ψ ↦ ψ, a set of all dimensions ℂ^ℝ

"""Applies the privileged basis identity x̂∣ψ⟩ = x∣ψ⟩."""
(*)(dim::Dimension, u::AbstractDimArray) = mul!(similar(u), dim, u)
function LinearAlgebra.mul!(du, dim::Dimension, u::AbstractDimArray)
    du .= DimensionalData.val.(dims.(DimIndices(u), Ref(DimensionalData.key2dim(dim)))) .* u
    du
end
LinearAlgebra.mul!(du::AbstractVecOrMat, dim::Dimension, u::Union{DimensionalData.AbstractDimVector, DimensionalData.AbstractDimMatrix}) =
    invoke(LinearAlgebra.mul!, Tuple{Any, Dimension, AbstractDimArray}, du, dim, u)
has_mul!(::Dimension) = true

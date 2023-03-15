export depends

abstract type Dimension{T <: Real} <: Operator{T} end
SymbolicUtils.istree(::Dimension) = false
getops(::Dimension) = ()
filter_type(T::Type{<: Dimension}, op::Operator) = Iterators.filter(el -> el isa T, AbstractTrees.Leaves(op))
# v4: symbolic time-independent solving
depends(op::Operator, x::Dimension) = x ∈ filter_type(Dimension, op)
"""Size as a map from an infinite set of ℂ^ℝ functions to ℂ^ℝ functions."""
Base.size(::Dimension) = (ℶ₂, ℶ₂) # Map from ψ ↦ ψ, a set of all dimensions ℂ^ℝ

# Concrete types for abstract algebraic rings:
for (symbol, ring) ∈ pairs((𝔽₂=Bool, ℤ=Int, ℚ=Rational, ℝ=Float64, ℂ=ComplexF64))
    @eval const $symbol = $ring
    @eval export $symbol
    @eval getsymbol(::Type{$ring}) = $(Meta.quot(symbol))
end
getsymbol(T::Type) = Symbol(T)
const RealField = Union{ℚ, AbstractFloat} # A formally real field, in the abstract algebraic sense.

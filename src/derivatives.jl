for op in (:/, :*)
    @eval Base.$op(t::Tuple{Interpolations.WeightedIndex, Vararg{Interpolations.WeightedIndex}}, x::Number) = ($op(first(t), x), Base.tail(t)...)
end

struct Derivative{S <: Tuple, Weights <: Union{Nothing, AbstractDimArray}} <: Operator{ℂ}
    weights::Weights

    function SciMLOperators.cache_operator(::Derivative{S, Nothing}, ψ::State{M}) where {S <: Union{NTuple{1, Any}, NTuple{2, Any}}, M}
        @inline format(::Type{<: NTuple{1, Any}}, weightedindexes) = weightedindexes
        @inline format(::Type{<: NTuple{2, Any}}, weightedindexes) = Interpolations.symmatrix(weightedindexes)

        dimnums = dimnum.(Ref(ψ), Tuple(S.parameters))
        weights = NTuple{M, Interpolations.WeightedAdjIndex}[
            Base.getindex(format(S, Interpolations.weightedindexes(
                (
                    Interpolations.value_weights,
                    Interpolations.gradient_weights,
                    Interpolations.hessian_weights
                )[1:fieldcount(S) + 1],
                Interpolations.itpinfo(interpolate(ψ))...,
                convert(Tuple, coords)
            )), dimnums...) /
                prod(step.(Base.getindex.(Ref(dims(ψ)), dimnums)))
            for coords in CartesianIndices(ψ)]
        new{S, typeof(weights)}(weights)
    end

    Derivative{S}() where {S <: Union{NTuple{1, Any}, NTuple{2, Any}}} = new{S, Nothing}(nothing)
end
const ∂ = Derivative
export ∂
for i ∈ 2:10
    partial = Symbol("∂"*sup(i))
    @eval const $partial = ∂{NTuple{$i, Any}}
    @eval export $partial
end
(^)(::Type{∂{NTuple{n, S  }}}, m::ℤ) where {n, S} = ∂{NTuple{n*m, S  }}
(^)(::Type{∂{NTuple{n, Any}}}, m::ℤ) where  n     = ∂{NTuple{n*m, Any}}
(^)(::Type{∂                }, m::ℤ)              = ∂{NTuple{1,   Any}}^m

SciMLOperators.cache_operator(d::∂{S}, ψ::State) where S =
    # Fast path:
    dims(ψ) == dims(d.weights) ? d : cache_operator(∂{S}(), ψ)

Base.getindex(::Type{∂{NTuple{n, Any}}}, space::Space    ) where n = ∂{NTuple{n, space}}()
Base.getindex(::Type{∂                }, spaces::Space...)         = ∂{Tuple{spaces...}}()

"""Optimized for Hessians from second-order interpolation and finite differencing."""
∂{S}() where S = prod(
    s -> ∂{Tuple{s...}}(),
    S.parameters                          |>
        Base.Fix2(Iterators.partition, 2) |>
        collect                           |>
        reverse!
)

getops(d::∂{S}) where S = S.parameters
islinear(  ::∂) = true
isconstant(::∂) = true

Base.size(d::∂) = (ℶ₂, ℶ₂)

SymbolicUtils.operation(::∂) = ∂
# Base.show(io::IO, ::Type{<: ∂                   })         = print(io, "∂")
# Base.show(io::IO, ::Type{<: ∂{NTuple{n, <: Any}}}) where n = print(io, "∂"*sup(n))
function SymbolicUtils.show_call(io::IO, d::Type{<: ∂}, args::AbstractVector)
    print(io, "∂[")
    join(io, args, ", ")
    print(io, "]")
end
function SymbolicUtils.show_call(io::IO, d::Type{<: ∂{NTuple{n, <: Any}}}, args::AbstractVector) where n
    print(io, "∂", sup(n), "[")
    join(io, args[1], ", ")
    print(io, "]")
end

(*)(                          d::∂,                             ψ::State) = mul!(similar(ψ), d, ψ)
LinearAlgebra.mul!(dψ::State, d::∂{<: Tuple, Nothing         }, ψ::State) = mul!(dψ, cache_operator(d, ψ), ψ)
function LinearAlgebra.mul!(dψ::State, d::∂{<: Tuple, <: AbstractArray}, ψ::State)
    dψ .= (wis -> Interpolations.InterpGetindex(interpolate(ψ))[wis...]).(d.weights)
    dψ
end
has_mul!(::∂) = true

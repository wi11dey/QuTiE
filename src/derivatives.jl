struct Derivative{S <: Tuple, Weights} <: Operator{ℂ}
    weights::Weights

    function Derivative{Tuple{s}}(ψ::State) where s
        weights = Interpolations.weightedindexes.(
            Ref((
                Interpolations.value_weights,
                Interpolations.gradient_weights
            )),
            Interpolations.itpinfo(interpolate(ψ))...,
            DimIndices(ψ)
        ) .|>
            Base.Fix2(getindex, dimnum(ψ, s[]))
        new{Tuple{s}, typeof(weights)}(weights)
    end

    function Derivative{Tuple{s, t}}(ψ::State) where {s, t}
        weights = Interpolations.weightedindexes.(
            Ref((
                Interpolations.value_weights,
                Interpolations.gradient_weights,
                Interpolations.hessian_weights
            )),
            Interpolations.itpinfo(interpolate(ψ))...,
            DimIndices(ψ)
        )                            .|>
            Interpolations.symmatrix .|>
            hessian -> hessian[dimnum(ψ, s[]), dimnum(ψ, t[])]
        new{Tuple{s, t}, typeof(weights)}(weights)
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

Base.getindex(::Type{∂{NTuple{n, Any}}}, space::Space    ) where n = ∂{NTuple{n, space}}()
Base.getindex(::Type{∂                }, spaces::Space...)         = ∂{Tuple{spaces...}}()

cache_operator(d::Derivative{S}, ψ::State) where S = Derivative{S}(ψ)

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

(*)(                          d::∂,                             ψ::State) = mul!(similar(dψ),                d , ψ)
LinearAlgebra.mul!(dψ::State, d::∂{<: Tuple, Nothing         }, ψ::State) = mul!(        dψ , cache_operator(d), ψ)
LinearAlgebra.mul!(dψ::State, d::∂{<: Tuple, <: AbstractArray}, ψ::State) =
    dψ .= d.weights .|>
    wis -> Interpolations.InterpGetindex(interpolate(ψ))[wis...]

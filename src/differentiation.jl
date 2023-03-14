# v2: save weights on update_coefficients!
struct Differential{S <: Tuple, Weights} <: Operator{ℂ}
    weights::Weights

    cache_operator(d::Differential{Tuple{s}}, ψ::State) where s =
        new{Tuple{s}}(Interpolations.weighted_indexes.(
            Ref((
                Interpolations.value_weights,
                Interpolations.gradient_weights,
                Interpolations.hessian_weights
            )),
            Interpolations.itpinfo(interpolate(ψ))...,
            DimIndices(ψ)
        ) .|>
            Base.Fix2(Base.getindex, dimnum(ψ, s)))

    cache_operator(d::Differential{Tuple{s, t}}, ψ::State) where {s, t} =
        new{Tuple{s, t}}(Interpolations.weighted_indexes.(
            Ref((
                Interpolations.value_weights,
                Interpolations.gradient_weights,
                Interpolations.hessian_weights
            )),
            Interpolations.itpinfo(interpolate(ψ))...,
            DimIndices(ψ)
        )                            .|>
            Interpolations.symmatrix .|>
            m -> m[dimnum(ψ, s), dimnum(ψ, t)])

    Differential{S}() where {S <: NTuple{2, Any}} = new{S}(nothing)
end
const ∂ = Differential
export ∂
for i ∈ 2:10
    partial = Symbol("∂"*sup(i))
    @eval const $partial = ∂{NTuple{$i, Any}}
    @eval export $partial
end
(^)(::Type{∂{NTuple{n, S  }}}, m::ℤ) where {n, S} = ∂{NTuple{n*m, S  }}
(^)(::Type{∂{NTuple{n, Any}}}, m::ℤ) where  n     = ∂{NTuple{n*m, Any}}
(^)(::Type{∂                }, m::ℤ)              = ∂{NTuple{1,   Any}}^m

Base.getindex(::Type{∂{NTuple{n, Any}}}, space::Space) = ∂{NTuple{n, space}}()
Base.getindex(::Type{∂}, spaces::Space...) = ∂{Tuple{spaces...}}()

"""Optimized for Hessians from second-order interpolation and finite differencing."""
∂{S}() where S = prod(
    s -> ∂{Tuple{s...}}(),
    S.parameters                          |>
        Base.Fix2(Iterators.partition, 2) |>
        collect                           |>
        reverse!
)

getops(d::∂{S}) where S = S.parameters
islinear(  ::Differential) = true
isconstant(::Differential) = true

Base.size(d::∂) = (ℶ₂, ℶ₂)

# v2: use sup()
SymbolicUtils.operation(::∂) = ∂
Base.show(io::IO, ::Type{<: ∂}) = print(io, "∂")
function SymbolicUtils.show_call(io::IO, d::Type{<: ∂}, args::AbstractVector)
    print(io, d, "[")
    join(io, args, ", ")
    print(io, "]")
end

# v2: optimize for whole array access
struct Derivative{n, N, T} <: AbstractArray{ℂ, N}
    wrt::ℤ
    ψ::State{N}
    itp::AbstractInterpolation
end
(d::∂{n, T})(ψ::State{N}) where {n, N, T} = Derivative{n, N, T}(ψ.inv[d.wrt], ψ, interpolate(D.ψ))
Base.axes(D::Derivative, args...) = axes(D.ψ, args...)
function Base.to_indices(D::Derivative{1}, ax::Volume, indices)
    indices = to_indices(D.ψ, indices)
    map(Iterators.product(indices)) do coords
        Interpolations.weightedindexes(
            (
                Interpolations.value_weights,
                Interpolations.gradient_weights
            ),
            Interpolations.itpinfo(D.itp)...,
            coords
        )[D.wrt]
    end |> tuple
end
@propagate_inbounds Base.getindex(D::Derivative{1, N}, index::NTuple{N, Interpolations.WeightedIndex}) where N =
    Interpolations.InterpGetindex(D.itp)[index...]
Base.getindex(::ComposedOperator{NTuple{2, Derivative{1}}})

@inline (*)(                          d::∂, ψ::State) = mul!(similar(dψ), d, ψ)
@inline LinearAlgebra.mul!(dψ::State, d::∂, ψ::State) = dψ .= d(ψ)

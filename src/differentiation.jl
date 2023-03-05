# v2: save weights on update_coefficients!
struct Differential{n, T <: AbstractFloat} <: Operator{T}
    wrt::Dimension{T} # v6: ∂(::Time) for classical objects

    Base.getindex(::Type{Differential{n}}, wrt::Dimension{T}) where {n, T} = new{n, T}(wrt)
end
const ∂ = Differential
export ∂
(^)(::Type{∂{n}}, m::ℤ) where n = ∂{n*m}
for i ∈ 2:10
    partial = Symbol("∂"*sup(i))
    @eval const $partial = ∂{$i}
    @eval export $partial
end
Base.getindex(::Type{∂}, args...) = ∂{1}[args...]
Base.getindex(::Type{∂{n}}, args...) where n = ∂{1}[args...]^n
getops(d::∂) = (d.wrt,)
islinear(::Differential) = true

SymbolicUtils.operation(::∂{1}) = ∂
Base.size(d::∂) = size(d.wrt)

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

@inline (*)(                          d::∂, ψ::State) = mul!(similar(dψ), d, ψ)
@inline LinearAlgebra.mul!(dψ::State, d::∂, ψ::State) = dψ .= @view d(ψ)[:]

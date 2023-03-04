struct Differential{n, T <: AbstractFloat} <: LinearOperator{T}
    wrt::Dimension{T} # v6: ∂(::Time) for classical objects

    Base.getindex(::Type{∂{n}}, wrt::Dimension{T}) where {n, T} = new{n, T}(wrt)
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

SymbolicUtils.operation(::∂{1}) = ∂
Base.size(d::∂) = size(d.wrt)

@inline (*)(                          d::∂, ψ::State) = central_diff(     ψ; dims=d.wrt)
@inline LinearAlgebra.mul!(dψ::State, d::∂, ψ::State) = central_diff!(dψ, ψ; dims=d.wrt)

struct Derivative{n, N, T} <: AbstractArray{ℂ, N}
    wrt::ℤ
    ψ::State{N}
end
(d::∂{n, T})(ψ::State{N}) where {n, N, T} = Derivative{n, N, T}(ψ.inv[d.wrt], ψ)
@propagate_inbounds function Base.getindex(D::Derivative{1}, args...) =
    itp = convert(AbstractInterpolation, D.ψ)
    indices = to_indices(D.ψ, args)
    @boundscheck checkbounds(Bool, itp, indices...) || Base.throw_boundserror(D.ψ, x)
    map(Iterators.product(indices)) do coords
        # v2: reuse weights
        wis = Interpolations.weightedindexes(
            (
                Interpolations.value_weights,
                Interpolations.gradient_weights
            ),
            Interpolations.itpinfo(itp)...,
            coords
        )
        Interpolations.InterpGetindex(itp)[wis[D.wrt]...]
    end
end

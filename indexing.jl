Base.firstindex(ψ::State, space::Space) = first(axes(ψ, space))
Base.lastindex( ψ::State, space::Space) =  last(axes(ψ, space))

abstract type ConvertedIndex{N, T <: Real} <: AbstractArray{T, N} end
Base.to_index(index::ConvertedIndex) = index.i
IndexStyle(::ConvertedIndex) = IndexLinear()
Base.getindex(index::ConvertedIndex{0}) = index.i
Base.size(index::ConvertedIndex{0}) = ()
(::Type{>: T})(index::ConvertedIndex{0, T}) where T = index.i
Base.convert(::Type{>: T}, index::ConvertedIndex{0, T}) where T = T(index)
Base.size(index::ConvertedIndex{1}) = size(index.i)
@propagate_inbounds Base.getindex(index::ConvertedIndex{1}, j::ℤ) = index.i[j]
abstract type RawIndex{N, T <: Real} <: ConvertedIndex{N, T} end
struct RawPosition{T} <: RawIndex{0, T}
    i::T
end
struct RawRange{T} <: RawIndex{1, T}
    i::AbstractRange{T}
end
Base.convert(::Type{AbstractRange{>: T}}, index::RawRange{T}) where T = index.i
struct Sum <: ConvertedIndex{1, ℤ}
    i::Base.OneTo{ℤ}
end
Sum(len::ℤ) = Base.OneTo(len)
Base.convert(::Type{>: Base.OneTo{ℤ}}, index::Sum) = index.i

to_coordinate(i::Pair{<: Space}) = i
to_coordinate(i::Pair{<: Length}) = i.first.space => i.second
to_coordinate(i::Space) = i => (:)
to_coordinate(l::Length) = l.space => l.indices

# Fallback definitions (recursion broken by more specific methods):
@propagate_inbounds Base.to_index(ψ::State, i::Pair{<: Space }) = Base.to_index(ψ, axes(ψ, i.first) => i.second)
@propagate_inbounds Base.to_index(ψ::State, i::Pair{<: Length}) = Base.to_index(ψ, to_coordinate(i))
@inline Base.to_index(ψ::State, i::Pair{<: Length, Colon}) = i.first |> length |> Base.OneTo |> RawRange{ℤ}
@inline Base.to_index(ψ::State, i::Pair{<: Length, Missing}) = Sum(length(i.first))
@inline Base.to_index(ψ::State, i::(Pair{Length{T}, T} where T)) =
    nothing # v2
@inline Base.to_index(ψ::State, i::(Pair{Length{T}, <: AbstractRange{T}} where T)) =
    nothing # v2
@propagate_inbounds function Base.to_index(ψ::State, i::(Pair{Length{T}, Length{T}} where T))
    @boundscheck i.second.space === i.first || throw(DimensionMismatch())
    Base.to_index(ψ, i.first => (i.second.indices == axes(ψ, i.first).indices ? (:) : i.second.indices))
end
AbstractTrees.children(    ::Pair{<: Union{Length, Space}}) = ()
AbstractTrees.childrentype(::Pair{<: Union{Length, Space}}) = Tuple{}
# v2: optimize for whole array access
function Base.to_indices(
    ψ::State,
    ax::Volume,
    indices::Tuple{Vararg{Union{
        Pair{<: Union{Length, Space}},
        Space,
        Length,
        Volume,
        Colon,
        Type{..}
    }}})
    summing = true
    lookup = IdDict(
        to_coordinate(i)
        for i ∈ AbstractTrees.Leaves(indices)
            if !(i == (:) || i == ..) || (summing = false))
    Base.to_index.(
        Ref(ψ),
        ax .=> get.(
            Ref(lookup),
            Base.getfield.(ax, :space),
            summing ? missing : (:)))
end
Base.getindex(ψ::State{N}, indices::Vararg{ConvertedIndex, N}) where N = sum(Interpolations.interpolate(ψ)(indices...); dims=findall(index -> index isa Sum, indices))
Base.getindex(ψ::State{N}, indices::Vararg{RawIndex,       N}) where N =     Interpolations.interpolate(ψ)(indices...)
Base.getindex(ψ::State, ::Colon) = AbstractArray(ψ)
Base.getindex(ψ::State, ::Type{..}) = getindex(ψ, :)
Base.setindex!(ψ::State{N}, value, indices::Vararg{RawIndex{ℤ}, N}) where N = AbstractArray(ψ)[indices...] .= value
Base.setindex!(ψ::State  , value, ::Colon) = AbstractArray(ψ) .= value
Base.setindex!(ψ::State  , value, ::Type{..}) = setindex!(ψ, :, value)

include("symbolic_index.jl")

using Interpolations
import DimensionalData.LookupArrays

struct Interpolatable{T, D <: DimensionalData.Dimension{T}, IT <: AbstractInterpolation} <: DimensionalData.Dimension{T}
    val::D
    itp::IT
end

DimensionalData.set(dim::DimensionalData.Dimension, itp::AbstractInterpolation) = Interpolatable(dim, itp)
DimensionalData.name(itp::Interpolatable) = itp |> parent |> name

interpolated(itp::Interpolatable) = itp.itp
interpolated(dim::Dimension) = dim

struct InterpolationLookup{T, N, D} <: Unaligned{T, N}
    itp::AbstractInterpolation
    dim::D
end
DimensionalData.transformfunc(lookup::InterpolationLookup) = coords -> Interpolations.weightedindexes(
    (Interpolations.value_weights,),
    Interpolations.itpinfo(itp)...,
    coords
)
DimensionalData.transformdim(lookup::InterpolationLookup) = lookup.dim
DimensionalData.transformdim(::Type{InterpolationLookup{<: Any, <: Any, D}}) where D = D

struct Interpolated{T} <: LookupArrays.ArraySelector{T}
    val::T
end

DimensionalData.dims2indices(dims, indices::NonEmpty{Interpolated}) =
    dims2indices(dims .|> metadata .|> InterpolationLookup, indices)

LookupArrays.select_unalligned_indices(lookups::Tuple{Vararg{InterpolationLookup}}, sel::NonEmpty{Interpolated}) =
    transformfunc(lookups[1])(map(val, sel))

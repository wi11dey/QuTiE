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

struct InterpolationLookup{T, N} <: Unaligned{T, N}
    itp::AbstractInterpolation
end

struct Interpolated{T} <: LookupArrays.ArraySelector{T}
    val::T
end

DimensionalData.dims2indices(dims, ::Tuple{Vararg{Interpolated}}) =
    dims2indices(dims .|> metadata .|> InterpolationLookup)

function LookupArrays.select_unalligned_indices(lookups::Tuple{Vararg{InterpolationLookup}}, sel::Tuple{Interpolated, Vararg{Interpolated}})
    transformed = transformfunc(lookups[1])(map(val, sel))
end

struct Length{S, T} <: DimensionalData.Dimension{T}
    val::T
end
Length{S}(val::T) where {S, T} = Length{S, T}(val)
Length{S}() where S = Length{S}(:)

DimensionalData.name(::Type{<: Length{S}}) where S = name(S)
DimensionalData.basetypeof(::Type{<: Length{S}}) where S = Length{S}
DimensionalData.key2dim(s::Dimension) = Length{s}()
DimensionalData.dim2key(::Type{<: Length{S}}) where S = S

isfield(    ::Length{S}) where S = isfield(    S)
isbounded(  ::Length{S}) where S = isbounded(  S)
isperiodic( ::Length{S}) where S = isperiodic( S)
isclassical(::Length{S}) where S = isclassical(S)

Base.show(io::IO, l::Length) = print(io, "$(name(l))[$(l.val)]")

Base.getindex(s::Dimension, indices) = Length{s}(indices)

struct Length{S, T} <: DimensionalData.Dimension{T}
    val::T

    @inline function Length{S, T}(val::T) where {S, T}
        S isa Space || throw(ArgumentError("Lengths must be of Spaces"))
        new{S, T}(val)
    end
end
Length{S}(val::T) where {S, T} = Length{S, T}(val)
Length{S}() where S = Length{S}(:)

DimensionalData.name(::Type{<: Length{S}}) where S = name(S)
DimensionalData.basetypeof(::Type{<: Length{S}}) where S = Length{S}
DimensionalData.key2dim(s::Dimension) = Length{s}()
DimensionalData.dim2key(::Type{<: Length{S}}) where S = S

(::Type{Space})(::Length{S}) where S = S

Base.show(io::IO, l::Length) = print(io, "$(name(l))[$(l.val)]")

Base.getindex(s::Dimension, indices) = Length{s}(indices)

const Volume{N} = NTuple{N, Length}
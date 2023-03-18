const Length{S, T} = Dim{S, T}

DimensionalData.key2dim(s::Dimension) = Length{s}()

DimensionalData.Dimensions.basedims(s::Dimension) = DimensionalData.key2dim(s)
DimensionalData.Dimensions._w(      s::Dimension) = DimensionalData.key2dim(s)

Base.getindex(s::Dimension, indices...   ) = Length{s}(indices...)
Base.getindex(s::Dimension, indices::ℤ...) = Length{s}(indices...)

const Volume{N} = NTuple{N, Length}

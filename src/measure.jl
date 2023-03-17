const Length{S, T} = Dim{S, T}

key2dim(s::Dimension) = Length{s}()

Base.getindex(s::Dimension, indices...   ) = Length{s}(indices...)
Base.getindex(s::Dimension, indices::ℤ...) = Length{s}(indices...)

const Volume{N} = NTuple{N, Length}

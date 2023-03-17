struct Vacuum <: AbstractDimArray{ℂ, 0, Tuple{}, Array{ℂ, 0}} end
DimensionalData.parent(::Vacuum) = fill(zero(ℂ))
DimensionalData.dims(::Vacuum) = ()
DimensionalData.refdims(::Vacuum) = ()
DimensionalData.name(::Vacuum) = "∣⟩"
DimensionalData.metadata(::Vacuum) = DimensionalData.LookupArrays.NoMetadata()

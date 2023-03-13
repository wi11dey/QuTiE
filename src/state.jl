struct State{N, D <: Volume{N}, Orig <: AbstractArray{ℂ, N}, Interp <: AbstractInterpolation{ℂ, N}} <: AbstractDimArray{ℂ, N, D, Orig}
    data::DimArray{        ℂ, N, D, Orig}
    interpolated::DimArray{ℂ, N, D, Interpolation{ℂ, N, Interp}}

    function DimensionalData.rebuild(::Union{State, Vacuum}, data::AbstractArray, dims::Volume, refdims::Tuple, name, metdata)
        @boundscheck length(unique(dims)) == length(dims) || throw(DimensionMismatch("Duplicate dimensions"))
        sz = length.(dims)
        @boundscheck length(data) == sz || throw(DimensionMismatch("Mismatch between product of dimensions and length of data"))
        reshaped = Base.ReshapedArray(data, sz, ())
        spaces = Space.(dims)
        spec = map(spaces) do space
            isfield(space) || return NoInterp()
            BSpline(Quadratic(ifelse(isperiodic(space), Periodic, Natural)(OnCell())))
        end
        padded = Interpolations.padded_axes(dims, spec)
        itp = extrapolate(extrapolate(
            scale(Interpolations.BSplineInterpolation(
                ℂ,
                OffsetArray(Base.ReshapedArray(Vector{ℂ}(
                    undef,
                    padded
                    .|> length
                    |> prod
                ), sz, ()), padded),
                spec,
                dims
            ), dims...),
            ifelse.(isperiodic.(spaces), Periodic(), Throw())
        ), zero(ℂ)) |> Interpolation
        new{N, D, typeof(reshaped), typeof(itp)}(
            DimArray(reshaped, dims; refdims=refdims, name=name, metadata=metadata),
            DimArray(itp, set.(dims, DimensionalData.NoLookup()))
        )
    end
end
@propagate_inbounds State{N, D}(dims::D, data::Vector{ℂ}) where {N, D} = rebuild(Vacuum(), data)
State{N}(dims::Volume{N}) where N = @inbounds State{N}(dims, Vector{ℂ}(undef, dims .|> length |> prod))
function State{N}(ψ₀::Operator) where N
end

Base.parent(ψ::State) = ψ.data

for method in :(dims, refdims, data, name, metadata, layerdims)
    @eval DimensionalData.$method(ψ::State) = ψ |> parent |> $method
end

struct Interpolated <: DimensionalData.ArraySelector
end

@generated Base.getindex(ψ::State, indices...) =
    # Multiplexes 2 child DimArrays:
    (any(indices.parameters .<: Interpolated)
     ? :(ψ.interpolated[indices...])
     : :(ψ.data[        indices...]))

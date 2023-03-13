struct State{N, D <: Volume{N}, Orig <: AbstractArray{ℂ, N}, Interp <: AbstractInterpolation{ℂ, N}} <: AbstractDimArray{ℂ, N, D, Orig}
    original::DimArray{        ℂ, N, D, Orig}
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
        itp = scale(Interpolations.BSplineInterpolation(
            ℂ,
            OffsetArray(Base.ReshapedArray(Vector{ℂ}(
                undef,
                padded     .|>
                    length  |>
                    prod
            ), sz, ()), padded),
            spec,
            dims
        ), dims...)                                                                   |>
            Base.Fix2(extrapolate, ifelse.(isperiodic.(spaces), Periodic(), Throw())) |>
            Base.Fix2(extrapolate, zero(ℂ))                                           |>
            Interpolation
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

Base.parent(ψ::State) = ψ.original

for method in :(dims, refdims, data, name, metadata, layerdims).args
    @eval DimensionalData.$method(ψ::State) = ψ |> parent |> DimensionalData.$method
end

function Interpolations.prefilter!(ψ::State)
    itp = ψ.interpolated |> # DimArray
        parent           |> # Interpolation
        parent           |> # AbstractExtrapolation
        parent           |> # AbstractExtrapolation
        parent           |> # ScaledInterpolation
        parent              # BSplineInterpolation
    orig = ψ   |> # State
        parent |> # DimArray
        parent    # AbstractArray
    if axes(orig) == axes(itp.coefs)
        copyto!(itp.coefs, orig)
    else
        fill!(itp.coefs, zero(ℂ))
        copyto!(
            itp.coefs,
            orig |> axes |> CartesianIndices,
            orig,
            orig |> axes |> CartesianIndices
        )
    end
    Interpolations.prefilter!(real(ℂ), itp.coefs, Interpolations.itptype(itp))
end


struct Interpolated{T, S <: DimensionalData.Selector{T}} <: DimensionalData.Selector{T}
    parent::S
end
Base.parent(sel::Interpolated) = sel.parent

import DimensionalData.LookupArrays
for method in :(val, first, last, atol, rtol).args
    @eval LookupArrays.$method(sel::Interpolated) = sel |> parent |> LookupArrays.$method
end
for method in :(selectindices, hasselection).args
    @eval DimensionalData.$method(l::DimensionalData.LookupArray, sel::Interpolated; kw...) =
        DimensionalData.$method(l, parent(sel); kw...)
end

@generated Base.getindex(ψ::State, indices...) =
    # Multiplexes 2 child DimArrays:
    (any(indices.parameters .<: Interpolated)
     ? :(ψ.interpolated[indices...])
     : :(ψ.data[        indices...]))

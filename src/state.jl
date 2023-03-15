using OffsetArrays

struct State{N, D, Orig <: AbstractArray{ℂ, N}, Interp <: AbstractInterpolation{ℂ, N}} <: AbstractDimArray{ℂ, N, D, Orig}
    original::DimArray{    ℂ, N}
    interpolated::DimArray{ℂ, N}

    function DimensionalData.rebuild(::Union{State, Vacuum}, data::AbstractArray, dims::Volume{N}, refdims::Tuple, name, metdata) where N
        @boundscheck length(unique(dims)) == length(dims) || throw(DimensionMismatch("Duplicate dimensions"))
        sz = length.(dims)
        @boundscheck length(data) == prod(sz) || throw(DimensionMismatch("Mismatch between product of dimensions and length of data"))
        reshaped = Base.ReshapedArray(data, sz, ())
        spaces = Dimension.(dims)
        spec = map(spaces) do space
            isfield(space) || return NoInterp()
            BSpline(Quadratic(ifelse(isperiodic(space), Periodic, Natural)(OnCell())))
        end
        ax = Base.OneTo.(sz)
        padded = Interpolations.padded_axes(ax, spec)
        padded_sz = length.(padded)
        itp = scale(Interpolations.BSplineInterpolation(
            ℝ,
            OffsetArray(Base.ReshapedArray(Vector{ℂ}(
                undef,
                prod(padded_sz)
            ), padded_sz, ()), padded),
            spec,
            ax
        ), DimensionalData.val.(dims)...) |>
            Base.Fix2(extrapolate, ifelse.(
                isperiodic.(spaces),
                Ref(Periodic()),
                Ref(Throw())
            )) |>
                Base.Fix2(extrapolate, zero(ℂ)) |>
                Interpolation
        original = DimArray(reshaped, dims; refdims=refdims, name=name, metadata=metadata)
        new{N, typeof(DimensionalData.dims(original)), typeof(reshaped), typeof(parent(itp))}(
            original,
            DimArray(itp, set.(dims, Ref(DimensionalData.NoLookup())))
        )
    end
end
@propagate_inbounds State(dims::Volume, data) = rebuild(Vacuum(), data, dims)
State(::UndefInitializer, dims::Volume) = @inbounds State(dims, Vector{ℂ}(undef, dims .|> length |> prod))
State(dims) = State(undef, dims)
State(init, op::Operator) =
    Iterators.map(filter_type(Space, op)) do space
        # TODO
        space[-10.0:0.1:10.0]
    end        |>
        Volume |>
        Base.Fix1(State, init)

Base.parent(ψ::State) = ψ.original

for method in :(dims, refdims, data, name, metadata, layerdims).args
    @eval DimensionalData.$method(ψ::State) = ψ |> parent |> DimensionalData.$method
end

@inline Interpolations.interpolate(ψ::State) =
    ψ.interpolated |> # DimArray
    parent         |> # Interpolation
    parent            # AbstractExtrapolation
function Interpolations.prefilter!(ψ::State)
    itp = ψ         |> # State
        interpolate |> # AbstractExtrapolation
        parent      |> # AbstractExtrapolation
        parent      |> # ScaledInterpolation
        parent         # BSplineInterpolation
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

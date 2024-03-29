using OffsetArrays

struct State{N, D <: Tuple, R <: Tuple, Orig <: AbstractArray{ℂ, N}, Na, Me, Interp <: AbstractDimArray{ℂ, N}} <: AbstractDimArray{ℂ, N, D, Orig}
    original::DimArray{ℂ, N, D, R, Orig, Na, Me}
    interpolated::Interp

    function DimensionalData.rebuild(ψ::Union{State, Vacuum},
                                     data::AbstractArray,
                                     dims::NTuple{N, Any},
                                     refdims,
                                     name,
                                     metadata) where N
        isempty(dims) && return Vacuum()
        sz = length.(dims)
        @boundscheck length(data) == prod(sz) || throw(DimensionMismatch("Mismatch between product of dimensions and length of data"))
        reshaped = Base.ReshapedArray(convert(AbstractArray{ℂ}, data) |> vec |> parent, sz, ())
        original = DimArray(reshaped, dims; refdims=refdims, metadata=metadata)
        if DimensionalData.dims(ψ) == dims
            # Fast path:
            interpolated = ψ.interpolated
            new{typeof(ψ).parameters...}(original, interpolated)
        else
            spaces = dim2key.(dims)
            spec = map(spaces) do space
                isfield(space) || return NoInterp()
                BSpline(Cubic(ifelse(isperiodic(space), Periodic, Natural)(OnCell())))
            end
            ax = Base.OneTo.(sz)
            padded = Interpolations.padded_axes(ax, spec)
            itp = scale(Interpolations.BSplineInterpolation(
                ℝ,
                OffsetArray(Base.ReshapedArray(Vector{ℂ}(
                    undef,
                    prod(length.(padded))
                ), length.(padded), ()), padded),
                spec,
                ax
            ), parent.(parent.(dims))...) |>
                Base.Fix2(extrapolate, ifelse.(
                    isperiodic.(spaces),
                    Ref(Periodic()),
                    Ref(Throw())
                )) |>
                    Base.Fix2(extrapolate, zero(ℂ)) |>
                    Interpolation
            interpolated = DimArray(itp, set.(dims, Ref(DimensionalData.NoLookup())))
            new{
                N,
                typeof.(Base.getfield.(Ref(original), (
                    :dims,
                    :refdims,
                    :data,
                    :name,
                    :metadata
                )))...,
                typeof(interpolated)
            }(
                original,
                interpolated
            )
        end
    end
end
@propagate_inbounds DimensionalData.rebuild(
    ψ::Union{State, Vacuum};
    data,
    dims,
    refdims,
    name,
    metadata
) = rebuild(
    ψ,
    data,
    dims,
    refdims,
    name,
    metadata
)
@propagate_inbounds State(
    data::AbstractArray,
    dims::Tuple;
    refdims=(),
    name=DimensionalData.NoName(),
    metadata=DimensionalData.NoMetadata()
) = rebuild(
    Vacuum(),
    data,
    dims,
    refdims,
    name,
    metadata
)
State(::UndefInitializer, dims::Volume) = @inbounds State(Vector{ℂ}(undef, dims .|> length |> prod), dims)
State(::UndefInitializer, op::Operator) =
    filter_type(Space, op)            |>
    unique                           .|>
    (space -> space[-10.0:0.1:10.0])  |> # TODO
    Volume                            |>
    Base.Fix1(State, undef)
Base.ones(op::Operator) = fill!(State(undef, op), one(ℂ))

DimensionalData.show_after(io::IO, mime::MIME, ψ::State) = DimensionalData.show_after(io, mime, ψ.original)
for method in :(parent, dims, refdims, data, name, metadata, layerdims).args
    @eval DimensionalData.$method(ψ::State) = DimensionalData.$method(ψ.original)
end

@inline Interpolations.interpolate(ψ::State) =
    ψ.interpolated |> # DimArray
    parent         |> # Interpolation
    parent         |> # AbstractExtrapolation
    parent         |> # AbstractExtrapolation
    parent         |> # ScaledInterpolation
    parent            # BSplineInterpolation
function Interpolations.prefilter!(ψ::State)
    itp = interpolate(ψ)
    orig = parent(ψ)
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
    Interpolations.prefilter!(real(ℂ), itp.coefs, Interpolations.itpflag(itp))
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
     : :(ψ.original[    indices...]))
# Base.getindex(ψ::State, i::Integer, j::Integer, k::Integer...) =
#     ψ.original[i, j, k...]

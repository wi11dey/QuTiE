# const Reshaped{N} = Base.ReshapedArray{ℂ, N, SubArray{ℂ, 1, Vector{ℂ}, Tuple{Base.Slice{Base.OneTo{ℤ}}}, true}, Tuple{}}
# const Interpolation{N, Spec} = Interpolations.BSplineInterpolation{ℂ, N, OffsetArray{ℂ, N, Reshaped{N}}, Spec, NTuple{N, Base.OneTo{ℤ}}}
# const Extrapolation{N, Spec} = Interpolations.FilledExtrapolation{ℂ, N, Interpolations.{Interpolation{N, Spec}}, Spec, ℂ}
struct State{N, D <: Volume{N}, Orig <: AbstractArray{ℂ, N}, Interp <: AbstractInterpolation{ℂ, N}} <: AbstractDimArray{ℂ, N, D, Grandparent{N}}
    data::DimArray{        ℂ, N, D, Orig}
    interpolated::DimArray{ℂ, N, D, IT  }

    @propagate_inbounds function State{N, D}(dims::D, data::Vector{ℂ}) where {N, D}
        @boundscheck length(unique(dims)) == length(dims) || throw(DimensionMismatch("Duplicate dimensions"))
        sz = length.(dims)
        @boundscheck length(data) == sz || throw(DimensionMismatch("Mismatch between product of dimensions and length of data"))

        @inline reshape_view(v::AbstractVector, dims::Dims) = Base.ReshapedArray(v, dims, ())
        @inline reshape_view(v::AbstractVector, dims::Dims) = Base.ReshapedArray(v, dims, ())

        reshaped = reshape_view(data, sz)
        spaces = Space.(dims)
        spec = map(spaces) do space
            isfield(space) || return NoInterp()
            BSpline(Quadratic(ifelse(isperiodic(space), Periodic, Natural)(OnCell())))
        end
        padded = Interpolations.padded_axes(dims, spec)
        itp = extrapolate(extrapolate(
            Interpolations.BSplineInterpolation(
                ℂ,
                OffsetArray(reshape_view(Vector{ℂ}(
                    undef,
                    padded
                    .|> length
                    |> prod
                ), sz), padded),
                spec,
                dims
            ),
            ifelse.(isperiodic.(spaces), Periodic(), Throw())
        ), zero(ℂ))
        new{N, D, typeof(reshaped), typeof(itp)}(
            DimArray(reshaped, dims),
            DimArray(itp, set.(dims, DimensionalData.NoLookup()))
        )
    end
    State{N}(ψ::State{N}) where N = new{N}(ψ.data, ψ.interpolated)
end
State{N}(ax::Volume{N}) where N = State{N}(ax, Vector{ℂ}(undef, ax .|> length |> prod))
Base.copy(ψ::State) = State(ψ)
Base.axes(ψ::State) = ψ.ax
function Base.vec( ψ::State)
    ψ.data    # DimArray
    |> parent # ReshapedArray
    |> parent # Vector
end
Base.size(ψ::State) = length.(axes(ψ))
Base.length(ψ::State) = prod(size(ψ))
Base.IteratorSize(::Type{State{N}}) where N = Base.HasShape{N}()
Base.IteratorEltype(::Type{<: State}) = Base.HasEltype()
Base.eltype(::Type{<: State}) = ℂ
Base.dims(ψ::State) = set.(ψ.dims, Ref(ψ.itp))

function Base.axes(ψ::State{N}, space::Space) where N
    @boundscheck space ∈ ψ.inv || throw(DimensionMismatch())
    @inbounds ψ.ax[ψ.inv[space]]
end
Base.size(ψ::State, space::Space) = length(axes(ψ, space))

Base.similar(::Type{>: State{N}}, ax::Volume{N}) where N = State(ax)

Base.fill!(ψ::State, value::ℂ) = fill!(vec(ψ), value)
Base.fill(value::ℂ, ax::NonEmptyVolume) = fill!(similar(State, ax), value)
Base.zeros(T::Type{ℂ}, ax::NonEmptyVolume) = fill(zero(ℂ), ax)
Base.ones( T::Type{ℂ}, ax::NonEmptyVolume) = fill( one(ℂ), ax)
Base.zeros(ax::NonEmptyVolume) = zeros(ℂ, ax)
Base.ones( ax::NonEmptyVolume) =  ones(ℂ, ax)

(::Type{>: AbstractArray{ℂ, N}})(ψ::State{N}) where N = reshape(vec(ψ), size(ψ))
Base.convert(::Type{>: AbstractArray{ℂ, N}}, ψ::State{N}) where N = AbstractArray{ℂ, N}(ψ)

Base.iterate(ψ::State, args...) = iterate(AbstractArray(ψ), args...)
Base.broadcastable(ψ::State) = AbstractArray(ψ)

Interpolations.interpolate(ψ::State) =
    extrapolate(
        interpolate(
            AbstractArray(ψ),
            ifelse.(isperiodic.(axes(ψ)),
                    Periodic(OnCell()),
                    Natural( OnCell()))
            .|> Quadratic
            .|> BSpline),
        ifelse.(isperiodic.(axes(ψ)),
                Periodic(),
                zero(ℂ)))

include("indexing.jl")

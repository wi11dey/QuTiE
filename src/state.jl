# const Reshaped{N} = Base.ReshapedArray{ℂ, N, SubArray{ℂ, 1, Vector{ℂ}, Tuple{Base.Slice{Base.OneTo{ℤ}}}, true}, Tuple{}}
# const Interpolation{N, Spec} = Interpolations.BSplineInterpolation{ℂ, N, OffsetArray{ℂ, N, Reshaped{N}}, Spec, NTuple{N, Base.OneTo{ℤ}}}
# const Extrapolation{N, Spec} = Interpolations.FilledExtrapolation{ℂ, N, Interpolations.{Interpolation{N, Spec}}, Spec, ℂ}
struct State{N, D <: Volume{N}, Orig <: AbstractArray{ℂ, N}, Interp <: AbstractInterpolation{ℂ, N}} <: AbstractDimArray{ℂ, N, D, Grandparent{N}}
    data::DimArray{        ℂ, N, D, Orig}
    interpolated::DimArray{ℂ, N, D, IT  }

    @propagate_inbounds function State{N}(ax::D, data::Vector{ℂ}) where N
        @boundscheck length(unique(ax)) == length(ax) || throw(DimensionMismatch("Duplicate dimensions."))
        reshaped = reshape(@inbounds(view(data, :)), length.(ax))
        spec = ifelse.(isfield.(ax),
                       ifelse.(isperiodic.(ax),
                               Periodic(OnCell()),
                               Natural( OnCell()))
                       .|> Quadratic
                       .|> BSpline,
                       NoInterp())
        padded = Interpolations.padded_axes(ax, spec)
        prefiltered = reshape(view(Vector{ℂ}(undef, padded .|> length |> prod), :), padded)
        itp = Interpolations.BSplineInterpolation(ℂ, prefiltered, spec, ax)
        new{N}(DimArray(reshaped, ax),
               DimArray(extrapolate(extrapolate(
                   itp,
                   map(ax) do l
                       isperiodic(l) && Periodic()
                       
                   end
                   ifelse.(isperiodic.(ax),
                           Periodic(),
                           ifelse())
               ), zero(ℂ)), set.(ax, DimensionalData.NoLookup())),
               extrapolate(interpolated,
                           ifelse.(isperiodic.(axes(ψ)),
                           Periodic(),
                           zero(ℂ))))
    end
    State{N}(ψ::State{N}) where N = new{N}(ψ.ax, copy(ψ.data), ψ.inv)
end
State{N}(ax::Volume{N}) where N = @inbounds State{N}(ax, Vector{ℂ}(undef, ax .|> length |> prod))
Base.copy(ψ::State) = State(ψ)
Base.axes(ψ::State) = ψ.ax
Base.vec( ψ::State) = @inbounds @view ψ.data[:]
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

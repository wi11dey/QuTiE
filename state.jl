struct State{N} <: AbstractArray{ℂ, N}
    ax::Volume{N}
    data::Vector{ℂ}

    inv::IdDict{Space, ℤ}

    function State{N}(ax::Volume{N}, data::Vector{ℂ}) where N
        @boundscheck length(data) == ax .|> length |> prod || throw(DimensionMismatch())
        @boundscheck length(unique(ax)) == length(ax) || throw(DimensionMismatch("Duplicate dimensions."))
        new{N}(
            ax,
            push!(data, zero(ℂ)),

            Base.getindex.(ax, :space) |> enumerate .|> reverse |> IdDict
        )
    end
    State{N}(ψ::State{N}) where N = new{N}(ψ.ax, copy(ψ.data), ψ.inv)
end
State{N}(ax::Volume{N}) where N = @inbounds State{N}(ax, Vector{ℂ}(undef, ax .|> length |> prod))
Base.copy(ψ::State) = State(ψ)
Base.axes(ψ::State) = ψ.ax
Base.vec( ψ::State) = @inbounds @view ψ.data[begin:end - 1]
Base.size(ψ::State) = length.(axes(ψ))
Base.length(ψ::State) = prod(size(ψ))
Base.IteratorSize(::Type{State{N}}) where N = Base.HasShape{N}()
Base.IteratorEltype(::Type{<: State}) = Base.HasEltype()
Base.eltype(::Type{<: State}) = ℂ

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

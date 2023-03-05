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

function Base.axes(ψ::State{N}, space::Space) where N
    @boundscheck space ∈ ψ.inv || throw(DimensionMismatch())
    @inbounds ψ.ax[ψ.inv[space]]
end

Base.similar(::Type{>: State{N}}, ax::Volume{N}) where N = State(ax)

Base.fill!(ψ::State, value::ℂ) = fill!(vec(ψ), value)
Base.fill(value::ℂ, ax::NonEmptyVolume) = fill!(similar(State, ax), value)
Base.zeros(T::Type{ℂ}, ax::NonEmptyVolume) = fill(zero(ℂ), ax)
Base.ones( T::Type{ℂ}, ax::NonEmptyVolume) = fill( one(ℂ), ax)
Base.zeros(ax::NonEmptyVolume) = zeros(ℂ, ax)
Base.ones( ax::NonEmptyVolume) =  ones(ℂ, ax)

include("indexing.jl")

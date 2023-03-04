const Volume{N} = NTuple{N, Length}
const NonEmptyVolume = Tuple{Length, Vararg{Length}} # May still be the null set if all lengths zero. All unqualified references to `Volume` should be NonEmptyVolume to avoid matching `Tuple{}`
Base.IteratorEltype(::Type{<: AbstractTrees.TreeIterator{<: NonEmptyVolume}}) = Base.HasEltype()
Base.eltype(::Type{<: AbstractTrees.TreeIterator{<: NonEmptyVolume}}) = Length
×(factors::Union{Length, NonEmptyVolume}...) = Volume(AbstractTrees.Leaves(factors))

Base.show(io::IO, vol::NonEmptyVolume) = join(io, vol, " × ")

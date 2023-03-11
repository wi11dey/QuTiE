export ×

const Volume{N} = NTuple{N, Length}
×(factors::Union{Length, NonEmptyVolume}...) = Volume(AbstractTrees.Leaves(factors))

Base.show(io::IO, vol::NonEmptyVolume) = join(io, vol, " × ")

struct Length{T, name} <: AbstractRange{T}
    space::Space{T, name}
    indices::Axis{name, AbstractRange{T}}
end
Length(space::Space{T, name}) where {T, name} = Length{T, name}(space, Axis(space))

Base.length(l::Length) = length(l.indices)
Base.first( l::Length) =  first(l.indices)
Base.last(  l::Length) =   last(l.indices)

isbounded(  l::Length) = isbounded(  l.space)
isperiodic( l::Length) = isperiodic( l.space)
isclassical(l::Length) = isclassical(l.space)

Base.show(io::IO, l::Length{T, name}) where {T, name} = print(io, "$name[$(l.indices)]")

Base.getindex(space::Space{T, name}, indices::AbstractRange{T}) where {T, name} = Length{T, name}(space, Axis{name}(indices))

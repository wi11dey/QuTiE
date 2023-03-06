struct Length{T} <: AbstractRange{T}
    space::Space{T}
    indices::AbstractRange{T}
end
# v3: function boundary detection by binary search
# v4: symbolic function boundary detection
# v4/5: intelligently select first dimension symbolically
"""Default range for provided space."""
function Length{T}(space::Space{T}) where T
    # TODO
    Length{T}(space, -10.0:10.0)
end
Length{T}(space::Space{T}) where {T <: Integer} = Length{T}(space, -10:10)
(::Type{Length})(space::Space{T}) where T = Length{T}(space)
Base.convert(::Type{T}, space::Space) where {T <: Length} = T(space)

Base.length(l::Length) = length(l.indices)
Base.first( l::Length) =  first(l.indices)
Base.last(  l::Length) =   last(l.indices)

isbounded(  l::Length) = isbounded(  l.space)
isperiodic( l::Length) = isperiodic( l.space)
isclassical(l::Length) = isclassical(l.space)

function Base.show(io::IO, l::Length)
    name = get(get(io, :spaces, IdDict{Space, Char}()), l.space, nothing)
    if isnothing(name)
        print(io, "Length{$(getsymbol(eltype(l.space)))}($(l.space.lower)..$(l.space.upper), $(l.indices))")
        return
    end
    print(io, "$name[$(l.indices)]")
end

AbstractTrees.children(::Length) = ()
AbstractTrees.childrentype(::Type{<: Length}) = Tuple{}

Base.getindex(space::Space, indices::AbstractRange) = Length(space, indices)

Base.firstindex(ψ::State, space::Space) = first(axes(ψ, space))
Base.lastindex( ψ::State, space::Space) =  last(axes(ψ, space))

include("symbolic_index.jl")

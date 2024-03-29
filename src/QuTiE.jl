module QuTiE

import Base: +, -, *, /, ^, ==, @propagate_inbounds
import TermInterface, SymbolicUtils
using LinearAlgebra
# using ModelingToolkit, SymbolicUtils # v4/5
# using MarchingCubes, ConstructiveGeometry, Compose # v6/7

include("define.jl")
include("algebra.jl")
include("scripts.jl")
include("strong_limit_cardinals.jl")
include("operators.jl")
include("dirac_delta.jl")
include("dimension.jl")
include("measure.jl")
include("commute.jl")
include("time.jl")
include("space.jl")
include("qubit.jl")
include("interpolation.jl")
include("vacuum.jl")
include("state.jl")
include("derivatives.jl")
include("lie_groups.jl")
include("classical.jl")

# function trim!(ψ::State)
#     r = axes(ψ)[1]
#     while abs(ψ.data[begin]) < ψ.ε
#         popfirst!(ψ.data)
#     end
# end

# function LinearAlgebra.normalize!(ψ::State)
# end

# """
# Tensor product of multiple states.
# """
# @propagate_inbounds Base.kron(ψ::State, φ::State) = State(axes(ψ)×axes(φ), kron(vec(ψ), vec(φ)))
# const ⊗ = kron

# function LinearAlgebra.dot(ψ::State, φ::State)
#     if axes(ψ) == axes(φ)
#         return ψ.data⋅φ.data
#     end
# end

# using MakieCore

end

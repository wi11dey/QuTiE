#!/usr/bin/env julia

module QuTiE

import Base: +, -, *, /, ^, ==, @propagate_inbounds
import TermInterface, SymbolicUtils
import AbstractTrees
using LinearAlgebra
using Interpolations
# using ModelingToolkit, SymbolicUtils # v4/5
# using MarchingCubes, ConstructiveGeometry, Compose # v6/7

export 𝑖, ∜, ×, depends, isbounded, isclassical

const 𝑖 = im
∜(x::ℝ) = x^(1/4)

include("algebra.jl")
include("scripts.jl")
include("infinites.jl")
include("operators.jl")
include("dirac_delta.jl")
include("dimension.jl")
include("commute.jl")
include("time.jl")
include("space.jl")
include("qubits.jl")
include("length.jl")
include("volume.jl")
include("state.jl")

Base.axes(op::Operator) = filter_type(Space, op) |> unique |> Volume

abstract type RawIndex{T <: Real, N} <: AbstractArray{T, N} end
IndexStyle(::RawIndex) = IndexLinear()
struct RawPosition{T} <: RawIndex{T, 0}
    pos::T
end
Base.getindex(pos::RawPosition) = pos.pos
Base.size(::RawPosition) = ()
Base.to_index(pos::RawPosition) = pos.pos
(::Type{>: T})(pos::RawPosition{T}) where T = pos.pos
Base.convert(::Type{>: T}, pos::RawPosition{T}) where T = T(pos)
struct RawRange{T} <: RawIndex{T, 1}
    range::AbstractRange{T}
end
Base.getindex(range::RawRange, i::ℤ) = range.range[i]
Base.size(range::RawRange) = size(range.range)
Base.to_index(range::RawRange) = range.range
Base.convert(::Type{AbstractRange{>: T}}, range::RawRange{T}) where T = T(range.range)

Base.convert(::Type{>: Pair{Space{T}}}, i::Pair{Length{T}}) where T = i.first.space => i.second

# Fallback definitions (recursion broken by more specific methods):
@propagate_inbounds Base.to_index(ψ::State, i::Pair{<: Space}) =
    Base.to_index(ψ, axes(ψ, i.first) => i.second)
@propagate_inbounds Base.to_index(ψ::State, i::Pair{<: Length}) =
    Base.to_index(ψ, convert(Pair{Space}, i))
@inline Base.to_index(ψ::State, i::Pair{<: Length, Colon}) = i.first |> length |> Base.OneTo |> RawRange{ℤ}
@inline Base.to_index(ψ::State, i::Pair{<: Length, Missing}) = (:)
@inline Base.to_index(ψ::State, i::(Pair{Length{T}, T} where T)) =
    nothing
@inline Base.to_index(ψ::State, i::(Pair{Length{T}, <: AbstractRange{T}} where T)) =
    nothing
@propagate_inbounds function Base.to_index(ψ::State, i::(Pair{Length{T}, Length{T}} where T)) =
    @boundscheck i.second.space === i.first || throw(DimensionMismatch())
    Base.to_index(ψ, i.first => (i.second.indices == axes(ψ, i.first).indices ? (:) : i.second.indices))
end
AbstractTrees.children(::Pair{<: Space}) = ()
AbstractTrees.childrentype(::Pair{<: Space}) = Tuple{}
function Base.to_indices(
    ψ::State{N},
    ax::Volume{N},
    indices::Tuple{Vararg{Union{
        Pair{<: Union{Length, Space}},
        Length,
        Volume,
        Colon,
        Type{..}
    }}}
    ) where N =
    summing = true
    lookup = IdDict(
        convert(Pair{Space}, i)
        for i ∈ AbstractTrees.Leaves(indices)
            if !(i == : || i == ..) || (summing = false))
    Base.to_index.(
        Ref(ψ),
        ax .=> get.(
            Ref(lookup),
            Base.getfield.(ax, :space),
            summing ? missing : (:)
        )
    )
end

@inline Base.convert(::Type{>: AbstractExtrapolation}, ψ::State) = extrapolate(
    interpolate(
        reshape(
            ψ |> vec,
            ψ |> axes .|> length
        ),
        map(axes(ψ)) do l
            BSpline(Quadratic((l.space.periodic ? Periodic : Natural)(OnCell())))
        end
    ),
    map(axes(ψ)) do l
        l.space.periodic ? Periodic : zero(ℂ)
    end
)

Base.getindex(ψ::State, indices::RawIndex...) =
    convert(AbstractInterpolation, ψ)(to_indices(ψ, indices)...)

Base.setindex!(ψ::State, indices::RawIndex{ℤ}...) = nothing

Base.similar(::Type{State}, ax::Volume{N}) where N = similar(State{N}, ax)
Base.similar(::Type{State{N}}, ax::Volume{N}) where N = State(ax)

Base.fill!(ψ::State, value::ℂ) = fill!(vec(ψ), value)
Base.fill(value::ℂ, ax::NonEmptyVolume) = fill!(similar(State, ax), value)
Base.zeros(T::Type{ℂ}, ax::NonEmptyVolume) = fill(zero(ℂ), ax)
Base.ones( T::Type{ℂ}, ax::NonEmptyVolume) = fill( one(ℂ), ax)
Base.zeros(ax::NonEmptyVolume) = zeros(ℂ, ax)
Base.ones( ax::NonEmptyVolume) =  ones(ℂ, ax)

include("differentiation.jl")

function trim!(ψ::State)
    r = axes(ψ)[1]
    while abs(ψ.data[begin]) < ψ.ε
        popfirst!(ψ.data)
    end
end

function LinearAlgebra.normalize!(ψ::State)
end

"""
Tensor product of multiple states.
"""
@propagate_inbounds Base.kron(ψ::State, φ::State) = State(axes(ψ)×axes(φ), kron(vec(ψ), vec(φ)))
const ⊗ = kron

function LinearAlgebra.dot(ψ::State, φ::State)
    if axes(ψ) == axes(φ)
        return ψ.data⋅φ.data
    end
end

# using MakieCore

include("lie_groups.jl")
include("classical.jl")

end

using ..QuTiE
using DifferentialEquations
# using AlgebraOfGraphics
using LaTeXStrings
# using Makie
# try
#     using GLMakie
# catch
#     using CairoMakie
# end
using Revise
using PhysicalConstants: CODATA2018, PhysicalConstant

for name in names(CODATA2018, all=true)
    @eval if CODATA2018.$name isa PhysicalConstant
        import PhysicalConstants.CODATA2018: $name
        export $name
    end
end

const ħ² = ħ^2

__revise_mode__ = :evalassign

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) ≠ 1 || ARGS[1] ∈ ("-h", "--help", "-?")
        error("Usage: ./$PROGRAM_FILE spec.jl")
    end
    includet(ARGS[1])
    if !isdefined(:ψ₀)
        error("Must define initial state ψ₀")
    end
    if isdefined(:H) || isdefined(:ℋ)
        if isdefined(:H) && isdefined(:ℋ)
            error("Cannot define both H and ℋ")
        end
        if isdefined(:L) || isdefined(:ℒ)
            error("Cannot define both Hamiltonian and Lagrangian")
        end
        if isdefined(:ℋ)
            # Canonicalize:
            H = ℋ
        end
        # Hamiltonian formulation:
        ψ = Observable([ψ₀*ones(axes(H))])
        integrator = init(ODEProblem(-im*H/ħ, ψ[][begin], (0, ∞)))
    end
    if isdefined(:L) || isdefined(:ℒ)
        if isdefined(:L) && isdefined(:ℒ)
            error("Cannot define both L and ℒ")
        end
        if isdefined(:H) || isdefined(:ℋ)
            error("Cannot define both Lagrangian and Hamiltonian")
        end
        if isdefined(:ℒ)
            # Canonicalize:
            L = ℒ
        end
        # v2: Lagrangian formulation
    end
    if !isdefined(:📊) && isdefined(:output)
        📊 = output
    end
    if !isdefined(:📊)
        📊 = visual(Wireframe)
    end
    if isdefined(:x)
        # Cartesian default.
        if isdefined(:y)
            if isdefined(:z)
            end
        end
    elseif isdefined(:r) || isdefined(:ρ)
        if isdefined(:θ) && isdefined(:φ)
            # Spherical default.
        elseif (isdefined(:θ) || isdefined(:φ)) && isdefined(:z)
            #
        end
    else
    end
    ψ = Observable(ψ₀*ones(axes(H)))
    draw(data(ψ)*📊)
    for (ψ, t) ∈ tuples(integrator)
        @show ψ, t
    end
end

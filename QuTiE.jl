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

abstract type ConvertedIndex{N, T <: Real} <: AbstractArray{T, N} end
Base.to_index(index::ConvertedIndex) = index.i
IndexStyle(::ConvertedIndex) = IndexLinear()
Base.getindex(index::ConvertedIndex{0}) = index.i
Base.size(index::ConvertedIndex{0}) = ()
(::Type{>: T})(index::ConvertedIndex{0, T}) where T = index.i
Base.convert(::Type{>: T}, index::ConvertedIndex{0, T}) where T = T(index)
Base.size(index::ConvertedIndex{1}) = size(index.i)
@propagate_inbounds Base.getindex(index::ConvertedIndex{1}, j::ℤ) = index.i[j]
abstract type RawIndex{N, T <: Real} <: ConvertedIndex{N, T} end
struct RawPosition{T} <: RawIndex{0, T}
    i::T
end
struct RawRange{T} <: RawIndex{T, 1}
    i::AbstractRange{T}
end
Base.convert(::Type{AbstractRange{>: T}}, index::RawRange{T}) where T = index.i
struct Sum <: ConvertedIndex
    i::Base.OneTo{ℤ}
end
Sum(len::ℤ) = Base.OneTo(len)
Base.convert(::Type{>: Base.OneTo{ℤ}}, index::Sum) = index.i

to_coordinate(i::Pair{<: Space}) = i
to_coordinate(i::Pair{<: Length}) = i.first.space => i.second
to_coordinate(i::Space) = i => (:)
to_coordinate(l::Length) = l.space => l.indices

# Fallback definitions (recursion broken by more specific methods):
@propagate_inbounds Base.to_index(ψ::State, i::Pair{<: Space }) = Base.to_index(ψ, axes(ψ, i.first) => i.second)
@propagate_inbounds Base.to_index(ψ::State, i::Pair{<: Length}) = Base.to_index(ψ, to_coordinate(i))
@inline Base.to_index(ψ::State, i::Pair{<: Length, Colon}) = i.first |> length |> Base.OneTo |> RawRange{ℤ}
@inline Base.to_index(ψ::State, i::Pair{<: Length, Missing}) = Sum(length(i.first))
@inline Base.to_index(ψ::State, i::(Pair{Length{T}, T} where T)) =
    nothing # v2
@inline Base.to_index(ψ::State, i::(Pair{Length{T}, <: AbstractRange{T}} where T)) =
    nothing # v2
@propagate_inbounds function Base.to_index(ψ::State, i::(Pair{Length{T}, Length{T}} where T)) =
    @boundscheck i.second.space === i.first || throw(DimensionMismatch())
    Base.to_index(ψ, i.first => (i.second.indices == axes(ψ, i.first).indices ? (:) : i.second.indices))
end
AbstractTrees.children(::Pair{<: Union{Length, Space}}) = ()
AbstractTrees.childrentype(::Pair{<: Union{Length, Space}}) = Tuple{}
# v2: optimize for whole array access
function Base.to_indices(
    ψ::State,
    ax::Volume,
    indices::Tuple{Vararg{Union{
        Pair{<: Union{Length, Space}},
        Space,
        Length,
        Volume,
        Colon,
        Type{..}
    }}}) =
        summing = true
    lookup = IdDict(
        to_coordinate(i)
        for i ∈ AbstractTrees.Leaves(indices)
            if !(i == : || i == ..) || (summing = false))
    Base.to_index.(
        Ref(ψ),
        ax .=> get.(
            Ref(lookup),
            Base.getfield.(ax, :space),
            summing ? missing : (:)))
end
(::Type{>: AbstractArray{ℂ, N}})(ψ::State{N}) where N =
    reshape(
        ψ |> vec,
        ψ |> axes .|> length
    )
Base.convert(::Type{>: AbstractArray{ℂ, N}}, ψ::State{N}) where N =
    AbstractArray{ℂ, N}(ψ)

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
Base.getindex(ψ::State{N}, indices::Vararg{ConvertedIndex, N}) where N = sum(Interpolations.interpolate(ψ)(indices...); dims=findall(index -> index isa Sum, indices))
Base.getindex(ψ::State{N}, indices::Vararg{RawIndex,       N}) where N =     Interpolations.interpolate(ψ)(indices...)
Base.setindex!(ψ::State{N}, indices::Vararg{RawIndex{ℤ}, N}, value) where N =
    AbstractArray(ψ)[indices...] .= value

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

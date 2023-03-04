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

struct Differential{n, T <: AbstractFloat} <: LinearOperator{T}
    wrt::Dimension{T} # v6: ∂(::Time) for classical objects

    Base.getindex(::Type{∂{n}}, wrt::Dimension{T}) where {n, T} = new{n, T}(wrt)
end
const ∂ = Differential
export ∂
(^)(::Type{∂{n}}, m::ℤ) where n = ∂{n*m}
for i ∈ 2:10
    partial = Symbol("∂"*sup(i))
    @eval const $partial = ∂{$i}
    @eval export $partial
end
Base.getindex(::Type{∂}, args...) = ∂{1}[args...]
Base.getindex(::Type{∂{n}}, args...) where n = ∂{1}[args...]^n
getops(d::∂) = (d.wrt,)

SymbolicUtils.operation(::∂{1}) = ∂
Base.size(d::∂) = size(d.wrt)

Base.axes(op::Operator) = filter_type(Space, op) |> unique |> Volume

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

@inline (*)(                          d::∂, ψ::State) = central_diff(     ψ; dims=d.wrt)
@inline LinearAlgebra.mul!(dψ::State, d::∂, ψ::State) = central_diff!(dψ, ψ; dims=d.wrt)
# dψ .= (diff(  cat((   wrt.periodic ? identity : zeros∘axes)(ψ[d.wrt => end  ]), ψ; dims=d.wrt), d.wrt)
#        + diff(cat(ψ, (wrt.periodic ? identity : zeros∘axes)(ψ[d.wrt => begin])   ; dims=d.wrt), d.wrt))/2

function Base.axes(ψ::State{N}, space::Space) where N
    @boundscheck space ∈ ψ.inv || throw(DimensionMismatch())
    @inbounds ψ.ax[ψ.inv[space]]
end
Base.firstindex(ψ::State, space::Space) = first(axes(ψ, space))
Base.lastindex( ψ::State, space::Space) =  last(axes(ψ, space))

include("symbolic_index.jl")

@inline Base.to_index(ψ::State, ::Missing) = missing
@inline Base.to_index(ψ::State, i::(Pair{Space{T}, T} where T)) = i.second
@inline Base.to_index(ψ::State, i::(Pair{Space{T}, <: AbstractRange{T}} where T)) = i.second
@inline Base.to_index(ψ::State, i::Pair{<: Space, Colon}) = axes(ψ, i.first).indices
@propagate_inbounds function Base.to_index(ψ::State, i::(Pair{Space{T}, Length{T}} where T)) =
    @boundscheck i.second.space === i.first || throw(DimensionMismatch())
    Base.to_index(ψ, i.first => i.second.indices)
end
abstract type ScaledIndex{T <: Real} end
struct ScaledPosition <: ScaledIndex
    i::ℤ
end
struct ScaledRange <: ScaledIndex
    range::OrdinalRange{ℤ, ℤ}
end
AbstractTrees.children(::Pair{<: Space}) = ()
AbstractTrees.childrentype(::Pair{<: Space}) = Tuple{}
function Base.to_indices(
    ψ::State{N},
    ax::Volume{N},
    indices::Tuple{Vararg{Union{
        Pair{>: Space},
        Length,
        Volume,
        Colon,
        Type{..}
    }}}
    ) where N =
        summing = true
    lookup = IdDict(
        convert(Pair, i)
        for i ∈ AbstractTrees.Leaves(indices)
            if !(i == : || i == ..) || (summing = false))
    Interpolations.coordlookup.(
        Base.getfield.(axes(ψ), :indices),
        to_index.(Ref(ψ), get.(Ref(lookup), ax, missing))
    )
end

@inline Base.convert(::Type{>: AbstractExtrapolation}, ψ::State) = extrapolate(
    scale(
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
            l.indices
        end
    ),
    map(axes(ψ)) do l
        l.space.periodic ? Periodic : zero(ℂ)
    end
)

Base.getindex(ψ::State, indices...) =
    convert(AbstractInterpolation, ψ)(to_indices(ψ, indices)...)

Base.similar(::Type{State}, ax::Volume{N}) where N = similar(State{N}, ax)
Base.similar(::Type{State{N}}, ax::Volume{N}) where N = State(ax)

Base.fill!(ψ::State, value::ℂ) = fill!(vec(ψ), value)
Base.fill(value::ℂ, ax::NonEmptyVolume) = fill!(similar(State, ax), value)
Base.zeros(T::Type{ℂ}, ax::NonEmptyVolume) = fill(zero(ℂ), ax)
Base.ones( T::Type{ℂ}, ax::NonEmptyVolume) = fill( one(ℂ), ax)
Base.zeros(ax::NonEmptyVolume) = zeros(ℂ, ax)
Base.ones( ax::NonEmptyVolume) =  ones(ℂ, ax)

struct Derivative{n, N, T} <: AbstractArray{ℂ, N}
    wrt::ℤ
    ψ::State{N}
end
(d::∂{n, T})(ψ::State{N}) where {n, N, T} = Derivative{n, N, T}(ψ.inv[d.wrt], ψ)
@propagate_inbounds function Base.getindex(D::Derivative{1}, args...) =
    itp = convert(AbstractInterpolation, D.ψ)
    indices = to_indices(D.ψ, args)
    @boundscheck checkbounds(Bool, itp, indices...) || Base.throw_boundserror(D.ψ, x)
    map(Iterators.product(indices)) do coords
        # v2: reuse weights
        wis = Interpolations.weightedindexes(
            (
                Interpolations.value_weights,
                Interpolations.gradient_weights
            ),
            Interpolations.itpinfo(itp)...,
            coords
        )
        Interpolations.InterpGetindex(itp)[wis[D.wrt]...]
    end
end

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

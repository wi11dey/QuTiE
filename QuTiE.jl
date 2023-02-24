#!/usr/bin/env julia

module QuTiE

import Base: -, ^, ==
import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, AbstractSciMLLinearOperator as LinearOperator, getops
import AbstractTrees
using LinearAlgebra
# using ModelingToolkit, SymbolicUtils # v4/5
# using MarchingCubes, ConstructiveGeometry, Compose # v6/7

export Time, Space, .., ℤ, ℚ, ℝ, ℂ, 𝑖, ∜, ∞, ∂, ∂², ∂³, δ, Qubit, qubits, depends, isbounded, isclassical

# Concrete types for abstract algebraic rings:
const ℤ = Int
const ℚ = Rational
const ℝ = Float64
const ℂ = Complex{ℝ}

const 𝑖 = im

∜(x::ℝ) = x^(1/4)

const Field = Union{ℚ, AbstractFloat} # In the abstract algebraic sense.

const ∞ = Val(Inf)
(-)(::Val{ Inf}) = Val(-Inf)
(-)(::Val{-Inf}) = Val( Inf)
Base.isinf(::Val{ Inf}) = true
Base.isinf(::Val{-Inf}) = true
Base.isfinite(::Val{ Inf}) = false
Base.isfinite(::Val{-Inf}) = false
Base.isnan(::Val{ Inf}) = false
Base.isnan(::Val{-Inf}) = false
==(::Val{Inf}, x::Field) = Inf == x
==(x::Field, ::Val{Inf}) = x == Inf
==(::Val{-Inf}, x::Field) = -Inf == x
==(x::Field, ::Val{-Inf}) = x == -Inf
Base.isless(::Val{Inf}, x) = isless(Inf, x)
Base.isless(x, ::Val{Inf}) = isless(x, Inf)
Base.isless(::Val{Inf}, ::Val{Inf}) = false
Base.isless(::Val{-Inf}, x) = isless(-Inf, x)
Base.isless(x, ::Val{-Inf}) = isless(x, -Inf)
Base.isless(::Val{-Inf}, ::Val{-Inf}) = false
Base.isless(::Val{ Inf}, ::Val{-Inf}) = false
Base.isless(::Val{ Inf}, ::Val{-Inf}) = false
(T::Type{<: Field})(::Val{ Inf}) = T( Inf)
(T::Type{<: Field})(::Val{-Inf}) = T(-Inf)
Base.convert(T::Type{<: Field}, val::Union{Val{Inf}, Val{-Inf}}) = T(val)
Base.promote_rule(T::Type{<: Field},   ::Type{<: Union{Val{Inf}, Val{-Inf}}}) = T
Base.promote_rule(T::Type{<: Integer}, ::Type{<: Union{Val{Inf}, Val{-Inf}}}) = AbstractFloat

Base.show(io::IO, ::Val{Inf}) = print(io, "∞")
Base.show(io::IO, ::Val{-Inf}) = print(io, "-∞")

const Compactification{T <: Number} = Union{T, Val{-Inf}, Val{Inf}} # Two-point compactification.
Base.typemin(::Type{>: Val{-Inf}}) = -∞
Base.typemax(::Type{>: Val{ Inf}}) =  ∞

(^)(op::Operator, n::Int) = prod(Iterators.repeated(op, n))
AbstractTrees.children(op::Operator) = getops(op)

abstract type Dimension{T <: Real} <: Operator{T} end
# v4: symbolic time-independent solving
depends(op::Operator, x::Dimension) = x ∈ AbstractTrees.Leaves(op)

getops(::Dimension) = ()

struct Time <: Dimension{ℝ} end
isclassical(::Time) = true # N + 1 dimensional formulation.

mutable struct Space{T} <: Dimension{T}
    const lower::Compactification{T}
    const upper::Compactification{T}
    const periodic::Bool
    const classical::Bool # ℂ^T Hilbert space if false.

    # v2: resample/interpolate when grid too large or not enough samples
    const a::Union{T, Nothing} # (Maximum) lattice spacing.
    const ε::real(ℂ) # Minimum modulus.
    const canary::T # Storage types should store enough cells to have at least this much canary border.

    function Space{T}(lower,
                      upper;
                      periodic=false,
                      classical=false, # v6
                      first=nothing,
                      step=nothing,
                      last=nothing,
                      a=nothing,
                      ε=1e-5,
                      canary=nothing) where T
        bounded = isfinite(lower) && isfinite(upper)
        !bounded && periodic && throw(ArgumentError("Unbounded space cannot be periodic"))
        lower == upper && throw(ArgumentError("Null space"))

        lower, upper = min(lower, upper), max(lower, upper)

        if isnothing(canary)
            if bounded
                if T <: Integer
                    canary = one(T)
                else
                    canary = eps(T)
                end
            else
                canary = zero(T)
            end
        end

        new(
            lower,
            upper,
            periodic,
            classical,

            a,
            ε,
            canary
        )
    end
end

function Base.show(io::IO, space::Space)
    print(io, "Space")
    symbol = eltype(space)
    for ring in [:ℤ, :ℚ, :ℝ, :ℂ]
        if symbol == getfield(QuTiE, ring)
            symbol = ring
            break
        end
    end
    print(io, "{$symbol}")
    print(io, "($(space.lower)..$(space.upper), periodic = $(space.periodic), classical = $(space.classical), a = $(space.a), ε = $(space.ε), canary = $(space.canary))")
end

Space(upper) = Space(zero(upper), upper)
Space(lower, step, upper; keywords...) = Space(lower, upper; step=step, keywords...)
Space(lower, upper, range::AbstractRange; keywords...) = Space(lower, upper; first=first(range), step=step(range), last=last(range), keywords...)
Space(range::AbstractRange{T}; keywords...) where T = Space{T}(first(range), last(range); step=step(range), keywords...)
Space{Bool}() = Space{Bool}(0, 1)

const .. = Space

==(a::Space, b::Space) = a === b
Base.hash(space::Space) = objectid(space)
Base.eltype(::Type{Space{T}}) where T = T
Base.first(space::Space) = space.lower
Base.last( space::Space) = space.upper
Base.in(x::Field, space::Space{<: Integer}) = false
Base.in(x, space::Space) = first(space) ≤ x ≤ last(space)
isbounded(space::Space) = isfinite(first(space)) && isfinite(last(space))
Base.isfinite(space::Space) = bounded(space) && eltype(space) <: Integer
Base.isinf(space::Space) = !isfinite(space)
isclassical(space::Space) = space.classical

const Qubit = Space{Bool}
qubits(val::Val) = ntuple(_ -> Qubit(), val)
qubits(n::Integer) = qubits(Val(n))

struct Derivative{N, T <: AbstractFloat} <: Operator{T}
    wrt::Space{T}
end
(::Type{Derivative})(args...) = Derivative{1}(args...)
(^)(::Type{Derivative{N}}, n::Int) where N = Derivative{N*n}
getops(d::Derivative) = (d.wrt,)
    
const ∂  = Derivative{1}
const ∂² = ∂^2
const ∂³ = ∂^3
# (::Type{Derivative{1}})(wrt::Space{T}) where {T <: AbstractFloat} = FunctionOperator(isinplace=true, T) do (dψ, ψ, p, t)
#     dψ .= (diff([wrt.periodic ? ψ[end] : zero(T); ψ]) + diff([a; wrt.periodic ? ψ[begin] : zero(T)]))/2
# end

# v6
struct ∂t end
(::Type{Derivative{1}})(::Time) = ∂t()

(::Type{Derivative{N}})(wrt) where N = ∂(wrt)^N

struct DiracDelta
    variety::Operator # Algebraic variety.
end
const δ = DiracDelta

# v4/5?
commutator(a::Operator, b::Operator) = nothing
anticommutator(a::Operator, b::Operator) = nothing

struct Length{T} <: AbstractRange{T}
    space::Space{T}
    indices::AbstractRange{T}
end
# Length(space::Space{T}, indices::AbstractRange{T}) where T =
#     Length(space, indices, first(space) == first(indices) && last(space) == last(indices) ? zero(T) : step(indices))
"""Default range for provided space."""
function Length{T}(space::Space{T}) where T
    
    Length{T}(space, -5:5)
end

Base.convert(::Type{Length{T}}, space::Space{T}) where T = Length{T}(space)
Base.convert(::Type{Length}, space::Space) = convert(Length{eltype(space)}, space)

AbstractTrees.children(::Length) = ()
AbstractTrees.childtype(::Type{<: Length}) = Tuple{}

const Volume{N} = NTuple{N, Length}
Base.IteratorEltype(::Type{<: AbstractTrees.TreeIterator{<: Volume}}) = Base.HasEltype()
Base.eltype(::Type{<: AbstractTrees.TreeIterator{<: Volume}}) = Length
×(a::Volume, b::Volume) = (a..., b...)
×(factors::Union{Length, Volume}...) = Volume(AbstractTrees.Leaves(factors))

# v3: function boundary detection by binary search
# v4: symbolic function boundary detection
# v4/5: intelligently select first dimension symbolically
Base.axes(op::Operator) = unique(dim for dim in AbstractTrees.Leaves(op) if dim isa Space) |> Volume

const Coordinate{T} = Pair{Space{T}, T}
struct Point{N} <: Base.AbstractCartesianIndex{N}
    ax::NTuple{N, Space}
    coords::NTuple{N, Real}
end
Base.keys(i::Point) = i.ax
Base.values(i::Point) = i.coords
Base.pairs(i::Point) = Iterators.map(=>, keys(i), values(i))
Point(coords::Coordinate...) = Point(zip(coords...)...) # Intended for relatively short lists of coordinates.

struct Points{N} <: AbstractArray{Point{N}, N}
    ax::Volume{N}
end
# LinearIndices(indices::Points)

mutable struct State{N} <: AbstractArray{ℂ, N}
    ax::Volume{N}
    const data::Vector{ℂ}

    const dimensions::IdDict{Space, Int}

    State{N}(ax::Volume{N}, data) where N = new{N}(ax, data, ax |> enumerate .|> reverse |> IdDict)
end
Base.axes(ψ::State) = ψ.ax

Base.keytype(ψ::State{N}) where N = Point{N}
Base.eachindex(ψ::State) = Points(axes(ψ))

Base.to_index(ψ::State, i::Coordinate) = nothing
Base.to_index(ψ::State, i::Pair{Space{T}, Colon} where T) = nothing
# Base.to_indices(ψ::State, i::HigherDimensionalSpaceIndex) = values(i)
# Base.to_indices(ψ::State, ax::HigherDimensionalSpace, indices::Tuple{Vararg{Pair{Space{T}, Union{T, Colon}} where T}}) = nothing
# function Base.getindex(ψ::State, i::HigherDimensionalSpaceIndex)
#     ψ.data[LinearIndices(eachindex.(Base.getfield.(axes(ψ), :indices)))[i.values...]]
# end
Base.getindex(ψ::State, indices::(Pair{Space{T}, <: Union{T, AbstractRange{T}, Colon}} where T)...) =
    ψ[HigherDimensionalSpaceIndex(axes(ψ), to_indices(ψ, indices))]

Base.similar(::Type{State}, ax::Volume{N}) where N = similar(State{N}, ax)
Base.similar(::Type{State{N}}, ax::Volume{N}) where N = State(ax, Vector{ℂ}(undef, length(CartesianIndices(ax))))

Base.fill(value::ComplexF64, ax::Volume) = fill!(similar(State, ax), value)
Base.zeros(T::Type{ComplexF64}, ax::Volume) = fill(zero(ComplexF64), ax)
Base.ones( T::Type{ComplexF64}, ax::Volume) = fill( one(ComplexF64), ax)
Base.zeros(ax::Volume) = zeros(ComplexF64, ax)
Base.ones( ax::Volume) =  ones(ComplexF64, ax)

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
Base.kron(ψ::State, φ::State) = State(axes(ψ)×axes(φ), kron(ψ.data, φ.data))
const ⊗ = kron

function LinearAlgebra.dot(ψ::State, φ::State)
    if axes(ψ) == axes(φ)
        return ψ.data⋅φ.data
    end
end

# using MakieCore

# v3
"""Returns the unique Lagrangian with gauge group U(n) and coupling constant g in terms of the given spaces."""
U(::Val{N}; g, spaces...) where N = nothing
U(n::Integer; args...) = U(Val(n); args...)

"""Returns the unique Lagrangian with gauge group SU(n) and coupling constant g in terms of the given spaces."""
SU(::Val{N}; g, spaces...) where N = nothing
SU(n::Integer; args...) = SU(Val(n); args...)

"""Returns the unique Lagrangian with gauge group SO(n) and coupling constant g in terms of the given spaces."""
SO(::Val{N}; g, spaces...) where N = nothing
SO(n::Integer; args...) = SO(Val(n); args...)

# v6
abstract type ClassicalObject end

# Propagate until reflection using Huygens' principle in N dimensions:
abstract type Wave{N} <: ClassicalObject end # For classical field theory.
struct PlaneWave{N} <: Wave{N} end
struct SphericalWave{N} <: Wave{N} end

# Solve using Euler-Lagrange equation in N generalized coordinates:
abstract type Particle{N} <: ClassicalObject end # For classical mechanics.

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
    for (ψ, t) in tuples(integrator)
        @show ψ, t
    end
end

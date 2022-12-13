#!/usr/bin/env julia

module QuTiE

import Base: -, ^, ==
import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, AbstractSciMLLinearOperator as LinearOperator, getops
import AbstractTrees
using LinearAlgebra

export Time, Space, ℤ, ℝ, ℂ, ∞, Qubit, qubits, depends, isbounded

const ℤ = Int
const ℝ = Float64
const ℂ = Complex{ℝ}

const Field = Union{Rational, AbstractFloat}

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
(T::Type{<: Field})(::Val{ Inf}) = T( Inf)
(T::Type{<: Field})(::Val{-Inf}) = T(-Inf)
Base.convert(T::Type{<: Field}, val::Union{Val{Inf}, Val{-Inf}}) = T(val)

const Compactification{T <: Number} = Union{T, Val{-Inf}, Val{Inf}}
Base.typemin(::Type{>: Val{-Inf}}) = -∞
Base.typemax(::Type{>: Val{ Inf}}) =  ∞

(^)(op::Operator, n::Int) = prod(Iterators.repeated(op, n))
AbstractTrees.children(op::Operator) = getops(op)

abstract type Coordinate{T <: Real} <: Operator{T} end
# v4: symbolic time-independent solving
depends(op::Operator, x::Coordinate) = x ∈ AbstractTrees.Leaves(op)

getops(::Coordinate) = ()

struct Time <: Coordinate{ℝ} end

mutable struct Space{T} <: Coordinate{T}
    const lower::Compactification{T}
    const upper::Compactification{T}
    const periodic::Bool

    # v2: resample/interpolate when grid too large or not enough samples
    const a::Union{T, Nothing} # (Maximum) lattice spacing.
    const ε::real(ℂ) # Minimum modulus.
    const border::T

    function Space{T}(lower,
                      upper;
                      periodic=false,
                      first=nothing,
                      step=nothing,
                      last=nothing,
                      a=nothing,
                      ε=1e-5,
                      samples=1:typemax(Int) #= Minimum and maximum number of samples. =#) where T
        (isinf(lower) || isinf(upper)) && periodic && throw(ArgumentError("Unbounded space cannot be periodic"))
        lower == upper && throw(ArgumentError("Null space"))

        lower, upper = min(lower, upper), max(lower, upper)

        isnothing(first) && (first = isfinite(lower) ? lower : -5)
        isnothing(last)  && (last  = isfinite(upper) ? upper :  5)
        isnothing(step) && (step = T <: Integer ? one(T) : (upper - lower)/100)

        new(
            lower,
            upper,
            periodic,

            a,
            ε,
            samples,

            isone(step) ? (first:last) : (first:step:last)
        )
    end
end
Space(upper) = Space(zero(upper), upper)
Space(lower, step, upper; keywords...) = Space(lower, upper; step=step, keywords...)
Space(lower, upper, range::AbstractRange; keywords...) = Space(lower, upper; first=first(range), step=step(range), last=last(range), keywords...)
Space(range::AbstractRange{T}; keywords...) where T = Space{T}(first(range), last(range); step=step(range), keywords...)
Space{Bool}() = Space{Bool}(0, 1)

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

const Qubit = Space{Bool}
qubits(val::Val) = ntuple(_ -> Qubit(), val)
qubits(n) = qubits(Val(n))

struct Derivative{N} end
(::Type{Derivative})(args...) = Derivative{1}(args...)
(^)(::Type{Derivative{N}}, n::Int) where N = Derivative{N*n}
const ∂  = Derivative{1}
const ∂² = ∂^2
const ∂³ = ∂^3
(::Type{Derivative{1}})(wrt::Space{T}) where {T <: Field} = FunctionOperator(isinplace=true, T) do (dψ, ψ, p, t)
    dψ .= (diff([wrt.periodic ? ψ[end] : zero(T); ψ]) + diff([a; wrt.periodic ? ψ[begin] : zero(T)]))/2
end
(::Type{Derivative{N}})(wrt::Space) where N = ∂(wrt)^N

struct Point{T <: NTuple{N, Real}} <: Base.AbstractCartesianIndex{N}
    ax::ProductSpace{N}
    coords::T

    @inline (::Type{Point})(ax::ProductSpace{N}, indices::NTuple{N, Real}) where N =
        new{Tuple{eltype.(typeof(ax).types)...}}(ax, indices)
end
Base.keys(i::Point) = i.ax
Base.values(i::Point) = i.indices
Base.pairs(i::Point) = Iterators.map(=>, keys(i), values(i))

struct SpatialRange{T} <: AbstractRange{T}
    space::Space{T}
    indices::AbstractRange{T}
    canary::T
end
SpatialRange(space::Space{T}, indices::AbstractRange{T}) where T =
    SpatialRange(space, indices, first(space) == first(indices) && last(space) == last(indices) ? zero(T) : step(indices))
"""Default range for provided space."""
function SpatialRange{T}(space::Space{T}) where T
    
end

Base.convert(::Type{SpatialRange{T}}, space::Space{T}) where T = SpatialRange{T}(space)
Base.convert(::Type{SpatialRange}, space::Space) = convert(SpatialRange{eltype(space)}, space)

AbstractTrees.children(::SpatialRange) = ()
AbstractTrees.childtype(::Type{<: SpatialRange}) = Tuple{}

const Region{N} = NTuple{N, SpatialRange}
Base.IteratorEltype(::Type{<: AbstractTrees.TreeIterator{<: Region}}) = Base.HasEltype()
Base.eltype(::Type{<: AbstractTrees.TreeIterator{<: Region}}) = SpatialRange
×(a::Region, b::Region) = (a..., b...)
×(factors::Union{SpatialRange, Region}...) = Region(AbstractTrees.Leaves(factors))

# v3: function boundary detection by binary search
# v4: symbolic function boundary detection
Base.axes(op::Operator) = Region(unique!(Base.Fix2(isa, Space), collect(AbstractTrees.Leaves(op))))

struct Points{N} <: AbstractArray{Point{N}, N}
    ax::Region{N}
end
LinearIndices(indices::Points)

mutable struct State{N} <: AbstractArray{ℂ, N}
    ax::Region{N}
    const data::Vector{ℂ}

    const dimensions::IdDict{Space, Int}

    State{N}(ax::Region{N}, data) where N = new{N}(ax, data, ax |> enumerate .|> reverse |> IdDict)
end
Base.axes(ψ::State) = ψ.ax

Base.keytype(ψ::State{N}) where N = Point{N}
Base.eachindex(ψ::State) = Points(axes(ψ))

Base.to_index(ψ::State, i::Pair{Space{T}, T} where T) = nothing
Base.to_index(ψ::State, i::Pair{Space{T}, Colon} where T) = nothing
Base.to_indices(ψ::State, i::HigherDimensionalSpaceIndex) = values(i)
Base.to_indices(ψ::State, ax::HigherDimensionalSpace, indices::Tuple{Vararg{Pair{Space{T}, Union{T, Colon}} where T}}) = nothing
function Base.getindex(ψ::State, i::HigherDimensionalSpaceIndex)
    ψ.data[LinearIndices(eachindex.(Base.getfield.(axes(ψ), :indices)))[i.values...]]
end
Base.getindex(ψ::State, indices::(Pair{Space{T}, <: Union{T, AbstractRange{T}, Colon}} where T)...) =
    ψ[HigherDimensionalSpaceIndex(axes(ψ), to_indices(ψ, indices))]

Base.similar(::Type{State}, ax::ProductSpace{N}) where N = similar(State{N}, ax)
Base.similar(::Type{State{N}}, ax::ProductSpace{N}) where N = State(ax, Vector{ℂ}(undef, length(CartesianIndices(ax))))

Base.fill(value::ComplexF64, ax::Region) = fill!(similar(State, ax), value)
Base.zeros(T::Type{ComplexF64}, ax::Region) = fill(zero(ComplexF64), ax)
Base.ones( T::Type{ComplexF64}, ax::Region) = fill( one(ComplexF64), ax)
Base.zeros(ax::Region) = zeros(ComplexF64, ax)
Base.ones( ax::Region) =  ones(ComplexF64, ax)

function trim!(ψ::State{1})
    r = axes(ψ)[1]
end
function trim!(ψ::State)
    for axis in axes(ψ)
        
    end
end

function LinearAlgebra.normalize!(ψ::State)
end

"""
Tensor product of multiple states.
"""
Base.kron(ψ::State, φ::State) = State(axes(ψ) × axes(φ), kron(ψ.data, φ.data))
const ⊗ = kron

function LinearAlgebra.dot(ψ::State, φ::State)
    if axes(ψ) == axes(φ)
        return ψ.data⋅φ.data
    end
end

end

using ..QuTiE
using Revise
using DifferentialEquations

using PhysicalConstants: CODATA2018, PhysicalConstant
for name in names(CODATA2018, all=true)
    @eval if CODATA2018.$name isa PhysicalConstant
        import PhysicalConstants.CODATA2018: $name
        export $name
    end
end

export ħ²
ħ² = ħ^2

__revise_mode__ = :evalassign

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1 || ARGS[1] ∈ ("-h", "--help", "-?")
        error("Usage: julia QuTiE.jl spec.jl")
    end
    includet(ARGS[1])
    if !isdefined(:H)
        error("Must define H for Hamiltonian formulation")
    end
    if !isdefined(:ψ₀)
        error("Must define initial state ψ₀")
    end
    integrator = init(ODEProblem(H, ψ₀*ones(axes(H)), (0, Inf)))
    for (ψ, t) in tuples(integrator)
        @show ψ, t
    end
end

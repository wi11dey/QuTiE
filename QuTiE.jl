#!/usr/bin/env julia

module QuTiE

import Base: -, ^, ==
import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, AbstractSciMLLinearOperator as LinearOperator, getops
import AbstractTrees

export Time, Space, ℤ, ℝ, ℂ, ∞, Qubit, Qubits

const ℤ = Int
const ℝ = Float64
const ℂ = Complex{ℝ}

const ∞ = Val(Inf)
(-)(::Val{ Inf}) = Val(-Inf)
(-)(::Val{-Inf}) = Val( Inf)
Base.isinf(::Val{ Inf}) = true
Base.isinf(::Val{-Inf}) = true
Base.isfinite(::Val{ Inf}) = false
Base.isfinite(::Val{-Inf}) = false
Base.isnan(::Val{ Inf}) = false
Base.isnan(::Val{-Inf}) = false
==(::Val{Inf}, x::AbstractFloat) = Inf == x
==(x::AbstractFloat, ::Val{Inf}) = x == Inf
==(::Val{-Inf}, x::AbstractFloat) = -Inf == x
==(x::AbstractFloat, ::Val{-Inf}) = x == -Inf
Base.isless(::Val{Inf}, x) = isless(Inf, x)
Base.isless(x, ::Val{Inf}) = isless(x, Inf)
Base.isless(::Val{Inf}, ::Val{Inf}) = false
Base.isless(::Val{-Inf}, x) = isless(-Inf, x)
Base.isless(x, ::Val{-Inf}) = isless(x, -Inf)
Base.isless(::Val{-Inf}, ::Val{-Inf}) = false
(T::Type{<: AbstractFloat})(::Val{ Inf}) = T( Inf)
(T::Type{<: AbstractFloat})(::Val{-Inf}) = T(-Inf)
Base.convert(T::Type{<: AbstractFloat}, val::Union{Val{Inf}, Val{-Inf}}) = T(val)

const Compactification{T <: Number} = Union{T, Val{-Inf}, Val{Inf}}
Base.typemin(::Type{>: Val{-Inf}}) = -∞
Base.typemax(::Type{>: Val{ Inf}}) =  ∞

(^)(op::Operator, n::Int) = prod(Iterators.repeated(op, n))
AbstractTrees.children(op::Operator) = getops(op)

abstract type Coordinate{T <: Real} <: Operator{T} end
depends(op::Operator, x::Coordinate) = x ∈ AbstractTrees.Leaves(op)

getops(::Coordinate) = ()
AbstractTrees.children(::Coordinate) = ()

struct Time <: Coordinate{ℝ} end

mutable struct Space{T} <: Coordinate{T}
    const lower::Compactification{T}
    const upper::Compactification{T}
    const periodic::Bool

    # v2: resample/interpolate when grid too large or not enough samples, along with minimum grid spacing, also optional samples option
    α::Union{T, Nothing}
    ε::real(ℂ)
    samples::UnitRange{Int}

    indices::AbstractRange{T}

    function Space{T}(lower,
                      upper;
                      periodic=false,
                      first=nothing,
                      step=nothing,
                      last=nothing,
                      α=nothing, # Scale factor (maximum grid spacing).
                      ε=1e-5, # Minimum complex modulus.
                      samples=1:typemax(Int) #= Minimum and maximum number of samples. =#) where T
        (isinf(lower) || isinf(upper)) && periodic && throw(ArgumentError("Unbounded space cannot be periodic"))
        lower == upper && throw(ArgumentError("Null space"))

        lower, upper = min(lower, upper), max(lower, upper)

        isnothing(first) && first = isfinite(lower) ? lower : -5
        isnothing(last)  && last  = isfinite(upper) ? upper :  5
        isnothing(step) && step = T <: Integer ? one(T) : (upper - lower)/100

        new(
            lower,
            upper,
            periodic,

            α,
            ε,
            samples,

            isone(step) ? first:last : first:step:last
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

struct Derivative{N} end
(::Type{Derivative})(args...) = Derivative{1}(args...)
(^)(::Type{Derivative{N}}, n::Int) where N = Derivative{N*n}
const ∂  = Derivative{1}
const ∂² = ∂^2
const ∂³ = ∂^3
(::Type{Derivative{1}})(wrt::Space{T <: AbstractFloat}) where T = FunctionOperator(isinplace=true, T) do (dψ, ψ, p, t)
    dψ .= (diff([wrt.periodic ? ψ[end] : zero(T); ψ]) + diff([a; wrt.periodic ? ψ[begin] : zero(T)]))/2
end
(::Type{Derivative{N}})(wrt::Space) where N = ∂(wrt)^N

struct ProductSpace{N, T} <: Operator{NTuple{N, T}}
    factors::NTuple{N, Space{T}}
end
ProductSpace{N, T}() where {N, T} = ProductSpace{N, T}(ntuple(_ -> Space{T}(), Val{N}))
ProductSpace(factors::Space...) = ProductSpace(factors)
ProductSpace(factors::Union{Space, ProductSpace}...) = ProductSpace(AbstractTrees.Leaves(factors)...)

getops(::Coordinate) = ()
AbstractTrees.children(product::ProductSpace) = product.factors
AbstractTrees.childtype(::Type{ProductSpace{N, T}}) where {N, T} = Space{T}
AbstractTrees.childtype(product::ProductSpace) = AbstractTrees.childtype(typeof(product))
Base.getindex(product::ProductSpace, n) = product.factors[n]

const × = ProductSpace

const Qubit = Space{Bool}
const Qubits{N} = ProductSpace{N, Bool}

const HigherDimensionalSpace{N} = NTuple{N, Space}

Base.axes(op::Operator) = filter(x -> x isa Space, AbstractTrees.Leaves(op)) |> unique! |> Tuple

function indicestype(T::Type{<: HigherDimensionalSpace})
    eltypes = eltype.(T.types)
    HigherDimensionalSpaceIndex{Union{eltypes...}, Tuple{(Pair{Space{elt}, elt} for elt in eltypes)...}}
end
indicestype(ax::HigherDimensionalSpace) = indicestype(typeof(ax))

struct HigherDimensionalSpaceIndices{U, T <: NTuple{N}} <: AbstractArray{HigherDimensionalSpaceIndex{U, T}, N}
    ax::HigherDimensionalSpace{N}

    function (::Type{HigherDimensionalSpaceIndices})(ax::A) where A
        new{indicestype(ax).parameters...}(ax)
    end
end

struct HigherDimensionalSpaceIndex{U, T <: NTuple{N, Pair{Space{V}, V} where {V <: U}}} <: Base.AbstractCartesianIndex{N}
    indices::IdDict{Space{U}, U}

    HigherDimensionalSpaceIndex{U, T}(indices::T) where T = new{U, T}(IdDict(indices))
    Base.getindex(i::HigherDimensionalSpaceIndex{U, T}, ax::HigherDimensionalSpace) where {U, T} = new{U, T}(filter(((x, _),) -> x ∈ ax, i.indices))
end

Base.getindex(i::HigherDimensionalSpaceIndex, space::Space) = i.indices[space]
Base.getindex(i::HigherDimensionalSpaceIndex, spaces::Space...) = i[spaces]

struct State{N} <: AbstractArray{ℂ, N}
    ax::HigherDimensionalSpace{N}
    data::Vector{ℂ}

    dimensions::IdDict{Space, Int}

    State(ax, data) = new(ax, data, ax |> enumerate .|> reverse |> IdDict)
end
Base.axes(ψ::State) = ψ.ax

Base.keytype(ψ::State) = indicestype(axes(ψ))
Base.eachindex(ψ::State) = HigherDimensionalSpaceIndices(axes(ψ))

function Base.getindex(ψ::State, indices::(Pair{Space{T}, Union{T, Colon}} where T)...)
    
end
# Base.getindex(ψ::State{N}, inds::NTuple{N, Int}...) where N = ψ.data[LinearIndices(ax)[inds...]]

Base.similar(::Type{State}, ax::HigherDimensionalSpace{N}) where N = similar(State{N}, ax)
Base.similar(::Type{State{N}}, ax::HigherDimensionalSpace{N}) where N = State(ax, Vector{ℂ}(undef, length(CartesianIndices(ax))))

Base.fill(value::ComplexF64, ax::Tuple{Vararg{Space}}) = fill!(similar(State, ax), value)
Base.zeros(T::Type{ComplexF64}, ax::Tuple{Vararg{Space}}) = fill(zero(ComplexF64), ax)
Base.ones( T::Type{ComplexF64}, ax::Tuple{Vararg{Space}}) = fill( one(ComplexF64), ax)
Base.zeros(ax::Tuple{Vararg{Space}}) = zeros(ComplexF64, ax)
Base.ones( ax::Tuple{Vararg{Space}}) =  ones(ComplexF64, ax)

"""
Tensor product of multiple states.
"""
function ⊗(Ψ::State...)
    φ = similar(State, Ψ .|> axes |> Iterators.flatten |> Tuple)
    for i in eachindex(φ)
        φ[i] = prod(ψ[i[axes(ψ)]] for ψ in Ψ)
    end
    φ
end

end

using ..QuTiE
using Revise
using PhysicalConstants.CODATA2018
using DifferentialEquations

export ħ²
ħ² = ħ^2

__revise_mode__ = :evalassign

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
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

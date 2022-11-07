#!/usr/bin/env julia

module QuTiE

import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, AbstractSciMLLinearOperator as LinearOperator
using OrderedCollections

export Time, Space, PeriodicSpace, ℤ, ℝ, ∞

const ℤ = Int
const ℝ = Float64
const ∞ = Inf

(^)(x::Operator, n::Int) = prod(Iterators.repeated(x, n))

abstract type Coordinate{T <: Real} <: Operator{T} end

struct Time <: Coordinate{ℝ} end

mutable struct Space{T, Periodic} <: Coordinate{T}
    const lower::T
    const upper::T

    indices::AbstractRange{T}
end
(::Type{Space{T}})(args...) where T = Space{T, false}(args...)
(space::Type{<: Space})(upper) = space(zero(upper), upper)

const PeriodicSpace{T} = Space{T, true}

struct Derivative{N} end
(::Type{Derivative})(args...) = Derivative{1}(args...)
(^)(::Type{Derivative{N}}, n::Int) where N = Derivative{N*n}
const ∂  = Derivative{1}
const ∂² = ∂^2
const ∂³ = ∂^3
(::Type{Derivative{1}})(wrt::Space{T, Periodic}) where {T, Periodic} = FunctionOperator(isinplace=true, T) do (dψ, ψ, p, t)
    dψ .= (diff([Periodic ? ψ[end] : zero(T); ψ]) + diff([a; Periodic ? ψ[begin] : zero(T)]))/2
end
(::Type{Derivative{N}})(wrt::Space) where N = ∂(wrt)^N

function Base.axes(op::Operator)
    dims = OrderedSet{Space}()
    walk(x::Space) = push!(x, dims)
    walk(op::Operator) = walk.(getops(op))
    walk(x) = nothing
    walk.(getops(op))
    (dims...,)
end

struct State{N} <: AbstractArray{ComplexF64, N}
    ax::NTuple{N, Space}
    data::Vector{ComplexF64}
end

Base.getindex(ψ::State{N}, inds::NTuple{N, Int}...) where N = ψ.data[LinearIndices(ax)[inds...]]

Base.similar(::Type{State}, ax::NTuple{N, Space}) where N = similar(State{N}, ax)
function Base.similar(::Type{State{N}}, ax::NTuple{N, Space}) where N
end

Base.fill(value::ComplexF64, ax::Tuple{Vararg{Space}}) = fill!(similar(State, ax), value)
Base.zeros(T::Type{ComplexF64}, ax::Tuple{Vararg{Space}}) = fill(zero(ComplexF64), ax)
Base.ones( T::Type{ComplexF64}, ax::Tuple{Vararg{Space}}) = fill( one(ComplexF64), ax)
Base.zeros(ax::Tuple{Vararg{Space}}) = zeros(ComplexF64, ax)
Base.ones( ax::Tuple{Vararg{Space}}) =  ones(ComplexF64, ax)

end

using ..QuTiE
using Revise
using PhysicalConstants.CODATA2018
using DifferentialEquations

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
    integrator = init(ODEProblem(H, ψ₀*ones(axes(H)), (0, ∞)))
    for (ψ, t) in tuples(integrator)
        @show ψ, t
    end
end

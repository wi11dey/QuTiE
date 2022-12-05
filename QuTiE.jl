#!/usr/bin/env julia

module QuTiE

import Base: -, ^, ==
import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, AbstractSciMLLinearOperator as LinearOperator
using OrderedCollections

export Time, Space, ℤ, ℝ, ℂ, ∞

const ℤ = Int
const ℝ = Float64
const ℂ = ComplexF64

const ∞ = Val(Inf)
(-)(::Val{ Inf}) = Val(-Inf)
(-)(::Val{-Inf}) = Val( Inf)
Base.isinf(::Val{ Inf}) = true
Base.isinf(::Val{-Inf}) = true
Base.isfinite(::Val{ Inf}) = false
Base.isfinite(::Val{-Inf}) = false
Base.isnan(::Val{ Inf}) = false
Base.isnan(::Val{-Inf}) = false
(==)(::Val{Inf}, x::AbstractFloat) = Inf == x
(==)(x::AbstractFloat, ::Val{Inf}) = x == Inf
(==)(::Val{-Inf}, x::AbstractFloat) = -Inf == x
(==)(x::AbstractFloat, ::Val{-Inf}) = x == -Inf
Base.isless(::Val{Inf}, x) = isless(Inf, x)
Base.isless(x, ::Val{Inf}) = isless(x, Inf)
Base.isless(::Val{Inf}, ::Val{Inf}) = false
Base.isless(::Val{-Inf}, x) = isless(-Inf, x)
Base.isless(x, ::Val{-Inf}) = isless(x, -Inf)
Base.isless(::Val{-Inf}, ::Val{-Inf}) = false

const Compactification{T} = Union{T, Val{-Inf}, Val{Inf}}
Base.typemin(::Type{>: Val{-Inf}}) = -∞
Base.typemax(::Type{>: Val{ Inf}}) =  ∞

(^)(x::Operator, n::Int) = prod(Iterators.repeated(x, n))

abstract type Coordinate{T <: Real} <: Operator{T} end

struct Time <: Coordinate{ℝ} end

Base.eps(x::Integer) = oneunit(x)

mutable struct Space{T} <: Coordinate{T}
    const lower::Compactification{T}
    const upper::Compactification{T}
    const periodic::Bool

    indices::AbstractRange{T}

    function Space{T}(lower::Compactification{T},
                      upper::Compactification{T};
                      periodic=false,
                      step::Union{T, Nothing}=nothing) where T
        if (isinf(lower) || isinf(upper)) && periodic
            throw(ArgumentError("Unbounded space cannot be periodic"))
        end
        if lower == upper
            throw(ArgumentError("Null space"))
        end
        new(min(lower, upper), max(lower, upper), periodic, step === 1 ? lower:upper : lower:step:upper)
    end
end
Space(upper) = Space(zero(upper), upper)
Space(lower, step, upper; keywords...) = Space(lower, upper; step=step, keywords...)
Space{Bool}() = Space{Bool}(0, 1)

struct ProductSpace{N, T} <: Operator{NTuple{N, T}}
    factors::NTuple{N, Space{T}}
end
ProductSpace{N, T}() where {N, T} = ProductSpace{N, T}(ntuple(_ -> Space{T}(), Val{N}))
ProductSpace(factors::Space...) = ProductSpace(factors)
ProductSpace(product::ProductSpace, additional::Space...) = ProductSpace((product.factors..., additional...))

Base.getindex(product::ProductSpace, n) = product.factors[n]

const × = ProductSpace

const Qubit = Space{Bool}
const Qubits{N} = ProductSpace{N, Bool}

struct Derivative{N} end
(::Type{Derivative})(args...) = Derivative{1}(args...)
(^)(::Type{Derivative{N}}, n::Int) where N = Derivative{N*n}
const ∂  = Derivative{1}
const ∂² = ∂^2
const ∂³ = ∂^3
(::Type{Derivative{1}})(wrt::Space{T <: AbstractFloat}) where T = FunctionOperator(isinplace=true, T) do (dψ, ψ, p, t)
    dψ .= (diff([Periodic ? ψ[end] : zero(T); ψ]) + diff([a; wrt.periodic ? ψ[begin] : zero(T)]))/2
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

struct State{N} <: AbstractArray{ℂ, N}
    ax::NTuple{N, Space}
    data::Vector{ℂ}
end
Base.axes(ψ::State) = ψ.ax

function ⊗(ψ::State{N}, φ::State{M}) where {N, M}
    ax = (axes(ψ)..., axes(φ)...)
    State(ax, [ψ[i[begin:N]...]*φ[i[N + 1:end]...] for i in CartesianIndices(ax)])
end
⊗(ψ::State{N}, φ::State{M}, rest::State...) = ⊗(ψ ⊗ φ, rest...)

Base.getindex(ψ::State{N}, inds::NTuple{N, Int}...) where N = ψ.data[LinearIndices(ax)[inds...]]

Base.similar(::Type{State}, ax::NTuple{N, Space}) where N = similar(State{N}, ax)
Base.similar(::Type{State{N}}, ax::NTuple{N, Space}) where N = State(ax, Vector{ℂ}(undef, length(CartesianIndices(ax))))

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
    integrator = init(ODEProblem(H, ψ₀*ones(axes(H)), (0, ∞)))
    for (ψ, t) in tuples(integrator)
        @show ψ, t
    end
end

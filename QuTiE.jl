#!/usr/bin/env julia

module QuTiE

import Base: -, ^, ==
import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, AbstractSciMLLinearOperator as LinearOperator, getops
import AbstractTrees
using LinearAlgebra

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

        isnothing(first) && (first = isfinite(lower) ? lower : -5)
        isnothing(last)  && (last  = isfinite(upper) ? upper :  5)
        isnothing(step) && (step = T <: Integer ? one(T) : (upper - lower)/100)

        new(
            lower,
            upper,
            periodic,

            α,
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

struct Derivative{N} end
(::Type{Derivative})(args...) = Derivative{1}(args...)
(^)(::Type{Derivative{N}}, n::Int) where N = Derivative{N*n}
const ∂  = Derivative{1}
const ∂² = ∂^2
const ∂³ = ∂^3
(::Type{Derivative{1}})(wrt::Space{T}) where {T <: AbstractFloat} = FunctionOperator(isinplace=true, T) do (dψ, ψ, p, t)
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

struct HigherDimensionalSpaceIndex{N} <: Base.AbstractCartesianIndex{N}
    indices::IdDict{Space, Real}

    function (::Type{HigherDimensionalSpaceIndex})(indices::(Pair{Space{T}, T} where T)...)
        indices = IdDict{Space, Real}(args...)
        new{length(indices)}(indices)
    end
    function Base.getindex(i::HigherDimensionalSpaceIndex, ax::HigherDimensionalSpace)
        indices = filter(((space, _),) -> space ∈ ax, i.indices)
        new{length(indices)}(indices)
    end
end
Base.getindex(i::HigherDimensionalSpaceIndex, ax::Space...) = i[ax]
Base.getindex(i::HigherDimensionalSpaceIndex, space::Space) = i.indices[space]

struct HigherDimensionalSpaceIndices{N} <: AbstractArray{HigherDimensionalSpaceIndex{N}, N}
    ax::HigherDimensionalSpace{N}
end

struct State{N} <: AbstractArray{ℂ, N}
    ax::HigherDimensionalSpace{N}
    data::Vector{ℂ}

    dimensions::IdDict{Space, Int}

    State{N}(ax::HigherDimensionalSpace{N}, data) where N = new{N}(ax, data, ax |> enumerate .|> reverse |> IdDict)
end
Base.axes(ψ::State) = ψ.ax

Base.keytype(ψ::State{N}) where N = HigherDimensionalSpaceIndex{N}
Base.eachindex(ψ::State) = HigherDimensionalSpaceIndices(axes(ψ))

Base.to_index(ψ::State, i::Pair{Space{T}, T} where T) = nothing
Base.to_index(ψ::State, i::Pair{Space{T}, Colon} where T) = nothing
Base.to_indices(ψ::State, ax::HigherDimensionalSpace, indices::Tuple{Vararg{Pair{Space{T}, Union{T, Colon}} where T}}) = nothing
Base.getindex(ψ::State, indices::(Pair{Space{T}, Union{T, Colon}} where T)...) =
    ψ.data[LinearIndices(eachindex.(Base.getfield.(axes(ψ), :indices)))[to_indices(ψ, indices)]]

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
Base.kron(ψ::State, φ::State) = State((axes(ψ), axes(φ)...), kron(ψ.data, φ.data))
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

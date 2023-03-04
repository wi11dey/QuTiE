#!/usr/bin/env julia

module QuTiE

import Base: +, -, *, /, ^, ==, @propagate_inbounds, to_index
import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, AbstractSciMLLinearOperator as LinearOperator, getops, ComposedOperator, ScaledOperator, ComposedScalarOperator, AddedOperator, FunctionOperator, AdjointOperator
import TermInterface, SymbolicUtils
import AbstractTrees
using LinearAlgebra
using Interpolations
# using ModelingToolkit, SymbolicUtils # v4/5
# using MarchingCubes, ConstructiveGeometry, Compose # v6/7

export Time, Space, .., 𝑖, ∜, ∞, δ, ×, Qubit, qubits, depends, isbounded, isclassical, commutes, commutator, anticommutator

const superscripts = collect("⁰¹²³⁴⁵⁶⁷⁸⁹")
const   subscripts = collect("₀₁₂₃₄₅₆₇₈₉")
sup(n::Integer) = join(superscripts[reverse!(digits(n)) .+ 1])
sub(n::Integer) = join(  subscripts[reverse!(digits(n)) .+ 1])

# Concrete types for abstract algebraic rings:
for (symbol, ring) ∈ pairs((ℤ=Int, ℚ=Rational, ℝ=Float64, ℂ=ComplexF64))
    @eval const $symbol = $ring
    @eval export $symbol
    @eval getsymbol(::Type{$ring}) = $(Meta.quot(symbol))
end
getsymbol(T::Type) = Symbol(T)

const 𝑖 = im

∜(x::ℝ) = x^(1/4)

const RealField = Union{ℚ, AbstractFloat} # A formally real field, in the abstract algebraic sense.

const ∞ = Val(Inf)
(-)(::Val{ Inf}) = Val(-Inf)
(-)(::Val{-Inf}) = Val( Inf)
Base.isinf(::Val{ Inf}) = true
Base.isinf(::Val{-Inf}) = true
Base.isfinite(::Val{ Inf}) = false
Base.isfinite(::Val{-Inf}) = false
Base.isnan(::Val{ Inf}) = false
Base.isnan(::Val{-Inf}) = false
==(::Val{Inf}, x::RealField) = Inf == x
==(x::RealField, ::Val{Inf}) = x == Inf
==(::Val{-Inf}, x::RealField) = -Inf == x
==(x::RealField, ::Val{-Inf}) = x == -Inf
Base.isless(::Val{Inf}, x) = isless(Inf, x)
Base.isless(x, ::Val{Inf}) = isless(x, Inf)
Base.isless(::Val{Inf}, ::Val{Inf}) = false
Base.isless(::Val{-Inf}, x) = isless(-Inf, x)
Base.isless(x, ::Val{-Inf}) = isless(x, -Inf)
Base.isless(::Val{-Inf}, ::Val{-Inf}) = false
Base.isless(::Val{ Inf}, ::Val{-Inf}) = false
Base.isless(::Val{-Inf}, ::Val{ Inf}) = true
(T::Type{<: RealField})(::Val{ Inf}) = T( Inf)
(T::Type{<: RealField})(::Val{-Inf}) = T(-Inf)
Base.convert(T::Type{<: RealField}, val::Union{Val{Inf}, Val{-Inf}}) = T(val)
Base.promote_rule(T::Type{<: RealField},   ::Type{<: Union{Val{Inf}, Val{-Inf}}}) = T
Base.promote_rule(T::Type{<: Integer}, ::Type{<: Union{Val{Inf}, Val{-Inf}}}) = AbstractFloat

Base.show(io::IO, ::Val{ Inf}) = print(io,  "∞")
Base.show(io::IO, ::Val{-Inf}) = print(io, "-∞")

struct Beth{α} end
const ℶ = Beth
(::Type{ℶ})(α::Integer) = ℶ{α}()
(^)(::Val{2}, ::ℶ{α}) where α = ℶ(α + 1) # Definition of ℶ by transfinite recursion.
(^)(base::ℤ, cardinal::ℶ) = Val(base)^cardinal
for i ∈ 0:10
    @eval const $(Symbol("ℶ"*sub(i))) = ℶ($i)
end
Base.show(io::IO, ::ℶ{α}) where α = print(io, "ℶ", sub(α))
# Assuming axiom of choice:
(+)(::ℶ{α}, ::ℶ{β}) where {α, β} = ℶ(max(α, β))
(*)(a::ℶ, b::ℶ) = a + b

const Compactification{T <: Number} = Union{T, typeof(-∞), typeof(∞)} # Two-point compactification.
Base.typemin(::Type{>: Val{-Inf}}) = -∞
Base.typemax(::Type{>: Val{ Inf}}) =  ∞

(^)(op::Operator, n::ℤ) = ComposedOperator(Iterators.repeated(op, n)...)
SymbolicUtils.istree(::Operator) = true
TermInterface.exprhead(::Operator) = :call
SymbolicUtils.operation(::Union{ComposedOperator, ComposedScalarOperator, ScaledOperator}) = (*)
SymbolicUtils.operation(::ScalarOperator) = identity
SymbolicUtils.operation(::AddedOperator) = (+)
SymbolicUtils.operation(::AdjointOperator) = adjoint
TermInterface.exprhead(::AdjointOperator) = Symbol("'")
SymbolicUtils.symtype(op::Operator) = eltype(op)
AbstractTrees.children(op::Operator) = getops(op)
SymbolicUtils.arguments(op::Operator) = getops(op) |> collect
SymbolicUtils.arguments(op::ScaledOperator) = [convert(Number, getops(op)[1]), getops(op)[2:end]...]

abstract type Dimension{T <: Real} <: Operator{T} end
SymbolicUtils.istree(::Dimension) = false
getops(::Dimension) = ()
"""Differs from `axes` in that it does not give concrete indices to any dimension."""
filter_type(T::Type,               op::Operator) = Iterators.filter(el -> el isa T, AbstractTrees.PostOrderDFS(op))
filter_type(T::Type{<: Dimension}, op::Operator) = Iterators.filter(el -> el isa T, AbstractTrees.Leaves(op))
# v4: symbolic time-independent solving
depends(op::Operator, x::Dimension) = x ∈ filter_type(Dimension, op)
"""Size as a map from an infinite set of ℂ^ℝ functions to ℂ^ℝ functions."""
Base.size(::Dimension) = (ℶ₂, ℶ₂) # Map from ψ ↦ ψ, a set of all dimensions ℂ^ℝ

struct Time <: Dimension{ℝ} end
isclassical(::Time) = true # N + 1 dimensional formulation.
Base.show(io::IO, ::Time) = print(io, "t")

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
Space(space::Space) = Space(
    space.lower,
    space.upper;
    periodic =space.periodic,
    classical=space.classical,

    a        =space.a,
    ε        =space.ε,
    canary   =space.canary
)
Base.copy(space::Space) = Space(space)
Space(upper) = Space(zero(upper), upper)
Space(lower, step, upper; keywords...) = Space(lower, upper; step=step, keywords...)
Space(range::AbstractRange{T}; keywords...) where T = Space{T}(first(range), last(range); a=step(range), keywords...)
Space{Bool}() = Space{Bool}(0, 1)

function Base.show(io::IO, space::Space)
    spaces = get(io, :spaces, nothing)
    if isnothing(spaces)
        print(io, "Space{$(getsymbol(eltype(space)))}($(space.lower)..$(space.upper)")
        if !get(io, :compact, false)
            print(io, ", periodic = $(space.periodic), classical = $(space.classical), a = $(space.a), ε = $(space.ε), canary = $(space.canary)")
        end
        print(io, ")")
        return
    end
    name = get(spaces, space, nothing)
    if isnothing(name)
        names = get(io, :names, nothing)
        if !isnothing(names)
            spaces[space] = newname = first(names)
            print(io, "($newname := ")
        end
        show(IOContext(io, :spaces => nothing), space)
        if !isnothing(names)
            print(io, ")")
        end
        return
    end
    print(io, name)
end

const .. = Space

==(a::Space, b::Space) = a === b
Base.hash(space::Space) = objectid(space)
Base.first(space::Space) = space.lower
Base.last( space::Space) = space.upper
Base.in(x::RealField, space::Space{<: Integer}) = false
Base.in(x, space::Space) = first(space) ≤ x ≤ last(space)
isbounded(space::Space) = isfinite(first(space)) && isfinite(last(space))
Base.isfinite(space::Space) = isbounded(space) && eltype(space) <: Integer
Base.isinf(space::Space) = !isfinite(space)
isclassical(space::Space) = space.classical
Base.length(space::Space{<: Integer}) = isfinite(space) ? last(space) - first(space) : ℶ₀
Base.length(space::Space) = ℶ₁

const Qubit = Space{Bool}
qubits(val::Val) = ntuple(_ -> Qubit(), val)
qubits(n::Integer) = qubits(Val(n))

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

function commutator(a::Operator, b::Operator)
    isdisjoint(    getindex.(∂, filter_type(Dimension, a)), filter_type(∂, b)) &&
        isdisjoint(getindex.(∂, filter_type(Dimension, b)), filter_type(∂, a)) &&
        return false # a and b commute.
    a*b - b*a
end
anticommutator(a::Operator, b::Operator) = a*b + b*a

# v4/5?
struct DiracDelta
    variety::Operator # Algebraic variety.
end
const δ = DiracDelta

struct Length{T} <: AbstractRange{T}
    space::Space{T}
    indices::AbstractRange{T}
end
# v3: function boundary detection by binary search
# v4: symbolic function boundary detection
# v4/5: intelligently select first dimension symbolically
"""Default range for provided space."""
function Length{T}(space::Space{T}) where T
    # TODO
    Length{T}(space, -10.0:10.0)
end
Length{T}(space::Space{T}) where {T <: Integer} = Length{T}(space, -10:10)
Base.length(l::Length) = length(l.indices)
Base.first( l::Length) =  first(l.indices)
Base.last(  l::Length) =   last(l.indices)

Base.convert(::Type{>: Length{T}}, space::Space{T}) where T = Length{T}(space)
Base.convert(::Type{>: Pair{Space{T}}}, l::Length{T}) where T = l.space => l.indices

function Base.show(io::IO, l::Length)
    name = get(get(io, :spaces, IdDict{Space, Char}()), l.space, nothing)
    if isnothing(name)
        print(io, "Length{$(getsymbol(eltype(l.space)))}($(l.space.lower)..$(l.space.upper), $(l.indices))")
        return
    end
    print(io, "$name[$(l.indices)]")
end

AbstractTrees.children(::Length) = ()
AbstractTrees.childrentype(::Type{<: Length}) = Tuple{}

Base.getindex(space::Space, indices::AbstractRange) = Length(space, indices)

const Volume{N} = NTuple{N, Length}
const NonEmptyVolume = Tuple{Length, Vararg{Length}} # May still be the null set if all lengths zero. All unqualified references to `Volume` should be NonEmptyVolume to avoid matching `Tuple{}`
Base.IteratorEltype(::Type{<: AbstractTrees.TreeIterator{<: NonEmptyVolume}}) = Base.HasEltype()
Base.eltype(::Type{<: AbstractTrees.TreeIterator{<: NonEmptyVolume}}) = Length
×(factors::Union{Length, NonEmptyVolume}...) = Volume(AbstractTrees.Leaves(factors))

Base.show(io::IO, vol::NonEmptyVolume) = join(io, vol, " × ")

Base.axes(op::Operator) = filter_type(Space, op) |> unique |> Volume
function Base.show(io::IO, op::Operator)
    names = get(io, :names, nothing)
    if isnothing(names)
        names = Iterators.Stateful(Char(i) for i ∈ Iterators.countfrom(0) if islowercase(Char(i)) && Char(i) ≠ 't')
        io = IOContext(io, :names => names)
    end
    spaces = get(io, :spaces, nothing)
    if isnothing(spaces)
        spaces = IdDict{Space, Char}()
        io = IOContext(io, :spaces => spaces)
    end
    if !get(io, :compact, false)
        printed = false
        for space ∈ filter_type(Space, op)
            get!(Ref(spaces), space) do
                newname = first(names)
                print(io, "$newname = ")
                show(IOContext(io, :spaces => nothing), space)
                println(";")
                printed = true
                return newname
            end
        end
        printed && println()
    end
    SymbolicUtils.show_term(io, op)
end
Base.show(io::IO, op::ScalarOperator) = print(io, convert(Number, op))

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

abstract type SymbolicIndex{T} <: SymbolicUtils.Symbolic{T} end
abstract type Index <: SymbolicIndex{Real} end
struct FirstIndex <: Index end
struct  LastIndex <: Index end
Base.firstindex(::State) = FirstIndex()
Base.lastindex( ::State) =  LastIndex()
Base.firstindex(::State, ::Integer) = FirstIndex()
Base.lastindex( ::State, ::Integer) =  LastIndex()
SymbolicUtils.istree(::Index) = false
struct IndexExpression <: SymbolicIndex{Real}
    f::Function
    args::Vector{Union{IndexExpression, Index, Number}}
end
const IndexScalar = Union{IndexExpression, Index, Number}
SymbolicUtils.istree(::IndexExpression) = true
TermInterface.exprhead(::IndexExpression) = :call
SymbolicUtils.operation(exp::IndexExpression) = exp.f
SymbolicUtils.arguments(exp::IndexExpression) = exp.args
(::Type{F <: Union{
    typeof(+),
    typeof(-),
    typeof(*),
    typeof(/),
    typeof(^)
}})(a, b) where F = IndexExpression(F, [a, b])
struct IndexRange{F <: AbstractRange} <: SymbolicIndex{F}
    args::Vector{IndexScalar}
end
(::Type{F <: AbstractRange})(args::IndexScalar...) where F = args |> collect |> IndexRange{F}
SymbolicUtils.istree(::IndexRange) = true
TermInterface.exprhead(::IndexRange) = :call
SymbolicUtils.operation(::IndexRange{F}) where F = F
SymbolicUtils.arguments(range::IndexRange) = range.args

function Base.axes(ψ::State{N}, space::Space) where N
    @boundscheck space ∈ ψ.inv || throw(DimensionMismatch())
    @inbounds ψ.ax[ψ.inv[space]]
end
Base.firstindex(ψ::State, space::Space) = first(axes(ψ, space))
Base.lastindex( ψ::State, space::Space) =  last(axes(ψ, space))

@inline to_index(ψ::State, ::Missing) = missing
@inline to_index(ψ::State, i::(Pair{Space{T}, T} where T)) = i.second
@inline to_index(ψ::State, i::(Pair{Space{T}, <: AbstractRange{T}} where T)) = i.second
@inline to_index(ψ::State, i::Pair{<: Space, Colon}) = axes(ψ, i.first).indices
@propagate_inbounds function to_index(ψ::State, i::(Pair{Space{T}, Length{T}} where T)) =
    @boundscheck i.second.space === i.first || throw(DimensionMismatch())
    to_index(ψ, i.first => i.second.indices)
end
function to_index(ψ::State, i::Pair{<: Space, <: SymbolicIndex})
    axis = axes(ψ, i.first)
    Base.to_index(ψ, i.first => SymbolicUtils.substitute(i.second, Dict{Index}(
        FirstIndex() => first(axis),
        LastIndex()  =>  last(axis)
    )))
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
    to_index.(Ref(ψ), get.(Ref(lookup), ax, missing))
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
Base.getindex()

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
    for (ψ, t) ∈ tuples(integrator)
        @show ψ, t
    end
end

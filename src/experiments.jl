# TODO look at Symbolic.jl and AbstractDiffEqOperator merging with ModelingToolkit.jl, and DiffEqOperators.jl (MethodOfLines.jl? looks like that automatically generates the discretization; may need more control and just discretize space myself), ContinuumArrays.jl, DomainSets.jl, GridArrays.jl, Grassmann.jl?, Infinities, integrate with QuantumOptics?
# JuliaDynamics.jl and InteractiveDynamics.jl for visualization
# Make Wavefunction take a Domain (and inherit from AbstractGrid1d), and generate the type of wavefunction based on type specialization of open/closed and circular domains, as well as split up for product domains. wavefunction.position will return a tuple of

[] intervals
() intervals
[) intervals
possible infinities on open
S^n manifolds
products of these
wavefunction is on a product of inputs

abstract type Manifold{D} end

struct ProductManifold{D} <: Manifold{D} end

const × = ProductManifold

struct Sphere{D} <: Manifold{D} end

const S = Sphere
(^)(::Type{Sphere}, n) = Sphere{n}

struct Coordinate{M <: Manifold{1}} end

x = Coordinate{ℝ}()
y = Coordinate{ℝ}()
z = Coordinate{ℝ}()

r, θ, ϕ = Coordinate{S^3}()

# coordinate transformations between manifolds, and importantly to Minkowski space (the laboratory frame)?
# user responsible for checking if metric is ok? Will assume Minkowski metric for rendering

import Base: ^
    using LinearAlgebra
import PhysicalConstants.CODATA2018: ReducedPlanckConstant, ħ
export Derivative, ∂, ReducedPlanckConstant, ħ

struct Time end
const t = Time

struct Parameter{N} end
Base.getindex(::Type{Parameter}, n::Int) = Parameter{n}

const p = Parameter

abstract type Basis{Cardinality, Name} end

abstract type HilbertSpace{Dimension, B <: Basis{Dimension}} end

const ∞ = Inf

const .. = HilbertSpace{∞}



abstract type Basis{Name} end

abstract type Operator{B <: Basis} <: AbstractLinearDiffEqOperator{ComplexF64} end

# abstract type AbstractBasis{B} <: Operator{B} end

abstract type HilbertSpace{Components, B <: Basis} <: Operator{B} end

basis(s::HilbertSpace{Components, B}) where {Components, B} = B

struct InfiniteHilbertSpace{Name, Start, Step, Stop, Auto} <: HilbertSpace{1, Basis{Name}}
    # XXX: Optimize conditions with Val?
    function ContinuousBasis(lower::Float64, step::Float64, upper::Float64, autosize::Bool)
        if !isfinite(step)
            throw(ArgumentError("step must be finite"))
        end

        if !autosize && !(isfinite(lower) && isfinite(upper))
            throw(ArgumentError("autosize must be enabled when basis is unbounded"))
        end

        new{gensym()}(lower, step, upper, autosize)
    end
end
ContinuousBasis(b::ContinuousBasis) = ContinuousBasis(b.lower, b.step, b.upper, b.autosize)

Base.copy(b::ContinuousBasis) = ContinuousBasis(b)

const .. = InfiniteHilbert
const ∞ = Inf
# const ℂ = ContinuousBasis(-∞, ∞)

struct FiniteHilbertSpace{Name} <: HilbertSpace{1, Basis{Name}}
    values::Vector{T}

    DiscreteBasis(args::Vararg{T}) where T = DiscreteBasis{T}(collect(args))

    DiscreteBasis(b::DiscreteBasis) = DiscreteBasis(b.values)
end
DiscreteBasis(itr) = DiscreteBasis(itr...)

Base.copy(b::DiscreteBasis) = DiscreteBasis(b)
Base.convert(::Type{FiniteHilbert}, basis::AbstractVector)

struct ProductSpace{Components, B} <: HilbertSpace{Components, B}
    components::Tuple{Vararg{AbstractBasis}}
end
ProductSpace(components...) = ProductBasis(components)
ProductBasis(product::ProductSpace{Components}, additional::HilbertSpace{1}...) where N = ProductBasis{Components + length(additional), }(product.components..., additional...)

Base.copy(b::ProductBasis) = ProductBasis(copy.(b.components))

(^)(b::AbstractBasis, n::Int) = ProductBasis(ntuple(i -> i > 1 ? copy(b) : b, n))

const × = ProductSpace

(v::HilbertSpace{Components, B})(state::Array{Components, ComplexF64}) where Components = State(x)
(v::HilbertSpace{Components})(state::AbstractArray{Components, ComplexF64}) where Components = v(convert(Array{Components, ComplexF64}, state))

struct State{Components, B <: Basis, Indices}
    data::Ref{Array{Components, N, ComplexF64}}
    derivatives::Vector{Array{N, ComplexG64}}
end

function resample!(ψ::State)
end

struct Derivative{S} <: Operator{ComponentBasis{S}}
    basis::ComponentBasis{S}
end

struct FunctionOperator{B, F, Args} <: Operator{B} end
Base.convert(::Type{FunctionOperator}, f::Function) = FunctionOperator{Union{}, f, Tuple{}}()

@generated function (op::Operator{B})(dψ, ψ::State{>:B, Indices}, p, t) where {B, CSCO}
    expr(::Type{Time}) = :(t)
    expr(::Type{Parameter{N}}) where N = :(p[$N])
    expr(::Type{FunctionOperator{B, F, Args}}) where {B, F, Args} = :($F($(expr.(Args)...)))

    quote
        for i in ψ.indices
            dψ[i] = $(expr(op))
        end
    end
end

(op::Operator)(dψ, ψ::State, p, t) = error("operator and state do not share a basis")



struct State{N} <: AbstractGrid{N,ComplexF64}
    domain::VcatDomain
    data::Array{N,ComplexF64}
end

struct Wavefunction <: AbstractVector{ComplexF64}
    data::Vector{ComplexF64}
    epsilon::Float64
    step::Float64
end
function trim!(w::Wavefunction)
    while abs(first(w.data)) < w.epsilon
        popfirst!(w.data)
    end
    while abs(last(w.data)) < w.epsilon
        pop!(w.data)
    end
end

const Operator = AbstractDiffEqOperator{ComplexF64}

struct Dimension <: Number
    dimension::Int
end

    struct Derivative{O} <: AbstractLinearDiffEqOperator{ComplexF64}
        dimension::Dimension
end
Derivative{O}(dimension::Int) where O = Derivative{O}(Dimension(dimension))
Derivative{O}() where O = Derivative{O}(1)
(::Type{Derivative})(args...) = Derivative{1}(args...)

(^)(::Type{Derivative{O}}, n::Int) where O = Derivative{O*n}

const ∂ = Derivative{1}

# Alternatively, with Symbolics.jl:
struct ∂{N} end
∂{N}(args...) where N = Differential(args...)^N
(::Type{∂})(args...) = ∂{1}
(^)(::Type{∂{N}}, n) where N = ∂{N*n}

schroedinger(H::Operator) = H/(im*ħ)

module Cartesian3DPosition
const x, y, z = Dimension.(1:3)

gradient() = ∂(x) + ∂(y) + ∂(z)
const ∇ = gradient
end

module Spherical3DPosition
const r, theta, phi = Dimension.(1:3)
const θ = theta
const ϕ = phi

const gradient = ∂(r) + (1/r)*∂(θ) + (1/(r*sin(θ)))*∂(ϕ)
const ∇ = gradient
end

x, y, z = ℂ^3
const ∇ = ∂.((x, y, z))

H = -ħ^2/(2m)*∇^2 + V(x, t)

p = -im*ħ*∂
V(x, t) = sin.(4x + t)
H = p^2(x)/(2m) + V(x, t)
H = -ħ^2/(2m)*∂^2(x) + sin.(4x + t)

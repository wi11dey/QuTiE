using SciMLOperators
import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, getops, ComposedOperator, ScaledOperator, ComposedScalarOperator, AddedOperator, IdentityOperator, FunctionOperator, AdjointOperator, InvertedOperator, islinear, isconstant, cache_operator, update_coefficients!, has_mul!

(^)(op::Operator, n::ℤ) = ComposedOperator(Iterators.repeated(op, n)...)
(^)(::Irrational{:ℯ}, op::Operator) = exp(op)
SymbolicUtils.istree(::Operator) = true
TermInterface.exprhead(::Operator) = :call
SymbolicUtils.operation(::Union{ComposedOperator, ComposedScalarOperator, ScaledOperator}) = (*)
SymbolicUtils.operation(::InvertedOperator) = inv
SymbolicUtils.operation(::IdentityOperator) = identity
SymbolicUtils.operation(::ScalarOperator) = identity
SymbolicUtils.operation(::AddedOperator) = (+)
SymbolicUtils.operation(::AdjointOperator) = adjoint
TermInterface.exprhead(::AdjointOperator) = Symbol("'")
SymbolicUtils.symtype(op::Operator) = eltype(op)
SymbolicUtils.arguments(op::Operator) = getops(op) |> collect
SymbolicUtils.arguments(op::ScaledOperator) = [convert(Number, getops(op)[1]), getops(op)[2:end]...]
@inline filter_type(::Type) = ()
@inline filter_type(::Type{T}, op,           ops::Operator...) where {T <: Operator} = filter_type(T,                ops...)
@inline filter_type(::Type{T}, op::Operator, ops::Operator...) where {T <: Operator} = filter_type(T, getops(op)..., ops...)
@inline filter_type(::Type{T}, op::T,        ops::Operator...) where {T <: Operator} = (op, filter_type(T, ops...)...)

SymbolicUtils.show_term(io::IO, ::IdentityOperator) = nothing

Base.show(io::IO, op::Operator) = SymbolicUtils.show_term(IOContext(io, :compact => true), op)
Base.show(io::IO, op::ScalarOperator) = print(io, convert(Number, op))

include("parallel_operators.jl")
include("broadcasted_operators.jl")

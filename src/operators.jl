import SciMLOperators: AbstractSciMLOperator as Operator,
AbstractSciMLScalarOperator as ScalarOperator,
getops,
ComposedOperator,
ScaledOperator,
ComposedScalarOperator,
AddedOperator,
FunctionOperator,
AdjointOperator,
InvertedOperator,
islinear,
isconstant,
cache_operator,
update_coefficients!

(^)(op::Operator, n::ℤ) = ComposedOperator(Iterators.repeated(op, n)...)
SymbolicUtils.istree(::Operator) = true
TermInterface.exprhead(::Operator) = :call
SymbolicUtils.operation(::Union{ComposedOperator, ComposedScalarOperator, ScaledOperator}) = (*)
SymbolicUtils.operation(::InvertedOperator) = inv
SymbolicUtils.operation(::ScalarOperator) = identity
SymbolicUtils.operation(::AddedOperator) = (+)
SymbolicUtils.operation(::AdjointOperator) = adjoint
TermInterface.exprhead(::AdjointOperator) = Symbol("'")
SymbolicUtils.symtype(op::Operator) = eltype(op)
AbstractTrees.children(op::Operator) = getops(op)
SymbolicUtils.arguments(op::Operator) = getops(op) |> collect
SymbolicUtils.arguments(op::ScaledOperator) = [convert(Number, getops(op)[1]), getops(op)[2:end]...]
filter_type(T::Type, op::Operator) = Iterators.filter(el -> el isa T, AbstractTrees.PostOrderDFS(op))

Base.show(io::IO, op::Operator) = SymbolicUtils.show_term(IOContext(io, :compact => true), op)
Base.show(io::IO, op::ScalarOperator) = print(io, convert(Number, op))

include("parallel_operators.jl")

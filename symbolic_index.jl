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

function Base.to_index(ψ::State, i::Pair{<: Space, <: SymbolicIndex})
    axis = axes(ψ, i.first)
    Base.to_index(ψ, i.first => SymbolicUtils.substitute(i.second, Dict{Index}(
        FirstIndex() => first(axis),
        LastIndex()  =>  last(axis)
    )))
end

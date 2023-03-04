abstract type SymbolicIndex{T} <: SymbolicUtils.Symbolic{T} end
abstract type SymbolicPosition <: SymbolicIndex{Real} end
struct SymbolicFirstIndex <: SymbolicPosition end
struct SymbolicLastIndex  <: SymbolicPosition end
Base.firstindex(::State) = SymbolicFirstIndex()
Base.lastindex( ::State) = SymbolicLastIndex( )
Base.firstindex(::State, ::Integer) = SymbolicFirstIndex()
Base.lastindex( ::State, ::Integer) = SymbolicLastIndex( )
SymbolicUtils.istree(::SymbolicFirstIndex) = false
SymbolicUtils.istree(::SymbolicLastIndex ) = false
struct SymbolicIndexExpression <: SymbolicPosition
    f::Function
    args::Vector{Union{SymbolicPosition, Number}}
end
SymbolicUtils.istree(::SymbolicIndexExpression) = true
TermInterface.exprhead(::SymbolicIndexExpression) = :call
SymbolicUtils.operation(exp::SymbolicIndexExpression) = exp.f
SymbolicUtils.arguments(exp::SymbolicIndexExpression) = exp.args
(::Type{F <: Union{
    typeof(+),
    typeof(-),
    typeof(*),
    typeof(/),
    typeof(^)
}})(a, b) where F = SymbolicIndexExpression(F, [a, b])
struct SymbolicRange{F <: AbstractRange} <: SymbolicIndex{F}
    args::Vector{SymbolicPosition}
end
(F::Union{Type{<: AbstractRange}, Colon})(args::SymbolicPosition...) where F = args |> collect |> SymbolicRange{F}
SymbolicUtils.istree(::SymbolicRange) = true
TermInterface.exprhead(::SymbolicRange) = :call
SymbolicUtils.operation(::SymbolicRange{F}) where F = F
SymbolicUtils.arguments(range::SymbolicRange) = range.args

Base.to_index(ψ::State, i::Pair{<: Space, <: SymbolicIndex}) =
    Base.to_index(ψ, i.first => SymbolicUtils.substitute(i.second, Dict(
        FirstIndex() => firstindex(ψ, i.first),
        LastIndex()  =>  lastindex(ψ, i.first)
    )))

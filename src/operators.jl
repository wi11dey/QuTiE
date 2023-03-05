import SciMLOperators: AbstractSciMLOperator as Operator, AbstractSciMLScalarOperator as ScalarOperator, getops, ComposedOperator, ScaledOperator, ComposedScalarOperator, AddedOperator, FunctionOperator, AdjointOperator, islinear

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
filter_type(T::Type, op::Operator) = Iterators.filter(el -> el isa T, AbstractTrees.PostOrderDFS(op))

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

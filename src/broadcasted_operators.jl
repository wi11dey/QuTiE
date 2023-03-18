struct BroadcastedPowerOperator{T, Base <: Number, Op <: Operator{T}} <: Operator{T}
    base::Base
    op::Op
end
Base.iszero(::BroadcastedPowerOperator) = false
isconstant(op::BroadcastedPowerOperator) = isconstant(op.op)
getops(op::BroadcastedPowerOperator) = op.base, op.op
Base.broadcasted(::typeof(^), base::Number, op::Operator) = BroadcastedPowerOperator(base, op)
(*)(op::BroadcastedPowerOperator, u::AbstractArray) = mul!(similar(u), op, u)
function LinearAlgebra.mul!(du::AbstractArray, op::BroadcastedPowerOperator, u::AbstractArray)
    mul!(du, op.op, u)
    du .= op.base.^du
    du
end
LinearAlgebra.mul!(du::AbstractVecOrMat, op::BroadcastedPowerOperator, u::AbstractVecOrMat) =
    invoke(LinearAlgebra.mul!, Tuple{AbstractArray, BroadcastedPowerOperator, AbstractVecOrMat}, du, op, u)
has_mul!(::BroadcastedPowerOperator) = true
SymbolicUtils.operation(::BroadcastedPowerOperator) = (.^)
function SymbolicUtils.show_call(io::IO, ::typeof(.^), args)
    SymbolicUtils.print_arg(io, args[1], paren=true)
    print(io, " .^ ")
    SymbolicUtils.print_arg(io, args[2], paren=true)
end

SciMLOperators.cache_operator(op::BroadcastedPowerOperator, u) =
    BroadcastedPowerOperator(op.base, cache_operator(op.op))

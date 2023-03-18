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
has_mul!(::BroadcastedPowerOperator) = true
SymbolicUtils.operation(::BroadcastedPowerOperator) = (.^)
function SymbolicUtils.show_call(io::IO, ::typeof(.^), args)
    SymbolicUtils.print_arg(io, args[1], paren=true)
    print(io, " .^ ")
    SymbolicUtils.print_arg(io, args[2], paren=true)
end

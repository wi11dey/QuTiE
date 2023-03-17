struct BroadcastedPowerOperator{T, Base <: Number, Op <: Operator{T}} <: Operator{T}
    base::Base
    op::Op
end
getops(op::BroadcastedPowerOperator) = op.base, op.op
Base.broadcasted(::typeof(^), base::Number, op::Operator) = BroadcastedPowerOperator(base, op)
# TODO: is this necessary?
(*)(op::BroadcastedPowerOperator, u::AbstractArray) = mul!(similar(u), op, u)
function LinearAlgebra.mul!(du::AbstractArray, op::BroadcastedPowerOperator, u::AbstractArray)
    mul!(du, op.op, u)
    du .= op.base.^du
    du
end

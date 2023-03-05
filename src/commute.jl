export commutator, anticommutator

function commutator(a::Operator, b::Operator)
    isdisjoint(    getindex.(∂, filter_type(Dimension, a)), filter_type(∂, b)) &&
        isdisjoint(getindex.(∂, filter_type(Dimension, b)), filter_type(∂, a)) &&
        return false # a and b commute.
    a*b - b*a
end
anticommutator(a::Operator, b::Operator) = a*b + b*a

export commutator, anticommutator

commutator(    a::Operator, b::Operator) = a*b - b*a
anticommutator(a::Operator, b::Operator) = a*b + b*a

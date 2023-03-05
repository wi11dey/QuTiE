export Qubit, qubits

const Qubit = Space{Bool}
qubits(val::Val) = ntuple(_ -> Qubit(), val)
qubits(n::Integer) = qubits(Val(n))

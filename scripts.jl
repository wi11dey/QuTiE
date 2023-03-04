const superscripts = collect("⁰¹²³⁴⁵⁶⁷⁸⁹")
const   subscripts = collect("₀₁₂₃₄₅₆₇₈₉")
sup(n::Integer) = join(superscripts[reverse!(digits(n)) .+ 1])
sub(n::Integer) = join(  subscripts[reverse!(digits(n)) .+ 1])

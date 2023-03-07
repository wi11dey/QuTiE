const superscripts = collect("⁰¹²³⁴⁵⁶⁷⁸⁹")
const   subscripts = collect("₀₁₂₃₄₅₆₇₈₉")
sup(n::Integer) = join(superscripts[reverse!(digits(n)) .+ 1])
sub(n::Integer) = join(  subscripts[reverse!(digits(n)) .+ 1])
getscript(s::String, scripts::AbstractVector{<: AbstractChar}) =
    indexin(match(Regex("^.*?(?<script>[$scripts]+)\$"), s)[:script], superscripts) .- 1
# getsup(s::String)
# getsub(s::String)

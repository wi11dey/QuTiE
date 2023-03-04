struct Beth{α} end
const ℶ = Beth
(::Type{ℶ})(α::Integer) = ℶ{α}()
(^)(::Val{2}, ::ℶ{α}) where α = ℶ(α + 1) # Definition of ℶ by transfinite recursion.
(^)(base::ℤ, cardinal::ℶ) = Val(base)^cardinal
for i ∈ 0:10
    @eval const $(Symbol("ℶ"*sub(i))) = ℶ($i)
end
Base.show(io::IO, ::ℶ{α}) where α = print(io, "ℶ", sub(α))
# Assuming axiom of choice:
(+)(::ℶ{α}, ::ℶ{β}) where {α, β} = ℶ(max(α, β))
(*)(a::ℶ, b::ℶ) = a + b

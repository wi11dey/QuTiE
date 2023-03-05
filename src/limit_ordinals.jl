export ∞

const ∞ = Val(Inf)
(-)(::Val{ Inf}) = Val(-Inf)
(-)(::Val{-Inf}) = Val( Inf)
Base.isinf(::Val{ Inf}) = true
Base.isinf(::Val{-Inf}) = true
Base.isfinite(::Val{ Inf}) = false
Base.isfinite(::Val{-Inf}) = false
Base.isnan(::Val{ Inf}) = false
Base.isnan(::Val{-Inf}) = false
==(::Val{Inf}, x::RealField) = Inf == x
==(x::RealField, ::Val{Inf}) = x == Inf
==(::Val{-Inf}, x::RealField) = -Inf == x
==(x::RealField, ::Val{-Inf}) = x == -Inf
Base.isless(::Val{Inf}, x) = isless(Inf, x)
Base.isless(x, ::Val{Inf}) = isless(x, Inf)
Base.isless(::Val{Inf}, ::Val{Inf}) = false
Base.isless(::Val{-Inf}, x) = isless(-Inf, x)
Base.isless(x, ::Val{-Inf}) = isless(x, -Inf)
Base.isless(::Val{-Inf}, ::Val{-Inf}) = false
Base.isless(::Val{ Inf}, ::Val{-Inf}) = false
Base.isless(::Val{-Inf}, ::Val{ Inf}) = true
(T::Type{<: RealField})(::Val{ Inf}) = T( Inf)
(T::Type{<: RealField})(::Val{-Inf}) = T(-Inf)
Base.convert(T::Type{<: RealField}, val::Union{Val{Inf}, Val{-Inf}}) = T(val)
Base.promote_rule(T::Type{<: RealField},   ::Type{<: Union{Val{Inf}, Val{-Inf}}}) = T
Base.promote_rule(T::Type{<: Integer}, ::Type{<: Union{Val{Inf}, Val{-Inf}}}) = AbstractFloat

Base.show(io::IO, ::Val{ Inf}) = print(io,  "∞")
Base.show(io::IO, ::Val{-Inf}) = print(io, "-∞")

const Compactification{T <: Number} = Union{T, typeof(-∞), typeof(∞)} # Two-point compactification.
Base.typemin(::Type{>: Val{-Inf}}) = -∞
Base.typemax(::Type{>: Val{ Inf}}) =  ∞

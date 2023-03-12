using Infinity
using MacroTools
using StaticArrays

export @space, ∞, isfield, isbounded, isperiodic, isclassical

struct Space{T, name} <: Dimension{T}
    lower::InfExtendedReal{T}
    upper::InfExtendedReal{T}
    periodic::Bool
    classical::Bool # ℂ^T Hilbert space if false.

    # v2: resample/interpolate when grid too large or not enough samples
    a::T # (Maximum) lattice spacing.
    ε::real(ℂ) # Minimum modulus.
    canary::T # Storage types should store enough cells to have at least this much canary border.

    function Space{T, name}(lower,
                            upper;
                            periodic=false,
                            classical=false, # v6
                            a=zero(T),
                            ε=1e-5,
                            canary=nothing) where {T, name}
        bounded = isfinite(lower) && isfinite(upper)
        !bounded && periodic && throw(ArgumentError("Unbounded space cannot be periodic"))
        lower == upper && throw(ArgumentError("Null space"))

        lower, upper = min(lower, upper), max(lower, upper)

        if isnothing(canary)
            if bounded
                if T <: Integer
                    canary = one(T)
                else
                    canary = eps(T)
                end
            else
                canary = zero(T)
            end
        end

        new{T, name}(
            lower,
            upper,
            periodic,
            classical,

            a,
            ε,
            canary
        )
    end
end
Space(space::Space) = Space(
    space.lower,
    space.upper;
    periodic =space.periodic,
    classical=space.classical,

    a        =space.a,
    ε        =space.ε,
    canary   =space.canary
)
Base.copy(space::Space) = Space(space)
Space(upper) = Space(zero(upper), upper)
Space(lower, step, upper; keywords...) = Space(lower, upper; step=step, keywords...)
Space(range::AbstractRange{T}; keywords...) where T = Space{T}(first(range), last(range); a=step(range), keywords...)
Space{T, name}() where {T <: Real, name} = Space{T, name}(-∞, ∞)

DimensionalData.name(::Space{T, nme}) where {T, nme} = nme

# v2: define names as new functions/types to keep information on the module in which dimensions were defined
macro space(expr)
    power = 1
    args = []
    @capture(expr, name_Symbol := definition_)
    @capture(definition, T_Symbol)                       && @goto parsed
    @capture(definition, T_Symbol^power_Int)             && @goto parsed
    @capture(definition, T_Symbol^(power_Int*(args__,))) && @goto parsed
    @capture(definition, (T_Symbol^power_Int)(args__))   && @goto parsed
    @capture(definition, T_Symbol(args__)^power_Int)     && @goto parsed
    @capture(definition, T_Symbol(args__))               && @goto parsed
    error("""Incorrect usage of @space. Use like:

@space x := ℝ(0, ∞)
@space 𝐫 := ℝ^3
""")
    @label parsed
    power = something(power, 1)
    power > 0 || error("Dimensions must be greater than zero")
    map!(args, args) do arg
        Meta.isexpr(arg, :parameters) || return arg
        Expr(:parameters, (kw isa Symbol ? Expr(:kw, kw, true) : kw for kw in arg.args)...)
    end
    value = if power > 1
        :($SVector($((:($Space{$T, $("$name"*sub(i) |> gensym |> Meta.quot)}($(args...))) for i ∈ 1:power)...)))
    else
        :($Space{$T, $(name |> gensym |> Meta.quot)}($(args...)))
    end
    esc(:(const $name = $value))
end

function Base.show(io::IO, space::Space)
    print(io, match(r"^##(?<tag>.+)#\d+$", space |> name |> string)[:tag])
    if !get(io, :compact, false)
        print(io, " := $(getsymbol(eltype(space)))($(space.lower.val), $(space.upper.val); periodic = $(space.periodic), classical = $(space.classical), a = $(space.a), ε = $(space.ε), canary = $(space.canary))")
    end
end

==(a::Space, b::Space) = a === b
Base.hash(space::Space) = objectid(space)
Base.first(space::Space) = space.lower
Base.last( space::Space) = space.upper
Base.in(x::RealField, space::Space{<: Integer}) = false
Base.in(x, space::Space) = first(space) ≤ x ≤ last(space)
isfield(space::Space) = !(eltype(space) <: Integer)
Base.isfinite(space::Space) = isbounded(space) && !isfield(space)
Base.isinf(space::Space) = !isfinite(space)
Base.length(space::Space{<: Integer}) = isfinite(space) ? last(space) - first(space) : ℶ₀
Base.length(space::Space) = ℶ₁
Base.extrema(space::Space) = first(space), last(space)
isbounded(space::Space) = all(isfinite, extrema(space))
isclassical(space::Space) = space.classical
isperiodic(space::Space) = space.periodic

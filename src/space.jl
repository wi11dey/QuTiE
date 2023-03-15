using Infinity

export Space, ∞, isfield, isbounded, isperiodic, isclassical

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
Space{T, name}(space::Space;
               periodic =space.periodic,
               classical=space.classical,
               a        =space.a,
               ε        =space.ε,
               canary   =space.canary) where {T, name} = Space{T, name}(
                   space.lower,
                   space.upper;
                   periodic,
                   classical,

                   a,
                   ε,
                   canary
               )
Space{T}(args...; kwargs...) where T = Space{T, gensym()}(args...; kwargs...)
Space(upper) = Space(zero(upper), upper)
Space(lower, step, upper; keywords...) = Space(lower, upper; step=step, keywords...)
Space(range::AbstractRange{T}; keywords...) where T = Space{T}(first(range), last(range); a=step(range), keywords...)

DimensionalData.name(::Space{T, nme}) where {T, nme} = nme

define(name::Symbol, space::Space{T}) where T = Space{T, name}(space)

function Base.show(io::IO, space::Space)
    if space |> name |> Meta.isidentifier
        print(io, space |> name)
    end
    if !get(io, :compact, false)
        if space |> name |> Meta.isidentifier
            print(io, " := ")
        end
        print(io, "$(getsymbol(eltype(space)))($(space.lower.val), $(space.upper.val); periodic = $(space.periodic), classical = $(space.classical), a = $(space.a), ε = $(space.ε), canary = $(space.canary))")
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
isconstant(::Space) = true

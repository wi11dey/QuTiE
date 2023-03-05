export Space, .., isbounded, isperiodic, isclassical

mutable struct Space{T} <: Dimension{T}
    const lower::Compactification{T}
    const upper::Compactification{T}
    const periodic::Bool
    const classical::Bool # ℂ^T Hilbert space if false.

    # v2: resample/interpolate when grid too large or not enough samples
    const a::Union{T, Nothing} # (Maximum) lattice spacing.
    const ε::real(ℂ) # Minimum modulus.
    const canary::T # Storage types should store enough cells to have at least this much canary border.

    function Space{T}(lower,
                      upper;
                      periodic=false,
                      classical=false, # v6
                      a=nothing,
                      ε=1e-5,
                      canary=nothing) where T
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

        new(
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
Space{Bool}() = Space{Bool}(0, 1)

function Base.show(io::IO, space::Space)
    spaces = get(io, :spaces, nothing)
    if isnothing(spaces)
        print(io, "Space{$(getsymbol(eltype(space)))}($(space.lower)..$(space.upper)")
        if !get(io, :compact, false)
            print(io, ", periodic = $(space.periodic), classical = $(space.classical), a = $(space.a), ε = $(space.ε), canary = $(space.canary)")
        end
        print(io, ")")
        return
    end
    name = get(spaces, space, nothing)
    if isnothing(name)
        names = get(io, :names, nothing)
        if !isnothing(names)
            spaces[space] = newname = first(names)
            print(io, "($newname := ")
        end
        show(IOContext(io, :spaces => nothing), space)
        if !isnothing(names)
            print(io, ")")
        end
        return
    end
    print(io, name)
end

const .. = Space

==(a::Space, b::Space) = a === b
Base.hash(space::Space) = objectid(space)
Base.first(space::Space) = space.lower
Base.last( space::Space) = space.upper
Base.in(x::RealField, space::Space{<: Integer}) = false
Base.in(x, space::Space) = first(space) ≤ x ≤ last(space)
isbounded(space::Space) = isfinite(first(space)) && isfinite(last(space))
Base.isfinite(space::Space) = isbounded(space) && eltype(space) <: Integer
Base.isinf(space::Space) = !isfinite(space)
isclassical(space::Space) = space.classical
Base.length(space::Space{<: Integer}) = isfinite(space) ? last(space) - first(space) : ℶ₀
Base.length(space::Space) = ℶ₁
isperiodic(space::Space) = space.periodic

using Interpolations

struct Interpolation{T, N, IT <: AbstractInterpolation{T, N}} <: AbstractArray{T, N}
    parent::IT
end
Base.parent(itp::Interpolation) = wrapper.parent
Base.size(itp::Interpolation) = itp |> parent |> size
Base.getindex(itp::Interpolation, i...) = parent(itp)(i...)

Interpolations.coefficients(etp::AbstractExtrapolation) = etp |> parent |> Interpolations.coefficients

function Base.checkbounds(etp::AbstractExtrapolation, x...)
    Interpolations.inbounds_position(Interpolations.etpflag(etp), Interpolations.bounds(parent(etp)), x, etp, x)
    return
end

Base.checkbounds(::Type{𝔽₂}, etp::AbstractExtrapolation, x::Union{Number, Vector}...) =
    try
        checkbounds(etp, x...)
    catch e
        e isa BoundsError || rethrow()
        return false
    else
        return true
    end

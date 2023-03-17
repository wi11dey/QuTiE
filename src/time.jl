export Time

mutable struct Time <: Dimension{ℝ}
    t::ℝ

    Time() = new(NaN)
end
update_coefficients!(τ::Time, u, p, t) = τ.t = t
isclassical(::Time) = true  # N + 1 dimensional formulation.
isperiodic( ::Time) = false # No time travel.
isconstant( ::Time) = false # By construction.
Base.first(::Time) = 0
Base.last( ::Time) = ∞
Base.show(io::IO, ::Time) = print(io, "t")

DimensionalData.name(::Time) = :t

(*)(                   τ::Time, _) = τ.t
LinearAlgebra.mul!(du, τ::Time, u) = du .= τ*u

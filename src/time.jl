export Time

struct Time <: Dimension{ℝ}
    t::ℝ

    update_coefficients!(τ::Time, u, p, t) = new(t)

    Time() = new(NaN)
end
isclassical(::Time) = true  # N + 1 dimensional formulation.
isperiodic( ::Time) = false # No time travel.
isconstant( ::Time) = false # By construction.
Base.show(io::IO, ::Time) = print(io, "t")

(*)(                   τ::Time, _) = τ.t
LinearAlgebra.mul!(du, τ::Time, u) = du .= τ*u

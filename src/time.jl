export Time

struct Time <: Dimension{ℝ} end
isclassical(::Time) = true # N + 1 dimensional formulation.
isperiodic( ::Time) = false # No time travel.
Base.show(io::IO, ::Time) = print(io, "t")

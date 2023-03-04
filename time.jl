export Time

struct Time <: Dimension{ℝ} end
isclassical(::Time) = true # N + 1 dimensional formulation.
Base.show(io::IO, ::Time) = print(io, "t")

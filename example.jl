t = Time()
x = Space{ℝ}(-∞, ∞)

# Harmonic potential:
k = 1
V(x, t) = -k*x

∇ = ∂(x)
H = -ħ^2/(2m)*∇^2 + V(x, t)
# Alternatively:
H = -ħ²/(2m)*∂²(x) + V(x, t)

# Initial Gaussian probability distribution for particle:
ψ₀=ℯ^(-x^2)

@define x := Space{ℝ}(-∞, ∞)

# Harmonic potential:
k = 1
V(x, t) = -k*x

m = 2

∇ = ∂[x]
H = -ħ²/(2m)*∇^2 + V(x, t)
# Alternatively:
H = -ħ²/(2m)*∂²[x] + V(x, t)

# Initial Gaussian probability distribution for particle:
ψ₀ = 1/∜(2π)*ℯ.^(-x^2/4)

# 📊 = mapping() * visual(Wireframe)

# v6
abstract type ClassicalObject end

# Propagate until reflection using Huygens' principle in N dimensions:
abstract type Wave{N} <: ClassicalObject end # For classical field theory.
struct PlaneWave{N} <: Wave{N} end
struct SphericalWave{N} <: Wave{N} end

# Solve using Euler-Lagrange equation in N generalized coordinates:
abstract type Particle{N} <: ClassicalObject end # For classical mechanics.

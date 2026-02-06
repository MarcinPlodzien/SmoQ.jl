# Date: 2026
#
#=
================================================================================
    cpuVariationalQuantumCircuitGradients.jl - Gradient Computation for VQC
================================================================================

PURPOSE:
--------
Compute gradients of cost functions with respect to variational parameters.
Enzyme.jl is the default autodiff backend for maximum performance.

GRADIENT METHODS:
-----------------
1. calculate_gradients (Enzyme autodiff - DEFAULT, exact, fast)
2. calculate_gradients_parameter_shift (for verification, no autodiff dependency)
3. calculate_gradients_spsa (stochastic, 2 evaluations regardless of n_params)

ENZYME INTEGRATION:
-------------------
Enzyme performs reverse-mode automatic differentiation at the LLVM level,
making it extremely fast for Julia code. It differentiates through our
bitwise gate operations efficiently.

USAGE:
------
```julia
include("cpuVariationalQuantumCircuitGradients.jl")
using .CPUVariationalQuantumCircuitGradients

# Define cost function
function my_cost(θ)
    ψ = make_ket("|0>", N)
    apply_circuit!(ψ, circuit, θ)
    return real(expect_local(ψ, 1, N, :z))  # Minimize ⟨Z₁⟩
end

# Compute gradients with Enzyme (default)
grad = calculate_gradients(my_cost, θ)

# Verify with parameter-shift rule
grad_ps = calculate_gradients_parameter_shift(my_cost, θ)
@assert isapprox(grad, grad_ps, atol=1e-5)
```

================================================================================
=#

module CPUVariationalQuantumCircuitGradients

using LinearAlgebra

# Enzyme for autodiff (optional - graceful fallback if not installed)
const ENZYME_AVAILABLE = try
    @eval using Enzyme
    true
catch
    false
end

export calculate_gradients
export calculate_gradients_parameter_shift
export calculate_gradients_spsa
export calculate_gradients_finite_difference

# ==============================================================================
# ENZYME AUTODIFF (DEFAULT)
# ==============================================================================

"""
    calculate_gradients(cost_fn, θ::Vector{Float64}) -> Vector{Float64}

Compute exact gradients using Enzyme reverse-mode autodiff.
This is the default and fastest method.

# Arguments
- `cost_fn`: Cost function f(θ) → Float64
- `θ`: Parameter vector

# Returns
- Gradient vector ∇f(θ)

# Example
```julia
grad = calculate_gradients(cost_fn, θ)
```
"""
function calculate_gradients(cost_fn, θ::Vector{Float64})
    if ENZYME_AVAILABLE
        return _enzyme_gradient(cost_fn, θ)
    else
        @warn "Enzyme not available, falling back to parameter-shift rule"
        return calculate_gradients_parameter_shift(cost_fn, θ)
    end
end

"""
Internal Enzyme gradient computation.
"""
function _enzyme_gradient(cost_fn, θ::Vector{Float64})
    # Allocate gradient storage
    dθ = zeros(Float64, length(θ))
    
    # Enzyme reverse-mode autodiff
    # autodiff(Reverse, f, Active, Duplicated(x, dx))
    # After call: dθ contains ∂f/∂θ
    Enzyme.autodiff(Enzyme.Reverse, cost_fn, Enzyme.Active, Enzyme.Duplicated(θ, dθ))
    
    return dθ
end

# ==============================================================================
# PARAMETER-SHIFT RULE (For Verification)
# ==============================================================================

"""
    calculate_gradients_parameter_shift(cost_fn, θ; shift=π/2) -> Vector{Float64}

Compute gradients using the parameter-shift rule (exact for Pauli rotation gates).

This method requires 2 circuit evaluations per parameter.

# Mathematical Formula
For rotation gates Rₓ(θ), Rᵧ(θ), Rᵤ(θ):
    ∂C/∂θₖ = [C(θₖ + π/2) - C(θₖ - π/2)] / 2

# Arguments
- `cost_fn`: Cost function f(θ) → Float64
- `θ`: Parameter vector
- `shift`: Shift amount (default π/2 for Pauli rotations)

# Returns
- Gradient vector ∇f(θ)

# Note
Use this to verify Enzyme gradients are correct.
"""
function calculate_gradients_parameter_shift(cost_fn, θ::Vector{Float64}; shift::Float64=π/2)
    n_params = length(θ)
    grad = zeros(Float64, n_params)
    
    θ_plus = copy(θ)
    θ_minus = copy(θ)
    
    for k in 1:n_params
        # Shift parameter k forward
        θ_plus[k] = θ[k] + shift
        θ_minus[k] = θ[k] - shift
        
        # Evaluate cost at shifted parameters
        c_plus = cost_fn(θ_plus)
        c_minus = cost_fn(θ_minus)
        
        # Parameter-shift formula
        grad[k] = (c_plus - c_minus) / (2 * sin(shift))
        
        # Reset
        θ_plus[k] = θ[k]
        θ_minus[k] = θ[k]
    end
    
    return grad
end

# ==============================================================================
# SPSA (Simultaneous Perturbation Stochastic Approximation)
# ==============================================================================

"""
    calculate_gradients_spsa(cost_fn, θ, c; Δ=nothing) -> Vector{Float64}

Estimate gradients using SPSA (only 2 circuit evaluations total!).

This is extremely efficient for many parameters, but provides a noisy estimate.
Best used with SPSA optimizer which averages over many iterations.

# Mathematical Formula
    gₖ ≈ [C(θ + cΔ) - C(θ - cΔ)] / (2c Δₖ)

where Δ is a Bernoulli ±1 random vector.

# Arguments
- `cost_fn`: Cost function f(θ) → Float64
- `θ`: Parameter vector
- `c`: Perturbation magnitude
- `Δ`: Optional perturbation vector (samples Bernoulli if not provided)

# Returns
- Gradient estimate ĝ(θ)

# Note
SPSA requires only 2 function evaluations regardless of the number of parameters!
This makes it ideal for hardware deployment or very large parameter counts.
"""
function calculate_gradients_spsa(cost_fn, θ::Vector{Float64}, c::Float64; 
                                   Δ::Union{Nothing, Vector{Float64}}=nothing)
    n_params = length(θ)
    
    # Generate Bernoulli ±1 perturbation if not provided
    if isnothing(Δ)
        Δ = 2.0 .* (rand(n_params) .> 0.5) .- 1.0
    end
    
    # Perturbed parameter vectors
    θ_plus = θ .+ c .* Δ
    θ_minus = θ .- c .* Δ
    
    # Only 2 function evaluations!
    c_plus = cost_fn(θ_plus)
    c_minus = cost_fn(θ_minus)
    
    # SPSA gradient estimate
    grad = (c_plus - c_minus) ./ (2.0 * c .* Δ)
    
    return grad
end

"""
    sample_spsa_perturbation(n_params) -> Vector{Float64}

Sample a Bernoulli ±1 perturbation vector for SPSA.
"""
function sample_spsa_perturbation(n_params::Int)
    return 2.0 .* (rand(n_params) .> 0.5) .- 1.0
end

# ==============================================================================
# FINITE DIFFERENCE (Fallback)
# ==============================================================================

"""
    calculate_gradients_finite_difference(cost_fn, θ; ε=1e-5) -> Vector{Float64}

Compute gradients using central finite differences.

# Mathematical Formula
    ∂C/∂θₖ ≈ [C(θₖ + ε) - C(θₖ - ε)] / (2ε)

# Arguments
- `cost_fn`: Cost function f(θ) → Float64
- `θ`: Parameter vector
- `ε`: Step size (default 1e-5)

# Returns
- Gradient approximation

# Note
This is a fallback method. Use Enzyme or parameter-shift for exact gradients.
"""
function calculate_gradients_finite_difference(cost_fn, θ::Vector{Float64}; ε::Float64=1e-5)
    n_params = length(θ)
    grad = zeros(Float64, n_params)
    
    θ_work = copy(θ)
    
    for k in 1:n_params
        θ_work[k] = θ[k] + ε
        c_plus = cost_fn(θ_work)
        
        θ_work[k] = θ[k] - ε
        c_minus = cost_fn(θ_work)
        
        grad[k] = (c_plus - c_minus) / (2 * ε)
        
        θ_work[k] = θ[k]
    end
    
    return grad
end

# ==============================================================================
# GRADIENT VERIFICATION
# ==============================================================================

"""
    verify_gradients(cost_fn, θ; atol=1e-4) -> (match::Bool, max_diff::Float64)

Compare Enzyme gradients with parameter-shift rule for verification.

# Returns
- `match`: true if gradients agree within tolerance
- `max_diff`: maximum absolute difference between methods
"""
function verify_gradients(cost_fn, θ::Vector{Float64}; atol::Float64=1e-4)
    grad_enzyme = calculate_gradients(cost_fn, θ)
    grad_ps = calculate_gradients_parameter_shift(cost_fn, θ)
    
    max_diff = maximum(abs.(grad_enzyme .- grad_ps))
    match = max_diff < atol
    
    if !match
        @warn "Gradient mismatch! max_diff = $max_diff"
    end
    
    return match, max_diff
end

end # module CPUVariationalQuantumCircuitGradients

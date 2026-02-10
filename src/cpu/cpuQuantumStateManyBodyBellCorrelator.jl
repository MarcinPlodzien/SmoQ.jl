# Date: 2026
#
#=
================================================================================
    cpuQuantumStateManyBodyBellCorrelator.jl - Many-Body Bell Correlations
================================================================================

PURPOSE:
--------
Matrix-free optimization of many-body Bell correlator for quantum states:

    ℰ = max_θ |Tr[ρ · 𝓑(θ)]|²

and two Q-correlators:
    Q_bell = log₂(ℰ) + N
    Q_ent  = log₄(ℰ) + N = (1/2)log₂(ℰ) + N

The Bell operator:
    𝓑(θ) = ⊗ⱼ Uⱼ(θⱼ,φⱼ) σ⁺ⱼ Uⱼ†(θⱼ,φⱼ)

where Uⱼ = Rz(φⱼ)Ry(θⱼ) and σ⁺ = |1⟩⟨0|.

MAIN API:
---------
    get_bell_correlator(state; kwargs...) -> (Q_bell, Q_ent, θ_opt)

INPUT MODES:
------------
1. Pure state |ψ⟩:       ⟨ψ|𝓑|ψ⟩
2. Density matrix ρ:     Tr[ρ·𝓑]
3. Trajectory ensemble:  (1/M) Σᵢ ⟨ψᵢ|𝓑|ψᵢ⟩  (MCWF mode)

PHYSICAL INTERPRETATION:
------------------------
- For entangled states (GHZ, graph states): Q_bell ≈ N, Q_ent > 0
- For product states: Q_bell <= 0, Q_ent <= 0 (no genuine multipartite entanglement)
- Maximum Q_bell = N-2 saturated by maximally entangled states
- Bell correlator is meaningful for N ≥ 3 (genuine multipartite entanglement)

OPTIMIZATION METHODS:
---------------------
The Bell correlator optimization is a non-convex problem with 2N parameters.
We implement and recommend the following methods:

1. SPSA + Adam (default, implemented here):
   - Simultaneous Perturbation Stochastic Approximation with Adam momentum
   - Gradient-free: requires only 2 function evaluations per iteration
   - Robust to noise, works well for MCWF trajectories
   - Recommended for general use

2. L-BFGS (fallback, via Optim.jl):
   - Quasi-Newton method using finite differences
   - Fast convergence for smooth objectives
   - Better for pure states, less suitable for noisy MCWF

3. Other options worth considering (not yet implemented):
   - CMA-ES: Covariance Matrix Adaptation Evolution Strategy
     Good for non-convex landscapes, but slower per iteration
   - Nelder-Mead: Simplex-based derivative-free method
     Simple but can get stuck in local minima
   - Basin-hopping: Global optimization with local refinement
     Best for finding global optimum, but expensive

REFERENCES (chronological order):
---------------------------------
The many-body Bell correlator implemented here has been developed and
applied in the following peer-reviewed publications:

[1] Jan Chwedeńczuk,
    "Many-body Bell inequalities for bosonic qubits"
    SciPost Physics Core 5, 025 (2022)
    DOI: 10.21468/SciPostPhysCore.5.2.025
    arXiv: 2203.02545

[2] Marcin Płodzień, Maciej Lewenstein, Emilia Witkowska, Jan Chwedeńczuk,
    "One-axis twisting as a method of generating many-body Bell correlations"
    Physical Review Letters 129, 250402 (2022)
    DOI: 10.1103/PhysRevLett.129.250402
    arXiv: 2206.10542

[3] Marcin Płodzień, Tomasz Wasak, Emilia Witkowska, Maciej Lewenstein,
    Jan Chwedeńczuk,
    "Generation of scalable many-body Bell correlations in spin chains
    with short-range two-body interactions"
    Physical Review Research 6, 023050 (2024)
    DOI: 10.1103/PhysRevResearch.6.023050
    arXiv: 2306.03163

[4] Marcin Płodzień, Jan Chwedeńczuk, Maciej Lewenstein,
    Grzegorz Rajchel-Mieldzioć,
    "Entanglement classification and non-k-separability certification
    via Greenberger-Horne-Zeilinger-class fidelity"
    Physical Review A 110, 032428 (2024)
    DOI: 10.1103/PhysRevA.110.032428
    arXiv: 2406.10662

[5] Marcin Płodzień, Jan Chwedeńczuk, Maciej Lewenstein,
    "Inherent quantum resources in stationary spin chains"
    Physical Review A 111, 012417 (2025)
    DOI: 10.1103/PhysRevA.111.012417
    arXiv: 2405.16974

[6] Marcin Płodzień, Maciej Lewenstein, Jan Chwedeńczuk,
    "Many-body quantum resources of graph states"
    Reports on Progress in Physics 88, 017001 (2025)
    DOI: 10.1088/1361-6633/adecc0
    arXiv: 2410.12487

================================================================================
=#

module CPUQuantumStateManyBodyBellCorrelator

using LinearAlgebra
using Printf
using Random
using Base.Threads

# Import matrix-free gates for optimized bell expectation
using ..CPUQuantumChannelGates

# Try to import Enzyme for autodiff (optional)
const ENZYME_AVAILABLE = try
    @eval using Enzyme
    @eval using Optimisers
    true
catch
    false
end

export get_bell_correlator, BellCorrelatorResult
export bell_expectation, bell_correlator, bell_expectation_fast
export compute_Q_bell, compute_Q_ent
export bootstrap_bell_error

# =============================================================================
# RESULT STRUCT
# =============================================================================

"""
Result of Bell correlator optimization.

# Fields
- `Q_bell::Float64`: log₂(ℰ) + N
- `Q_ent::Float64`: log₄(ℰ) + N = (1/2)log₂(ℰ) + N
- `θ_opt::Vector{Float64}`: Optimal angles [θ₁,φ₁,...,θₙ,φₙ]
- `ℰ_max::Float64`: Maximum |⟨𝓑⟩|²
- `N::Int`: Number of qubits
- `iterations::Int`: Optimizer iterations
- `converged::Bool`: Whether optimizer converged
"""
struct BellCorrelatorResult
    Q_bell::Float64
    Q_ent::Float64
    θ_opt::Vector{Float64}
    ℰ_max::Float64
    N::Int
    iterations::Int
    converged::Bool
end

# =============================================================================
# ROTATED σ⁺ OPERATOR (matrix element computation)
# =============================================================================

"""
    rotated_sigma_plus_element(θ, φ, bra_bit, ket_bit) -> ComplexF64

Compute matrix element ⟨bra_bit| Rz(φ)Ry(θ) σ⁺ Ry(-θ)Rz(-φ) |ket_bit⟩.

σ⁺ = |1⟩⟨0| in standard convention (raising operator).
Local rotation U = Rz(φ)Ry(θ) parameterizes Bloch sphere.
"""
function rotated_sigma_plus_element(θ::Float64, φ::Float64,
                                     bra_bit::Int, ket_bit::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)

    # U = Rz(φ)Ry(θ)
    # U matrix elements:
    # U[0,0] = cos(θ/2)e^{-iφ/2}, U[0,1] = -sin(θ/2)e^{-iφ/2}
    # U[1,0] = sin(θ/2)e^{iφ/2},  U[1,1] = cos(θ/2)e^{iφ/2}

    # σ⁺ = |1⟩⟨0| → (Uσ⁺U†)_{ij} = U_{i,1} * conj(U_{j,0})

    exp_m = exp(-im * φ / 2)
    exp_p = exp(im * φ / 2)

    # U_{bra_bit, 1}
    if bra_bit == 0
        U_i1 = -s * exp_m
    else  # bra_bit == 1
        U_i1 = c * exp_p
    end

    # conj(U_{ket_bit, 0})
    if ket_bit == 0
        U_j0_conj = conj(c * exp_m)
    else  # ket_bit == 1
        U_j0_conj = conj(s * exp_p)
    end

    return U_i1 * U_j0_conj
end

"""
    bell_operator_element(angles, bra, ket, N) -> ComplexF64

Compute ⟨bra|𝓑(θ,φ)|ket⟩ for N-qubit Bell operator.
"""
function bell_operator_element(angles::AbstractVector{Float64},
                                bra::Int, ket::Int, N::Int)
    result = ComplexF64(1.0)

    @inbounds for k in 1:N
        θ = angles[2k - 1]
        φ = angles[2k]
        bra_bit = (bra >> (k-1)) & 1
        ket_bit = (ket >> (k-1)) & 1
        result *= rotated_sigma_plus_element(θ, φ, bra_bit, ket_bit)
    end

    return result
end

# =============================================================================
# BELL EXPECTATION VALUE
# =============================================================================

"""
    bell_expectation(ψ::Vector{ComplexF64}, angles::Vector{Float64}) -> ComplexF64

Compute ⟨ψ|𝓑(θ)|ψ⟩ for pure state.
"""
function bell_expectation(ψ::Vector{ComplexF64}, angles::Vector{Float64})
    d = length(ψ)
    N = Int(log2(d))
    @assert length(angles) == 2N "Need 2N angles (θ,φ per qubit)"

    result = ComplexF64(0.0)

    @inbounds for bra in 0:(d-1)
        for ket in 0:(d-1)
            elem = bell_operator_element(angles, bra, ket, N)
            result += conj(ψ[bra+1]) * ψ[ket+1] * elem
        end
    end

    return result
end

"""
    bell_expectation(ρ::Matrix{ComplexF64}, angles::Vector{Float64}) -> ComplexF64

Compute Tr[ρ·𝓑(θ)] for density matrix.
"""
function bell_expectation(ρ::Matrix{ComplexF64}, angles::Vector{Float64})
    d = size(ρ, 1)
    N = Int(log2(d))
    @assert length(angles) == 2N "Need 2N angles (θ,φ per qubit)"

    result = ComplexF64(0.0)

    @inbounds for s in 0:(d-1)
        for t in 0:(d-1)
            B_ts = bell_operator_element(angles, t, s, N)
            result += ρ[s+1, t+1] * B_ts
        end
    end

    return result
end

"""
    bell_expectation(trajectories::Vector{Vector{ComplexF64}}, angles) -> ComplexF64

Compute (1/M) Σᵢ ⟨ψᵢ|𝓑(θ)|ψᵢ⟩ for MCWF ensemble.

IMPORTANT: Average expectation FIRST, then |...|² for proper MCWF.
"""
function bell_expectation(trajectories::Vector{Vector{ComplexF64}},
                          angles::Vector{Float64})
    M = length(trajectories)
    @assert M > 0 "Need at least one trajectory"

    result = ComplexF64(0.0)
    for ψ in trajectories
        result += bell_expectation(ψ, angles)
    end

    return result / M
end

"""
    bell_correlator(state, angles) -> Float64

Compute ℰ = |⟨𝓑(θ)⟩|².
"""
function bell_correlator(state, angles::Vector{Float64})
    return abs2(bell_expectation(state, angles))
end

# =============================================================================
# OPTIMIZED MATRIX-FREE BELL EXPECTATION (O(2^N) instead of O(4^N))
# =============================================================================

"""
    bell_expectation_fast(ψ::Vector{ComplexF64}, angles::Vector{Float64}) -> ComplexF64

OPTIMIZED matrix-free computation of ⟨ψ|𝓑(θ)|ψ⟩ using bitwise Ry/Rz gates.

ALGORITHM:
----------
The key insight is that σ⁺ = |1⟩⟨0|, so σ⁺⊗N = |1...1⟩⟨0...0|.

For the rotated Bell operator 𝓑 = ⊗ⱼ(Uⱼσ⁺Uⱼ†) where Uⱼ = Rz(φⱼ)Ry(θⱼ):
  ⟨ψ|𝓑|ψ⟩ = Σᵢⱼ conj(ψᵢ) * ⟨i|𝓑|j⟩ * ψⱼ

We can compute this efficiently by:
1. Transform |ψ⟩ → |ψ'⟩ = (⊗Uⱼ†)|ψ⟩ using matrix-free rotations
2. Compute ⟨ψ'|σ⁺⊗N|ψ'⟩ = conj(ψ'[|1...1⟩]) * ψ'[|0...0⟩]

COMPLEXITY: O(N × 2^N) instead of O(4^N) - exponentially faster!
"""
function bell_expectation_fast(ψ::Vector{ComplexF64}, angles::Vector{Float64})
    d = length(ψ)
    N = Int(log2(d))
    @assert length(angles) == 2N "Need 2N angles (θ,φ per qubit)"

    # Create working copy of state
    ψ_work = copy(ψ)

    # Apply U† = ⊗ⱼ(Ry(-θⱼ)Rz(-φⱼ)) to the state
    # Note: U = Rz(φ)Ry(θ), so U† = Ry(-θ)Rz(-φ)
    @inbounds for k in 1:N
        θ = angles[2k - 1]
        φ = angles[2k]
        CPUQuantumChannelGates.apply_rz_psi!(ψ_work, k, -φ, N)
        CPUQuantumChannelGates.apply_ry_psi!(ψ_work, k, -θ, N)
    end

    # Now compute ⟨ψ'|σ⁺⊗N|ψ'⟩
    # σ⁺⊗N = |1...1⟩⟨0...0| = |d-1⟩⟨0| in index notation
    # ⟨ψ'|σ⁺⊗N|ψ'⟩ = conj(ψ'[d]) * ψ'[1]  (1-indexed: |0...0⟩ = index 1, |1...1⟩ = index d)
    return conj(ψ_work[d]) * ψ_work[1]
end

"""
    bell_expectation_fast(ρ::Matrix{ComplexF64}, angles::Vector{Float64}) -> ComplexF64

OPTIMIZED Tr[ρ·𝓑(θ)] for density matrix using matrix-free rotations.

ALGORITHM:
----------
Transform ρ: ρ' = (⊗Uⱼ†) ρ (⊗Uⱼ)
Then: Tr[ρ'·σ⁺⊗N] = ρ'[|1...1⟩, |0...0⟩] = ρ'[d, 1]

COMPLEXITY: O(N × 4^N) for the rotation (applied to rows and cols)
            Still better than O(4^N) per-element reconstruction.
"""
function bell_expectation_fast(ρ::Matrix{ComplexF64}, angles::Vector{Float64})
    d = size(ρ, 1)
    N = Int(log2(d))
    @assert length(angles) == 2N "Need 2N angles (θ,φ per qubit)"

    # Create working copy
    ρ_work = copy(ρ)

    # Apply (⊗Uⱼ†) ρ (⊗Uⱼ) = transform with U† on both sides
    # For DM: apply_ry_rho! and apply_rz_rho! already do ρ' = U ρ U†
    # So we need to apply with negative angles
    @inbounds for k in 1:N
        θ = angles[2k - 1]
        φ = angles[2k]
        CPUQuantumChannelGates.apply_rz_rho!(ρ_work, k, -φ, N)
        CPUQuantumChannelGates.apply_ry_rho!(ρ_work, k, -θ, N)
    end

    # Tr[ρ'·σ⁺⊗N] = ρ'[|0...0⟩, |1...1⟩] = ρ'[1, d] (matrix element)
    # Note: σ⁺ = |1⟩⟨0|, so Tr[ρ σ⁺] = ρ[0,1] = ρ[1,2] in 1-indexed
    # For N qubits: σ⁺⊗N = |1...1⟩⟨0...0|, so Tr[ρ σ⁺⊗N] = ρ[|0...0⟩, |1...1⟩]
    return ρ_work[1, d]
end

"""
    bell_expectation_fast(trajectories::Vector{Vector{ComplexF64}}, angles) -> ComplexF64

Compute (1/M) Σᵢ ⟨ψᵢ|𝓑(θ)|ψᵢ⟩ for MCWF ensemble.
Note: Threading already happens in trajectory generation.
"""
function bell_expectation_fast(trajectories::Vector{Vector{ComplexF64}},
                                angles::Vector{Float64})
    M = length(trajectories)
    @assert M > 0 "Need at least one trajectory"

    result = ComplexF64(0.0)
    for ψ in trajectories
        result += bell_expectation_fast(ψ, angles)
    end

    return result / M
end

"""
    bell_correlator_fast(state, angles) -> Float64

Compute ℰ = |⟨𝓑(θ)⟩|² using optimized matrix-free expectation.
"""
function bell_correlator_fast(state, angles::Vector{Float64})
    return abs2(bell_expectation_fast(state, angles))
end

export bell_correlator_fast

# =============================================================================
# Q CORRELATOR FUNCTIONS
# =============================================================================

"""
Compute Q_bell = log₂(ℰ) + N.
"""
function compute_Q_bell(ℰ::Float64, N::Int)
    return ℰ > 0 ? log2(ℰ) + N : -Inf
end

"""
Compute Q_ent = log₄(ℰ) + N = (1/2)log₂(ℰ) + N.
"""
function compute_Q_ent(ℰ::Float64, N::Int)
    return ℰ > 0 ? 0.5 * log2(ℰ) + N : -Inf
end

# =============================================================================
# BOOTSTRAP ERROR ESTIMATION FOR MCWF
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  ERROR BARS FOR MCWF BELL CORRELATOR: DETAILED EXPLANATION                  │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# PROBLEM STATEMENT:
# ──────────────────
# In MCWF, we have M quantum trajectories {ψ₁, ψ₂, ..., ψₘ} that represent
# a mixed state ρ = (1/M) Σᵢ |ψᵢ⟩⟨ψᵢ| without explicitly constructing ρ.
#
# For the Bell correlator, we compute:
#
#   zᵢ = ⟨ψᵢ|𝓑(θ)|ψᵢ⟩        (complex number for each trajectory)
#   ⟨𝓑⟩ = (1/M) Σᵢ zᵢ         (average of complex numbers)
#   ℰ = |⟨𝓑⟩|²                (modulus squared → real, non-negative)
#
# The key challenge is: How do we estimate error bars for ℰ?
#
# WHY STANDARD ERROR PROPAGATION FAILS:
# ─────────────────────────────────────
# The function f(z) = |z|² is NONLINEAR in the complex argument z.
# We cannot simply compute std(zᵢ) and propagate, because:
#   - zᵢ is complex (has real and imaginary parts)
#   - |a + b|² ≠ |a|² + |b|² (cross terms matter!)
#   - The variance of |z̄|² depends on correlations between Re(zᵢ) and Im(zᵢ)
#
# SOLUTION: BOOTSTRAP RESAMPLING
# ──────────────────────────────
# Bootstrap is a nonparametric method that works for ANY function of averages:
#
#   For b = 1 to B (e.g., B = 100):
#     1. Resample: Draw M indices {i₁, i₂, ..., iₘ} with replacement from {1,...,M}
#     2. Compute resampled average: ⟨𝓑⟩_b = (1/M) Σⱼ zᵢⱼ
#     3. Compute resampled ℰ_b = |⟨𝓑⟩_b|²
#
#   Result: {ℰ₁, ℰ₂, ..., ℰ_B} → distribution of ℰ
#   Error bar: σ_ℰ = std({ℰ₁, ..., ℰ_B})
#
# This gives us: ℰ = ℰ_mean ± σ_ℰ
#
# PROPAGATING ERROR TO Q_bell AND Q_ent:
# ──────────────────────────────────────
# Given ℰ = ℰ_mean ± σ_ℰ, we need to compute:
#
#   Q_bell = log₂(ℰ) + N
#   Q_ent  = log₄(ℰ) + N = (1/2)log₂(ℰ) + N
#
# Using first-order error propagation (Taylor expansion):
#
#   σ_f = |df/dx| × σ_x
#
# For Q_bell = log₂(ℰ) + N:
#   dQ_bell/dℰ = 1/(ℰ × ln(2))
#
#   ┌──────────────────────────────────────┐
#   │  σ_Q_bell = σ_ℰ / (ℰ × ln(2))        │
#   └──────────────────────────────────────┘
#
# For Q_ent = (1/2)log₂(ℰ) + N:
#   dQ_ent/dℰ = 1/(2 × ℰ × ln(2))
#
#   ┌──────────────────────────────────────┐
#   │  σ_Q_ent = σ_ℰ / (2 × ℰ × ln(2))     │
#   └──────────────────────────────────────┘
#
# FINAL RESULT:
# ─────────────
#   Q_bell = log₂(ℰ_mean) + N  ±  σ_ℰ/(ℰ_mean × ln2)
#   Q_ent  = log₄(ℰ_mean) + N  ±  σ_ℰ/(2 × ℰ_mean × ln2)
#
# NOTE: Error propagation is valid when σ_ℰ << ℰ_mean.
# For very noisy data (large σ_ℰ/ℰ_mean), consider using the full
# bootstrap distribution of Q values instead.
#
# =============================================================================

"""
    bootstrap_bell_error(trajectories, θ_opt; n_bootstrap=100)
        -> (ℰ_mean, ℰ_std, Q_bell_std, Q_ent_std)

Compute error bars for MCWF Bell correlator using bootstrap resampling.

# The Challenge
For MCWF, we compute:
- `zᵢ = ⟨ψᵢ|𝓑(θ)|ψᵢ⟩` for each trajectory (complex number)
- `⟨𝓑⟩ = (1/M) Σᵢ zᵢ` (average of complex numbers)
- `ℰ = |⟨𝓑⟩|²` (nonlinear function!)

Since |·|² is nonlinear, we cannot use standard error propagation on zᵢ.

# The Solution: Bootstrap
1. Resample M trajectories with replacement → new ensemble
2. Compute ℰ_boot for resampled ensemble
3. Repeat B times → get distribution of ℰ values
4. std(ℰ values) = σ_ℰ is the error bar for ℰ

# Propagation to Q values
Given ℰ = ℰ_mean ± σ_ℰ:
- `Q_bell = log₂(ℰ) + N` → `σ_Q_bell = σ_ℰ / (ℰ × ln2)`
- `Q_ent = log₄(ℰ) + N`  → `σ_Q_ent = σ_ℰ / (2ℰ × ln2)`

# Arguments
- `trajectories::Vector{Vector{ComplexF64}}`: MCWF trajectory ensemble (M state vectors)
- `θ_opt::Vector{Float64}`: Optimal angles from optimization (2N values)
- `n_bootstrap::Int=100`: Number of bootstrap samples (B)

# Returns
Named tuple with:
- `ℰ_mean::Float64`: Mean ℰ from bootstrap distribution
- `ℰ_std::Float64`: Standard deviation of ℰ (σ_ℰ)
- `Q_bell_std::Float64`: Propagated error bar for Q_bell
- `Q_ent_std::Float64`: Propagated error bar for Q_ent

# Example
```julia
# After optimization:
Q_bell, Q_ent, θ_opt = get_bell_correlator(trajectories; max_iter=300)

# Get error bars:
ℰ_mean, σ_ℰ, σ_Q_bell, σ_Q_ent = bootstrap_bell_error(trajectories, θ_opt)

# Report with error bars:
println("Q_bell = \$(round(Q_bell, digits=2)) ± \$(round(σ_Q_bell, digits=2))")
println("Q_ent  = \$(round(Q_ent, digits=2)) ± \$(round(σ_Q_ent, digits=2))")
```
"""
function bootstrap_bell_error(trajectories::Vector{Vector{ComplexF64}},
                               θ_opt::Vector{Float64};
                               n_bootstrap::Int = 100)
    M = length(trajectories)
    N = Int(log2(length(trajectories[1])))

    # Storage for bootstrap ℰ samples
    ℰ_samples = zeros(n_bootstrap)

    # Bootstrap loop
    for b in 1:n_bootstrap
        # Step 1: Resample M trajectories WITH REPLACEMENT
        # This is the key to bootstrap - some trajectories appear multiple times,
        # others may not appear at all
        indices = rand(1:M, M)
        resampled = [trajectories[i] for i in indices]

        # Step 2: Compute ℰ for resampled ensemble
        # This uses the SAME optimal angles θ_opt from the original optimization
        # (We don't re-optimize for each bootstrap sample)
        ℰ_samples[b] = bell_correlator(resampled, θ_opt)
    end

    # Step 3: Compute statistics of ℰ distribution
    ℰ_mean = mean(ℰ_samples)
    ℰ_std = std(ℰ_samples)

    # Step 4: Propagate error to Q values using derivatives
    #
    # Q_bell = log₂(ℰ) + N = log(ℰ)/log(2) + N
    #   → dQ_bell/dℰ = 1/(ℰ × ln(2))
    #   → σ_Q_bell = |dQ_bell/dℰ| × σ_ℰ = σ_ℰ / (ℰ × ln(2))
    #
    # Q_ent = (1/2)log₂(ℰ) + N
    #   → dQ_ent/dℰ = 1/(2 × ℰ × ln(2))
    #   → σ_Q_ent = σ_ℰ / (2 × ℰ × ln(2))
    #
    if ℰ_mean > 0
        Q_bell_std = ℰ_std / (ℰ_mean * log(2))
        Q_ent_std = ℰ_std / (2 * ℰ_mean * log(2))
    else
        # If ℰ_mean ≤ 0, log is undefined → infinite error
        Q_bell_std = Inf
        Q_ent_std = Inf
    end

    return ℰ_mean, ℰ_std, Q_bell_std, Q_ent_std
end

# Helper functions (avoid Statistics.jl dependency)
mean(x) = sum(x) / length(x)
std(x) = sqrt(sum((xi - mean(x))^2 for xi in x) / (length(x) - 1))

# =============================================================================
# SPSA + ADAM OPTIMIZER
# =============================================================================

"""SPSA+Adam optimizer for Bell correlator maximization."""
Base.@kwdef mutable struct SPSAAdamOptimizer
    α::Float64 = 0.15       # Learning rate
    c::Float64 = 0.02       # Perturbation magnitude
    β1::Float64 = 0.9       # Adam first moment decay
    β2::Float64 = 0.999     # Adam second moment decay
    ε::Float64 = 1e-8       # Numerical stability
    max_iter::Int = 500     # Maximum iterations
    tol::Float64 = 1e-6     # Convergence tolerance
    verbose::Bool = false   # Print progress
end

"""Maximize f(θ) using SPSA+Adam."""
function optimize_spsa_adam(f, θ_init::Vector{Float64};
                            opt::SPSAAdamOptimizer = SPSAAdamOptimizer())
    n = length(θ_init)
    θ = copy(θ_init)

    m = zeros(n)
    v = zeros(n)

    f_best = f(θ)
    θ_best = copy(θ)

    for t in 1:opt.max_iter
        Δ = 2 .* (rand(n) .> 0.5) .- 1

        f_plus = f(θ .+ opt.c .* Δ)
        f_minus = f(θ .- opt.c .* Δ)

        g = -(f_plus - f_minus) ./ (2 * opt.c) .* Δ

        m = opt.β1 .* m .+ (1 - opt.β1) .* g
        v = opt.β2 .* v .+ (1 - opt.β2) .* (g .^ 2)

        m_hat = m ./ (1 - opt.β1^t)
        v_hat = v ./ (1 - opt.β2^t)

        θ .-= opt.α .* m_hat ./ (sqrt.(v_hat) .+ opt.ε)

        f_current = f(θ)
        if f_current > f_best
            f_best = f_current
            θ_best .= θ
        end

        if opt.verbose && t % 100 == 0
            @printf("SPSA iter %4d: f = %.6f (best = %.6f)\n", t, f_current, f_best)
        end

        if abs(f_plus - f_minus) < opt.tol
            return θ_best, f_best, t, true
        end
    end

    return θ_best, f_best, opt.max_iter, false
end

# =============================================================================
# MAIN API: get_bell_correlator
# =============================================================================

"""
    get_bell_correlator(state; kwargs...) -> (Q_bell, Q_ent, θ_opt)

Optimize Bell correlator and return Q values and optimal angles.

# Arguments
- `state`: Pure state vector, density matrix, or vector of MCWF trajectories

# Keyword Arguments
- `method::Symbol=:spsa_adam`: Optimizer (:spsa_adam, :lbfgs, :enzyme)
- `max_iter::Int=500`: Maximum optimizer iterations
- `θ_init::Union{Nothing,Vector{Float64}}=nothing`: Initial angles (random if not given)
- `verbose::Bool=false`: Print optimizer progress
- `return_full::Bool=false`: Return full BellCorrelatorResult struct

# Returns
- Tuple `(Q_bell, Q_ent, θ_opt)` where θ_opt = [θ₁,φ₁,...,θₙ,φₙ]
- If `return_full=true`, returns `BellCorrelatorResult` struct

# Example
```julia
ψ = make_ghz_state(4)
Q_bell, Q_ent, θ_opt = get_bell_correlator(ψ)
println("Q_bell = \$Q_bell, Q_ent = \$Q_ent")
```
"""
function get_bell_correlator(state;
                              method::Symbol = :spsa_adam,
                              max_iter::Int = 500,
                              θ_init::Union{Nothing, Vector{Float64}} = nothing,
                              verbose::Bool = false,
                              return_full::Bool = false)

    N = _get_N_from_state(state)

    if θ_init === nothing
        θ_init = 2π * rand(2N)
    end
    @assert length(θ_init) == 2N "Need 2N angles"

    # Use optimized matrix-free version for O(N × 2^N) performance
    obj = θ -> bell_correlator_fast(state, θ)

    if method == :best
        # Try multiple optimizers and pick the one with highest ℰ
        best_θ, best_ℰ, best_iters, best_conv = _optimize_lbfgs(obj, θ_init; max_iter=max_iter, verbose=false)

        θ2, ℰ2, _, _ = _optimize_bfgs(obj, θ_init; max_iter=max_iter, verbose=false)
        if ℰ2 > best_ℰ
            best_θ, best_ℰ = θ2, ℰ2
        end

        θ3, ℰ3, _, _ = _optimize_nelder_mead(obj, θ_init; max_iter=max_iter, verbose=false)
        if ℰ3 > best_ℰ
            best_θ, best_ℰ = θ3, ℰ3
        end

        θ_opt, ℰ_max, iters, converged = best_θ, best_ℰ, best_iters, true
    elseif method == :spsa_adam
        opt = SPSAAdamOptimizer(max_iter=max_iter, verbose=verbose)
        θ_opt, ℰ_max, iters, converged = optimize_spsa_adam(obj, θ_init; opt=opt)
    elseif method == :lbfgs
        θ_opt, ℰ_max, iters, converged = _optimize_lbfgs(obj, θ_init;
                                                          max_iter=max_iter, verbose=verbose)
    elseif method == :bfgs
        θ_opt, ℰ_max, iters, converged = _optimize_bfgs(obj, θ_init;
                                                         max_iter=max_iter, verbose=verbose)
    elseif method == :nelder_mead
        θ_opt, ℰ_max, iters, converged = _optimize_nelder_mead(obj, θ_init;
                                                                max_iter=max_iter, verbose=verbose)
    elseif method == :blackbox
        θ_opt, ℰ_max, iters, converged = _optimize_blackbox(obj, θ_init;
                                                             max_iter=max_iter, verbose=verbose)
    elseif method == :nlopt_bobyqa
        θ_opt, ℰ_max, iters, converged = _optimize_nlopt(obj, θ_init;
                                                          max_iter=max_iter, verbose=verbose)
    elseif method == :autograd_adam
        if !ENZYME_AVAILABLE
            error("Enzyme not available. Install with: ] add Enzyme Optimisers")
        end
        θ_opt, ℰ_max, iters, converged = _optimize_enzyme_adam(obj, θ_init;
                                                                max_iter=max_iter, verbose=verbose)
    else
        error("Unknown method: $method. Available: :best, :lbfgs, :bfgs, :nelder_mead, :blackbox, :nlopt_bobyqa, :autograd_adam, :spsa_adam")
    end

    Q_bell = compute_Q_bell(ℰ_max, N)
    Q_ent = compute_Q_ent(ℰ_max, N)

    if return_full
        return BellCorrelatorResult(Q_bell, Q_ent, θ_opt, ℰ_max, N, iters, converged)
    else
        return (Q_bell, Q_ent, θ_opt)
    end
end

# Helper to get N
function _get_N_from_state(ψ::Vector{ComplexF64})
    return Int(log2(length(ψ)))
end

function _get_N_from_state(ρ::Matrix{ComplexF64})
    return Int(log2(size(ρ, 1)))
end

function _get_N_from_state(trajectories::Vector{Vector{ComplexF64}})
    return Int(log2(length(trajectories[1])))
end

# L-BFGS optimizer (requires Optim.jl loaded in Main)
function _optimize_lbfgs(f, θ_init; max_iter=500, verbose=false)
    try
        result = Main.Optim.optimize(θ -> -f(θ), θ_init, Main.Optim.LBFGS(),
                                     Main.Optim.Options(iterations=max_iter, show_trace=verbose))

        return Main.Optim.minimizer(result), -Main.Optim.minimum(result),
               Main.Optim.iterations(result), Main.Optim.converged(result)
    catch e
        @warn "Optim.jl not available or failed: $e. Falling back to SPSA+Adam"
        return optimize_spsa_adam(f, θ_init)
    end
end

# BFGS (full memory quasi-Newton)
function _optimize_bfgs(f, θ_init; max_iter=500, verbose=false)
    try
        result = Main.Optim.optimize(θ -> -f(θ), θ_init, Main.Optim.BFGS(),
                                     Main.Optim.Options(iterations=max_iter, show_trace=verbose))

        return Main.Optim.minimizer(result), -Main.Optim.minimum(result),
               Main.Optim.iterations(result), Main.Optim.converged(result)
    catch e
        @warn "BFGS failed: $e. Falling back to L-BFGS"
        return _optimize_lbfgs(f, θ_init; max_iter=max_iter, verbose=verbose)
    end
end

# Nelder-Mead (simplex, derivative-free)
function _optimize_nelder_mead(f, θ_init; max_iter=500, verbose=false)
    try
        result = Main.Optim.optimize(θ -> -f(θ), θ_init, Main.Optim.NelderMead(),
                                     Main.Optim.Options(iterations=max_iter, show_trace=verbose))

        return Main.Optim.minimizer(result), -Main.Optim.minimum(result),
               Main.Optim.iterations(result), Main.Optim.converged(result)
    catch e
        @warn "NelderMead failed: $e. Falling back to L-BFGS"
        return _optimize_lbfgs(f, θ_init; max_iter=max_iter, verbose=verbose)
    end
end

# BlackBoxOptim.jl - Differential Evolution (global optimizer)
function _optimize_blackbox(f, θ_init; max_iter=500, verbose=false)
    try
        n = length(θ_init)
        search_range = [(0.0, 2π) for _ in 1:n]

        result = Main.BlackBoxOptim.bboptimize(θ -> -f(θ);
                                SearchRange=search_range,
                                NumDimensions=n,
                                MaxFuncEvals=max_iter * 10,
                                TraceMode=verbose ? :verbose : :silent,
                                Method=:adaptive_de_rand_1_bin_radiuslimited)

        θ_opt = Main.BlackBoxOptim.best_candidate(result)
        ℰ_max = -Main.BlackBoxOptim.best_fitness(result)

        return θ_opt, ℰ_max, max_iter, true
    catch e
        @warn "BlackBoxOptim.jl not available: $e. Falling back to L-BFGS"
        return _optimize_lbfgs(f, θ_init; max_iter=max_iter, verbose=verbose)
    end
end

# NLopt.jl - BOBYQA (Bound Optimization BY Quadratic Approximation)
function _optimize_nlopt(f, θ_init; max_iter=500, verbose=false)
    try
        n = length(θ_init)
        opt = Main.NLopt.Opt(:LN_BOBYQA, n)

        Main.NLopt.lower_bounds!(opt, zeros(n))
        Main.NLopt.upper_bounds!(opt, fill(2π, n))
        Main.NLopt.max_objective!(opt, (θ, grad) -> f(θ))
        Main.NLopt.maxeval!(opt, max_iter)
        Main.NLopt.xtol_rel!(opt, 1e-6)

        ℰ_max, θ_opt, ret = Main.NLopt.optimize(opt, θ_init)

        return θ_opt, ℰ_max, max_iter, ret in [:SUCCESS, :FTOL_REACHED, :XTOL_REACHED]
    catch e
        @warn "NLopt.jl not available: $e. Falling back to L-BFGS"
        return _optimize_lbfgs(f, θ_init; max_iter=max_iter, verbose=verbose)
    end
end

"""
    _optimize_enzyme_adam(f, θ_init; max_iter, verbose)

Optimize using Enzyme autodiff with Adam optimizer.
Uses reverse-mode AD for exact gradients, then Adam for updates.
"""
function _optimize_enzyme_adam(f, θ_init; max_iter=500, verbose=false, lr=0.01, tol=1e-6)
    θ = copy(θ_init)
    n_params = length(θ)

    # Adam optimizer state
    adam = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(adam, θ)

    best_val = f(θ)
    best_θ = copy(θ)
    converged = false

    for iter in 1:max_iter
        # Compute gradient using Enzyme (we want to maximize f, so negate for gradient descent)
        dθ = zeros(Float64, n_params)
        θ_copy = copy(θ)

        # Define negated objective for minimization
        neg_f = x -> -f(x)

        try
            Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(neg_f), Enzyme.Active,
                           Enzyme.Duplicated(θ_copy, dθ))
        catch e
            @warn "Enzyme autodiff failed at iter $iter: $e"
            break
        end

        # Update with Adam (negate gradient since we computed gradient of -f)
        opt_state, θ = Optimisers.update!(opt_state, θ, dθ)

        # Evaluate new value
        val = f(θ)

        if val > best_val
            best_val = val
            best_θ = copy(θ)
        end

        # Check convergence
        grad_norm = sqrt(sum(dθ.^2))
        if grad_norm < tol
            converged = true
            break
        end

        if verbose && iter % 50 == 0
            @printf("  Enzyme+Adam iter %4d: ℰ = %.6f, |∇| = %.2e\n", iter, val, grad_norm)
        end
    end

    return best_θ, best_val, max_iter, converged
end

# =============================================================================
# FILENAME GENERATION HELPER
# =============================================================================

"""
    make_bell_filename(; state_type, N, representation, kwargs...) -> String

Generate informative filename encoding all parameters.

# Example
```julia
make_bell_filename(
    state_type = "ghz_z",
    N = 4,
    representation = :dm,        # or :mcwf, :pure
    n_trajectories = 100,        # only for MCWF
    is_mixed = true,
    noise_model = "dephasing",
    p_noise = 0.1
)
# Returns: "ghz_z_N04_dm_mixed_dephasing_p0.10"
```
"""
function make_bell_filename(;
                             state_type::String,
                             N::Int,
                             representation::Symbol,
                             n_trajectories::Int = 0,
                             is_mixed::Bool = false,
                             noise_model::String = "none",
                             p_noise::Float64 = 0.0)

    parts = String[]

    push!(parts, state_type)
    push!(parts, @sprintf("N%02d", N))

    if representation == :mcwf
        push!(parts, @sprintf("mcwf_M%d", n_trajectories))
    elseif representation == :dm
        push!(parts, "dm")
    else
        push!(parts, "pure")
    end

    push!(parts, is_mixed ? "mixed" : "pure")

    if noise_model != "none" && p_noise > 0
        push!(parts, noise_model)
        push!(parts, @sprintf("p%.2f", p_noise))
    end

    return join(parts, "_")
end

export make_bell_filename

end # module CPUQuantumStateManyBodyBellCorrelator

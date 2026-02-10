# Date: 2026
#
#=
================================================================================
    cpuQuantumStateMeasurements.jl - Projective Measurements & Reset (CPU)
================================================================================

OVERVIEW
--------
Matrix-free projective measurements in computational and Pauli bases.
Implements the quantum measurement postulate: measuring observable O collapses
the state to an eigenstate |λₖ⟩ with probability |⟨λₖ|ψ⟩|².

This module provides:
  1. Projective measurements (with collapse)
  2. Sampling (without collapse, for statistics)
  3. State reset (force qubits to target states)

MAIN FUNCTIONS
--------------
PROJECTIVE MEASUREMENT (measure + collapse):
  projective_measurement!(ψ, qubits, basis, N)      → (outcomes, ψ)
  projective_measurement!(ψ, qubits, bases, N)      → per-qubit basis
  projective_measurement!(ψ, basis, N)              → all qubits
  projective_measurement!(ψ, O::Matrix, N)          → general observable
  projective_measurement_all!(ψ, N)                 → FAST Z-basis all qubits

SAMPLING (no collapse):
  sample_state(ψ, qubits, basis, N)                 → outcomes only

RESET (force to target states):
  reset_state!(ψ, qubits, targets, N)               → force qubits to |0⟩,|1⟩,|+⟩,|−⟩
  reset_state!(ψ, target, N)                        → reset all qubits

MEASUREMENT BASES
-----------------
  :z           — Computational basis |0⟩, |1⟩           [matrix-free]
  :x           — Hadamard basis |+⟩ = (|0⟩+|1⟩)/√2     [matrix-free]
  :y           — Y-eigenbasis |+i⟩ = (|0⟩+i|1⟩)/√2    [matrix-free]
  [:x, :z, :y] — Different basis per qubit              [matrix-free]
  O::Matrix    — General observable (diagonalize)       [O(dim³)]

RESET TARGETS
-------------
  :zero        — |0⟩
  :one         — |1⟩
  :plus        — |+⟩ = (|0⟩+|1⟩)/√2
  :minus       — |−⟩ = (|0⟩-|1⟩)/√2

PHYSICS BACKGROUND
------------------
Projective measurement of observable O = Σₖ λₖ|λₖ⟩⟨λₖ|:
  1. State |ψ⟩ collapses to eigenstate |λₖ⟩ with probability P(k) = |⟨λₖ|ψ⟩|²
  2. Outcome is eigenvalue λₖ
  3. Entanglement with other qubits is BROKEN

For Pauli basis measurements (X, Y), we use basis rotations:
  X-basis: H|ψ⟩ → measure Z → H|ψ'⟩
  Y-basis: HS†|ψ⟩ → measure Z → SH|ψ'⟩

PERFORMANCE
-----------
  projective_measurement! (per-qubit):     O(N × 2^N)
  projective_measurement_all! (Z-basis):   O(2^N)  ← N× faster!
  sample_state:                            O(N × 2^N) (copies state)
  reset_state!:                            O(2^N)

BIT CONVENTION
--------------
LITTLE-ENDIAN: Qubit k has bit position (k-1).
  State index i = Σₖ bₖ × 2^(k-1) where bₖ ∈ {0,1} is qubit k's state.
  Qubit 1 = LSB, qubit N = MSB.

USAGE EXAMPLES
--------------

# ─────────────────────────────────────────────────────────────────────────────
# projective_measurement!(ψ, qubits, basis, N)
# ─────────────────────────────────────────────────────────────────────────────
# Measure specific qubits in a given basis, state collapses.

```julia
# Setup: 2-qubit Bell state
ψ = ComplexF64[1/√2, 0, 0, 1/√2]  # |00⟩ + |11⟩
N = 2

# Measure qubit 1 in Z-basis
outcomes, ψ = projective_measurement!(ψ, [1], :z, N)
# outcomes = [0] or [1]
# If outcome=0: ψ = [1, 0, 0, 0] = |00⟩
# If outcome=1: ψ = [0, 0, 0, 1] = |11⟩

# Measure qubit 2 in X-basis
outcomes, ψ = projective_measurement!(ψ, [2], :x, N)
# outcomes = [0] means |+⟩, [1] means |−⟩
```

# ─────────────────────────────────────────────────────────────────────────────
# projective_measurement!(ψ, qubits, bases::Vector, N)
# ─────────────────────────────────────────────────────────────────────────────
# Different basis for each qubit.

```julia
# Measure qubit 1 in X-basis, qubit 2 in Z-basis
outcomes, ψ = projective_measurement!(ψ, [1, 2], [:x, :z], N)
# outcomes[1] ∈ {0,1} for X-basis (|+⟩ or |−⟩)
# outcomes[2] ∈ {0,1} for Z-basis (|0⟩ or |1⟩)
```

# ─────────────────────────────────────────────────────────────────────────────
# projective_measurement!(ψ, O::Matrix, N)
# ─────────────────────────────────────────────────────────────────────────────
# Measure general observable O, returns eigenvalue.

```julia
# Measure Pauli X on 1 qubit
σx = ComplexF64[0 1; 1 0]
ψ = ComplexF64[1/√2, 1/√2]  # |+⟩ state
eigenvalue, ψ = projective_measurement!(ψ, σx, 1)
# eigenvalue = +1 or -1 (eigenvalues of σx)
# ψ collapses to |+⟩ or |−⟩
```

# ─────────────────────────────────────────────────────────────────────────────
# projective_measurement_all!(ψ, N)  [FAST]
# ─────────────────────────────────────────────────────────────────────────────
# Measure ALL qubits in Z-basis simultaneously. O(2^N), N× faster!

```julia
ψ = rand(ComplexF64, 2^10); ψ ./= norm(ψ)  # Random 10-qubit state
bitstring, ψ = projective_measurement_all!(ψ, 10)
# bitstring = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0] (example)
# ψ = computational basis state |0110100110⟩
```

# ─────────────────────────────────────────────────────────────────────────────
# sample_state(ψ, qubits, basis, N)
# ─────────────────────────────────────────────────────────────────────────────
# Sample outcomes WITHOUT collapsing. Good for statistics.

```julia
ψ = ComplexF64[1/√2, 0, 0, 1/√2]  # Bell state
counts = Dict(0 => 0, 1 => 0)
for _ in 1:10000
    outcomes = sample_state(ψ, [1], :z, 2)
    counts[outcomes[1]] += 1
end
# counts ≈ Dict(0 => 5000, 1 => 5000)  (50/50 for Bell state)
# Original ψ is UNCHANGED!
```

# ─────────────────────────────────────────────────────────────────────────────
# reset_state!(ψ, qubits, targets, N)
# ─────────────────────────────────────────────────────────────────────────────
# Force qubits to specific states. Breaks entanglement.

```julia
# Reset qubit 1 to |0⟩, qubit 3 to |+⟩
reset_state!(ψ, [1, 3], [:zero, :plus], N)

# Reset specific qubits to |1⟩
reset_state!(ψ, [2, 4], :one, N)

# Reset ALL qubits to |0⟩
reset_state!(ψ, :zero, N)
# ψ = |00...0⟩
```

DEPENDENCIES
------------
Imports from cpuQuantumChannelGates.jl:
  - apply_hadamard_psi!  (for X-basis rotation)
  - apply_s_psi!         (for Y-basis rotation)
  - apply_sdagger_psi!   (for Y-basis rotation)

================================================================================
=#

module CPUQuantumStateMeasurements

using LinearAlgebra

# Import gates from the gates module (avoid duplication)
using ..CPUQuantumChannelGates: apply_hadamard_psi!, apply_s_psi!, apply_sdagger_psi!

# Main unified interface
export projective_measurement!, projective_measurement_all!, sample_state
export reset_state!, reset_qubit!

# Helpers
export sample_from_probabilities

# ==============================================================================
# HELPER: SAMPLING
# ==============================================================================

"""
    sample_from_probabilities(probs::Vector{Float64}) -> Int

Sample an index from probability distribution using inverse CDF.
Returns index in 1:length(probs).
"""
function sample_from_probabilities(probs::Vector{Float64})
    r = rand()
    cumsum_p = 0.0
    @inbounds for i in eachindex(probs)
        cumsum_p += probs[i]
        if r <= cumsum_p
            return i
        end
    end
    return length(probs)
end

# ==============================================================================
# CORE Z-BASIS MEASUREMENT (matrix-free, bitwise)
# ==============================================================================

"""
    _measure_z_single!(ψ, k, N) -> outcome

Measure qubit k in Z-basis with collapse. Returns 0 or 1.
"""
function _measure_z_single!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)

    # Compute P(0)
    prob_0 = 0.0
    @inbounds for i in 0:(dim-1)
        if (i & mask) == 0
            prob_0 += abs2(ψ[i+1])
        end
    end

    # Sample outcome
    outcome = rand() < prob_0 ? 0 : 1

    # Collapse and renormalize
    norm_sq = 0.0
    @inbounds for i in 0:(dim-1)
        bit_k = (i >> (k-1)) & 1
        if bit_k != outcome
            ψ[i+1] = zero(ComplexF64)
        else
            norm_sq += abs2(ψ[i+1])
        end
    end

    norm_factor = 1.0 / sqrt(norm_sq)
    @inbounds for i in 1:dim
        ψ[i] *= norm_factor
    end

    return outcome
end

"""Z-basis measurement for density matrix."""
function _measure_z_single!(ρ::Matrix{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)

    # Compute P(0) from diagonal
    prob_0 = 0.0
    @inbounds for i in 0:(dim-1)
        if (i & mask) == 0
            prob_0 += real(ρ[i+1, i+1])
        end
    end

    outcome = rand() < prob_0 ? 0 : 1
    prob_outcome = outcome == 0 ? prob_0 : (1.0 - prob_0)

    # Project: zero out inconsistent rows/columns
    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 != outcome
            for j in 0:(dim-1)
                ρ[i+1, j+1] = zero(ComplexF64)
                ρ[j+1, i+1] = zero(ComplexF64)
            end
        end
    end

    # Renormalize
    if prob_outcome > 1e-15
        ρ ./= prob_outcome
    end

    return outcome
end

# ==============================================================================
# UNIFIED INTERFACE: projective_measurement!
# ==============================================================================

"""
    projective_measurement!(ψ::Vector{ComplexF64}, qubits::Vector{Int}, basis, N::Int)
    projective_measurement!(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, basis, N::Int)

Projective measurement of specified qubits in given basis.

# Arguments
- `ψ` or `ρ`: State (modified in-place)
- `qubits`: Vector of qubit indices to measure (1-based)
- `basis`: :z, :x, :y, Vector of symbols, or Matrix (general observable)
- `N`: Total number of qubits

# Returns
- `outcomes`: Vector of 0s and 1s for each measured qubit
- State is modified in-place (collapsed)

# Basis options
- `:z` — Computational basis (all qubits)
- `:x` — Hadamard basis (all qubits)
- `:y` — Y-eigenbasis (all qubits)
- `[:x, :z, :y]` — Different basis per qubit
- `O::Matrix` — General observable (requires diagonalization)
"""
function projective_measurement!(ψ::Vector{ComplexF64}, qubits::Vector{Int}, basis::Symbol, N::Int)
    outcomes = Int[]

    for k in qubits
        if basis == :z
            push!(outcomes, _measure_z_single!(ψ, k, N))

        elseif basis == :x
            # X-basis: H → Z-measure → H
            apply_hadamard_psi!(ψ, k, N)
            push!(outcomes, _measure_z_single!(ψ, k, N))
            apply_hadamard_psi!(ψ, k, N)

        elseif basis == :y
            # Y-basis: S† → H → Z-measure → H → S
            apply_sdagger_psi!(ψ, k, N)
            apply_hadamard_psi!(ψ, k, N)
            push!(outcomes, _measure_z_single!(ψ, k, N))
            apply_hadamard_psi!(ψ, k, N)
            apply_s_psi!(ψ, k, N)
        else
            error("Unknown basis: $basis. Use :z, :x, :y, or provide Vector/Matrix.")
        end
    end

    return outcomes, ψ
end

# Per-qubit basis specification
function projective_measurement!(ψ::Vector{ComplexF64}, qubits::Vector{Int}, bases::Vector{Symbol}, N::Int)
    @assert length(qubits) == length(bases) "Must specify basis for each qubit"
    outcomes = Int[]

    for (k, basis) in zip(qubits, bases)
        if basis == :z
            push!(outcomes, _measure_z_single!(ψ, k, N))
        elseif basis == :x
            apply_hadamard_psi!(ψ, k, N)
            push!(outcomes, _measure_z_single!(ψ, k, N))
            apply_hadamard_psi!(ψ, k, N)
        elseif basis == :y
            apply_sdagger_psi!(ψ, k, N)
            apply_hadamard_psi!(ψ, k, N)
            push!(outcomes, _measure_z_single!(ψ, k, N))
            apply_hadamard_psi!(ψ, k, N)
            apply_s_psi!(ψ, k, N)
        end
    end

    return outcomes, ψ
end

# General observable (requires diagonalization)
function projective_measurement!(ψ::Vector{ComplexF64}, O::Matrix{ComplexF64}, N::Int)
    dim = 1 << N
    @assert size(O) == (dim, dim) "Observable must be $dim × $dim"

    # Diagonalize: O = U D U†
    F = eigen(Hermitian(O))
    eigenvalues = F.values
    U = F.vectors

    # Transform to eigenbasis
    ψ_eigenbasis = U' * ψ

    # Compute probabilities
    probs = [abs2(ψ_eigenbasis[i]) for i in 1:dim]

    # Sample eigenstate
    outcome_idx = sample_from_probabilities(probs)
    eigenvalue = eigenvalues[outcome_idx]

    # Collapse to eigenstate in original basis
    fill!(ψ, zero(ComplexF64))
    @inbounds for i in 1:dim
        ψ[i] = U[i, outcome_idx]
    end

    return eigenvalue, ψ
end

# Density matrix versions
function projective_measurement!(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, basis::Symbol, N::Int)
    outcomes = Int[]

    for k in qubits
        if basis == :z
            push!(outcomes, _measure_z_single!(ρ, k, N))
        elseif basis == :x || basis == :y
            # For ρ, we need to apply U ρ U† for basis change
            # This is more complex - implement as needed
            error("X/Y basis for density matrix not yet implemented. Use :z or pure states.")
        end
    end

    return outcomes, ρ
end

# ==============================================================================
# SAMPLE WITHOUT COLLAPSE
# ==============================================================================

"""
    sample_state(ψ::Vector{ComplexF64}, qubits::Vector{Int}, basis::Symbol, N::Int)

Sample measurement outcomes WITHOUT collapsing the state.
Returns vector of outcomes (0 or 1 for each qubit).
"""
function sample_state(ψ::Vector{ComplexF64}, qubits::Vector{Int}, basis::Symbol, N::Int)
    # Make a copy to avoid modifying original
    ψ_copy = copy(ψ)
    outcomes, _ = projective_measurement!(ψ_copy, qubits, basis, N)
    return outcomes
end

"""Sample from density matrix diagonal (Z-basis only)."""
function sample_state(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, basis::Symbol, N::Int)
    if basis != :z
        error("Only Z-basis sampling implemented for density matrices")
    end

    dim = 1 << N
    probs = [real(ρ[i, i]) for i in 1:dim]
    outcome_idx = sample_from_probabilities(probs)
    outcome_bits = outcome_idx - 1

    return [(outcome_bits >> (k-1)) & 1 for k in qubits]
end

# ==============================================================================
# CONVENIENCE: MEASURE ALL QUBITS
# ==============================================================================

"""Measure all N qubits in given basis (qubit-by-qubit, O(N×2^N))."""
function projective_measurement!(ψ::Vector{ComplexF64}, basis::Symbol, N::Int)
    return projective_measurement!(ψ, collect(1:N), basis, N)
end

function projective_measurement!(ρ::Matrix{ComplexF64}, basis::Symbol, N::Int)
    return projective_measurement!(ρ, collect(1:N), basis, N)
end

# ==============================================================================
# FAST ALL-QUBIT MEASUREMENT (Z-basis only)
# ==============================================================================

"""
    projective_measurement_all!(ψ::Vector{ComplexF64}, N::Int) -> (bitstring, ψ)

Fast projective measurement of ALL qubits in Z-basis simultaneously.

# Performance
O(2^N) - single sample from full distribution, then collapse to basis state.
This is N× faster than qubit-by-qubit measurement!

# Returns
- `bitstring`: Vector of 0s and 1s, length N (bitstring[k] = outcome of qubit k)
- `ψ`: Collapsed to computational basis state

# Example
```julia
ψ = rand(ComplexF64, 2^10); ψ ./= norm(ψ)
bitstring, ψ = projective_measurement_all!(ψ, 10)  # ~20× faster than qubit-by-qubit
```
"""
function projective_measurement_all!(ψ::Vector{ComplexF64}, N::Int)
    dim = 1 << N

    # Compute probabilities
    probs = [abs2(ψ[i]) for i in 1:dim]

    # Sample outcome index (1-based)
    outcome_idx = sample_from_probabilities(probs)
    outcome_bits = outcome_idx - 1  # 0-based for bit operations

    # Extract bitstring
    bitstring = [(outcome_bits >> (k-1)) & 1 for k in 1:N]

    # Collapse to basis state
    fill!(ψ, zero(ComplexF64))
    ψ[outcome_idx] = 1.0

    return bitstring, ψ
end

"""Fast Z-basis measurement for density matrix."""
function projective_measurement_all!(ρ::Matrix{ComplexF64}, N::Int)
    dim = 1 << N

    # Sample from diagonal (populations)
    probs = [real(ρ[i, i]) for i in 1:dim]
    outcome_idx = sample_from_probabilities(probs)
    outcome_bits = outcome_idx - 1

    # Extract bitstring
    bitstring = [(outcome_bits >> (k-1)) & 1 for k in 1:N]

    # Collapse to |outcome⟩⟨outcome|
    fill!(ρ, zero(ComplexF64))
    ρ[outcome_idx, outcome_idx] = 1.0

    return bitstring, ρ
end

# ==============================================================================
# RESET STATE (Force qubits to given state)
# ==============================================================================

export reset_state!

"""
    reset_state!(ψ::Vector{ComplexF64}, qubits::Vector{Int}, target_states::Vector{Symbol}, N::Int)
    reset_state!(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, target_states::Vector{Symbol}, N::Int)

Reset specified qubits to given states. Other qubits retain their conditional state.

# Algorithm (pure state)
1. Measure specified qubits (projective, stochastic)
2. Force measured qubits to target state by replacing amplitudes
3. Renormalize

# Arguments
- `ψ` or `ρ`: State (modified in-place)
- `qubits`: Vector of qubit indices to reset (1-based)
- `target_states`: Vector of target states (:zero, :one, :plus, :minus)
- `N`: Total number of qubits

# Returns
- Modified state with qubits reset to targets

# Example
```julia
# Reset qubits 1 and 3 to |0⟩ and |+⟩
reset_state!(ψ, [1, 3], [:zero, :plus], N)
```
"""
function reset_state!(ψ::Vector{ComplexF64}, qubits::Vector{Int}, target_states::Vector{Symbol}, N::Int)
    @assert length(qubits) == length(target_states) "Must specify target for each qubit"

    for (k, target) in zip(qubits, target_states)
        # First measure in Z-basis to collapse
        _measure_z_single!(ψ, k, N)

        # Then force to target state
        _force_qubit_state!(ψ, k, target, N)
    end

    return ψ
end

# Convenience: all same target
function reset_state!(ψ::Vector{ComplexF64}, qubits::Vector{Int}, target::Symbol, N::Int)
    targets = fill(target, length(qubits))
    return reset_state!(ψ, qubits, targets, N)
end

# Reset all qubits
function reset_state!(ψ::Vector{ComplexF64}, target::Symbol, N::Int)
    return reset_state!(ψ, collect(1:N), target, N)
end

"""
Force qubit k to target state (matrix-free, bitwise).
Assumes qubit has already been measured/collapsed.
"""
function _force_qubit_state!(ψ::Vector{ComplexF64}, k::Int, target::Symbol, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)

    if target == :zero
        # Force to |0⟩: zero out all |1⟩ states for qubit k
        @inbounds for i in 0:(dim-1)
            if (i & mask) != 0  # Bit k is 1
                j = xor(i, mask)  # Partner with bit k = 0
                ψ[j+1] += ψ[i+1]  # Move amplitude to |0⟩
                ψ[i+1] = zero(ComplexF64)
            end
        end
        _normalize!(ψ)

    elseif target == :one
        # Force to |1⟩
        @inbounds for i in 0:(dim-1)
            if (i & mask) == 0  # Bit k is 0
                j = xor(i, mask)  # Partner with bit k = 1
                ψ[j+1] += ψ[i+1]
                ψ[i+1] = zero(ComplexF64)
            end
        end
        _normalize!(ψ)

    elseif target == :plus
        # Force to |+⟩ = (|0⟩ + |1⟩)/√2
        # First force to |0⟩, then apply Hadamard
        _force_qubit_state!(ψ, k, :zero, N)
        apply_hadamard_psi!(ψ, k, N)

    elseif target == :minus
        # Force to |−⟩ = (|0⟩ - |1⟩)/√2
        _force_qubit_state!(ψ, k, :one, N)
        apply_hadamard_psi!(ψ, k, N)

    else
        error("Unknown target state: $target. Use :zero, :one, :plus, :minus.")
    end

    return ψ
end

"""Helper: normalize state vector."""
function _normalize!(ψ::Vector{ComplexF64})
    norm_sq = sum(abs2, ψ)
    if norm_sq > 1e-15
        ψ ./= sqrt(norm_sq)
    end
    return ψ
end

# Density matrix version
function reset_state!(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, target_states::Vector{Symbol}, N::Int)
    @assert length(qubits) == length(target_states) "Must specify target for each qubit"

    for (k, target) in zip(qubits, target_states)
        if target == :zero
            _force_qubit_rho!(ρ, k, 0, N)
        elseif target == :one
            _force_qubit_rho!(ρ, k, 1, N)
        else
            error("For density matrices, only :zero and :one targets implemented")
        end
    end

    return ρ
end

"""
    _force_qubit_rho!(ρ, k, b, N) -> ρ

Force qubit k to |b⟩⟨b| in density matrix by projection.

# Algorithm
Zero out all matrix elements ρ[i,j] where qubit k's bit ≠ b in either i or j.
Then renormalize to Tr(ρ) = 1.

# Mathematical operation
ρ → P_b ρ P_b / Tr(P_b ρ P_b)
where P_b = |b⟩⟨b| on qubit k, I on others.

# Example: 2-qubit entangled state
```julia
# Start with Bell state ρ = |Φ+⟩⟨Φ+| = (|00⟩+|11⟩)(⟨00|+⟨11|)/2
#     |00⟩ |01⟩ |10⟩ |11⟩
# |00⟩  1/2   0    0   1/2
# |01⟩   0    0    0    0
# |10⟩   0    0    0    0
# |11⟩  1/2   0    0   1/2

# After _force_qubit_rho!(ρ, 1, 0, 2):  (force qubit 1 to |0⟩)
# Zero out rows/cols where qubit 1 bit = 1 (i.e., |10⟩, |11⟩)
# Result: ρ = |00⟩⟨00| (pure state!)
#     |00⟩ |01⟩ |10⟩ |11⟩
# |00⟩   1    0    0    0
# |01⟩   0    0    0    0
# |10⟩   0    0    0    0
# |11⟩   0    0    0    0
```
"""
function _force_qubit_rho!(ρ::Matrix{ComplexF64}, k::Int, b::Int, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)

    # Zero out all coherences and populations with wrong bit value
    @inbounds for i in 0:(dim-1)
        bit_i = (i >> (k-1)) & 1
        for j in 0:(dim-1)
            bit_j = (j >> (k-1)) & 1
            if bit_i != b || bit_j != b
                ρ[i+1, j+1] = zero(ComplexF64)
            end
        end
    end

    # Renormalize
    tr = real(sum(ρ[i, i] for i in 1:dim))
    if tr > 1e-15
        ρ ./= tr
    end

    return ρ
end

function reset_state!(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, target::Symbol, N::Int)
    targets = fill(target, length(qubits))
    return reset_state!(ρ, qubits, targets, N)
end

function reset_state!(ρ::Matrix{ComplexF64}, target::Symbol, N::Int)
    return reset_state!(ρ, collect(1:N), target, N)
end

# ==============================================================================
# RESET QUBIT ALIAS (explicit qubit list and ket list)
# ==============================================================================

"""
    reset_qubit!(ψ, qubits::Vector{Int}, kets::Vector{Symbol}, N::Int)
    reset_qubit!(ρ, qubits::Vector{Int}, kets::Vector{Symbol}, N::Int)

Explicit alias for reset_state! - reset specified qubits to specified kets.

# Arguments
- `ψ` or `ρ`: State (modified in-place)
- `qubits`: List of qubit indices [1, 3, 5, ...]
- `kets`: List of target kets [:zero, :one, :plus, :minus, ...]
- `N`: Total number of qubits

# Example
```julia
# Reset qubits 1,3,5 to |0⟩, |1⟩, |+⟩
reset_qubit!(ψ, [1, 3, 5], [:zero, :one, :plus], N)
```
"""
reset_qubit!(ψ::Vector{ComplexF64}, qubits::Vector{Int}, kets::Vector{Symbol}, N::Int) =
    reset_state!(ψ, qubits, kets, N)

reset_qubit!(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, kets::Vector{Symbol}, N::Int) =
    reset_state!(ρ, qubits, kets, N)

# Single ket for all qubits
reset_qubit!(ψ::Vector{ComplexF64}, qubits::Vector{Int}, ket::Symbol, N::Int) =
    reset_state!(ψ, qubits, ket, N)

reset_qubit!(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, ket::Symbol, N::Int) =
    reset_state!(ρ, qubits, ket, N)

end # module

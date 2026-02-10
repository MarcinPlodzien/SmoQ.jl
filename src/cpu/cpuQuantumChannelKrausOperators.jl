# Date: 2026
#
#=
================================================================================
    cpuQuantumChannelKraus.jl - Matrix-Free Kraus Noise Channels (CPU)
================================================================================

OVERVIEW
--------
Matrix-free implementations of common single-qubit quantum noise channels.
All operations use bitwise indexing to update density matrix elements directly,
avoiding construction of full 2^N × 2^N Kraus operators.

COMPLEXITY
----------
Per qubit: O(4^N) for density matrix (unavoidable - all elements affected)
But: No Kraus matrix construction, no matrix multiplication, just element updates!

API
---
All channels use a unified signature with qubit list. Multiple dispatch
distinguishes density matrix (Matrix) from pure state (Vector):

  # Density matrix
  apply_channel_dephasing!(ρ::Matrix, 0.1, [1, 3], N)        # qubits 1,3
  apply_channel_dephasing!(ρ::Matrix, 0.1, [2], N)           # single qubit
  apply_channel_dephasing!(ρ::Matrix, 0.1, collect(1:N), N)  # all qubits

  # Pure state (MCWF - stochastic)
  apply_channel_dephasing!(ψ::Vector, 0.1, [1, 3], N)

CHANNELS
--------

Dephasing (Phase Damping):
  apply_channel_dephasing!(state, p, qubits, N)
      Effect: Coherences decay, populations unchanged
      ρ_01 → √(1-p) ρ_01
  Physical: T2 decay

Depolarizing:
  apply_channel_depolarizing!(state, p, qubits, N)
      Effect: State mixed toward maximally mixed
      ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
  Physical: Uniform random errors

Amplitude Damping (T1 Decay):
  apply_channel_amplitude_damping!(state, γ, qubits, N)
      Effect: Excited state decays to ground
      ρ₁₁ → (1-γ)·ρ₁₁, ρ₀₀ → ρ₀₀ + γ·ρ₁₁
  Physical: Spontaneous emission

Bit Flip:
  apply_channel_bit_flip!(state, p, qubits, N)
      Effect: Random X errors
      ρ → (1-p)ρ + p·XρX
  Physical: Classical bit errors

Phase Flip:
  apply_channel_phase_flip!(state, p, qubits, N)
      Effect: Random Z errors
      ρ → (1-p)ρ + p·ZρZ
  Physical: Random phase errors

BIT CONVENTION
--------------
LITTLE-ENDIAN: Qubit k has bit position (k-1).

================================================================================
=#

module CPUQuantumChannelKraus

using LinearAlgebra
using Random

export apply_channel_dephasing!, apply_channel_depolarizing!
export apply_channel_amplitude_damping!
export apply_channel_bit_flip!, apply_channel_phase_flip!

# ==============================================================================
# INTERNAL: Single-qubit implementations (density matrix)
# ==============================================================================

"""Internal: Apply dephasing to single qubit."""
function _apply_dephasing_single!(ρ::Matrix{ComplexF64}, p::Float64, qubit::Int)
    if p ≈ 0.0
        return ρ
    end

    damp = sqrt(1 - p)
    dim = size(ρ, 1)

    @inbounds for j in 0:(dim-1)
        bit_j = (j >> (qubit - 1)) & 1
        for i in 0:(dim-1)
            bit_i = (i >> (qubit - 1)) & 1
            if bit_i != bit_j
                ρ[i+1, j+1] *= damp
            end
        end
    end
    return ρ
end

"""Internal: Apply depolarizing to single qubit."""
function _apply_depolarizing_single!(ρ::Matrix{ComplexF64}, p::Float64, qubit::Int)
    if p ≈ 0.0
        return ρ
    end

    mask = 1 << (qubit - 1)
    dim = size(ρ, 1)
    p3 = p / 3.0
    one_minus_p = 1.0 - p

    ρ_new = copy(ρ)

    @inbounds for j in 0:(dim-1)
        bit_j = (j >> (qubit - 1)) & 1
        for i in 0:(dim-1)
            bit_i = (i >> (qubit - 1)) & 1

            i_flip = xor(i, mask)
            j_flip = xor(j, mask)

            val = ρ[i+1, j+1]
            val_flip = ρ[i_flip+1, j_flip+1]

            sign = 1 - 2 * ((bit_i + bit_j) & 1)

            if sign == 1
                ρ_new[i+1, j+1] = one_minus_p * val + p3 * (2*val_flip + val)
            else
                ρ_new[i+1, j+1] = one_minus_p * val - p3 * val
            end
        end
    end

    ρ .= ρ_new
    return ρ
end

"""Internal: Apply amplitude damping to single qubit."""
function _apply_amplitude_damping_single!(ρ::Matrix{ComplexF64}, γ::Float64, qubit::Int)
    if γ ≈ 0.0
        return ρ
    end

    mask = 1 << (qubit - 1)
    dim = size(ρ, 1)
    sqrt_1_minus_γ = sqrt(1 - γ)
    one_minus_γ = 1 - γ

    # Phase 1: Scale coherences
    @inbounds for j in 0:(dim-1)
        bit_j = (j >> (qubit - 1)) & 1
        for i in 0:(dim-1)
            bit_i = (i >> (qubit - 1)) & 1
            if bit_i != bit_j
                ρ[i+1, j+1] *= sqrt_1_minus_γ
            end
        end
    end

    # Phase 2: Transfer population from excited to ground
    @inbounds for j in 0:(dim-1)
        bit_j = (j >> (qubit - 1)) & 1
        for i in 0:(dim-1)
            bit_i = (i >> (qubit - 1)) & 1
            if bit_i == 1 && bit_j == 1
                i_ground = xor(i, mask)
                j_ground = xor(j, mask)
                ρ[i_ground+1, j_ground+1] += γ * ρ[i+1, j+1]
                ρ[i+1, j+1] *= one_minus_γ
            end
        end
    end

    return ρ
end

"""Internal: Apply bit flip to single qubit."""
function _apply_bit_flip_single!(ρ::Matrix{ComplexF64}, p::Float64, qubit::Int)
    if p ≈ 0.0
        return ρ
    end

    mask = 1 << (qubit - 1)
    dim = size(ρ, 1)
    one_minus_p = 1 - p

    @inbounds for j in 0:(dim-1)
        for i in 0:(dim-1)
            if (i & mask) == 0 && (j & mask) == 0
                i00, j00 = i, j
                i01, j01 = i, j | mask
                i10, j10 = i | mask, j
                i11, j11 = i | mask, j | mask

                val_00 = ρ[i00+1, j00+1]
                val_01 = ρ[i01+1, j01+1]
                val_10 = ρ[i10+1, j10+1]
                val_11 = ρ[i11+1, j11+1]

                ρ[i00+1, j00+1] = one_minus_p * val_00 + p * val_11
                ρ[i01+1, j01+1] = one_minus_p * val_01 + p * val_10
                ρ[i10+1, j10+1] = one_minus_p * val_10 + p * val_01
                ρ[i11+1, j11+1] = one_minus_p * val_11 + p * val_00
            end
        end
    end

    return ρ
end

"""Internal: Apply phase flip to single qubit."""
function _apply_phase_flip_single!(ρ::Matrix{ComplexF64}, p::Float64, qubit::Int)
    if p ≈ 0.0
        return ρ
    end

    coherence_scale = 1 - 2*p
    dim = size(ρ, 1)

    @inbounds for j in 0:(dim-1)
        bit_j = (j >> (qubit - 1)) & 1
        for i in 0:(dim-1)
            bit_i = (i >> (qubit - 1)) & 1
            if bit_i != bit_j
                ρ[i+1, j+1] *= coherence_scale
            end
        end
    end

    return ρ
end

# ==============================================================================
# INTERNAL: Single-qubit implementations (pure state - MCWF)
# ==============================================================================

"""Internal: Dephasing on single qubit (MCWF).

For MCWF to match density matrix dephasing EXACTLY:
- Density matrix: ρ_01 → √(1-p) × ρ_01
- For |+⟩ state: ⟨X⟩ = √(1-p)

The simplest way to achieve this in MCWF is to apply a random phase
that averages to the same coherence decay. We apply Z with probability p,
which gives ⟨X⟩ = (1-p)×1 + p×(-1) = 1-2p.

But density matrix gives √(1-p), so we need different approach:
Apply a deterministic amplitude scaling is not valid for pure states.

CORRECT MCWF: Apply random Z with prob q where (1-2q) = √(1-p)
→ q = (1 - √(1-p))/2

For p=0.2: q = (1-√0.8)/2 = (1-0.894)/2 = 0.053
Expected: (1-2×0.053) = 0.894 = √0.8 ✓
"""
function _apply_dephasing_single!(ψ::Vector{ComplexF64}, p::Float64, qubit::Int)
    if p ≈ 0.0
        return ψ
    end

    # Apply Z with probability q where (1-2q) = √(1-p)
    # This gives ensemble average ⟨X⟩ = √(1-p), matching density matrix
    q = (1.0 - sqrt(1.0 - p)) / 2.0

    if rand() < q
        # Apply Z to this qubit
        dim = length(ψ)
        @inbounds for i in 0:(dim-1)
            if ((i >> (qubit - 1)) & 1) == 1
                ψ[i+1] = -ψ[i+1]
            end
        end
    end

    return ψ
end

"""Internal: Depolarizing on single qubit (MCWF)."""
function _apply_depolarizing_single!(ψ::Vector{ComplexF64}, p::Float64, qubit::Int)
    r = rand()
    if r < p/3
        # Apply X
        mask = 1 << (qubit - 1)
        dim = length(ψ)
        @inbounds for i in 0:(dim-1)
            if (i & mask) == 0
                j = i | mask
                ψ[i+1], ψ[j+1] = ψ[j+1], ψ[i+1]
            end
        end
    elseif r < 2*p/3
        # Apply Y (= iXZ, but just flip and phase)
        mask = 1 << (qubit - 1)
        dim = length(ψ)
        @inbounds for i in 0:(dim-1)
            if (i & mask) == 0
                j = i | mask
                ψ[i+1], ψ[j+1] = -im*ψ[j+1], im*ψ[i+1]
            end
        end
    elseif r < p
        # Apply Z
        dim = length(ψ)
        @inbounds for i in 0:(dim-1)
            if ((i >> (qubit - 1)) & 1) == 1
                ψ[i+1] = -ψ[i+1]
            end
        end
    end
    # else: do nothing (identity with probability 1-p)
    return ψ
end

"""Internal: Bit flip on single qubit (MCWF)."""
function _apply_bit_flip_single!(ψ::Vector{ComplexF64}, p::Float64, qubit::Int)
    if rand() < p
        mask = 1 << (qubit - 1)
        dim = length(ψ)

        @inbounds for i in 0:(dim-1)
            if (i & mask) == 0
                j = i | mask
                ψ[i+1], ψ[j+1] = ψ[j+1], ψ[i+1]
            end
        end
    end
    return ψ
end

"""Internal: Phase flip on single qubit (MCWF)."""
function _apply_phase_flip_single!(ψ::Vector{ComplexF64}, p::Float64, qubit::Int)
    if rand() < p
        dim = length(ψ)
        @inbounds for i in 0:(dim-1)
            if ((i >> (qubit - 1)) & 1) == 1
                ψ[i+1] = -ψ[i+1]
            end
        end
    end
    return ψ
end

"""Internal: Amplitude damping on single qubit (MCWF)."""
function _apply_amplitude_damping_single!(ψ::Vector{ComplexF64}, γ::Float64, qubit::Int)
    mask = 1 << (qubit - 1)
    dim = length(ψ)

    pop_1 = 0.0
    @inbounds for i in 0:(dim-1)
        if (i & mask) != 0
            pop_1 += abs2(ψ[i+1])
        end
    end

    p_jump = γ * pop_1

    if rand() < p_jump && pop_1 > 1e-12
        @inbounds for i in 0:(dim-1)
            if (i & mask) != 0
                i_ground = xor(i, mask)
                ψ[i_ground+1] += ψ[i+1]
                ψ[i+1] = 0.0
            end
        end
    else
        sqrt_1_minus_γ = sqrt(1 - γ)
        @inbounds for i in 0:(dim-1)
            if (i & mask) != 0
                ψ[i+1] *= sqrt_1_minus_γ
            end
        end
    end

    norm_ψ = sqrt(sum(abs2, ψ))
    if norm_ψ > 1e-12
        ψ ./= norm_ψ
    end

    return ψ
end

# ==============================================================================
# PUBLIC API: Channels (dispatch on Matrix vs Vector)
# ==============================================================================

"""
    apply_channel_dephasing!(ρ::Matrix, p, qubits, N)
    apply_channel_dephasing!(ψ::Vector, p, qubits, N)

Apply dephasing channel to specified qubits with dephasing probability p.

# Effect (density matrix)
- Populations unchanged
- Coherences scaled by √(1-p) for each qubit

# Effect (pure state - MCWF)
- With probability p, project onto |0⟩ or |1⟩ (stochastic)

# Examples
```julia
# Density matrix
apply_channel_dephasing!(ρ, 0.1, [1, 3], N)        # qubits 1 and 3
apply_channel_dephasing!(ρ, 0.1, [2], N)           # single qubit 2
apply_channel_dephasing!(ρ, 0.1, collect(1:N), N)  # all qubits

# Pure state (MCWF)
apply_channel_dephasing!(ψ, 0.1, [1], N)
```
"""
function apply_channel_dephasing!(ρ::Matrix{ComplexF64}, p::Float64, qubits::Vector{Int}, N::Int)
    @assert 0 <= p <= 1 "Dephasing probability must be in [0,1]"
    @assert all(1 .<= qubits .<= N) "All qubit indices must be in 1..N"

    for q in qubits
        _apply_dephasing_single!(ρ, p, q)
    end
    return ρ
end

function apply_channel_dephasing!(ψ::Vector{ComplexF64}, p::Float64, qubits::Vector{Int}, N::Int)
    for q in qubits
        _apply_dephasing_single!(ψ, p, q)
    end
    return ψ
end

"""
    apply_channel_depolarizing!(ρ::Matrix, p, qubits, N)
    apply_channel_depolarizing!(ψ::Vector, p, qubits, N)

Apply depolarizing channel to specified qubits.

# Effect (density matrix)
ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ) for each qubit

# Examples
```julia
apply_channel_depolarizing!(ρ, 0.05, [1, 2], N)
apply_channel_depolarizing!(ρ, 0.1, collect(1:N), N)
```
"""
function apply_channel_depolarizing!(ρ::Matrix{ComplexF64}, p::Float64, qubits::Vector{Int}, N::Int)
    @assert 0 <= p <= 1 "Depolarizing probability must be in [0,1]"
    @assert all(1 .<= qubits .<= N) "All qubit indices must be in 1..N"

    for q in qubits
        _apply_depolarizing_single!(ρ, p, q)
    end
    return ρ
end

function apply_channel_depolarizing!(ψ::Vector{ComplexF64}, p::Float64, qubits::Vector{Int}, N::Int)
    for q in qubits
        _apply_depolarizing_single!(ψ, p, q)
    end
    return ψ
end

"""
    apply_channel_amplitude_damping!(ρ::Matrix, γ, qubits, N)
    apply_channel_amplitude_damping!(ψ::Vector, γ, qubits, N)

Apply amplitude damping channel to specified qubits.

# Effect (density matrix)
- Ground state gains population: ρ₀₀ → ρ₀₀ + γ·ρ₁₁
- Excited state decays: ρ₁₁ → (1-γ)·ρ₁₁
- Coherences decay: ρ₀₁ → √(1-γ)·ρ₀₁

# Examples
```julia
apply_channel_amplitude_damping!(ρ, 0.1, [1], N)
apply_channel_amplitude_damping!(ρ, 0.05, collect(1:N), N)
```
"""
function apply_channel_amplitude_damping!(ρ::Matrix{ComplexF64}, γ::Float64, qubits::Vector{Int}, N::Int)
    @assert 0 <= γ <= 1 "Damping parameter must be in [0,1]"
    @assert all(1 .<= qubits .<= N) "All qubit indices must be in 1..N"

    for q in qubits
        _apply_amplitude_damping_single!(ρ, γ, q)
    end
    return ρ
end

function apply_channel_amplitude_damping!(ψ::Vector{ComplexF64}, γ::Float64, qubits::Vector{Int}, N::Int)
    for q in qubits
        _apply_amplitude_damping_single!(ψ, γ, q)
    end
    return ψ
end

"""
    apply_channel_bit_flip!(ρ::Matrix, p, qubits, N)
    apply_channel_bit_flip!(ψ::Vector, p, qubits, N)

Apply bit flip channel to specified qubits.

# Effect (density matrix)
ρ → (1-p)ρ + p·XρX for each qubit

# Examples
```julia
apply_channel_bit_flip!(ρ, 0.01, [1, 2, 3], N)
apply_channel_bit_flip!(ρ, 0.05, collect(1:N), N)
```
"""
function apply_channel_bit_flip!(ρ::Matrix{ComplexF64}, p::Float64, qubits::Vector{Int}, N::Int)
    @assert 0 <= p <= 1 "Flip probability must be in [0,1]"
    @assert all(1 .<= qubits .<= N) "All qubit indices must be in 1..N"

    for q in qubits
        _apply_bit_flip_single!(ρ, p, q)
    end
    return ρ
end

function apply_channel_bit_flip!(ψ::Vector{ComplexF64}, p::Float64, qubits::Vector{Int}, N::Int)
    for q in qubits
        _apply_bit_flip_single!(ψ, p, q)
    end
    return ψ
end

"""
    apply_channel_phase_flip!(ρ::Matrix, p, qubits, N)
    apply_channel_phase_flip!(ψ::Vector, p, qubits, N)

Apply phase flip channel to specified qubits.

# Effect (density matrix)
ρ → (1-p)ρ + p·ZρZ for each qubit

# Examples
```julia
apply_channel_phase_flip!(ρ, 0.1, [1], N)
apply_channel_phase_flip!(ρ, 0.05, collect(1:N), N)
```
"""
function apply_channel_phase_flip!(ρ::Matrix{ComplexF64}, p::Float64, qubits::Vector{Int}, N::Int)
    @assert 0 <= p <= 1 "Flip probability must be in [0,1]"
    @assert all(1 .<= qubits .<= N) "All qubit indices must be in 1..N"

    for q in qubits
        _apply_phase_flip_single!(ρ, p, q)
    end
    return ρ
end

function apply_channel_phase_flip!(ψ::Vector{ComplexF64}, p::Float64, qubits::Vector{Int}, N::Int)
    for q in qubits
        _apply_phase_flip_single!(ψ, p, q)
    end
    return ψ
end

end # module

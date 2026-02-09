# Date: 2026
#
#=
================================================================================
    cpuQuantumChannelLindbladianEvolutionExact.jl

    EXACT Lindbladian evolution using DENSITY MATRIX formalism.

    dρ/dt = -i[H,ρ] + γ Σᵢ D[σ_zᵢ]ρ

    where D[L]ρ = L ρ L† - ½{L†L, ρ} is the Lindblad dissipator.

    JUMP OPERATOR: Lₖ = σ_zₖ (pure dephasing)

    This module implements:
    1. Density matrix evolution with exact unitary propagator U = exp(-iHdt)
    2. Bitwise σ_z dissipator for pure dephasing on each qubit
    3. First-order Euler integration of the Lindblad master equation
================================================================================
=#

module CPUQuantumChannelLindbladianEvolutionDensityMatrix

using LinearAlgebra

export lindblad_dm_step!
export lindblad_dm_step_trotter!
export apply_sigma_z_dissipator!

# ==============================================================================
# σ_z DISSIPATOR FOR DENSITY MATRIX (Bitwise, Matrix-Free)
#
# For the Lindblad dissipator D[L]ρ = L ρ L† - ½{L†L, ρ}, with L = σ_z:
#   D[σ_z]ρ = σ_z ρ σ_z - ½{I, ρ} = σ_z ρ σ_z - ρ
#
# Since σ_z = diag(1, -1) and σ_z² = I, this simplifies to:
#   (D[σ_z]ρ)ᵢⱼ = (-1)^(bᵢ⊕bⱼ) ρᵢⱼ - ρᵢⱼ
#
# MATRIX ELEMENT RULES for density matrix ρᵢⱼ:
# - If bit_k(i) = bit_k(j) (diagonal blocks): no change
# - If bit_k(i) ≠ bit_k(j) (off-diagonal coherences): ρᵢⱼ *= (1 - 2γdt)
# ==============================================================================

"""
    apply_sigma_z_dissipator!(ρ, k, γdt, N)

Apply σ_z dissipator (pure dephasing) to density matrix for qubit k.
Implements: D[σ_z]ρ = σ_z ρ σ_z - ρ

This is a first-order Euler step. Parameter γdt = γ × dt.
Pure dephasing only affects off-diagonal elements (coherences).
"""
function apply_sigma_z_dissipator!(ρ::Matrix{ComplexF64}, k::Int, γdt::Float64, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)
    decay_factor = 1 - 2γdt  # Dephasing decay

    @inbounds for j in 0:(dim-1), i in 0:(dim-1)
        bi = (i >> (k-1)) & 1  # Extract bit k from row index
        bj = (j >> (k-1)) & 1  # Extract bit k from column index

        if bi != bj
            # Off-diagonal coherence: pure dephasing
            ρ[i+1, j+1] *= decay_factor
        end
        # Diagonal blocks (bi == bj): no change
    end
    return ρ
end

"""
    population_excited_rho(ρ, k, N)

Compute Tr(σ⁺σ⁻ ρ) = probability that qubit k is in |1⟩ (excited state).
"""
function population_excited_rho(ρ::Matrix{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)
    result = 0.0
    @inbounds for i in 0:(dim-1)
        if (i & mask) != 0
            result += real(ρ[i+1, i+1])
        end
    end
    return result
end

# ==============================================================================
# LINDBLAD DENSITY MATRIX EVOLUTION STEP
# ==============================================================================

"""
    lindblad_dm_step!(ρ, U, N, γ, dt)

Single step of Lindblad master equation for density matrix (EXACT method).

ALGORITHM:
----------
1. Coherent evolution: ρ → U ρ U†
2. Dissipator for each qubit: ρ → ρ + γ×dt × D[σ_zₖ]ρ

This is a first-order Euler step:
  ρ(t+dt) ≈ U ρ(t) U† + γ×dt × Σₖ D[σ_zₖ]ρ
"""
function lindblad_dm_step!(ρ::Matrix{ComplexF64}, U::Matrix{ComplexF64}, N::Int, γ::Float64, dt::Float64)
    # Step 1: Coherent evolution ρ → U ρ U†
    ρ .= U * ρ * U'

    # Step 2: Apply σ_z dissipator on each qubit (pure dephasing)
    @inbounds for k in 1:N
        apply_sigma_z_dissipator!(ρ, k, γ * dt, N)
    end

    return ρ
end

"""
    lindblad_dm_step_trotter!(ρ, gates, N, γ, dt)

Single step of Lindblad master equation for density matrix using TROTTER gates.

ALGORITHM:
----------
1. Coherent evolution: ρ → U_trotter ρ U_trotter† (via column/row gate application)
2. Dissipator for each qubit: ρ → ρ + γ×dt × D[σ⁻ₖ]ρ
"""
function lindblad_dm_step_trotter!(ρ::Matrix{ComplexF64}, gates, N::Int, γ::Float64, dt::Float64)
    # Step 1: Apply Trotter gates to density matrix
    # ρ → U_trotter ρ U_trotter†
    # This requires applying gates to columns (left multiply) and rows (right multiply by U†)
    dim = size(ρ, 1)

    # Apply U to each column: ρ → U ρ
    for gate in gates
        U = gate.U
        indices = gate.indices
        nq = length(indices)

        if nq == 1
            qubit = indices[1]
            step = 1 << (qubit - 1)
            u00, u01 = U[1,1], U[1,2]
            u10, u11 = U[2,1], U[2,2]

            for col in 1:dim
                for base_idx in 0:(dim-1)
                    if (base_idx & step) == 0
                        idx0 = base_idx + 1
                        idx1 = base_idx + step + 1
                        a0, a1 = ρ[idx0, col], ρ[idx1, col]
                        ρ[idx0, col] = u00 * a0 + u01 * a1
                        ρ[idx1, col] = u10 * a0 + u11 * a1
                    end
                end
            end
        elseif nq == 2
            q1, q2 = indices[1], indices[2]
            if q1 > q2
                q1, q2 = q2, q1
            end
            step1 = 1 << (q1 - 1)
            step2 = 1 << (q2 - 1)

            for col in 1:dim
                for base_idx in 0:(dim-1)
                    if (base_idx & step1) == 0 && (base_idx & step2) == 0
                        idx00 = base_idx + 1
                        idx01 = base_idx + step1 + 1
                        idx10 = base_idx + step2 + 1
                        idx11 = base_idx + step1 + step2 + 1

                        a00, a01, a10, a11 = ρ[idx00, col], ρ[idx01, col], ρ[idx10, col], ρ[idx11, col]

                        ρ[idx00, col] = U[1,1]*a00 + U[1,2]*a01 + U[1,3]*a10 + U[1,4]*a11
                        ρ[idx01, col] = U[2,1]*a00 + U[2,2]*a01 + U[2,3]*a10 + U[2,4]*a11
                        ρ[idx10, col] = U[3,1]*a00 + U[3,2]*a01 + U[3,3]*a10 + U[3,4]*a11
                        ρ[idx11, col] = U[4,1]*a00 + U[4,2]*a01 + U[4,3]*a10 + U[4,4]*a11
                    end
                end
            end
        end
    end

    # Apply U† to each row: ρ → ρ U†
    for gate in gates
        U = conj.(gate.U)  # U† has conjugated elements
        indices = gate.indices
        nq = length(indices)

        if nq == 1
            qubit = indices[1]
            step = 1 << (qubit - 1)
            u00, u01 = U[1,1], U[1,2]
            u10, u11 = U[2,1], U[2,2]

            for row in 1:dim
                for base_idx in 0:(dim-1)
                    if (base_idx & step) == 0
                        idx0 = base_idx + 1
                        idx1 = base_idx + step + 1
                        a0, a1 = ρ[row, idx0], ρ[row, idx1]
                        ρ[row, idx0] = u00 * a0 + u01 * a1
                        ρ[row, idx1] = u10 * a0 + u11 * a1
                    end
                end
            end
        elseif nq == 2
            q1, q2 = indices[1], indices[2]
            if q1 > q2
                q1, q2 = q2, q1
            end
            step1 = 1 << (q1 - 1)
            step2 = 1 << (q2 - 1)

            for row in 1:dim
                for base_idx in 0:(dim-1)
                    if (base_idx & step1) == 0 && (base_idx & step2) == 0
                        idx00 = base_idx + 1
                        idx01 = base_idx + step1 + 1
                        idx10 = base_idx + step2 + 1
                        idx11 = base_idx + step1 + step2 + 1

                        a00, a01, a10, a11 = ρ[row, idx00], ρ[row, idx01], ρ[row, idx10], ρ[row, idx11]

                        ρ[row, idx00] = U[1,1]*a00 + U[1,2]*a01 + U[1,3]*a10 + U[1,4]*a11
                        ρ[row, idx01] = U[2,1]*a00 + U[2,2]*a01 + U[2,3]*a10 + U[2,4]*a11
                        ρ[row, idx10] = U[3,1]*a00 + U[3,2]*a01 + U[3,3]*a10 + U[3,4]*a11
                        ρ[row, idx11] = U[4,1]*a00 + U[4,2]*a01 + U[4,3]*a10 + U[4,4]*a11
                    end
                end
            end
        end
    end

    # Step 2: Apply σ_z dissipator on each qubit (pure dephasing)
    @inbounds for k in 1:N
        apply_sigma_z_dissipator!(ρ, k, γ * dt, N)
    end

    return ρ
end

end # module

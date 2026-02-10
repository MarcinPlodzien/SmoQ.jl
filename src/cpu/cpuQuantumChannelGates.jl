# Date: 2026
#
#=
================================================================================
    cpuQuantumChannelGates.jl - Matrix-Free Quantum Gates (CPU)
================================================================================

OVERVIEW
--------
Matrix-free (bitwise) implementation of quantum gates for pure states and
density matrices. Operates in-place where possible for efficiency.

SINGLE-QUBIT GATES (CLIFFORD)
-----------------------------
- apply_rx_psi!, apply_rx_rho!  : Rx(θ) = exp(-iθX/2)
- apply_ry_psi!, apply_ry_rho!  : Ry(θ) = exp(-iθY/2)
- apply_rz_psi!, apply_rz_rho!  : Rz(θ) = exp(-iθZ/2)
- apply_hadamard_psi!, apply_hadamard_rho!   : H gate
- apply_pauli_x_psi!, apply_pauli_z_psi!     : X, Z gates
- apply_s_psi!, apply_sdagger_psi!           : S, S† gates

SINGLE-QUBIT GATES (NON-CLIFFORD) ⚡ Source of Magic
----------------------------------------------------
- apply_t_psi!, apply_t_rho!         : T gate = diag(1, e^{iπ/4})
- apply_tdagger_psi!, apply_tdagger_rho! : T† gate

TWO-QUBIT GATES (CLIFFORD)
--------------------------
- apply_cnot_psi!, apply_cnot_rho!   : CNOT gate
- apply_cz_psi!, apply_cz_rho!       : CZ gate
- apply_rxx_psi!, apply_rxx_rho!     : Rxx(θ) = exp(-iθXX/2)
- apply_ryy_psi!, apply_ryy_rho!     : Ryy(θ) = exp(-iθYY/2)
- apply_rzz_psi!, apply_rzz_rho!     : Rzz(θ) = exp(-iθZZ/2)

THREE-QUBIT GATES (NON-CLIFFORD) ⚡
-----------------------------------
- apply_ccz_psi!     : CCZ gate (phase -1 when all three qubits |111⟩)
- apply_toffoli_psi! : Toffoli (CCNOT) gate

LINDBLAD JUMP OPERATORS (Non-Unitary)
-------------------------------------
- apply_sigma_minus_psi!        : σ⁻ = |0⟩⟨1| lowering operator
- apply_sigma_plus_psi!         : σ⁺ = |1⟩⟨0| raising operator
- population_excited_psi        : ⟨σ⁺σ⁻⟩ = P(|1⟩)
- population_ground_psi         : ⟨σ⁻σ⁺⟩ = P(|0⟩)
- apply_sigma_minus_dissipator_rho! : D[σ⁻]ρ dissipator for density matrix

BIT CONVENTION
--------------
LITTLE-ENDIAN: Qubit k has bit position (k-1).
Qubit 1 = LSB, qubit N = MSB in state index.

GATE MATRICES
-------------
Rx(θ) = | cos(θ/2)   -i sin(θ/2) |
        | -i sin(θ/2)  cos(θ/2)  |

Ry(θ) = | cos(θ/2)  -sin(θ/2) |
        | sin(θ/2)   cos(θ/2) |

Rz(θ) = | e^{-iθ/2}    0      |
        |    0      e^{iθ/2}  |

H = (1/√2) | 1   1 |
           | 1  -1 |

T = | 1    0       |    T† = | 1     0        |
    | 0  e^{iπ/4}  |         | 0  e^{-iπ/4}   |

σ⁻ = | 0  1 |    σ⁺ = | 0  0 |
     | 0  0 |         | 1  0 |

================================================================================
=#

module CPUQuantumChannelGates

using LinearAlgebra

# Exports - Single qubit gates (pure state)
export apply_rx_psi!, apply_ry_psi!, apply_rz_psi!
export apply_hadamard_psi!, apply_pauli_x_psi!, apply_pauli_y_psi!, apply_pauli_z_psi!
export apply_s_psi!, apply_sdagger_psi!

# Exports - Single qubit gates (density matrix)
export apply_rx_rho!, apply_ry_rho!, apply_rz_rho!
export apply_hadamard_rho!, apply_pauli_x_rho!, apply_pauli_z_rho!

# Exports - Two qubit gates (pure state)
export apply_cnot_psi!, apply_cz_psi!, apply_cry_psi!
export apply_rxx_psi!, apply_ryy_psi!, apply_rzz_psi!

# Exports - Two qubit gates (density matrix)
export apply_cnot_rho!, apply_cz_rho!, apply_cry_rho!
export apply_rxx_rho!, apply_ryy_rho!, apply_rzz_rho!

# Exports - Geometry helpers
export GridGeometry, qubit_index

# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================

"""
    GridGeometry(Lx, Ly)

Helper struct for 2D grid qubit indexing.
- Lx: Number of sites per rail (chain length)
- Ly: Number of rails

Qubit (x, y) → linear index = x + (y-1)*Lx
Where x ∈ 1:Lx (position along rail), y ∈ 1:Ly (rail index)
"""
struct GridGeometry
    Lx::Int  # Sites per rail
    Ly::Int  # Number of rails
end

"""
    qubit_index(g::GridGeometry, x, y) -> Int

Convert 2D grid position (x, y) to linear qubit index.
"""
function qubit_index(g::GridGeometry, x::Int, y::Int)
    @assert 1 <= x <= g.Lx "x must be in 1..Lx"
    @assert 1 <= y <= g.Ly "y must be in 1..Ly"
    return x + (y - 1) * g.Lx
end

# ==============================================================================
# SINGLE-QUBIT GATES - PURE STATE
# ==============================================================================

"""
    apply_rx_psi!(ψ::Vector{ComplexF64}, k::Int, θ::Float64, N::Int)

Apply Rx(θ) gate to qubit k in-place.

Rx(θ) = | cos(θ/2)   -i sin(θ/2) |
        | -i sin(θ/2)  cos(θ/2)  |
"""
function apply_rx_psi!(ψ::Vector{ComplexF64}, k::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)
    is = -im * s

    step = 1 << (k - 1)  # Bit flip distance
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0  # bit k is 0
            j = i + step  # j has bit k = 1
            ψ_i = ψ[i+1]
            ψ_j = ψ[j+1]

            # Apply Rx
            ψ[i+1] = c * ψ_i + is * ψ_j
            ψ[j+1] = is * ψ_i + c * ψ_j
        end
    end
    return ψ
end

"""
    apply_ry_psi!(ψ::Vector{ComplexF64}, k::Int, θ::Float64, N::Int)

Apply Ry(θ) gate to qubit k in-place.

Ry(θ) = | cos(θ/2)  -sin(θ/2) |
        | sin(θ/2)   cos(θ/2) |
"""
function apply_ry_psi!(ψ::Vector{ComplexF64}, k::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)

    step = 1 << (k - 1)
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0
            j = i + step
            ψ_i = ψ[i+1]
            ψ_j = ψ[j+1]

            # Apply Ry
            ψ[i+1] = c * ψ_i - s * ψ_j
            ψ[j+1] = s * ψ_i + c * ψ_j
        end
    end
    return ψ
end

"""
    apply_rz_psi!(ψ::Vector{ComplexF64}, k::Int, θ::Float64, N::Int)

Apply Rz(θ) gate to qubit k in-place.

Rz(θ) = | e^{-iθ/2}    0      |
        |    0      e^{iθ/2}  |
"""
function apply_rz_psi!(ψ::Vector{ComplexF64}, k::Int, θ::Float64, N::Int)
    phase_0 = exp(-im * θ / 2)
    phase_1 = exp(im * θ / 2)
    # Branchless lookup: ((i >> (k-1)) & 1) gives 0 or 1, +1 for 1-based indexing
    phases = (phase_0, phase_1)
    dim = 1 << N

    @inbounds @simd for i in 0:(dim-1)
        ψ[i+1] *= phases[((i >> (k-1)) & 1) + 1]
    end
    return ψ
end

"""
    apply_hadamard_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply Hadamard gate to qubit k in-place.

H = (1/√2) | 1   1 |
           | 1  -1 |
"""
function apply_hadamard_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    s = 1 / sqrt(2)
    step = 1 << (k - 1)
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0
            j = i + step
            ψ_i = ψ[i+1]
            ψ_j = ψ[j+1]

            ψ[i+1] = s * (ψ_i + ψ_j)
            ψ[j+1] = s * (ψ_i - ψ_j)
        end
    end
    return ψ
end

"""
    apply_pauli_x_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply Pauli X (bit flip) to qubit k in-place.
"""
function apply_pauli_x_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    step = 1 << (k - 1)
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0
            j = i + step
            ψ[i+1], ψ[j+1] = ψ[j+1], ψ[i+1]
        end
    end
    return ψ
end

"""
    apply_pauli_y_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply Pauli Y to qubit k in-place.
Y = | 0  -i |
    | i   0 |
"""
function apply_pauli_y_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    step = 1 << (k - 1)
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0
            j = i + step
            v0, v1 = ψ[i+1], ψ[j+1]
            # Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
            ψ[i+1] = -im * v1
            ψ[j+1] = im * v0
        end
    end
    return ψ
end

"""
    apply_pauli_z_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply Pauli Z (phase flip) to qubit k in-place.
"""
function apply_pauli_z_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 1
            ψ[i+1] = -ψ[i+1]
        end
    end
    return ψ
end

"""
    apply_s_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply S gate (phase gate, √Z) to qubit k in-place.

S = | 1   0 |
    | 0   i |

S = Rz(π/2) up to global phase.
"""
function apply_s_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 1  # Bit k is 1
            ψ[i+1] *= im
        end
    end
    return ψ
end

"""
    apply_sdagger_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply S† gate (conjugate of S gate) to qubit k in-place.

S† = | 1   0  |
     | 0  -i  |

S† = Rz(-π/2) up to global phase.
"""
function apply_sdagger_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 1  # Bit k is 1
            ψ[i+1] *= -im
        end
    end
    return ψ
end

# ==============================================================================
# NON-CLIFFORD GATES - T GATE (π/8 GATE)
#
# The T gate is the CANONICAL non-Clifford gate - the source of "magic".
# T = diag(1, e^{iπ/4}) = S^{1/2} = fourth root of Z
#
# T|+⟩ is a "magic state" used in fault-tolerant quantum computing.
# T cannot be synthesized from Clifford gates alone.
# ==============================================================================

"""
    apply_t_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply T gate (π/8 gate) to qubit k in-place. This is a NON-CLIFFORD gate.

T = | 1    0       |  = Rz(π/4) up to global phase
    | 0  e^{iπ/4}  |

T is the key gate for universal quantum computation beyond Clifford circuits.
T|+⟩ creates "magic" - a resource for quantum advantage.
"""
function apply_t_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    phase = exp(im * π / 4)  # e^{iπ/4} = (1 + i)/√2

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 1  # Bit k is 1
            ψ[i+1] *= phase
        end
    end
    return ψ
end

"""
    apply_tdagger_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply T† gate (conjugate of T gate) to qubit k in-place.

T† = | 1    0        |  = Rz(-π/4) up to global phase
     | 0  e^{-iπ/4}  |
"""
function apply_tdagger_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    phase = exp(-im * π / 4)  # e^{-iπ/4}

    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 1
            ψ[i+1] *= phase
        end
    end
    return ψ
end

"""
    apply_t_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)

Apply T gate to density matrix: ρ' = T ρ T†
"""
function apply_t_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    phase = exp(im * π / 4)
    phase_conj = conj(phase)

    @inbounds for i in 0:(dim-1)
        pi = ((i >> (k-1)) & 1 == 1) ? phase : 1.0
        for j in 0:(dim-1)
            pj = ((j >> (k-1)) & 1 == 1) ? phase_conj : 1.0
            ρ[i+1, j+1] *= pi * pj
        end
    end
    return ρ
end

"""
    apply_tdagger_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)

Apply T† gate to density matrix.
"""
function apply_tdagger_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    phase = exp(-im * π / 4)
    phase_conj = conj(phase)

    @inbounds for i in 0:(dim-1)
        pi = ((i >> (k-1)) & 1 == 1) ? phase : 1.0
        for j in 0:(dim-1)
            pj = ((j >> (k-1)) & 1 == 1) ? phase_conj : 1.0
            ρ[i+1, j+1] *= pi * pj
        end
    end
    return ρ
end

# ==============================================================================
# NON-CLIFFORD GATES - CCZ / TOFFOLI
#
# CCZ (Controlled-Controlled-Z) applies -1 phase when all 3 qubits are |1⟩.
# Toffoli = (H⊗I⊗I) CCZ (H⊗I⊗I) - CCNOT is equivalent up to basis change.
# Both are non-Clifford 3-qubit gates.
# ==============================================================================

"""
    apply_ccz_psi!(ψ::Vector{ComplexF64}, q1::Int, q2::Int, q3::Int, N::Int)

Apply CCZ (Controlled-Controlled-Z) gate: apply -1 phase when ALL three qubits are |1⟩.

CCZ is symmetric under permutation of the three qubits.
This is a non-Clifford gate used in magic state injection and error correction.
"""
function apply_ccz_psi!(ψ::Vector{ComplexF64}, q1::Int, q2::Int, q3::Int, N::Int)
    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        b1 = (i >> (q1-1)) & 1
        b2 = (i >> (q2-1)) & 1
        b3 = (i >> (q3-1)) & 1

        if b1 == 1 && b2 == 1 && b3 == 1
            ψ[i+1] = -ψ[i+1]
        end
    end
    return ψ
end

"""
    apply_toffoli_psi!(ψ::Vector{ComplexF64}, c1::Int, c2::Int, target::Int, N::Int)

Apply Toffoli (CCNOT) gate: flip target if BOTH controls are |1⟩.

Toffoli = | I  0 |  (in 8x8 matrix form, with 2x2 X in bottom-right)
          | 0  X |
"""
function apply_toffoli_psi!(ψ::Vector{ComplexF64}, c1::Int, c2::Int, target::Int, N::Int)
    dim = 1 << N
    target_step = 1 << (target - 1)

    @inbounds for i in 0:(dim-1)
        bc1 = (i >> (c1-1)) & 1
        bc2 = (i >> (c2-1)) & 1
        bt = (i >> (target-1)) & 1

        # Only act when both controls are 1 AND target is 0 (avoid double swap)
        if bc1 == 1 && bc2 == 1 && bt == 0
            j = i + target_step  # flip target
            ψ[i+1], ψ[j+1] = ψ[j+1], ψ[i+1]
        end
    end
    return ψ
end

# Export non-Clifford gates
export apply_t_psi!, apply_tdagger_psi!
export apply_t_rho!, apply_tdagger_rho!
export apply_ccz_psi!, apply_toffoli_psi!

# ==============================================================================
# SINGLE-QUBIT GATES - DENSITY MATRIX
# ==============================================================================

"""
    apply_rx_rho!(ρ::Matrix{ComplexF64}, k::Int, θ::Float64, N::Int)

Apply Rx(θ) to density matrix: ρ' = Rx(θ) ρ Rx(θ)†
"""
function apply_rx_rho!(ρ::Matrix{ComplexF64}, k::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)
    is = -im * s
    is_dag = im * s  # conjugate

    step = 1 << (k - 1)
    dim = 1 << N

    # Apply to rows: ρ_new = Rx * ρ
    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0
            j = i + step
            for col in 1:dim
                ρ_i = ρ[i+1, col]
                ρ_j = ρ[j+1, col]
                ρ[i+1, col] = c * ρ_i + is * ρ_j
                ρ[j+1, col] = is * ρ_i + c * ρ_j
            end
        end
    end

    # Apply to columns: ρ_new = ρ * Rx†
    @inbounds for j in 0:(dim-1)
        if (j >> (k-1)) & 1 == 0
            jj = j + step
            for row in 1:dim
                ρ_j = ρ[row, j+1]
                ρ_jj = ρ[row, jj+1]
                ρ[row, j+1] = c * ρ_j + is_dag * ρ_jj
                ρ[row, jj+1] = is_dag * ρ_j + c * ρ_jj
            end
        end
    end

    return ρ
end

"""
    apply_ry_rho!(ρ::Matrix{ComplexF64}, k::Int, θ::Float64, N::Int)

Apply Ry(θ) to density matrix: ρ' = Ry(θ) ρ Ry(θ)†
"""
function apply_ry_rho!(ρ::Matrix{ComplexF64}, k::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)

    step = 1 << (k - 1)
    dim = 1 << N

    # Apply to rows
    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0
            j = i + step
            for col in 1:dim
                ρ_i = ρ[i+1, col]
                ρ_j = ρ[j+1, col]
                ρ[i+1, col] = c * ρ_i - s * ρ_j
                ρ[j+1, col] = s * ρ_i + c * ρ_j
            end
        end
    end

    # Apply to columns: ρ * Ry†
    # Ry† = | c   s |  so column transformation:
    #       |-s   c |  new_j = c * old_j - s * old_jj, new_jj = s * old_j + c * old_jj
    @inbounds for j in 0:(dim-1)
        if (j >> (k-1)) & 1 == 0
            jj = j + step
            for row in 1:dim
                ρ_j = ρ[row, j+1]
                ρ_jj = ρ[row, jj+1]
                ρ[row, j+1] = c * ρ_j - s * ρ_jj
                ρ[row, jj+1] = s * ρ_j + c * ρ_jj
            end
        end
    end

    return ρ
end

"""
    apply_rz_rho!(ρ::Matrix{ComplexF64}, k::Int, θ::Float64, N::Int)

Apply Rz(θ) to density matrix.
"""
function apply_rz_rho!(ρ::Matrix{ComplexF64}, k::Int, θ::Float64, N::Int)
    phase_0 = exp(-im * θ / 2)
    phase_1 = exp(im * θ / 2)

    dim = 1 << N

    # Rz is diagonal, so ρ' = Rz ρ Rz†
    # ρ'[i,j] = phase[bit_k(i)] * ρ[i,j] * conj(phase[bit_k(j)])
    @inbounds for i in 0:(dim-1)
        phase_i = ((i >> (k-1)) & 1 == 0) ? phase_0 : phase_1
        for j in 0:(dim-1)
            phase_j = ((j >> (k-1)) & 1 == 0) ? conj(phase_0) : conj(phase_1)
            ρ[i+1, j+1] *= phase_i * phase_j
        end
    end

    return ρ
end

"""
    apply_hadamard_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)

Apply Hadamard to density matrix.
"""
function apply_hadamard_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)
    s = 1 / sqrt(2)
    step = 1 << (k - 1)
    dim = 1 << N

    # Apply to rows
    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0
            j = i + step
            for col in 1:dim
                ρ_i = ρ[i+1, col]
                ρ_j = ρ[j+1, col]
                ρ[i+1, col] = s * (ρ_i + ρ_j)
                ρ[j+1, col] = s * (ρ_i - ρ_j)
            end
        end
    end

    # Apply to columns (H† = H)
    @inbounds for j in 0:(dim-1)
        if (j >> (k-1)) & 1 == 0
            jj = j + step
            for row in 1:dim
                ρ_j = ρ[row, j+1]
                ρ_jj = ρ[row, jj+1]
                ρ[row, j+1] = s * (ρ_j + ρ_jj)
                ρ[row, jj+1] = s * (ρ_j - ρ_jj)
            end
        end
    end

    return ρ
end

"""
    apply_pauli_x_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)

Apply Pauli X to density matrix.
"""
function apply_pauli_x_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)
    step = 1 << (k - 1)
    dim = 1 << N

    # Apply to rows
    @inbounds for i in 0:(dim-1)
        if (i >> (k-1)) & 1 == 0
            j = i + step
            for col in 1:dim
                ρ[i+1, col], ρ[j+1, col] = ρ[j+1, col], ρ[i+1, col]
            end
        end
    end

    # Apply to columns
    @inbounds for j in 0:(dim-1)
        if (j >> (k-1)) & 1 == 0
            jj = j + step
            for row in 1:dim
                ρ[row, j+1], ρ[row, jj+1] = ρ[row, jj+1], ρ[row, j+1]
            end
        end
    end

    return ρ
end

"""
    apply_pauli_z_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)

Apply Pauli Z to density matrix.
"""
function apply_pauli_z_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)
    dim = 1 << N

    # Z is diagonal with entries ±1
    # ρ'[i,j] = (-1)^{bit_k(i)} * ρ[i,j] * (-1)^{bit_k(j)}
    @inbounds for i in 0:(dim-1)
        sign_i = ((i >> (k-1)) & 1 == 0) ? 1 : -1
        for j in 0:(dim-1)
            sign_j = ((j >> (k-1)) & 1 == 0) ? 1 : -1
            ρ[i+1, j+1] *= sign_i * sign_j
        end
    end

    return ρ
end

# ==============================================================================
# LINDBLAD JUMP OPERATORS - PURE STATE
#
# PHYSICS BACKGROUND:
# -------------------
# The Lindblad master equation describes open quantum system dynamics:
#   dρ/dt = -i[H,ρ] + Σⱼ γⱼ (Lⱼ ρ Lⱼ† - ½{Lⱼ†Lⱼ, ρ})
#
# For spontaneous emission, the jump operator is σ⁻ = |0⟩⟨1| (lowering operator):
#   σ⁻|1⟩ = |0⟩    (qubit decays from excited to ground)
#   σ⁻|0⟩ = 0      (nothing happens if already in ground)
#
# The raising operator is σ⁺ = |1⟩⟨0| = (σ⁻)†.
#
# BITWISE IMPLEMENTATION:
# -----------------------
# State |i₁ i₂ ... iₙ⟩ is stored at index sum(iₖ × 2^(k-1)).
# For σ⁻ on qubit k:
#   mask = 1 << (k-1) = 2^(k-1)
#   If bit k is set: state |...1ₖ...⟩ → |...0ₖ...⟩
#   Index transformation: i → xor(i, mask) (XOR flips bit k)
# ==============================================================================

"""
    apply_sigma_minus_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply σ⁻ = |0⟩⟨1| (lowering/annihilation operator) to qubit k using BITWISE ops.

ALGORITHM:
----------
For each basis state i where bit k is SET (qubit in |1⟩):
  1. Find target index j = xor(i, mask) where bit k is UNSET (|0⟩)
  2. Transfer amplitude: ψ[j] = ψ[i] (qubit "decays" |1⟩→|0⟩)
  3. Zero source: ψ[i] = 0 (amplitude moved, not copied)

States with bit k already 0 are unaffected (σ⁻|0⟩ = 0).

WARNING: This operator is NOT unitary. Use for MCWF quantum jumps.
"""
function apply_sigma_minus_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)  # Binary mask with only bit k set

    @inbounds for i in 0:(dim-1)
        if (i & mask) != 0  # Bit k is set (qubit in |1⟩)
            j = xor(i, mask)    # XOR flips bit k: target has bit k = 0
            ψ[j+1] = ψ[i+1]  # Transfer amplitude |1⟩→|0⟩
            ψ[i+1] = 0.0     # Zero source amplitude
        end
    end
    return ψ
end

"""
    apply_sigma_plus_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)

Apply σ⁺ = |1⟩⟨0| (raising/creation operator) to qubit k using BITWISE ops.

This is the Hermitian conjugate of σ⁻:
  σ⁺|0⟩ = |1⟩    (qubit excited from ground to excited)
  σ⁺|1⟩ = 0      (nothing happens if already excited)
"""
function apply_sigma_plus_psi!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)

    @inbounds for i in 0:(dim-1)
        if (i & mask) == 0  # Bit k is 0 (qubit in |0⟩)
            j = i | mask     # Set bit k: target has bit k = 1
            ψ[j+1] = ψ[i+1]  # Transfer amplitude |0⟩→|1⟩
            ψ[i+1] = 0.0
        end
    end
    return ψ
end

"""
    population_excited_psi(ψ::Vector{ComplexF64}, k::Int, N::Int)

Compute ⟨ψ|σ⁺σ⁻|ψ⟩ = probability that qubit k is in |1⟩ (excited state).

This equals: Σᵢ |ψᵢ|² for all basis states i where bit k is SET.
Used for computing quantum jump probabilities in MCWF.
"""
function population_excited_psi(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)
    result = 0.0

    @inbounds for i in 0:(dim-1)
        if (i & mask) != 0  # Qubit k is in |1⟩
            result += abs2(ψ[i+1])
        end
    end
    return result
end

"""
    population_ground_psi(ψ::Vector{ComplexF64}, k::Int, N::Int)

Compute ⟨ψ|σ⁻σ⁺|ψ⟩ = probability that qubit k is in |0⟩ (ground state).
"""
function population_ground_psi(ψ::Vector{ComplexF64}, k::Int, N::Int)
    return 1.0 - population_excited_psi(ψ, k, N)
end

# ==============================================================================
# LINDBLAD DISSIPATOR - DENSITY MATRIX
#
# For the Lindblad dissipator D[L]ρ = L ρ L† - ½{L†L, ρ}, with L = σ⁻:
#   D[σ⁻]ρ = σ⁻ ρ σ⁺ - ½{σ⁺σ⁻, ρ}
#
# In the computational basis, σ⁺σ⁻ = |1⟩⟨1| (projector onto excited state).
#
# MATRIX ELEMENT RULES for density matrix ρᵢⱼ:
# - If bit_k(i)=1 AND bit_k(j)=1:
#     ρ_{flip(i),flip(j)} += γdt × ρᵢⱼ   (σ⁻ρσ⁺ term: population transfer)
#     ρᵢⱼ *= (1 - γdt)                    (decay from |1⟩|1⟩ block)
# - If bit_k(i)=1 XOR bit_k(j)=1 (coherence between |0⟩ and |1⟩):
#     ρᵢⱼ *= (1 - ½γdt)                   (dephasing)
# - If bit_k(i)=0 AND bit_k(j)=0:
#     No change (ground state block is stable under decay)
# ==============================================================================

"""
    apply_sigma_minus_dissipator_rho!(ρ::Matrix{ComplexF64}, k::Int, γdt::Float64, N::Int)

Apply σ⁻ dissipator to density matrix using BITWISE operations.
Implements: D[σ⁻]ρ = σ⁻ ρ σ⁺ - ½{σ⁺σ⁻, ρ}

This is a first-order Euler step of the Lindblad master equation.
Parameter γdt = γ × dt where γ is the decay rate and dt is the time step.
"""
function apply_sigma_minus_dissipator_rho!(ρ::Matrix{ComplexF64}, k::Int, γdt::Float64, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)

    @inbounds for j in 0:(dim-1), i in 0:(dim-1)
        bi = (i >> (k-1)) & 1  # Extract bit k from row index
        bj = (j >> (k-1)) & 1  # Extract bit k from column index

        if bi == 1 && bj == 1
            # Both |1⟩: population transfer |1⟩|1⟩→|0⟩|0⟩ + decay
            i0 = xor(i, mask)  # Flip bit k to 0
            j0 = xor(j, mask)
            ρ[i0+1, j0+1] += γdt * ρ[i+1, j+1]  # σ⁻ρσ⁺ term
            ρ[i+1, j+1] *= (1 - γdt)             # Decay of |1⟩|1⟩ block
        elseif bi == 1 || bj == 1
            # Coherence |0⟩⟨1| or |1⟩⟨0|: dephasing
            ρ[i+1, j+1] *= (1 - 0.5γdt)
        end
        # bi=0 && bj=0: ground state block unchanged under σ⁻ decay
    end
    return ρ
end

# Export Lindblad operators
export apply_sigma_minus_psi!, apply_sigma_plus_psi!
export population_excited_psi, population_ground_psi
export apply_sigma_minus_dissipator_rho!

# ==============================================================================
# TWO-QUBIT GATES - PURE STATE
# ==============================================================================

"""
    apply_cnot_psi!(ψ::Vector{ComplexF64}, control::Int, target::Int, N::Int)

Apply CNOT gate: flip target if control is |1⟩.
"""
function apply_cnot_psi!(ψ::Vector{ComplexF64}, control::Int, target::Int, N::Int)
    dim = 1 << N
    target_step = 1 << (target - 1)

    @inbounds for i in 0:(dim-1)
        # Only act if control bit is 1 and target bit is 0
        if ((i >> (control-1)) & 1 == 1) && ((i >> (target-1)) & 1 == 0)
            j = i + target_step  # flip target
            ψ[i+1], ψ[j+1] = ψ[j+1], ψ[i+1]
        end
    end
    return ψ
end

"""
    apply_cz_psi!(ψ::Vector{ComplexF64}, i::Int, j::Int, N::Int)

Apply CZ gate: apply -1 phase when both qubits are |1⟩.
"""
function apply_cz_psi!(ψ::Vector{ComplexF64}, qi::Int, qj::Int, N::Int)
    dim = 1 << N
    # Branchless: sign = 1 when either bit is 0, sign = -1 when both bits are 1
    # bi & bj = 1 only when both are 1, so: 1 - 2*(bi & bj) gives +1 or -1
    @inbounds @simd for idx in 0:(dim-1)
        bi = (idx >> (qi-1)) & 1
        bj = (idx >> (qj-1)) & 1
        sign = 1 - 2 * (bi & bj)
        ψ[idx+1] *= sign
    end
    return ψ
end

"""
    apply_cry_psi!(ψ::Vector{ComplexF64}, control::Int, target::Int, θ::Float64, N::Int)

Apply Controlled-Ry (CRy) gate: apply Ry(θ) to target if control is |1⟩.
Matrix-free bitwise implementation.

CRy = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Ry(θ)

When control is |0⟩: do nothing
When control is |1⟩: apply Ry(θ) = |cos(θ/2)  -sin(θ/2)|
                                   |sin(θ/2)   cos(θ/2)|
"""
function apply_cry_psi!(ψ::Vector{ComplexF64}, control::Int, target::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)

    dim = 1 << N
    target_step = 1 << (target - 1)

    @inbounds for i in 0:(dim-1)
        # Only act if control bit is 1 and target bit is 0
        if ((i >> (control-1)) & 1 == 1) && ((i >> (target-1)) & 1 == 0)
            j = i + target_step  # j has target bit = 1

            # Apply Ry rotation to the (i, j) pair
            ψ_i = ψ[i+1]  # |...control=1...target=0...⟩
            ψ_j = ψ[j+1]  # |...control=1...target=1...⟩

            # Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            # Ry(θ)|1⟩ = -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
            ψ[i+1] = c * ψ_i - s * ψ_j
            ψ[j+1] = s * ψ_i + c * ψ_j
        end
    end
    return ψ
end

"""
    apply_cry_rho!(ρ::Matrix{ComplexF64}, control::Int, target::Int, θ::Float64, N::Int)

Apply CRy to density matrix: ρ' = CRy(θ) ρ CRy(θ)†
Since CRy is Hermitian-parametrized (real Ry), CRy† = CRy(-θ).
"""
function apply_cry_rho!(ρ::Matrix{ComplexF64}, control::Int, target::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)

    dim = 1 << N
    target_step = 1 << (target - 1)

    # Apply CRy to rows (left multiply by CRy)
    @inbounds for i in 0:(dim-1)
        if ((i >> (control-1)) & 1 == 1) && ((i >> (target-1)) & 1 == 0)
            j = i + target_step
            for col in 1:dim
                ρ_i = ρ[i+1, col]
                ρ_j = ρ[j+1, col]
                ρ[i+1, col] = c * ρ_i - s * ρ_j
                ρ[j+1, col] = s * ρ_i + c * ρ_j
            end
        end
    end

    # Apply CRy† to columns (right multiply by CRy†)
    # CRy† has c and -s (since Ry(-θ) = Ry(θ)†)
    @inbounds for i in 0:(dim-1)
        if ((i >> (control-1)) & 1 == 1) && ((i >> (target-1)) & 1 == 0)
            j = i + target_step
            for row in 1:dim
                ρ_i = ρ[row, i+1]
                ρ_j = ρ[row, j+1]
                ρ[row, i+1] = c * ρ_i + s * ρ_j
                ρ[row, j+1] = -s * ρ_i + c * ρ_j
            end
        end
    end
    return ρ
end

"""
    apply_rxx_psi!(ψ::Vector{ComplexF64}, i::Int, j::Int, θ::Float64, N::Int)

Apply Rxx(θ) = exp(-iθXX/2) to qubits i and j.

Matrix form (in computational basis |00⟩, |01⟩, |10⟩, |11⟩):
| cos(θ/2)     0          0       -i sin(θ/2) |
|    0      cos(θ/2)  -i sin(θ/2)     0       |
|    0     -i sin(θ/2)  cos(θ/2)     0       |
|-i sin(θ/2)   0          0        cos(θ/2)  |
"""
function apply_rxx_psi!(ψ::Vector{ComplexF64}, qi::Int, qj::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)
    is = -im * s

    dim = 1 << N
    step_i = 1 << (qi - 1)
    step_j = 1 << (qj - 1)

    # Process each group of 4 states sharing the same "other" qubits
    @inbounds for base in 0:(dim-1)
        # Skip if qi or qj bit is 1 (we'll handle from the 00 case)
        if ((base >> (qi-1)) & 1 != 0) || ((base >> (qj-1)) & 1 != 0)
            continue
        end

        # 4 states in this block
        i00 = base
        i01 = base + step_j
        i10 = base + step_i
        i11 = base + step_i + step_j

        ψ00 = ψ[i00+1]
        ψ01 = ψ[i01+1]
        ψ10 = ψ[i10+1]
        ψ11 = ψ[i11+1]

        # Apply Rxx
        ψ[i00+1] = c * ψ00 + is * ψ11
        ψ[i01+1] = c * ψ01 + is * ψ10
        ψ[i10+1] = is * ψ01 + c * ψ10
        ψ[i11+1] = is * ψ00 + c * ψ11
    end
    return ψ
end

"""
    apply_ryy_psi!(ψ::Vector{ComplexF64}, i::Int, j::Int, θ::Float64, N::Int)

Apply Ryy(θ) = exp(-iθYY/2) to qubits i and j.

Matrix form:
| cos(θ/2)     0          0       i sin(θ/2) |
|    0      cos(θ/2)  -i sin(θ/2)    0       |
|    0     -i sin(θ/2)  cos(θ/2)     0       |
| i sin(θ/2)   0          0        cos(θ/2)  |
"""
function apply_ryy_psi!(ψ::Vector{ComplexF64}, qi::Int, qj::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)
    is = -im * s
    is_neg = im * s

    dim = 1 << N
    step_i = 1 << (qi - 1)
    step_j = 1 << (qj - 1)

    @inbounds for base in 0:(dim-1)
        if ((base >> (qi-1)) & 1 != 0) || ((base >> (qj-1)) & 1 != 0)
            continue
        end

        i00 = base
        i01 = base + step_j
        i10 = base + step_i
        i11 = base + step_i + step_j

        ψ00 = ψ[i00+1]
        ψ01 = ψ[i01+1]
        ψ10 = ψ[i10+1]
        ψ11 = ψ[i11+1]

        # Apply Ryy
        ψ[i00+1] = c * ψ00 + is_neg * ψ11
        ψ[i01+1] = c * ψ01 + is * ψ10
        ψ[i10+1] = is * ψ01 + c * ψ10
        ψ[i11+1] = is_neg * ψ00 + c * ψ11
    end
    return ψ
end

"""
    apply_rzz_psi!(ψ::Vector{ComplexF64}, i::Int, j::Int, θ::Float64, N::Int)

Apply Rzz(θ) = exp(-iθZZ/2) to qubits i and j.

Diagonal gate:
|00⟩ → e^{-iθ/2} |00⟩
|01⟩ → e^{+iθ/2} |01⟩
|10⟩ → e^{+iθ/2} |10⟩
|11⟩ → e^{-iθ/2} |11⟩
"""
function apply_rzz_psi!(ψ::Vector{ComplexF64}, qi::Int, qj::Int, θ::Float64, N::Int)
    phase_same = exp(-im * θ / 2)   # bits equal (00 or 11)
    phase_diff = exp(im * θ / 2)    # bits different (01 or 10)

    dim = 1 << N

    @inbounds for idx in 0:(dim-1)
        bit_i = (idx >> (qi-1)) & 1
        bit_j = (idx >> (qj-1)) & 1
        if bit_i == bit_j
            ψ[idx+1] *= phase_same
        else
            ψ[idx+1] *= phase_diff
        end
    end
    return ψ
end

# ==============================================================================
# TWO-QUBIT GATES - DENSITY MATRIX
# ==============================================================================

"""
    apply_cnot_rho!(ρ::Matrix{ComplexF64}, control::Int, target::Int, N::Int)

Apply CNOT to density matrix.
"""
function apply_cnot_rho!(ρ::Matrix{ComplexF64}, control::Int, target::Int, N::Int)
    # CNOT is its own inverse, so CNOT† = CNOT
    dim = 1 << N
    target_step = 1 << (target - 1)

    # Apply to rows
    @inbounds for i in 0:(dim-1)
        if ((i >> (control-1)) & 1 == 1) && ((i >> (target-1)) & 1 == 0)
            j = i + target_step
            for col in 1:dim
                ρ[i+1, col], ρ[j+1, col] = ρ[j+1, col], ρ[i+1, col]
            end
        end
    end

    # Apply to columns
    @inbounds for j in 0:(dim-1)
        if ((j >> (control-1)) & 1 == 1) && ((j >> (target-1)) & 1 == 0)
            jj = j + target_step
            for row in 1:dim
                ρ[row, j+1], ρ[row, jj+1] = ρ[row, jj+1], ρ[row, j+1]
            end
        end
    end

    return ρ
end

"""
    apply_cz_rho!(ρ::Matrix{ComplexF64}, i::Int, j::Int, N::Int)

Apply CZ to density matrix.
"""
function apply_cz_rho!(ρ::Matrix{ComplexF64}, qi::Int, qj::Int, N::Int)
    dim = 1 << N

    # CZ is diagonal with ±1 entries
    @inbounds for i in 0:(dim-1)
        sign_i = (((i >> (qi-1)) & 1 == 1) && ((i >> (qj-1)) & 1 == 1)) ? -1 : 1
        for j in 0:(dim-1)
            sign_j = (((j >> (qi-1)) & 1 == 1) && ((j >> (qj-1)) & 1 == 1)) ? -1 : 1
            ρ[i+1, j+1] *= sign_i * sign_j
        end
    end
    return ρ
end

"""
    apply_rxx_rho!(ρ::Matrix{ComplexF64}, i::Int, j::Int, θ::Float64, N::Int)

Apply Rxx(θ) to density matrix.
"""
function apply_rxx_rho!(ρ::Matrix{ComplexF64}, qi::Int, qj::Int, θ::Float64, N::Int)
    # Build full unitary and apply: ρ' = U ρ U†
    # For now, use matrix multiplication approach
    dim = 1 << N

    # Build Rxx matrix for 2 qubits
    c = cos(θ / 2)
    s = sin(θ / 2)
    is = -im * s

    # Apply to a copy of ρ via the pure state gate applied to rows and columns
    step_i = 1 << (qi - 1)
    step_j = 1 << (qj - 1)

    # Apply U to rows
    ρ_temp = copy(ρ)
    @inbounds for base in 0:(dim-1)
        if ((base >> (qi-1)) & 1 != 0) || ((base >> (qj-1)) & 1 != 0)
            continue
        end

        i00 = base
        i01 = base + step_j
        i10 = base + step_i
        i11 = base + step_i + step_j

        for col in 1:dim
            ρ00 = ρ[i00+1, col]
            ρ01 = ρ[i01+1, col]
            ρ10 = ρ[i10+1, col]
            ρ11 = ρ[i11+1, col]

            ρ_temp[i00+1, col] = c * ρ00 + is * ρ11
            ρ_temp[i01+1, col] = c * ρ01 + is * ρ10
            ρ_temp[i10+1, col] = is * ρ01 + c * ρ10
            ρ_temp[i11+1, col] = is * ρ00 + c * ρ11
        end
    end

    # Apply U† to columns
    is_dag = conj(is)
    @inbounds for base in 0:(dim-1)
        if ((base >> (qi-1)) & 1 != 0) || ((base >> (qj-1)) & 1 != 0)
            continue
        end

        j00 = base
        j01 = base + step_j
        j10 = base + step_i
        j11 = base + step_i + step_j

        for row in 1:dim
            ρ00 = ρ_temp[row, j00+1]
            ρ01 = ρ_temp[row, j01+1]
            ρ10 = ρ_temp[row, j10+1]
            ρ11 = ρ_temp[row, j11+1]

            ρ[row, j00+1] = c * ρ00 + is_dag * ρ11
            ρ[row, j01+1] = c * ρ01 + is_dag * ρ10
            ρ[row, j10+1] = is_dag * ρ01 + c * ρ10
            ρ[row, j11+1] = is_dag * ρ00 + c * ρ11
        end
    end

    return ρ
end

"""
    apply_ryy_rho!(ρ::Matrix{ComplexF64}, i::Int, j::Int, θ::Float64, N::Int)

Apply Ryy(θ) to density matrix.
"""
function apply_ryy_rho!(ρ::Matrix{ComplexF64}, qi::Int, qj::Int, θ::Float64, N::Int)
    c = cos(θ / 2)
    s = sin(θ / 2)
    is = -im * s
    is_neg = im * s

    dim = 1 << N
    step_i = 1 << (qi - 1)
    step_j = 1 << (qj - 1)

    ρ_temp = copy(ρ)

    # Apply U to rows
    @inbounds for base in 0:(dim-1)
        if ((base >> (qi-1)) & 1 != 0) || ((base >> (qj-1)) & 1 != 0)
            continue
        end

        i00 = base
        i01 = base + step_j
        i10 = base + step_i
        i11 = base + step_i + step_j

        for col in 1:dim
            ρ00 = ρ[i00+1, col]
            ρ01 = ρ[i01+1, col]
            ρ10 = ρ[i10+1, col]
            ρ11 = ρ[i11+1, col]

            ρ_temp[i00+1, col] = c * ρ00 + is_neg * ρ11
            ρ_temp[i01+1, col] = c * ρ01 + is * ρ10
            ρ_temp[i10+1, col] = is * ρ01 + c * ρ10
            ρ_temp[i11+1, col] = is_neg * ρ00 + c * ρ11
        end
    end

    # Apply U† to columns
    is_dag = conj(is)
    is_neg_dag = conj(is_neg)

    @inbounds for base in 0:(dim-1)
        if ((base >> (qi-1)) & 1 != 0) || ((base >> (qj-1)) & 1 != 0)
            continue
        end

        j00 = base
        j01 = base + step_j
        j10 = base + step_i
        j11 = base + step_i + step_j

        for row in 1:dim
            ρ00 = ρ_temp[row, j00+1]
            ρ01 = ρ_temp[row, j01+1]
            ρ10 = ρ_temp[row, j10+1]
            ρ11 = ρ_temp[row, j11+1]

            ρ[row, j00+1] = c * ρ00 + is_neg_dag * ρ11
            ρ[row, j01+1] = c * ρ01 + is_dag * ρ10
            ρ[row, j10+1] = is_dag * ρ01 + c * ρ10
            ρ[row, j11+1] = is_neg_dag * ρ00 + c * ρ11
        end
    end

    return ρ
end

"""
    apply_rzz_rho!(ρ::Matrix{ComplexF64}, i::Int, j::Int, θ::Float64, N::Int)

Apply Rzz(θ) to density matrix.
"""
function apply_rzz_rho!(ρ::Matrix{ComplexF64}, qi::Int, qj::Int, θ::Float64, N::Int)
    phase_same = exp(-im * θ / 2)
    phase_diff = exp(im * θ / 2)

    dim = 1 << N

    @inbounds for i in 0:(dim-1)
        bit_i1 = (i >> (qi-1)) & 1
        bit_i2 = (i >> (qj-1)) & 1
        phase_i = (bit_i1 == bit_i2) ? phase_same : phase_diff

        for j in 0:(dim-1)
            bit_j1 = (j >> (qi-1)) & 1
            bit_j2 = (j >> (qj-1)) & 1
            phase_j = (bit_j1 == bit_j2) ? conj(phase_same) : conj(phase_diff)

            ρ[i+1, j+1] *= phase_i * phase_j
        end
    end
    return ρ
end

end # module

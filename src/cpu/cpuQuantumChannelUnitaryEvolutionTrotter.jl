# Date: 2026
#
#=
================================================================================
    cpuQuantumChannelUnitaryEvolutionTrotter.jl - Matrix-Free Trotter (CPU)
================================================================================

OVERVIEW
--------
Matrix-free Trotter-Suzuki time evolution using bitwise gate application.

KEY INSIGHT
-----------
Instead of exp(-iHt)|ψ⟩ with a 2^N × 2^N matrix, we decompose:
    H = Σₖ Hₖ  (local terms)
    exp(-iHt) ≈ Πₖ exp(-iHₖ dt)  (Trotter)

Each exp(-iHₖ dt) is a small 2×2 or 4×4 local unitary that can be applied
directly to the state vector using bitwise index manipulation.

COMPLEXITY
----------
- Matrix-based: O(2^N × 2^N) per step
- Bitwise: O(2^N × G) where G = number of gates << 2^N

EXPORTS
-------
- FastTrotterGate struct
- precompute_trotter_gates_bitwise_cpu(params, dt)
- apply_fast_trotter_step_cpu!(psi, gates, N)
- apply_fast_trotter_gates_to_matrix_rows_cpu!(M, gates, N)
- evolve_trotter_rho_cpu!(rho, gates, N, n_substeps)
- evolve_trotter_psi_cpu!(psi, gates, N, n_substeps)

================================================================================
=#

module CPUQuantumChannelUnitaryEvolutionTrotter

using LinearAlgebra
using Base.Threads

export FastTrotterGate
export precompute_trotter_gates_bitwise_cpu
export apply_one_qubit_gate_bitwise_cpu!, apply_two_qubit_gate_bitwise_cpu!
export apply_fast_trotter_step_cpu!
export apply_fast_trotter_gates_to_matrix_rows_cpu!
export evolve_trotter_rho_cpu!, evolve_trotter_psi_cpu!
export solve_trotter_bitwise_cpu

# ==============================================================================
# TROTTER GATE STRUCTURE
# ==============================================================================

"""
    FastTrotterGate

Represents a local unitary gate for matrix-free Trotter evolution.
- indices: Qubit indices (1-indexed), length 1 or 2
- U: Local unitary matrix (2×2 for 1-qubit, 4×4 for 2-qubit)
"""
struct FastTrotterGate
    indices::Vector{Int}
    U::Matrix{ComplexF64}
end

# ==============================================================================
# GATE APPLICATION: PURE STATES (VECTORS)
# ==============================================================================

"""
    apply_one_qubit_gate_bitwise_cpu!(psi, U, qubit, N)

Apply a 1-qubit gate U to qubit `qubit` in an N-qubit state vector.
Modifies psi in-place. Matrix-free bitwise implementation.
"""
function apply_one_qubit_gate_bitwise_cpu!(psi::AbstractVector{ComplexF64}, U::Matrix{ComplexF64}, qubit::Int, N::Int)
    dim = 1 << N
    step = 1 << (qubit - 1)

    u00, u01 = U[1,1], U[1,2]
    u10, u11 = U[2,1], U[2,2]

    # Sequential loop over state indices (avoids nested threading issues)
    for base_idx in 0:(dim - 1)
        if (base_idx & step) == 0
            idx0 = base_idx + 1
            idx1 = base_idx + step + 1

            @inbounds begin
                a0, a1 = psi[idx0], psi[idx1]
                psi[idx0] = u00 * a0 + u01 * a1
                psi[idx1] = u10 * a0 + u11 * a1
            end
        end
    end

    return psi
end

"""
    apply_two_qubit_gate_bitwise_cpu!(psi, U, q1, q2, N)

Apply a 2-qubit gate U to qubits (q1, q2) in an N-qubit state vector.
Modifies psi in-place. Matrix-free bitwise implementation.
"""
function apply_two_qubit_gate_bitwise_cpu!(psi::AbstractVector{ComplexF64}, U::Matrix{ComplexF64}, q1::Int, q2::Int, N::Int)
    dim = 1 << N

    if q1 > q2
        q1, q2 = q2, q1
    end

    step1 = 1 << (q1 - 1)
    step2 = 1 << (q2 - 1)

    U11, U12, U13, U14 = U[1,1], U[1,2], U[1,3], U[1,4]
    U21, U22, U23, U24 = U[2,1], U[2,2], U[2,3], U[2,4]
    U31, U32, U33, U34 = U[3,1], U[3,2], U[3,3], U[3,4]
    U41, U42, U43, U44 = U[4,1], U[4,2], U[4,3], U[4,4]

    # Sequential loop over state indices (avoids nested threading issues)
    for base_idx in 0:(dim - 1)
        if (base_idx & step1) == 0 && (base_idx & step2) == 0
            idx00 = base_idx + 1
            idx01 = base_idx + step1 + 1
            idx10 = base_idx + step2 + 1
            idx11 = base_idx + step1 + step2 + 1

            @inbounds begin
                a00, a01, a10, a11 = psi[idx00], psi[idx01], psi[idx10], psi[idx11]

                psi[idx00] = U11*a00 + U12*a01 + U13*a10 + U14*a11
                psi[idx01] = U21*a00 + U22*a01 + U23*a10 + U24*a11
                psi[idx10] = U31*a00 + U32*a01 + U33*a10 + U34*a11
                psi[idx11] = U41*a00 + U42*a01 + U43*a10 + U44*a11
            end
        end
    end

    return psi
end

"""
    apply_fast_trotter_step_cpu!(psi, gates, N)

Apply a sequence of FastTrotterGates to state vector psi.
Gates are applied in order (first-order Trotter).
"""
function apply_fast_trotter_step_cpu!(psi::AbstractVector{ComplexF64}, gates::Vector{FastTrotterGate}, N::Int)
    @inbounds for gate in gates
        nq = length(gate.indices)
        if nq == 1
            apply_one_qubit_gate_bitwise_cpu!(psi, gate.U, gate.indices[1], N)
        elseif nq == 2
            apply_two_qubit_gate_bitwise_cpu!(psi, gate.U, gate.indices[1], gate.indices[2], N)
        else
            error("Gates with more than 2 qubits not supported")
        end
    end
    return psi
end

# ==============================================================================
# GATE APPLICATION: DENSITY MATRICES (ROWS)
# ==============================================================================

"""
    apply_fast_trotter_gates_to_matrix_rows_cpu!(M, gates, N)

Apply Trotter gates to the ROWS of matrix M (Right Multiplication: M -> M * U^T).
Optimized for column-major memory layout.
"""
function apply_fast_trotter_gates_to_matrix_rows_cpu!(M::Matrix{ComplexF64}, gates::Vector{FastTrotterGate}, N::Int)
    dim = size(M, 2)
    dim_rows = size(M, 1)

    @inbounds for gate in gates
        if length(gate.indices) == 1
            qubit = gate.indices[1]
            step = 1 << (qubit - 1)
            U = gate.U
            u00, u01 = U[1,1], U[1,2]
            u10, u11 = U[2,1], U[2,2]

            for k in 0:(dim - 1)
                if (k & step) == 0
                    idx0 = k + 1
                    idx1 = k + step + 1

                    @simd for i in 1:dim_rows
                        a0 = M[i, idx0]
                        a1 = M[i, idx1]
                        M[i, idx0] = u00 * a0 + u01 * a1
                        M[i, idx1] = u10 * a0 + u11 * a1
                    end
                end
            end
        else
            _apply_two_qubit_gate_to_rows_cpu!(M, gate, dim_rows)
        end
    end
    return M
end

function _apply_two_qubit_gate_to_rows_cpu!(M, gate, dim_rows)
    q1, q2 = gate.indices[1], gate.indices[2]
    bit1, bit2 = q1 - 1, q2 - 1
    step1, step2 = 1 << bit1, 1 << bit2
    dim = size(M, 2)
    U = gate.U

    U11, U12, U13, U14 = U[1,1], U[1,2], U[1,3], U[1,4]
    U21, U22, U23, U24 = U[2,1], U[2,2], U[2,3], U[2,4]
    U31, U32, U33, U34 = U[3,1], U[3,2], U[3,3], U[3,4]
    U41, U42, U43, U44 = U[4,1], U[4,2], U[4,3], U[4,4]

    @inbounds for k in 0:(dim - 1)
        if ((k >> bit1) & 1) == 0 && ((k >> bit2) & 1) == 0
            c00 = k + 1
            c10 = k + step1 + 1
            c01 = k + step2 + 1
            c11 = k + step1 + step2 + 1

            @simd for i in 1:dim_rows
                v00 = M[i, c00]
                v01 = M[i, c10]
                v10 = M[i, c01]
                v11 = M[i, c11]

                M[i, c00] = U11*v00 + U12*v01 + U13*v10 + U14*v11
                M[i, c10] = U21*v00 + U22*v01 + U23*v10 + U24*v11
                M[i, c01] = U31*v00 + U32*v01 + U33*v10 + U34*v11
                M[i, c11] = U41*v00 + U42*v01 + U43*v10 + U44*v11
            end
        end
    end
end

# ==============================================================================
# HIGH-LEVEL EVOLUTION FUNCTIONS
# ==============================================================================

"""
    evolve_trotter_psi_cpu!(psi, gates, N, n_substeps=1)

In-place Trotter evolution for pure state: |ψ⟩ → U|ψ⟩
Applies all gates n_substeps times.
"""
function evolve_trotter_psi_cpu!(psi::AbstractVector{ComplexF64}, gates::Vector{FastTrotterGate},
                                  N::Int, n_substeps::Int=1)
    for _ in 1:n_substeps
        apply_fast_trotter_step_cpu!(psi, gates, N)
    end
    return psi
end

"""
    evolve_trotter_rho_cpu!(rho, gates, N, n_substeps=1)

In-place Trotter evolution for density matrix: ρ → U ρ U†
Uses column-row strategy:
1. Apply U to each column (left multiply)
2. Apply U† to each row (right multiply)
"""
function evolve_trotter_rho_cpu!(rho::Matrix{ComplexF64}, gates::Vector{FastTrotterGate},
                                  N::Int, n_substeps::Int=1)
    dim = size(rho, 1)

    # Create conjugated gates for U†
    conj_gates = [FastTrotterGate(g.indices, conj.(g.U)) for g in gates]

    for _ in 1:n_substeps
        # 1. Apply U to each column: ρ = U * ρ
        for j in 1:dim
            col = view(rho, :, j)
            apply_fast_trotter_step_cpu!(col, gates, N)
        end

        # 2. Apply U† to each row: ρ = ρ * U†
        apply_fast_trotter_gates_to_matrix_rows_cpu!(rho, conj_gates, N)
    end

    return rho
end

"""
    solve_trotter_bitwise_cpu(psi0, gates, N, t_list, dt)

Matrix-free bitwise Trotter evolution over time.
Returns states at each time point.
"""
function solve_trotter_bitwise_cpu(psi0::Vector{ComplexF64}, gates::Vector{FastTrotterGate},
                                    N::Int, t_list::Vector{Float64}, dt::Float64)
    states = Vector{Vector{ComplexF64}}(undef, length(t_list))
    states[1] = copy(psi0)

    psi = copy(psi0)
    t_current = t_list[1]

    for i in 2:length(t_list)
        t_target = t_list[i]
        while t_current < t_target - dt/2
            apply_fast_trotter_step_cpu!(psi, gates, N)
            t_current += dt
        end
        states[i] = copy(psi)
    end

    return t_list, states
end

# ==============================================================================
# GATE PRECOMPUTATION
# ==============================================================================

"""
    precompute_trotter_gates_bitwise_cpu(params, dt) -> Vector{FastTrotterGate}

Computes exp(-iH_local*dt) for each 2-qubit coupling and 1-qubit field.
"""
function precompute_trotter_gates_bitwise_cpu(params, dt::Float64)
    gates = FastTrotterGate[]

    # Pauli matrices
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]

    # 2-qubit gates for each coupling (Heisenberg: exp(-i*(Jx XX + Jy YY + Jz ZZ)*dt))
    for coup in vcat(params.x_bonds, params.y_bonds)
        Jx, Jy, Jz = coup.Jxx, coup.Jyy, coup.Jzz

        XX = kron(σx, σx)
        YY = kron(σy, σy)
        ZZ = kron(σz, σz)
        H_local = Jx * XX + Jy * YY + Jz * ZZ

        U = exp(-im * H_local * dt)
        push!(gates, FastTrotterGate([coup.i, coup.j], U))
    end

    # 1-qubit gates for local fields
    for field in params.fields
        hx, hy, hz = field.hx, field.hy, field.hz
        if abs(hx) > 1e-14 || abs(hy) > 1e-14 || abs(hz) > 1e-14
            H_local = hx * σx + hy * σy + hz * σz
            U = exp(-im * H_local * dt)
            push!(gates, FastTrotterGate([field.idx], U))
        end
    end

    return gates
end

end # module

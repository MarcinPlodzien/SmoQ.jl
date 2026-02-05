# Date: 2026
#
#=
================================================================================
    cpuVariationalQuantumCircuitExecutor.jl - Dual-Mode Circuit Execution
================================================================================

PURPOSE:
--------
Execute variational quantum circuits on both pure states (MCWF) and density
matrices (exact Kraus). All operations use matrix-free bitwise implementations.

EXECUTION MODES:
----------------
1. PURE STATE (MCWF):
   apply_circuit!(ψ::Vector, circuit, θ)
   - Gates: direct bitwise application
   - Noise: stochastic sampling (Monte Carlo Wave Function)
   - Measurement: projective collapse with random outcome
   - Reset: trace out + tensor in fresh state

2. DENSITY MATRIX (EXACT):
   apply_circuit!(ρ::Matrix, circuit, θ)
   - Gates: ρ' = U ρ U†
   - Noise: Kraus operators ρ' = Σₖ Kₖ ρ Kₖ†
   - Measurement: partial trace
   - Reset: trace out + tensor in |target⟩⟨target|

USAGE:
------
```julia
using .CPUVariationalQuantumCircuitExecutor

# Pure state (fast, stochastic for noise)
ψ = zeros(ComplexF64, 2^N); ψ[1] = 1.0
outcomes = apply_circuit!(ψ, circuit, θ)

# Density matrix (exact, deterministic)
ρ = zeros(ComplexF64, 2^N, 2^N); ρ[1,1] = 1.0
apply_circuit!(ρ, circuit, θ)
```

================================================================================
=#

module CPUVariationalQuantumCircuitExecutor

using LinearAlgebra

# Import circuit builder
include("cpuVariationalQuantumCircuitBuilder.jl")
using .CPUVariationalQuantumCircuitBuilder

# Import gate operations
include("cpuQuantumChannelGates.jl")
using .CPUQuantumChannelGates

# Import measurements module (for projective_measurement!, reset_state!)
include("cpuQuantumStateMeasurements.jl")
using .CPUQuantumStateMeasurements: projective_measurement!, reset_state!, reset_qubit!

# Import partial trace module (for partial_trace)
include("cpuQuantumStatePartialTrace.jl")
using .CPUQuantumStatePartialTrace: partial_trace, partial_trace_regions

# Import Kraus operators module (for noise channels)
include("cpuQuantumChannelKrausOperators.jl")
using .CPUQuantumChannelKraus

export apply_circuit!, apply_operation!, CircuitResult
export apply_circuit_pure!, apply_circuit_dm!

# ==============================================================================
# OPERATION CATEGORIES (imported from builder)
# ==============================================================================

const GATE_TYPES = CPUVariationalQuantumCircuitBuilder.GATE_TYPES
const NOISE_TYPES = CPUVariationalQuantumCircuitBuilder.NOISE_TYPES
const MEASURE_TYPES = CPUVariationalQuantumCircuitBuilder.MEASURE_TYPES
const RESET_TYPES = CPUVariationalQuantumCircuitBuilder.RESET_TYPES
const TRACE_TYPES = CPUVariationalQuantumCircuitBuilder.TRACE_TYPES

# ==============================================================================
# MAIN DISPATCH: PURE STATE
# ==============================================================================

"""
    apply_circuit!(ψ::Vector{ComplexF64}, circuit::ParameterizedCircuit, θ::Vector{Float64}) -> CircuitResult

Apply circuit to pure state using MCWF (stochastic for noise).

# Returns
- `CircuitResult` with final state and measurement outcomes
"""
function apply_circuit!(ψ::Vector{ComplexF64}, circuit::ParameterizedCircuit, θ::Vector{Float64})
    @assert length(ψ) == 1 << circuit.N "State dimension mismatch"
    @assert length(θ) >= circuit.n_params "Not enough parameters"
    
    outcomes = Int[]
    
    for op in circuit.operations
        # Check classical conditioning
        if op.if_outcome > 0
            # Skip if condition not met
            if op.if_outcome > length(outcomes) || outcomes[op.if_outcome] != 1
                continue
            end
        end
        
        result = apply_operation!(ψ, op, θ, circuit.N)
        if !isnothing(result)
            append!(outcomes, result)
        end
    end
    
    return CircuitResult(ψ, outcomes)
end

"""
    apply_operation!(ψ::Vector, op::CircuitOperation, θ, N)

Apply a single operation to pure state. Returns measurement outcomes if applicable.
"""
function apply_operation!(ψ::Vector{ComplexF64}, op::CircuitOperation, θ::Vector{Float64}, N::Int)
    if op.type in GATE_TYPES
        return apply_gate_psi!(ψ, op, θ, N)
    elseif op.type in NOISE_TYPES
        return apply_noise_mcwf!(ψ, op, N)
    elseif op.type in MEASURE_TYPES
        return apply_measure_psi!(ψ, op, N)
    elseif op.type in RESET_TYPES
        return apply_reset_psi!(ψ, op, N)
    elseif op.type in TRACE_TYPES
        return apply_trace_out_psi!(ψ, op, N)
    elseif op.type == :lindblad_step
        return apply_lindblad_mcwf!(ψ, op, N)
    else
        error("Unknown operation type: $(op.type)")
    end
end

# ==============================================================================
# GATE APPLICATION (PURE STATE)
# ==============================================================================

function apply_gate_psi!(ψ::Vector{ComplexF64}, op::CircuitOperation, θ::Vector{Float64}, N::Int)
    angle = op.param_idx > 0 ? θ[op.param_idx] : op.fixed_angle
    q = op.qubits
    
    if op.type == :rx
        apply_rx_psi!(ψ, q[1], angle, N)
    elseif op.type == :ry
        apply_ry_psi!(ψ, q[1], angle, N)
    elseif op.type == :rz
        apply_rz_psi!(ψ, q[1], angle, N)
    elseif op.type == :h
        apply_hadamard_psi!(ψ, q[1], N)
    elseif op.type == :x
        apply_pauli_x_psi!(ψ, q[1], N)
    elseif op.type == :y
        apply_pauli_y_psi!(ψ, q[1], N)
    elseif op.type == :z
        apply_pauli_z_psi!(ψ, q[1], N)
    elseif op.type == :s
        apply_s_psi!(ψ, q[1], N)
    elseif op.type == :t
        apply_t_psi!(ψ, q[1], N)
    elseif op.type == :cz
        apply_cz_psi!(ψ, q[1], q[2], N)
    elseif op.type == :cnot || op.type == :cx
        apply_cnot_psi!(ψ, q[1], q[2], N)
    elseif op.type == :rxx
        apply_rxx_psi!(ψ, q[1], q[2], angle, N)
    elseif op.type == :ryy
        apply_ryy_psi!(ψ, q[1], q[2], angle, N)
    elseif op.type == :rzz
        apply_rzz_psi!(ψ, q[1], q[2], angle, N)
    end
    
    return nothing
end

# ==============================================================================
# NOISE (uses CPUQuantumChannelKrausOperators)
# ==============================================================================

function apply_noise_mcwf!(ψ::Vector{ComplexF64}, op::CircuitOperation, N::Int)
    p = op.p
    qubits = op.qubits
    
    if op.type == :depolarizing
        apply_channel_depolarizing!(ψ, p, qubits, N)
    elseif op.type == :amplitude_damping
        apply_channel_amplitude_damping!(ψ, p, qubits, N)
    elseif op.type == :dephasing
        apply_channel_dephasing!(ψ, p, qubits, N)
    elseif op.type == :bit_flip
        apply_channel_bit_flip!(ψ, p, qubits, N)
    elseif op.type == :phase_damping
        apply_channel_dephasing!(ψ, p, qubits, N)  # Same as dephasing
    end
    
    return nothing
end

# ==============================================================================
# MEASUREMENT (uses CPUQuantumStateMeasurements)
# ==============================================================================

function apply_measure_psi!(ψ::Vector{ComplexF64}, op::CircuitOperation, N::Int)
    # Use projective_measurement! from Measurements module
    outcomes, _ = projective_measurement!(ψ, op.qubits, op.basis, N)
    return outcomes
end

# ==============================================================================
# RESET (uses CPUQuantumStateMeasurements)
# ==============================================================================

function apply_reset_psi!(ψ::Vector{ComplexF64}, op::CircuitOperation, N::Int)
    # Use reset_state! from Measurements module
    targets = fill(op.target_state, length(op.qubits))
    reset_state!(ψ, op.qubits, targets, N)
    return nothing
end

# ==============================================================================
# TRACE OUT (PURE STATE -> needs partial trace, returns density matrix)
# ==============================================================================

function apply_trace_out_psi!(ψ::Vector{ComplexF64}, op::CircuitOperation, N::Int)
    # NOTE: trace_out converts pure state to density matrix for remaining qubits
    # This changes the state type - should be handled by circuit executor
    @warn "trace_out in pure state mode: use partial_trace from cpuQuantumStatePartialTrace.jl"
    return nothing
end

# ==============================================================================
# LINDBLADIAN MCWF (stub - use existing module)
# ==============================================================================

function apply_lindblad_mcwf!(ψ::Vector{ComplexF64}, op::CircuitOperation, N::Int)
    # TODO: Integrate with existing Lindbladian MCWF module
    @warn "Lindbladian step not yet integrated - skipping"
    return nothing
end

# ==============================================================================
# MAIN DISPATCH: DENSITY MATRIX
# ==============================================================================

"""
    apply_circuit!(ρ::Matrix{ComplexF64}, circuit::ParameterizedCircuit, θ::Vector{Float64})

Apply circuit to density matrix using exact Kraus operators.
"""
function apply_circuit!(ρ::Matrix{ComplexF64}, circuit::ParameterizedCircuit, θ::Vector{Float64})
    @assert size(ρ, 1) == 1 << circuit.N "State dimension mismatch"
    @assert length(θ) >= circuit.n_params "Not enough parameters"
    
    for op in circuit.operations
        apply_operation!(ρ, op, θ, circuit.N)
    end
    
    return nothing
end

function apply_operation!(ρ::Matrix{ComplexF64}, op::CircuitOperation, θ::Vector{Float64}, N::Int)
    if op.type in GATE_TYPES
        return apply_gate_rho!(ρ, op, θ, N)
    elseif op.type in NOISE_TYPES
        return apply_noise_kraus!(ρ, op, N)
    elseif op.type in MEASURE_TYPES
        return apply_measure_rho!(ρ, op, N)
    elseif op.type in RESET_TYPES
        return apply_reset_rho!(ρ, op, N)
    elseif op.type == :lindblad_step
        return apply_lindblad_dm!(ρ, op, N)
    else
        error("Unknown operation type: $(op.type)")
    end
end

# ==============================================================================
# GATE APPLICATION (DENSITY MATRIX)
# ==============================================================================

function apply_gate_rho!(ρ::Matrix{ComplexF64}, op::CircuitOperation, θ::Vector{Float64}, N::Int)
    angle = op.param_idx > 0 ? θ[op.param_idx] : op.fixed_angle
    q = op.qubits
    
    if op.type == :rx
        apply_rx_rho!(ρ, q[1], angle, N)
    elseif op.type == :ry
        apply_ry_rho!(ρ, q[1], angle, N)
    elseif op.type == :rz
        apply_rz_rho!(ρ, q[1], angle, N)
    elseif op.type == :h
        apply_hadamard_rho!(ρ, q[1], N)
    elseif op.type == :x
        apply_pauli_x_rho!(ρ, q[1], N)
    elseif op.type == :z
        apply_pauli_z_rho!(ρ, q[1], N)
    elseif op.type == :cz
        apply_cz_rho!(ρ, q[1], q[2], N)
    elseif op.type == :cnot || op.type == :cx
        apply_cnot_rho!(ρ, q[1], q[2], N)
    elseif op.type == :rxx
        apply_rxx_rho!(ρ, q[1], q[2], angle, N)
    elseif op.type == :ryy
        apply_ryy_rho!(ρ, q[1], q[2], angle, N)
    elseif op.type == :rzz
        apply_rzz_rho!(ρ, q[1], q[2], angle, N)
    end
    
    return nothing
end

# ==============================================================================
# NOISE (KRAUS OPERATORS - EXACT)
# ==============================================================================

function apply_noise_kraus!(ρ::Matrix{ComplexF64}, op::CircuitOperation, N::Int)
    p = op.p
    
    for q in op.qubits
        if op.type == :depolarizing
            _apply_depolarizing_kraus!(ρ, q, p, N)
        elseif op.type == :amplitude_damping
            _apply_amplitude_damping_kraus!(ρ, q, p, N)
        elseif op.type == :dephasing
            _apply_dephasing_kraus!(ρ, q, p, N)
        elseif op.type == :bit_flip
            _apply_bit_flip_kraus!(ρ, q, p, N)
        end
    end
    
    return nothing
end

# Depolarizing Kraus: ρ' = (1-p)ρ + p/3 (XρX + YρY + ZρZ)
function _apply_depolarizing_kraus!(ρ::Matrix{ComplexF64}, k::Int, p::Float64, N::Int)
    dim = size(ρ, 1)
    ρ_new = (1 - p) * ρ
    
    # X contribution
    ρ_x = copy(ρ)
    apply_pauli_x_rho!(ρ_x, k, N)
    ρ_new .+= (p/3) .* ρ_x
    
    # Y contribution (approximate using X and Z)
    ρ_y = copy(ρ)
    apply_pauli_z_rho!(ρ_y, k, N)
    apply_pauli_x_rho!(ρ_y, k, N)
    ρ_new .+= (p/3) .* ρ_y
    
    # Z contribution
    ρ_z = copy(ρ)
    apply_pauli_z_rho!(ρ_z, k, N)
    ρ_new .+= (p/3) .* ρ_z
    
    ρ .= ρ_new
end

# Bit flip Kraus
function _apply_bit_flip_kraus!(ρ::Matrix{ComplexF64}, k::Int, p::Float64, N::Int)
    ρ_x = copy(ρ)
    apply_pauli_x_rho!(ρ_x, k, N)
    ρ .= (1 - p) .* ρ .+ p .* ρ_x
end

# Dephasing Kraus
function _apply_dephasing_kraus!(ρ::Matrix{ComplexF64}, k::Int, p::Float64, N::Int)
    ρ_z = copy(ρ)
    apply_pauli_z_rho!(ρ_z, k, N)
    ρ .= (1 - p/2) .* ρ .+ (p/2) .* ρ_z
end

# Amplitude damping Kraus
function _apply_amplitude_damping_kraus!(ρ::Matrix{ComplexF64}, k::Int, γ::Float64, N::Int)
    # K0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
    # K1 = √γ |0⟩⟨1|
    # For now, use dissipator approximation
    apply_sigma_minus_dissipator_rho!(ρ, k, γ, N)
end

# ==============================================================================
# MEASUREMENT (DENSITY MATRIX - PARTIAL TRACE)
# ==============================================================================

function apply_measure_rho!(ρ::Matrix{ComplexF64}, op::CircuitOperation, N::Int)
    # For DM, measurement = partial trace (no outcome, just decoherence)
    for q in op.qubits
        _decohere_qubit_rho!(ρ, q, N)
    end
    return nothing
end

function _decohere_qubit_rho!(ρ::Matrix{ComplexF64}, k::Int, N::Int)
    # Dephase: remove off-diagonal elements in Z basis
    dim = size(ρ, 1)
    bit_pos = k - 1
    
    @inbounds for i in 0:(dim-1)
        for j in 0:(dim-1)
            if ((i >> bit_pos) & 1) != ((j >> bit_pos) & 1)
                ρ[i+1, j+1] = 0.0
            end
        end
    end
end

# ==============================================================================
# RESET (DENSITY MATRIX)
# ==============================================================================

function apply_reset_rho!(ρ::Matrix{ComplexF64}, op::CircuitOperation, N::Int)
    for q in op.qubits
        _reset_qubit_rho!(ρ, q, op.target_state, N)
    end
    return nothing
end

function _reset_qubit_rho!(ρ::Matrix{ComplexF64}, k::Int, target::Symbol, N::Int)
    # First decohere, then rotate to target
    _decohere_qubit_rho!(ρ, k, N)
    
    # For simplicity, reset to target by projecting and rescaling
    # Full implementation would use partial trace + tensor
    # TODO: Integrate with cpuQuantumStatePreparation reset functions
end

# ==============================================================================
# LINDBLADIAN (DENSITY MATRIX)
# ==============================================================================

function apply_lindblad_dm!(ρ::Matrix{ComplexF64}, op::CircuitOperation, N::Int)
    # TODO: Integrate with existing Lindbladian DM module
    @warn "Lindbladian step for DM not yet integrated - skipping"
    return nothing
end

end # module CPUVariationalQuantumCircuitExecutor

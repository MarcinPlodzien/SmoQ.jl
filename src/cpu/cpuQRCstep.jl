# Date: 2026
#
#=
================================================================================
    cpuQRCstep.jl - Modular QRC Step (CPU)
================================================================================
Implements a single step of the QRC protocol with 5 stages using MODULAR
building blocks from cpuQuantumStateCore and cpuQuantumGates.

  1. RESET: Reset input qubits to |0⟩ (via partial trace + collapse)
  2. ENCODE: Apply Ry(θ) gates to encode input values  
  3. TENSOR: Create tensor product of encoded input with reservoir
  4. EVOLVE: Apply unitary time evolution
  5. MEASURE: Extract observable expectation values

MODULAR DESIGN
--------------
Each stage uses general-purpose quantum operations:
- make_product_ket(:zero, L) → create |0⟩^⊗L
- apply_ry_psi!(ψ, k, θ, N) → encode u via rotation
- tensor_product_ket(ψ_A, ψ_B, N_A, N_B) → combine states
- Trotter/exact evolution → time dynamics
- Measurement functions → extract observables

Naming convention:
  - _rho suffix: Density matrix (DM) mode
  - _psi suffix: Pure state (MCWF) mode
  - _cpu suffix: CPU implementation

================================================================================
=#

module CPUQRCstep

using LinearAlgebra

# Import from parent modules (loaded by master script before this file)
using ..CPUQuantumStatePreparation: tensor_product_ket, tensor_product_ket!, tensor_product_rho, 
    make_product_ket, make_product_rho, make_ket, make_rho
using ..CPUQuantumChannelGates: apply_ry_psi!, apply_ry_rho!
using ..CPUQuantumStatePartialTrace: partial_trace


export qrc_step_modular_rho!, qrc_step_modular_psi!
export qrc_reset_psi, qrc_reset_psi!, qrc_reset_rho
export encode_input_ry_psi, encode_input_ry_psi!, encode_input_ry_rho


# ==============================================================================
# MODULAR INPUT ENCODING
# ==============================================================================

"""
    encode_input_ry_psi(u_window::Vector{Float64}, L::Int) -> Vector{ComplexF64}

Encode input values into L qubits using modular building blocks:
  1. Create |0⟩^⊗L using make_product_ket
  2. Apply Ry(θₖ) to each qubit k using apply_ry_psi!

Maps u ∈ [0,1] to rotation angle θ = π*u:
  - u = 0   → θ = 0   → |0⟩
  - u = 0.5 → θ = π/2 → (|0⟩+|1⟩)/√2 = |+⟩
  - u = 1   → θ = π   → |1⟩

This is mathematically equivalent to cos(θ/2)|0⟩ + sin(θ/2)|1⟩
"""
function encode_input_ry_psi(u_window::Vector{Float64}, L::Int)
    @assert length(u_window) >= L "Input window must have at least L values"
    
    # STEP 1: Create |0⟩^⊗L using modular state preparation
    ψ = make_product_ket(fill(:zero, L))
    
    # STEP 2: Apply Ry(π*u) to each qubit using modular gate
    for k in 1:L
        θ = π * u_window[k]
        apply_ry_psi!(ψ, k, θ, L)
    end
    
    return ψ
end

# 1-argument overload for convenience - derives L from vector length
encode_input_ry_psi(u_window::Vector{Float64}) = encode_input_ry_psi(u_window, length(u_window))

"""
    encode_input_ry_psi!(ψ_out, u_window, L) -> nothing

IN-PLACE version: Encode input values into pre-allocated state vector.
Avoids allocation - use in hot loops.

# Arguments
- `ψ_out`: Pre-allocated output buffer of length 2^L (will be overwritten)
- `u_window`: Input values in [0,1]
- `L`: Number of qubits
"""
function encode_input_ry_psi!(ψ_out::Vector{ComplexF64}, u_window::Vector{Float64}, L::Int)
    dim = 1 << L
    
    # Reset to |0...0⟩
    fill!(ψ_out, zero(ComplexF64))
    ψ_out[1] = one(ComplexF64)
    
    # Apply Ry(π*u) to each qubit
    for k in 1:L
        θ = π * u_window[k]
        apply_ry_psi!(ψ_out, k, θ, L)
    end
    
    return nothing
end



"""
    encode_input_ry_rho(u_window::Vector{Float64}, L::Int) -> Matrix{ComplexF64}

Encode input values into L-qubit density matrix using modular blocks:
  1. Create |0⟩⟨0|^⊗L using make_product_rho
  2. Apply Ry(θₖ) to each qubit k using apply_ry_rho!
"""
function encode_input_ry_rho(u_window::Vector{Float64}, L::Int)
    @assert length(u_window) >= L "Input window must have at least L values"
    
    # STEP 1: Create |0⟩⟨0|^⊗L
    ρ = make_product_rho(fill(:zero, L))
    
    # STEP 2: Apply Ry(π*u) to each qubit
    for k in 1:L
        θ = π * u_window[k]
        apply_ry_rho!(ρ, k, θ, L)
    end
    
    return ρ
end
# ==============================================================================
# QRC RESET OPERATIONS
# ==============================================================================

"""
    qrc_reset_psi(psi, psi_in, L_in, L_res, protocol) -> Vector{ComplexF64}

Perform QRC reset for pure state (MCWF) via projective measurement of INPUT qubits.

# MCWF Reset Protocol (to match DM partial trace):

The density matrix approach (DM) does:
  1. ρ_reservoir = Tr_input(ρ_current)  -- partial trace over input rail
  2. ρ_new = ρ_in_encoded ⊗ ρ_reservoir -- tensor with fresh encoded input

For MCWF with pure states, we cannot represent mixed states. Instead we use
projective measurement to stochastically sample from the reservoir distribution,
so that the ENSEMBLE AVERAGE over many trajectories equals the DM result.

# Mathematical Details:

Given a bipartite pure state on input (L_in qubits) ⊗ reservoir (L_res qubits):
    |ψ⟩ = Σ_{i,j} c_{ij} |i⟩_in ⊗ |j⟩_res

where i ∈ {1,...,2^L_in} and j ∈ {1,...,2^L_res} are computational basis indices.

The reduced density matrix of the reservoir is:
    ρ_res = Tr_in(|ψ⟩⟨ψ|) = Σ_i Σ_{j,k} c_{ij} c*_{ik} |j⟩⟨k|

The diagonal elements (populations) are:
    ρ_res[j,j] = Σ_i |c_{ij}|²

If we measure INPUT qubits in computational basis:
  - Probability of outcome i:  P(i) = Σ_j |c_{ij}|²
  - Post-measurement reservoir state if outcome i: |φ_res⟩ = (1/√P(i)) Σ_j c_{ij}|j⟩

Key insight: The ensemble average of the post-measurement reservoir state gives:
    E[|φ_res⟩⟨φ_res|] = Σ_i P(i) |φ_res(i)⟩⟨φ_res(i)| = ρ_res

This ensures MCWF matches DM in the limit of many trajectories!

# Implementation:

We use the reshape trick: for a state vector ψ of length dim_in × dim_res,
    M = reshape(ψ, (dim_in, dim_res))
gives M[i,j] = amplitude for |i⟩_in ⊗ |j⟩_res (little-endian indexing).

Then:
  - P(input=i) = Σ_j |M[i,j]|² = sum of row i squared magnitudes
  - Post-measurement reservoir = row i of M, normalized

# Arguments:
- `psi`: Current state vector (dim_in × dim_res)
- `psi_in`: New encoded input state (dim_in)
- `L_in`: Number of input qubits
- `L_res`: Number of reservoir qubits
- `protocol`: :traceout_rail_1 (measure input) or :traceout_rail_2 (measure reservoir)

# Returns:
- New state vector |ψ_in_new⟩ ⊗ |φ_res⟩
"""
function qrc_reset_psi(psi::Vector{ComplexF64}, psi_in::Vector{ComplexF64}, 
                           L_in::Int, L_res::Int, protocol::Symbol)
    dim_in = 1 << L_in
    dim_res = 1 << L_res
    
    if protocol == :traceout_rail_1 || protocol == :protocol1
        # =====================================================================
        # PROTOCOL 1: Measure INPUT rail, keep RESERVOIR
        # This matches DM's: ρ_res = Tr_input(ρ), then ρ_new = ρ_in ⊗ ρ_res
        # =====================================================================
        
        # Reshape state vector to matrix: M[i,j] = ⟨i_in, j_res | ψ⟩
        # Row index i = input basis state, Column index j = reservoir basis state
        M = reshape(psi, (dim_in, dim_res))
        
        # --- STEP 1: Sample input measurement outcome ---
        # P(input = i) = Σ_j |M[i,j]|² (norm² of row i)
        r = rand()  # Random number for inverse CDF sampling
        cumsum_p = 0.0
        total_prob = 0.0
        outcome = 1  # Default to first state
        
        # Compute total probability (should be 1 if normalized, but compute for safety)
        @inbounds for i in 1:dim_in
            for j in 1:dim_res
                total_prob += abs2(M[i, j])
            end
        end
        
        # Inverse CDF sampling: find first i where cumsum(P) >= r
        @inbounds for i in 1:dim_in
            row_prob = 0.0
            for j in 1:dim_res
                row_prob += abs2(M[i, j])
            end
            cumsum_p += row_prob / total_prob
            if r <= cumsum_p
                outcome = i
                break
            end
        end
        
        # --- STEP 2: Extract post-measurement reservoir state ---
        # |φ_res⟩ = (1/||row||) × M[outcome, :] 
        # This is the reservoir state conditioned on measuring input = outcome
        phi_res = zeros(ComplexF64, dim_res)
        norm_sq = 0.0
        @inbounds for j in 1:dim_res
            phi_res[j] = M[outcome, j]
            norm_sq += abs2(M[outcome, j])
        end
        
        # Normalize the reservoir state
        norm_phi = sqrt(norm_sq)
        if norm_phi > 1e-12
            @inbounds for j in 1:dim_res
                phi_res[j] /= norm_phi
            end
        else
            # Fallback: if probability is zero, reset to |0...0⟩
            phi_res[1] = 1.0
        end
        
        # --- STEP 3: Tensor new input with collapsed reservoir ---
        # |ψ_new⟩ = |ψ_in_encoded⟩ ⊗ |φ_res⟩
        return tensor_product_ket(psi_in, phi_res, L_in, L_res)
        
    elseif protocol == :traceout_rail_2 || protocol == :protocol2
        # =====================================================================
        # PROTOCOL 2: Measure RESERVOIR rail, keep INPUT (which becomes reservoir)
        # =====================================================================
        M = reshape(psi, (dim_in, dim_res))
        
        # Sample reservoir measurement outcome
        # P(reservoir = j) = Σ_i |M[i,j]|² (norm² of column j)
        r = rand()
        cumsum_p = 0.0
        total_prob = 0.0
        outcome = 1
        
        @inbounds for j in 1:dim_res
            for i in 1:dim_in
                total_prob += abs2(M[i, j])
            end
        end
        
        @inbounds for j in 1:dim_res
            col_prob = 0.0
            for i in 1:dim_in
                col_prob += abs2(M[i, j])
            end
            cumsum_p += col_prob / total_prob
            if r <= cumsum_p
                outcome = j
                break
            end
        end
        
        # Extract post-measurement input state (becomes new reservoir)
        phi_new_res = zeros(ComplexF64, dim_in)
        norm_sq = 0.0
        @inbounds for i in 1:dim_in
            phi_new_res[i] = M[i, outcome]
            norm_sq += abs2(M[i, outcome])
        end
        norm_phi = sqrt(norm_sq)
        if norm_phi > 1e-12
            @inbounds for i in 1:dim_in
                phi_new_res[i] /= norm_phi
            end
        else
            phi_new_res[1] = 1.0
        end
        
        return tensor_product_ket(psi_in, phi_new_res, L_in, L_in)
    else
        error("Unknown protocol: $protocol. Use :traceout_rail_1 or :traceout_rail_2")
    end
end


"""
    qrc_reset_psi!(ψ_out, psi, psi_in, phi_res_buf, L_in, L_res, protocol) -> nothing

IN-PLACE version: Perform QRC reset, writing result to pre-allocated ψ_out.
Uses pre-allocated phi_res_buf for intermediate reservoir state.
Avoids allocation - use in hot loops.

# Arguments
- `ψ_out`: Pre-allocated output buffer of length 2^(L_in+L_res)
- `psi`: Current state vector (read-only)
- `psi_in`: New encoded input state
- `phi_res_buf`: Pre-allocated buffer for intermediate reservoir state (length 2^L_res)
- `L_in`, `L_res`: Number of qubits in input and reservoir rails
- `protocol`: Reset protocol (:traceout_rail_1 or :traceout_rail_2)
"""
function qrc_reset_psi!(ψ_out::Vector{ComplexF64}, psi::Vector{ComplexF64}, 
                        psi_in::Vector{ComplexF64}, phi_res_buf::Vector{ComplexF64},
                        L_in::Int, L_res::Int, protocol::Symbol)
    dim_in = 1 << L_in
    dim_res = 1 << L_res
    
    if protocol == :traceout_rail_1 || protocol == :protocol1
        M = reshape(psi, (dim_in, dim_res))
        
        # Sample input measurement outcome
        r = rand()
        cumsum_p = 0.0
        total_prob = 0.0
        outcome = 1
        
        @inbounds for i in 1:dim_in
            for j in 1:dim_res
                total_prob += abs2(M[i, j])
            end
        end
        
        @inbounds for i in 1:dim_in
            row_prob = 0.0
            for j in 1:dim_res
                row_prob += abs2(M[i, j])
            end
            cumsum_p += row_prob / total_prob
            if r <= cumsum_p
                outcome = i
                break
            end
        end
        
        # Extract and normalize reservoir state into buffer
        norm_sq = 0.0
        @inbounds for j in 1:dim_res
            phi_res_buf[j] = M[outcome, j]
            norm_sq += abs2(M[outcome, j])
        end
        norm_phi = sqrt(norm_sq)
        if norm_phi > 1e-12
            @inbounds for j in 1:dim_res
                phi_res_buf[j] /= norm_phi
            end
        else
            fill!(phi_res_buf, zero(ComplexF64))
            phi_res_buf[1] = 1.0
        end
        
        # Tensor product in-place
        tensor_product_ket!(ψ_out, psi_in, phi_res_buf, L_in, L_res)
        
    elseif protocol == :traceout_rail_2 || protocol == :protocol2
        M = reshape(psi, (dim_in, dim_res))
        
        r = rand()
        cumsum_p = 0.0
        total_prob = 0.0
        outcome = 1
        
        @inbounds for j in 1:dim_res
            for i in 1:dim_in
                total_prob += abs2(M[i, j])
            end
        end
        
        @inbounds for j in 1:dim_res
            col_prob = 0.0
            for i in 1:dim_in
                col_prob += abs2(M[i, j])
            end
            cumsum_p += col_prob / total_prob
            if r <= cumsum_p
                outcome = j
                break
            end
        end
        
        # Extract and normalize - use first L_in entries of phi_res_buf
        norm_sq = 0.0
        @inbounds for i in 1:dim_in
            phi_res_buf[i] = M[i, outcome]
            norm_sq += abs2(M[i, outcome])
        end
        norm_phi = sqrt(norm_sq)
        if norm_phi > 1e-12
            @inbounds for i in 1:dim_in
                phi_res_buf[i] /= norm_phi
            end
        else
            fill!(view(phi_res_buf, 1:dim_in), zero(ComplexF64))
            phi_res_buf[1] = 1.0
        end
        
        tensor_product_ket!(ψ_out, psi_in, view(phi_res_buf, 1:dim_in), L_in, L_in)
    else
        error("Unknown protocol: $protocol. Use :traceout_rail_1 or :traceout_rail_2")
    end
    return nothing
end



"""Helper: sample integer index from probability distribution."""
function _sample_from_probs(probs::Vector{Float64})
    r = rand()
    cumsum_p = 0.0
    for i in 1:length(probs)
        cumsum_p += probs[i]
        if r <= cumsum_p
            return i
        end
    end
    return length(probs)
end

"""
    qrc_reset_rho(rho, rho_in, L, n_rails, protocol) -> Matrix{ComplexF64}

QRC reset for density matrix:
  1. Partial trace to remove input rail
  2. Tensor new input DM with reservoir DM

Uses modular partial_trace and tensor_product_rho.
"""
function qrc_reset_rho(rho::Matrix{ComplexF64}, rho_in::Matrix{ComplexF64},
                               L::Int, n_rails::Int, protocol::Symbol)
    N = L * n_rails
    
    if protocol == :traceout_rail_1
        # Trace out qubits 1..L, keep rest
        rho_reservoir = partial_trace(rho, collect(1:L), N)
        return tensor_product_rho(rho_in, rho_reservoir)
        
    elseif protocol == :traceout_rail_2
        # Trace out qubits L+1..2L, keep first L
        trace_qubits = collect((L+1):N)
        rho_reservoir = partial_trace(rho, trace_qubits, N)
        return tensor_product_rho(rho_in, rho_reservoir)
    else
        error("Unknown protocol: $protocol")
    end
end

# ==============================================================================
# MODULAR QRC STEP - PURE STATE (MCWF)
# ==============================================================================

"""
    qrc_step_modular_psi!(ψ, u_window, evolution, geometry, measure_fn) -> (ψ_new, features)

Single QRC step using MODULAR building blocks:

  1. **RESET**: Partial trace → collapse → basis state
  2. **ENCODE**: make_product_ket(:zero) → apply_ry_psi! for each qubit
  3. **TENSOR**: tensor_product_ket(ψ_in, ψ_reservoir)
  4. **EVOLVE**: U*ψ (exact) or Trotter gates (approximate)
  5. **MEASURE**: Extract observable expectation values

All stages use general-purpose quantum operations from the modular libraries.
"""
function qrc_step_modular_psi!(ψ::Vector{ComplexF64}, 
                               u_window::Vector{Float64},
                               evolution::NamedTuple,
                               geometry::NamedTuple,
                               evolve_fn,
                               measure_fn)
    
    L, n_rails, N = geometry.L, geometry.n_rails, geometry.N
    reset_type = geometry.reset_type
    dim_in = 1 << L
    dim_res = 1 << L
    
    # ─── STAGE 1: RESET (partial trace + collapse) ───
    if reset_type == :traceout_rail_1
        M = reshape(ψ, (dim_in, dim_res))
        probs = [sum(abs2.(M[i, :])) for i in 1:dim_in]
        outcome = sample_outcome(probs)
        ψ_reservoir = M[outcome, :]
        normalize_state!(ψ_reservoir)
    elseif reset_type == :traceout_rail_2
        M = reshape(ψ, (dim_in, dim_res))
        probs = [sum(abs2.(M[:, j])) for j in 1:dim_res]
        outcome = sample_outcome(probs)
        ψ_reservoir = M[:, outcome]
        normalize_state!(ψ_reservoir)
    else
        ψ_reservoir = ψ
    end
    
    # ─── STAGE 2: ENCODE using modular blocks ───
    # |0⟩^⊗L → apply Ry(π*u) to each qubit
    ψ_in = encode_input_ry_psi(u_window, L)
    
    # ─── STAGE 3: TENSOR using modular tensor product ───
    ψ_new = tensor_product_ket(ψ_in, ψ_reservoir, L, L)
    
    # ─── STAGE 4: EVOLVE ───
    if evolution.integrator == :exact
        ψ_new = evolution.U * ψ_new
    elseif evolution.integrator == :trotter
        evolve_fn(ψ_new, evolution.gates, N, evolution.n_substeps)
    end
    
    # ─── STAGE 5: MEASURE ───
    features = measure_fn(ψ_new, L, n_rails)
    
    return ψ_new, features
end


# ==============================================================================
# MODULAR QRC STEP - DENSITY MATRIX
# ==============================================================================

"""
    qrc_step_modular_rho!(ρ, u_window, evolution, geometry, evolve_fn, measure_fn) -> (ρ_new, features)

Single QRC step for DENSITY MATRIX using modular blocks:

  1. **RESET**: Partial trace to get ρ_reservoir
  2. **ENCODE**: make_product_rho(:zero) → apply_ry_rho! for each qubit
  3. **TENSOR**: tensor_product_rho(ρ_in, ρ_reservoir)
  4. **EVOLVE**: U·ρ·U† (exact) or Trotter (approximate)
  5. **MEASURE**: Tr(O·ρ) for each observable
"""
function qrc_step_modular_rho!(ρ::Matrix{ComplexF64}, 
                               u_window::Vector{Float64},
                               evolution::NamedTuple,
                               geometry::NamedTuple,
                               evolve_fn,
                               measure_fn)
    
    L, n_rails, N = geometry.L, geometry.n_rails, geometry.N
    reset_type = geometry.reset_type
    
    # ─── STAGE 1: RESET (partial trace) ───
    if reset_type == :traceout_rail_1
        ρ_reservoir = partial_trace(ρ, collect(1:L), N)
    elseif reset_type == :traceout_rail_2
        ρ_reservoir = partial_trace(ρ, collect((L+1):N), N)
    else
        ρ_reservoir = ρ
    end
    
    # ─── STAGE 2: ENCODE using modular blocks ───
    # |0⟩⟨0|^⊗L → apply Ry(π*u) to each qubit
    ρ_in = encode_input_ry_rho(u_window, L)
    
    # ─── STAGE 3: TENSOR using modular tensor product ───
    ρ_new = tensor_product_rho(ρ_in, ρ_reservoir)
    
    # ─── STAGE 4: EVOLVE ───
    if evolution.integrator == :exact
        ρ_new = evolution.U * ρ_new * evolution.U_adj
    elseif evolution.integrator == :trotter
        evolve_fn(ρ_new, evolution.gates, N, evolution.n_substeps)
    end
    
    # ─── STAGE 5: MEASURE ───
    features = measure_fn(ρ_new, L, n_rails)
    
    return ρ_new, features
end

end # module


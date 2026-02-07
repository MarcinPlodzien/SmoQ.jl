# Date: 2026
#
#=
################################################################################
#
#     cpuQuantumChannelLindbladianEvolutionMCWF.jl
#
#     MONTE CARLO WAVE FUNCTION (MCWF) / QUANTUM TRAJECTORY METHOD
#     for LINDBLAD MASTER EQUATION
#
################################################################################

================================================================================
 THEORETICAL BACKGROUND
================================================================================

LINDBLAD MASTER EQUATION (Open Quantum Systems):
------------------------------------------------
The Lindblad master equation describes dissipative evolution of a density matrix:

    dρ/dt = -i[H,ρ] + γ Σₖ D[Lₖ]ρ

where H is the Hamiltonian and D[L]ρ is the Lindblad dissipator:

    D[L]ρ = L ρ L† - ½{L†L, ρ}
          = L ρ L† - ½(L†L ρ + ρ L†L)

For spontaneous emission with jump operator L = σ⁻ = |0⟩⟨1|:
  - L†L = σ⁺σ⁻ = |1⟩⟨1| (projector onto excited state)
  - D[σ⁻]ρ describes decay from |1⟩ to |0⟩ with rate γ


MCWF STOCHASTIC UNRAVELING:
---------------------------
The MCWF method represents the density matrix as an ENSEMBLE AVERAGE of random
pure state trajectories:

    ρ(t) = E[|ψ(t)⟩⟨ψ(t)|]

where each trajectory |ψ(t)⟩ follows a stochastic evolution:
  - Most of the time: non-unitary evolution under EFFECTIVE Hamiltonian
  - Occasionally: random quantum JUMPS

Key insight: The average over many trajectories EXACTLY reproduces the Lindblad
master equation. This is mathematically rigorous, not an approximation!


================================================================================
 ALGORITHM: FIRST-ORDER MCWF (Quantum Jump Method)
================================================================================

For each time step dt, perform:

STEP 1: COHERENT EVOLUTION
--------------------------
Apply the unitary propagator:
    |ψ⟩ → U|ψ⟩  where U = exp(-iHdt)


STEP 2: NON-HERMITIAN DECAY (The Crucial Step!)
------------------------------------------------
Apply the EFFECTIVE non-Hermitian Hamiltonian:

    H_eff = H - (iγ/2) Σₖ L†ₖLₖ

The second term represents the "measurement backaction" - even when NO jump
occurs, we gain information that the system didn't decay, which biases the
state toward undecayed configurations.

For σ⁻ decay: each amplitude with qubit k in |1⟩ gets multiplied by:
    √(1 - γdt)

This is because:
    exp(-γdt/2 × n_k) ≈ (1 - γdt/2)^n_k ≈ √(1 - γdt)^n_k


STEP 3: COMPUTE DECAY PROBABILITY
---------------------------------
The total probability of a jump occurring is:
    dp = 1 - ||ψ̃||²

where ||ψ̃||² is the norm-squared after decay. This equals:
    dp = γ × dt × Σₖ ⟨ψ|L†ₖLₖ|ψ⟩


STEP 4: QUANTUM JUMP DECISION
-----------------------------
Draw random number r ∈ [0, 1):

IF r < dp (JUMP OCCURRED):
  • Select which operator Lₖ caused the jump with probability ∝ ⟨L†ₖLₖ⟩
  • Apply the jump: |ψ⟩ → Lₖ|ψ⟩
  • Renormalize: |ψ⟩ → |ψ⟩/||ψ||

IF r ≥ dp (NO JUMP):
  • The state already encodes the "no-jump" information via the decay step
  • Just renormalize: |ψ⟩ → ψ̃/||ψ̃||


STEP 5: RENORMALIZE
-------------------
After decay (and possible jump), the state must be normalized:
    |ψ⟩ → |ψ⟩/||ψ||


================================================================================
 COMMON BUGS AND PITFALLS
================================================================================

BUG #1: Missing the decay step
------------------------------
WRONG: Apply U, then just do jumps with probability γdt⟨L†L⟩
RIGHT: Apply U, then DECAY the state, then decide on jumps

Without the decay step, the "no-jump" trajectories don't have the correct
probability bias, leading to factor-of-2 errors in observables!

BUG #2: Not renormalizing on no-jump
------------------------------------
Even when no jump occurs, the state norm changed due to decay.
MUST renormalize in BOTH cases (jump and no-jump)!

BUG #3: Wrong jump selection weighting
--------------------------------------
When multiple jump operators are present, must select the operator
with probability PROPORTIONAL to γₖ×⟨L†ₖLₖ⟩, not uniform random.


================================================================================
 IMPLEMENTATION
================================================================================

This module implements:
  1. Bitwise matrix-free σ⁻ operator for quantum jumps
  2. Correct MCWF algorithm with effective Hamiltonian decay
  3. Support for both exact U and Trotter decomposition

The ensemble average of trajectories EXACTLY converges to Lindblad ρ(t).
=#

module CPUQuantumChannelLindbladianEvolutionMonteCarloWaveFunction

using LinearAlgebra
using Base.Threads

export lindblad_mcwf_step!
export apply_sigma_z!

# ==============================================================================
# σ_z JUMP OPERATOR FOR PURE STATE (Bitwise, Matrix-Free)
#
# PHYSICS:
# --------
# The σ_z operator = diag(1, -1) describes pure dephasing:
#   σ_z|0⟩ = +|0⟩   (ground state gets +1 phase)  
#   σ_z|1⟩ = -|1⟩   (excited state gets -1 phase)
#
# BITWISE IMPLEMENTATION:
# -----------------------
# For each basis state |i⟩:
#   - If bit k is UNSET (qubit in |0⟩): no change (×1)
#   - If bit k is SET (qubit in |1⟩): flip sign (×-1)
# ==============================================================================

"""
    apply_sigma_z!(ψ, k, N)

Apply σ_z (phase flip) to qubit k using BITWISE operations.
For pure dephasing jumps: flips sign of amplitudes where qubit k is in |1⟩.
"""
function apply_sigma_z!(ψ::Vector{ComplexF64}, k::Int, N::Int)
    dim = 1 << N
    mask = 1 << (k - 1)  # Binary mask with only bit k set
    
    @inbounds for i in 0:(dim-1)
        if (i & mask) != 0  # Bit k is set (qubit in |1⟩)
            ψ[i+1] = -ψ[i+1]  # Flip sign
        end
    end
    return ψ
end

# ==============================================================================
# MCWF LINDBLAD EVOLUTION STEP (σ_z DEPHASING)
#
# The MCWF algorithm implements Lindblad: dρ/dt = -i[H,ρ] + γ Σ_k D[σ_z_k]ρ
#
# For σ_z: L†L = I, so decay is UNIFORM across all amplitudes.
#
# ALGORITHM (First-Order):
# 1. Apply coherent evolution: |ψ⟩ → U|ψ⟩ where U = exp(-iHdt)
# 2. Apply non-Hermitian decay from H_eff = H - iγN/2:
#    ALL amplitudes get multiplied by sqrt(1 - Nγdt)
# 3. Compute decay probability: dp = 1 - ||ψ̃||²
# 4. Draw random r ∈ [0,1):
#    - If r < dp (JUMP): Select qubit k uniformly, apply σ_z_k
#    - If r ≥ dp (NO JUMP): State retains accumulated phase info
# 5. Renormalize: |ψ⟩ → |ψ⟩/||ψ||
#
# Key property: E[|ψ⟩⟨ψ|] = ρ_Lindblad (exact mathematical equivalence)
# ==============================================================================

"""
    lindblad_mcwf_step!(ψ, U, N, γ, dt)

Single MCWF step for Lindblad evolution with σ_z (dephasing) jump operators.
Uses the CORRECT effective Hamiltonian algorithm.

# Arguments
- `ψ::Vector{ComplexF64}`: Pure state vector (modified in-place)
- `U::Matrix{ComplexF64}`: Unitary propagator exp(-iHdt)
- `N::Int`: Number of qubits
- `γ::Float64`: Dephasing rate (same for all qubits)
- `dt::Float64`: Time step

# Algorithm (σ_z dephasing)
1. Coherent evolution: ψ → U ψ
2. Non-Hermitian decay: ψ *= sqrt(1 - Nγdt) (uniform for all amplitudes)
3. Jump decision with probability dp = 1 - ||ψ||²
4. If jump: select qubit uniformly, apply σ_z (phase flip)
5. Renormalize
"""
function lindblad_mcwf_step!(ψ::Vector{ComplexF64}, U::Matrix{ComplexF64}, N::Int, γ::Float64, dt::Float64)
    # Step 1: Coherent (unitary) evolution
    ψ .= U * ψ
    
    # Step 2: Non-Hermitian decay from effective Hamiltonian H_eff = H - iγN/2 × I
    # For σ_z: L†L = I for each qubit, so total decay is uniform
    # Decay factor: sqrt(1 - Nγdt) applied to ALL amplitudes
    decay_factor = sqrt(1.0 - N * γ * dt)
    ψ .*= decay_factor
    
    # Step 3: Compute total decay probability dp = 1 - ||ψ̃||²
    norm_sq = real(dot(ψ, ψ))
    dp = 1.0 - norm_sq  # Total probability of jump
    
    # Step 4: Jump decision
    r = rand()
    if r < dp && dp > 0
        # JUMP occurred - for σ_z, all qubits have equal jump probability
        # Select qubit uniformly at random
        selected_k = rand(1:N)
        
        # Apply σ_z on selected qubit (phase flip)
        apply_sigma_z!(ψ, selected_k, N)
    end
    # If r >= dp: NO JUMP - state already has correct "no-jump" information
    
    # Step 5: Renormalize (always needed after decay + possible jump)
    normalize!(ψ)
    
    return ψ
end

"""
    lindblad_mcwf_step_trotter!(ψ, gates, apply_trotter!, N, γ, dt)

Single MCWF step using Trotter gates for coherent evolution (MATRIX-FREE).
Uses σ_z (dephasing) jump operators.

# Arguments
- `ψ::Vector{ComplexF64}`: Pure state vector
- `gates::Vector{FastTrotterGate}`: Precomputed Trotter gates
- `apply_trotter!::Function`: Function to apply gates, e.g., apply_fast_trotter_step_cpu!
- `N::Int`: Number of qubits
- `γ::Float64`: Dephasing rate
- `dt::Float64`: Time step

# Algorithm (σ_z dephasing)
1. Apply Trotter gates (coherent evolution)
2. Apply non-Hermitian decay: ψ *= sqrt(1 - Nγdt) (uniform)
3. Compute decay probability dp = 1 - ||ψ||²
4. If jump: select qubit uniformly, apply σ_z (phase flip)
5. Renormalize

# Performance
- Coherent evolution: O(N × 2^N) vs O(4^N) for dense matrix
- σ_z jumps: O(2^N) bitwise, matrix-free
"""
function lindblad_mcwf_step_trotter!(ψ::Vector{ComplexF64}, gates, apply_trotter!::Function, 
                                      N::Int, γ::Float64, dt::Float64)
    # Step 1: Apply Trotter gates (coherent evolution - MATRIX-FREE)
    apply_trotter!(ψ, gates, N)
    
    # Step 2: Non-Hermitian decay for σ_z: uniform across all amplitudes
    decay_factor = sqrt(1.0 - N * γ * dt)
    ψ .*= decay_factor
    
    # Step 3: Compute total decay probability dp = 1 - ||ψ̃||²
    norm_sq = real(dot(ψ, ψ))
    dp = 1.0 - norm_sq
    
    # Step 4: Jump decision
    r = rand()
    if r < dp && dp > 0
        # JUMP occurred - for σ_z, all qubits have equal jump probability
        selected_k = rand(1:N)
        
        # Apply σ_z on selected qubit (phase flip)
        apply_sigma_z!(ψ, selected_k, N)
    end
    
    # Step 5: Renormalize
    normalize!(ψ)
    
    return ψ
end

export lindblad_mcwf_step_trotter!

# ==============================================================================
# BATCHED MCWF EVOLUTION (Multiple Trajectories)
#
# Key performance optimizations:
# 1. Precompute Trotter gates ONCE for all trajectories
# 2. Store trajectories in matrix form (dim × n_traj) for cache-friendly access
# 3. Threaded trajectory processing
# 4. Bitwise operations for all quantum jumps
# ==============================================================================

"""
    lindblad_mcwf_batched_step!(Ψ, gates, apply_trotter!, N, n_traj, γ, dt)

Batched MCWF step for multiple trajectories stored as columns of matrix Ψ.
Uses the CORRECT effective Hamiltonian algorithm with non-Hermitian decay.

# Arguments
- `Ψ::Matrix{ComplexF64}`: State matrix (dim × n_traj), each column is a trajectory
- `gates::Vector{FastTrotterGate}`: Precomputed Trotter gates
- `apply_trotter!::Function`: Gate application function
- `N::Int`: Number of qubits
- `n_traj::Int`: Number of trajectories
- `γ::Float64`: Decay rate
- `dt::Float64`: Time step

# Performance
- Uses Threads.@threads for parallel trajectory processing
- Gate application is O(N × 2^N) per trajectory (matrix-free)
- σ⁻ jumps are O(N × 2^N) per trajectory (bitwise)
"""
function lindblad_mcwf_batched_step!(Ψ::Matrix{ComplexF64}, gates, apply_trotter!::Function,
                                      N::Int, n_traj::Int, γ::Float64, dt::Float64)
    dim = size(Ψ, 1)
    decay_factor = sqrt(1.0 - γ * dt)  # Precompute decay factor
    
    @inbounds Threads.@threads for traj in 1:n_traj
        # Get view of this trajectory
        ψ = view(Ψ, :, traj)
        
        # Step 1: Apply Trotter gates (matrix-free coherent evolution)
        apply_trotter!(ψ, gates, N)
        
        # Step 2: Non-Hermitian decay from effective Hamiltonian
        for i in 0:(dim-1)
            n_excited = count_ones(i)
            if n_excited > 0
                Ψ[i+1, traj] *= decay_factor^n_excited
            end
        end
        
        # Step 3: Compute decay probability and make jump decision
        norm_sq = real(dot(ψ, ψ))
        dp = 1.0 - norm_sq
        
        r = rand()
        if r < dp && dp > 0
            # JUMP - select qubit with weighted probability
            cumulative = 0.0
            selected_k = 1
            
            for k in 1:N
                p_k = γ * dt * population_excited(ψ, k, N) / norm_sq
                cumulative += p_k
                if r < cumulative
                    selected_k = k
                    break
                end
            end
            
            apply_sigma_minus!(ψ, selected_k, N)
        end
        
        # Step 4: Renormalize
        nrm = norm(ψ)
        @simd for i in 1:dim
            Ψ[i, traj] /= nrm
        end
    end
    
    return Ψ
end

export lindblad_mcwf_batched_step!

"""
    evolve_mcwf_batched_trotter!(Ψ, gates, apply_trotter!, N, n_traj, n_steps, γ, dt; 
                                  observables_fn=nothing, results=nothing)

Run full batched MCWF evolution for multiple trajectories.

# Arguments
- `Ψ::Matrix{ComplexF64}`: Initial states (dim × n_traj), modified in-place
- `gates`: Trotter gates
- `apply_trotter!::Function`: Gate application function
- `N::Int`: Number of qubits
- `n_traj::Int`: Number of trajectories
- `n_steps::Int`: Number of time steps
- `γ::Float64`: Decay rate
- `dt::Float64`: Time step
- `observables_fn`: Optional function to compute observables at each step
- `results`: Optional storage for observables
"""
function evolve_mcwf_batched_trotter!(Ψ::Matrix{ComplexF64}, gates, apply_trotter!::Function,
                                       N::Int, n_traj::Int, n_steps::Int, γ::Float64, dt::Float64;
                                       observables_fn=nothing, results=nothing)
    for step in 1:n_steps
        lindblad_mcwf_batched_step!(Ψ, gates, apply_trotter!, N, n_traj, γ, dt)
        
        if observables_fn !== nothing && results !== nothing
            observables_fn(Ψ, step, results)
        end
    end
    return Ψ
end

export evolve_mcwf_batched_trotter!

# Extend population_excited and apply_sigma_minus! for views
function population_excited(ψ::SubArray{ComplexF64}, k::Int, N::Int)
    dim = length(ψ)
    mask = 1 << (k - 1)
    result = 0.0
    
    @inbounds for i in 0:(dim-1)
        if (i & mask) != 0
            result += abs2(ψ[i+1])
        end
    end
    return result
end

function apply_sigma_minus!(ψ::SubArray{ComplexF64}, k::Int, N::Int)
    dim = length(ψ)
    mask = 1 << (k - 1)
    
    @inbounds for i in 0:(dim-1)
        if (i & mask) != 0
            j = xor(i, mask)
            ψ[j+1] = ψ[i+1]
            ψ[i+1] = 0.0
        end
    end
    return ψ
end

end # module

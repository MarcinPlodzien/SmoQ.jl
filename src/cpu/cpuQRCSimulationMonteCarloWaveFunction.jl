# Date: 2026
#
module CPUQRCSimulationMonteCarloWaveFunction

# ==============================================================================
# MONTE CARLO WAVE FUNCTION (MCWF) ENGINE
# ==============================================================================
# This module implements the stochastic unravelling of the master equation (MCWF),
# also known as Quantum Trajectories.
#
# METHODOLOGY:
# Instead of evolving the density matrix \rho (dim N^2), we evolve a stochastic
# pure state |\psi> (dim N). The average of N_traj trajectories converges to \rho.
#
#   \rho(t) \approx \frac{1}{N_{traj}} \sum_{k=1}^{N_{traj}} |\psi_k(t)\rangle \langle \psi_k(t)|
#
# This allows simulation of much larger systems (N ~ 20) than exact DM (N ~ 10).
# ==============================================================================


using LinearAlgebra
using Statistics
using Printf
using ProgressMeter
using Random
using Base.Threads
 
using ..CPUHamiltonianBuilder: HamiltonianParams, CouplingTerm, FieldTerm
using ..CPUQuantumChannelUnitaryEvolutionTrotter: FastTrotterGate, apply_fast_trotter_step_cpu!, precompute_trotter_gates_bitwise_cpu
using ..CPUQRCstep: encode_input_ry_psi, encode_input_ry_psi!
using ..CPUQRCstep: qrc_reset_psi, qrc_reset_psi!
using ..CPUQuantumStateObservables: _expect_X, _expect_Y, _expect_Z, _expect_XX, _expect_YY, _expect_ZZ, 
    measure_all_observables_state_vector_cpu, measure_all_observables_state_vector_cpu!

export run_mcwf_simulation, check_integrator_type

# Helper for legacy bitwise-based code (returns Ket)
function spin_encode(u_win)
    psi = _encode_single(u_win[1])
    for i in 2:length(u_win)
        psi = tensor(psi, _encode_single(u_win[i]))
    end
    return psi
end

@inline function _encode_single(val)
    u_clamped = clamp(val, 0.0, 1.0)
    theta = 2.0 * acos(sqrt(1.0 - u_clamped))
    c, s = cos(theta/2), sin(theta/2)
    return Ket(SpinBasis(1//2), [ComplexF64(c), ComplexF64(s)])
end

# ==============================================================================
# MAIN SIMULATION ROUTINE
# ==============================================================================

"""
    run_mcwf_simulation(psi_init, input_seq, H, ops, n_traj, T_evol, dt, L_chain; ...)

Executes the QRC simulation using the Quantum Trajectory method.

ARGS:
    - psi_init: Initial state |0...0>
    - input_seq: Vector of scalar inputs u(t) in [0, 1].
    - H: The system Hamiltonian (Static or Layered).
    - ops: Vector of observables to measure at each step.
    - n_traj: Number of independent trajectories to enable ensemble averaging.
    - T_evol: Clock cycle duration (time between inputs).
    - dt: Integration step size (for RK4/Trotter).
    
KWARGS:
    - integrator_type: :exact_cpu (exp), :rk4 (Runge-Kutta), or :trotter_cpu.


RETURNS:
    - X: Matrix (Time x Features). The ensemble-averaged expectation values.
"""
function run_mcwf_simulation(psi_init, input_seq, H, ops, n_traj, T_evol, dt, L_chain; 
                             integrator_type=:exact_precomp_cpu, 
                             encoding=:batch, 
                             n_rails=2, 
                             reset_type::Symbol=:traceout_rail_1,
                             reset_qubit_to::Symbol=:ket_0, # :ket_0 (Z) or :ket_plus (X)
                             obs_method::Symbol=:standard   # :standard, :fast_local, :fast_full
                            ) 
    T = length(input_seq)
    N_total = L_chain * n_rails
    
    # Determine number of observables based on method
    # New naming: explicit/bitwise × local/local_and_corr
    is_explicit = obs_method in (:explicit_local, :explicit_local_and_corr)
    is_local_only = obs_method in (:explicit_local, :bitwise_local)
    
    if is_explicit
        n_ops = length(ops)
        feat_str = is_local_only ? "local" : "local+corr"
        println("[MCWF] Using EXPLICIT operators ($(n_ops) features, $(feat_str))")
    elseif is_local_only  # :bitwise_local
        n_ops = 3 * N_total  # X, Y, Z per qubit
        println("[MCWF] Using BITWISE LOCAL observables ($(n_ops) features)")
    else  # :bitwise_local_and_corr
        n_ops = 3*N_total + 3*(L_chain-1)*n_rails + 3*L_chain*(n_rails-1)
        println("[MCWF] Using BITWISE LOCAL+CORR observables ($(n_ops) features)")
    end
    
    padded_u = vcat(zeros(L_chain), input_seq) # Pre-pad for sliding window
    
    # --- 1. PRE-COMPUTATION PHASE ---
    # To maximize performance, we pre-calculate unitary operators where possible.
    # --------------------------------------------------------------------------
    
    H_mat = nothing
    U_op = nothing
    U_layers = [] 
    
    if integrator_type == :exact_cpu || integrator_type == :exact_precomp_cpu
        # MATRIX EXPONENTIATION (Optimal for N < 12)
        # U = exp(-i * H * T_evol)
        # We convert SparseOperator to dense Matrix for BLAS acceleration in exponentiation.
        # Handle LayeredHamiltonian vs regular Operator
        if hasproperty(H, :layers)
            H_mat = Matrix(H.data.data)
        else
            H_mat = Matrix(H.data)
        end
        println("[MCWF] Pre-computing Exact Unitary (size $(size(H_mat)))...")
        U_op = exp(-1im * H_mat * T_evol)
        
    elseif integrator_type == :trotter_cpu
        # MATRIX-FREE BITWISE TROTTER - local gate application without full matrices
        println("[MCWF] Pre-computing Fast Trotter Gates (bitwise local application)...")
        if hasproperty(H, :layers) && !isempty(H.layers)
            trotter_gates = FastTrotterGate[]
            for layer in H.layers
                for (indices, op_local) in layer
                    op_dense = Matrix(op_local.data)
                    U_local = exp(-1im * op_dense * dt)
                    push!(trotter_gates, FastTrotterGate(indices, U_local))
                end
            end
            U_layers = trotter_gates
        else
            error("Integrator :trotter_cpu requires a LayeredHamiltonian. Use construct_layered_hamiltonian().")
        end
    else
        error("Unsupported integrator: $integrator_type. Supported: :exact_cpu, :trotter_cpu")
    end

    # --- 2. TRAJECTORY LOOP (PARALLEL) ---
    # We utilize Julia's multi-threading to run independent trajectories on separate cores.
    # --------------------------------------------------------------------------
    
    # Shared Accumulator (Thread-Safe via Lock)
    X_sum = zeros(Float64, T, n_ops)
    acc_lock = ReentrantLock()
    
    # Pre-calculate basis dimensions for Reset operations
    dim_in = 2^L_chain
    dim_total = length(psi_init.data)
    dim_res = div(dim_total, dim_in)
    basis_res = (dim_res > 1) ? tensor(psi_init.basis.bases[L_chain+1:end]...) : nothing
    
    println("[MCWF] Launching $n_traj Trajectories on $(Threads.nthreads()) threads...")
    p = Progress(n_traj, 1, "MCWF Progress: ")
    
    Threads.@threads for traj_i in 1:n_traj
        try
            # Deepcopy initial state for this trajectory
            curr_psi = copy(psi_init)
            
            # Local buffer for this trajectory (Time x Features)
            # Avoids race conditions accumulating into shared X_sum directly
            local_X = zeros(Float64, T, n_ops)
            
            # Run the sequential evolution for this single quantum system
            _run_traj_loop!(
                local_X, curr_psi, padded_u, 
                dim_in, dim_res, basis_res, L_chain, T, n_ops, ops, 
                U_op, U_layers, H_mat, dt, 
                integrator_type, T_evol, 
                encoding, 
                reset_type, reset_qubit_to,
                obs_method, n_rails  # Pass obs_method for fast observables
            )
            
            # Atomic Accumulation
            lock(acc_lock) do
                X_sum .+= local_X
            end
            next!(p)
            
        catch e
            println("Error in Trajectory $traj_i: $e")
            rethrow(e)
        end
    end
    
    # Average the results
    return X_sum ./ n_traj
end

# ==============================================================================
# INNER TRAJECTORY LOOP
# ==============================================================================
# This function handles the time-stepping for a SINGLE trajectory.
# It is critical this function be allocation-free in the inner loops.

function _run_traj_loop!(traj_acc, curr_psi, padded_u, dim_in, dim_res, basis_res, L_chain, T, n_ops, ops, U_op, U_layers, H_mat, dt, 
                        integrator_type, T_evol, encoding, reset_type, reset_qubit_to,
                        obs_method, n_rails)

     # Pre-allocate RK4 buffers if needed to prevent GC pressure
     k1, k2, k3, k4, temp = (integrator_type == :rk4 || integrator_type == :rk4_sparse) ? 
        (zeros(ComplexF64, length(curr_psi.data)) for _ in 1:5) : 
        (nothing, nothing, nothing, nothing, nothing)

     N_total = length(curr_psi.basis.bases)

     for k in 1:T
        # --- A. INPUT ENCODING ---
        # 1. Extract Input Window u_k
        u_win = zeros(Float64, L_chain)
        if encoding == :batch
            u_win .= padded_u[(k+L_chain):-1:(k+1)]
        else # :broadcast
            u_win .= padded_u[k+L_chain]
        end

        # 2. Reset / Encode Logic
        # Rail 1 handles the primary input injection (Standard QRC Assumption).
        next_psi = nothing

        # METHOD: Projective Measurement & Reset
        # ... comments ...
        
        M = reshape(curr_psi.data, (dim_in, dim_res))
        
        # Encode new input |\psi_in>
        psi_in = spin_encode(u_win)
            

            if reset_type == :traceout_rail_1
                # PROTOCOL 1: Reset Rail 1 (Top/Input). Keep Rail 2 (Reservoir, indices L+1..N).
                # Physics: New Input connects to Old Reservoir.
                # Implementation for Trace Rail 1 (Measure Input - Rows)
                if L_chain >= 1
                    dim_in_chk = 2^L_chain
                    if dim_in_chk < length(curr_psi.data)
                         # 1. Calculate Probabilities for Input basis states (Rows of M)
                         # P(Input=i) = norm(Row i)^2
                         probs = [norm(view(M, i, :))^2 for i in 1:dim_in]
                         
                         # 2. Sample Outcome
                         r_val = rand()
                         cum_p = 0.0
                         chosen_i = dim_in
                         for i in 1:dim_in
                             cum_p += probs[i]
                             if r_val <= cum_p
                                 chosen_i = i
                                 break
                             end
                         end
                         
                         # 3. Project Reservoir State (Normalized Row)
                         phi_res_vec = M[chosen_i, :]
                         normalize_state!(phi_res_vec)
                         phi_res_ket = Ket(basis_res, Vector{ComplexF64}(phi_res_vec))
                         
                        next_psi = tensor(psi_in, phi_res_ket)
                     end
                else
                     next_psi = psi_in
                end

            elseif reset_type == :traceout_rail_2
                # PROTOCOL 2: Trace out Rail 2 (Reservoir). Input (Rail 1) becomes Reservoir.
                # New Input -> Rail 1.
                # Physics: Old Input becomes the New Reservoir.
                # Measure Reservoir (Cols) - Keep Input (Col Vector)
                
                # Note: Rotation logic removed for now as reset_qubit_to is :ket_0
                if reset_qubit_to == :ket_plus
                    # Rotate Reservoir (Rows) to align X basis with Z measurement
                    # Ry(pi/2) maps |+> -> |0>.
                    # We apply Ry(pi/2) (x) I to the state (Res (x) Input).
                    # M_new = U_rot * M
                    
                    c, s = 0.70710678118, 0.70710678118
                    u_rot_2x2 = [c -s; s c]
                    
                    n_res_qubits = Int(log2(dim_res))
                    if n_res_qubits > 0
                         # Build Kronecker Product of u_rot_2x2
                         U_rot_res = u_rot_2x2
                         for _ in 2:n_res_qubits
                             U_rot_res = kron(U_rot_res, u_rot_2x2)
                         end
                         # Apply Left Multiplication (Rows are Res)
                         M = U_rot_res * M
                    end
                end 

                if dim_res > 1
                    # P(Res=j) = norm(Col j)^2
                    probs = [norm(view(M, :, j))^2 for j in 1:dim_res]
                    
                    r_val = rand()
                    cum_p = 0.0
                    chosen_j = dim_res
                    for j in 1:dim_res
                        cum_p += probs[j]
                        if r_val <= cum_p
                            chosen_j = j
                            break
                        end
                    end
                    
                    # Keep Input State (Normalized Col)
                    normalized_col = M[:, chosen_j]
                    normalize_state!(normalized_col)
                    # This Column is in Input Basis.
                    # We transfer it to Reservoir Rail. we assume homogeneous basis.
                    phi_res_ket = Ket(basis_res, Vector{ComplexF64}(normalized_col))
                    
                    next_psi = tensor(psi_in, phi_res_ket)
                end
            end

        # If a reset occurred, next_psi is set. Otherwise, curr_psi continues.
        # This ensures that the state passed to the time evolution step is always the most recent.
        if next_psi !== nothing
            curr_psi = next_psi
        end
        
        # --- EVOLUTION STEP ---
        if integrator_type == :exact_cpu || integrator_type == :exact_precomp_cpu
            # Exact Unitary (Precomputed)
            curr_psi.data .= U_op * curr_psi.data
            
        elseif integrator_type == :trotter_cpu
            # Matrix-free Bitwise Trotter Evolution (local gate application)
            N_qubits = length(curr_psi.basis.bases)
            n_trotter_steps = max(1, floor(Int, T_evol / dt))
            for _ in 1:n_trotter_steps
                apply_fast_trotter_step_cpu!(curr_psi.data, U_layers, N_qubits)
            end
        else
            error("Unsupported integrator: $integrator_type. Supported: :exact_cpu, :trotter_cpu")
        end
        
        # Calculate Expectation Values <O> = <psi|O|psi>
        # Determine method type from obs_method symbol
        is_explicit = obs_method in (:explicit_local, :explicit_local_and_corr)
        is_local_only = obs_method in (:explicit_local, :bitwise_local)
        
        if is_explicit
            # Explicit: use bitwise operations expect() with pre-built operators
            for f in 1:n_ops
                traj_acc[k, f] += real(expect(ops[f], curr_psi))
            end
        else
            # Bitwise: compute directly from state vector amplitudes
            ψ = curr_psi.data
            obs_idx = 1
            
            # Local observables (X, Y, Z for each qubit)
            for q in 1:N_total
                traj_acc[k, obs_idx] += _expect_X(ψ, q, N_total)
                traj_acc[k, obs_idx+1] += _expect_Y(ψ, q, N_total)
                traj_acc[k, obs_idx+2] += _expect_Z(ψ, q, N_total)
                obs_idx += 3
            end
            
            # Correlators (only for :bitwise_local_and_corr)
            if !is_local_only
                # Intra-rail correlators
                for rail in 1:n_rails
                    offset = (rail - 1) * L_chain
                    for site in 1:(L_chain-1)
                        i = offset + site
                        j = offset + site + 1
                        traj_acc[k, obs_idx] += _expect_XX(ψ, i, j, N_total)
                        traj_acc[k, obs_idx+1] += _expect_YY(ψ, i, j, N_total)
                        traj_acc[k, obs_idx+2] += _expect_ZZ(ψ, i, j, N_total)
                        obs_idx += 3
                    end
                end
                # Rungs
                for rail in 1:(n_rails-1)
                    for site in 1:L_chain
                        i = (rail - 1) * L_chain + site
                        j = rail * L_chain + site
                        traj_acc[k, obs_idx] += _expect_XX(ψ, i, j, N_total)
                        traj_acc[k, obs_idx+1] += _expect_YY(ψ, i, j, N_total)
                        traj_acc[k, obs_idx+2] += _expect_ZZ(ψ, i, j, N_total)
                        obs_idx += 3
                    end
                end
            end
        end
    end
end



# ==============================================================================
# BITWISE TRAJECTORY LOOP  
# ==============================================================================

"""
    _run_traj_loop_bitwise!(traj_acc, curr_psi, padded_u, L, T, n_ops, ...)

Execute a single MCWF trajectory for QRC time series generation.

# QRC MCWF Algorithm Overview:

Monte Carlo Wave Function (MCWF) is a stochastic method to simulate open quantum
systems. Instead of evolving the full density matrix ρ (O(N⁴) memory/computation),
we evolve pure states |ψ⟩ (O(N²)) and average over many trajectories.

For QRC, each timestep k processes one input sample u[k]:

## STEP A: INPUT ENCODING
- Extract window of L input values: u_win = [u_{k-L+1}, ..., u_k]
- Encode to qubit rotations: |ψ_in⟩ = ⊗_{q=1}^L Ry(θ_q)|0⟩
- Rotation angle: θ = 2*acos(√(1-u)) maps u∈[0,1] to Bloch sphere

## STEP B: RESET (Critical for MCWF-DM equivalence!)
- Projective measurement of input qubits
- Sample measurement outcome i with prob P(i) = Σ_j |⟨i,j|ψ⟩|²
- Extract post-measurement reservoir: |φ_res⟩ ∝ Σ_j ⟨i|ψ⟩|j⟩
- Tensor new input with reservoir: |ψ_new⟩ = |ψ_in⟩ ⊗ |φ_res⟩

KEY INSIGHT: E[|ψ_new⟩⟨ψ_new|] = |ψ_in⟩⟨ψ_in| ⊗ ρ_reservoir
This ensures MCWF matches DM in the limit of many trajectories!

## STEP C: TIME EVOLUTION
- Apply Hamiltonian: |ψ(t)⟩ = exp(-iHt)|ψ_new⟩
- Either exact (precomputed matrix exp) or Trotter decomposition

## STEP D: OBSERVABLE MEASUREMENT
- Compute ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for each qubit (feature extraction)
- Compute ⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩ for adjacent pairs
- These form the feature vector X[k,:] for linear regression

# Note on Averaging:
The averaging over n_traj trajectories happens in run_mcwf_simulation(),
which calls this function and accumulates results.
"""
function _run_traj_loop_bitwise!(traj_acc::Matrix{Float64}, curr_psi::Vector{ComplexF64}, 
                                 padded_u::Vector{Float64}, L::Int, T::Int, n_ops::Int,
                                 H_action!, gates, integrator_type::Symbol, T_evol::Float64,
                                 encoding::Symbol, reset_type::Symbol, n_rails::Int, n_substeps::Int,
                                 H_sparse, U_exact, H_params,
                                 spectral_bounds)
    N = L * n_rails  # Total qubits = qubits_per_rail × n_rails
    dim_psi = 1 << N
    dim_in = 1 << L
    dim_res = 1 << L
    
    # =========================================================================
    # PRE-ALLOCATE BUFFERS (avoid allocation inside hot loop)
    # =========================================================================
    u_win = zeros(Float64, L)              # Input window buffer
    psi_in = zeros(ComplexF64, dim_in)     # Encoded input state buffer
    phi_res_buf = zeros(ComplexF64, dim_res)  # Reservoir state buffer
    next_psi = zeros(ComplexF64, dim_psi)  # Next state buffer
    obs_buf = zeros(Float64, n_ops)        # Observable results buffer
    
    for k in 1:T
        # =====================================================================
        # STEP A: INPUT ENCODING (using pre-allocated u_win)
        # =====================================================================
        if encoding == :batch || encoding == :batch_causal
            @inbounds for q in 1:L
                u_win[q] = padded_u[k + q]
            end
        else  # :broadcast
            @inbounds for q in 1:L
                u_win[q] = padded_u[k + L]
            end
        end

        # Encode in-place into psi_in buffer
        encode_input_ry_psi!(psi_in, u_win, L)
        
        # =====================================================================
        # STEP B: RESET (Projective Measurement) - IN-PLACE
        # =====================================================================
        qrc_reset_psi!(next_psi, curr_psi, psi_in, phi_res_buf, L, L, reset_type)
        
        # =====================================================================
        # STEP C: TIME EVOLUTION
        # =====================================================================
        if integrator_type == :trotter_cpu
            # Trotter decomposition - already in-place
            for _ in 1:n_substeps
                apply_fast_trotter_step_cpu!(next_psi, gates, N)
            end
            
        elseif integrator_type == :exact_cpu || integrator_type == :exact_precomp_cpu
            # Exact evolution - mul! for in-place (need temp buffer)
            mul!(curr_psi, U_exact, next_psi)
            next_psi, curr_psi = curr_psi, next_psi  # Swap references
        else
            error("Unsupported integrator: $integrator_type. Supported: :exact_cpu, :trotter_cpu")
        end
        
        # Update state for next iteration
        curr_psi .= next_psi
        
        # =====================================================================
        # STEP D: OBSERVABLE MEASUREMENT (IN-PLACE)
        # =====================================================================
        measure_all_observables_state_vector_cpu!(obs_buf, curr_psi, L, n_rails)
        @inbounds traj_acc[k, :] .= obs_buf
    end
end




"""
    run_mcwf_simulation(psi0, H_params, gates, padded_u, L, n_rails, T, n_traj,
                                integrator_type, T_evol, encoding, reset_type) -> Matrix{Float64}

Fully bitwise MCWF simulation.
Supports only :exact_cpu and :trotter_cpu integrators.
"""
function run_mcwf_simulation(psi0::Vector{ComplexF64}, H_params, gates,
                                      padded_u::Vector{Float64}, L::Int, n_rails::Int, 
                                      T::Int, n_traj::Int, integrator_type::Symbol,
                                      T_evol::Float64, encoding::Symbol, reset_type::Symbol,
                                      n_substeps::Int=5, H_sparse=nothing;
                                      batch_size::Int=min(n_traj, Threads.nthreads() * 4))
    N = L * n_rails
    n_ops = 3*N + 3*(L-1)*n_rails + 3*L*(n_rails-1)  # Local + intra + inter
    
    X_sum = zeros(Float64, T, n_ops)
    p = Progress(n_traj; desc="MCWF Bitwise Progress: ", showspeed=true)
    
    # Precompute exact unitary if needed
    U_exact = nothing
    if (integrator_type == :exact_cpu || integrator_type == :exact_precomp_cpu) && H_sparse !== nothing
        println("[MCWF] Precomputing exact unitary exp(-iHt)...")
        U_exact = exp(-1im * Matrix(H_sparse) * T_evol)
    end
    
    println("[MCWF] Launching $n_traj Trajectories (BITWISE mode, $N qubits, batch=$batch_size)...")
    
    # Process trajectories in batches
    n_batches = cld(n_traj, batch_size)
    
    for batch_idx in 1:n_batches
        batch_start = (batch_idx - 1) * batch_size + 1
        batch_end = min(batch_idx * batch_size, n_traj)
        batch_n = batch_end - batch_start + 1
        
        batch_sum = zeros(Float64, T, n_ops)
        acc_lock = ReentrantLock()
        
        Threads.@threads for traj_offset in 1:batch_n
            traj_i = batch_start + traj_offset - 1
            try
                curr_psi = copy(psi0)
                local_X = zeros(Float64, T, n_ops)
                
                _run_traj_loop_bitwise!(local_X, curr_psi, padded_u, L, T, n_ops,
                                        nothing, gates, integrator_type, T_evol,
                                        encoding, reset_type, n_rails, n_substeps,
                                        H_sparse, U_exact, H_params, nothing)
                
                lock(acc_lock) do
                    batch_sum .+= local_X
                end
                next!(p)
            catch e
                println("Error in Bitwise Trajectory $traj_i: $e")
                rethrow(e)
            end
        end
        
        X_sum .+= batch_sum
        GC.gc(false)
    end
    
    return X_sum ./ n_traj
end

export run_mcwf_simulation

end # module

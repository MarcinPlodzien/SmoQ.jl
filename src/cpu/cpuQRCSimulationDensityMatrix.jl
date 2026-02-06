# Date: 2026
#
#=
================================================================================
    CPUDensityMatrix.jl - Density Matrix QRC Engine (CPU Implementation)
================================================================================

OVERVIEW
--------
High-level orchestration for QRC workflow using Density Matrix formalism.
Manages the complete Reset-Evolve-Measure cycle for DM simulations.

KEY FUNCTIONALITIES
-------------------
1. Observable Construction: Automated generation of Local and Correlation operators
2. Input Encoding: Mapping time-series windows to tensor product states  
3. Time Evolution: Reset-Evolve-Measure cycle with multiple integrators
4. Diagnostics: SvN entropy, negativity, IPR tracking

SUPPORTED INTEGRATORS
---------------------
- :exact_cpu / :exact_precomp_cpu - Precomputed exp(-iHt)
- :trotter_cpu - Matrix-free Trotter decomposition
 

USAGE
-----
```julia
include("utils/cpu/CPUDensityMatrix.jl")
using .CPUDensityMatrix
features = run_dm_simulation(rho0, H_sparse, params, ...)
```

================================================================================
=#

module CPUQRCSimulationDensityMatrix

# ==============================================================================
# QRC EXPERIMENT HELPERS (DENSITY MATRIX ENGINE)
# ==============================================================================
# ENCODING CONVENTIONS (CRITICAL FOR FAIR PREDICTION EVALUATION)
# ==============================================================================
#
# The calling script (02_run_QRC_parameters_sweep.jl, 09_CPU_TEST.jl) is responsible
# for preparing the input sequence with appropriate padding. This module receives
# base encoding type (:batch or :broadcast) and expects the input to be pre-padded.
#
# ENCODING MODES (parsed by calling script, NOT this module):
#
#   :batch_causal (parsed → :batch + L zeros prepended)
#     • Input window at step k: [u_k, u_{k-1}, ..., u_{k-L+1}]
#     • Contains PAST and CURRENT values only
#     • Prediction target u_{k+τ} is NEVER in window
#     •  FAIR for genuine forecasting evaluation
#
#   :batch_lookahead (parsed → :batch + NO padding)
#     • Input window at step k: [u_{k+1}, u_{k+2}, ..., u_{k+L}]
#     • Contains FUTURE values
#     • For τ ≤ L: target u_{k+τ} IS IN THE WINDOW!
#     • DATA LEAKAGE: C(τ≤L) ≈ 0.999 - NOT genuine prediction!
#     • Valid only for: offline batch processing, feature extraction studies
#
#   :broadcast (parsed → :broadcast + L zeros prepended)
#     • All L input qubits receive same current value u_k
#     • Inherently causal - no future information possible
#     • Lower capacity but unambiguous fairness
#
# QUANTITATIVE DIFFERENCE (L=4, Heisenberg, Protocol 1):
#   batch_causal:    C_total ≈ 13.78, C(τ≤4) ≈ 0.87 (genuine)
#   batch_lookahead: C_total ≈ 15.97, C(τ≤4) ≈ 0.999 (data leakage!)
#   broadcast:       C_total ≈ 12.63, C(τ≤4) ≈ 0.83 (genuine)
#
# The ~2 extra capacity points from lookahead come ENTIRELY from trivial
# data-in-window extraction, NOT genuine forecasting ability.
#
# For publications on PREDICTION capacity: USE batch_causal ONLY!
#
# See NOTES.txt for extended discussion and decision flowchart.
# ==============================================================================

using SparseArrays
using LinearAlgebra
using Printf
using ProgressMeter
using OrdinaryDiffEq
using Statistics

# Use modules from parent context (included by master script)
using ..CPUHamiltonianBuilder: CouplingTerm, FieldTerm, HamiltonianParams, construct_sparse_hamiltonian
using ..CPUQuantumStateObservables: measure_all_observables_density_matrix_cpu
using ..CPUQRCstep: qrc_reset_rho
using ..CPUQuantumStateCharacteristic: inverse_participation_ratio, purity, von_neumann_entropy,
    entanglement_entropy_horizontal, entanglement_entropy_vertical,
    negativity_horizontal, negativity_vertical
using ..CPUQuantumChannelUnitaryEvolutionExact: precompute_exact_propagator_cpu, evolve_exact_rho_cpu!, evolve_exact_psi_cpu!
using ..CPUQuantumChannelUnitaryEvolutionTrotter: FastTrotterGate, precompute_trotter_gates_bitwise_cpu, evolve_trotter_rho_cpu!, evolve_trotter_psi_cpu!, apply_fast_trotter_step_cpu!, apply_fast_trotter_gates_to_matrix_rows_cpu!
using ..CPUQRCstep: encode_input_ry_psi, encode_input_ry_rho

export run_dm_simulation


# ==============================================================================
# ==============================================================================

"""
    run_dm_simulation(rho0, H_sparse, H_params, padded_u, L, n_rails, T, 
                               integrator, T_evol, encoding, reset_type, n_substeps)

Uses raw Matrix{ComplexF64} for density matrix and FastObservables for measurements.

Returns: Matrix{Float64} of shape (T, n_observables)
"""
function run_dm_simulation(rho0::Matrix{ComplexF64}, H_sparse, H_params,
                                    padded_u::Vector{Float64}, L::Int, n_rails::Int, 
                                    T::Int, integrator::Symbol, T_evol::Float64,
                                    encoding::Symbol, reset_type::Symbol, n_substeps::Int;
                                    # Diagnostic options (all optional, defaults = off)
                                    calculate_SvN_horizontal::Bool=false,
                                    calculate_SvN_vertical::Bool=false,
                                    calculate_IPR::Bool=false,
                                    calculate_purity::Bool=false,
                                    calculate_negativity_horizontal::Bool=false,
                                    calculate_negativity_vertical::Bool=false,
                                    diagnostic_sample_step::Int=10)
    N = L * n_rails
    dim = 1 << N
    n_ops = 3*N + 3*(L-1)*n_rails + 3*L*(n_rails-1)  # Local + intra + inter correlators
    features = zeros(Float64, T, n_ops)
    
    # Check if any diagnostics are enabled
    any_diagnostics = calculate_SvN_horizontal || calculate_SvN_vertical || 
                      calculate_IPR || calculate_purity ||
                      calculate_negativity_horizontal || calculate_negativity_vertical
    
    # Initialize diagnostic arrays
    n_diagnostic_samples = any_diagnostics ? div(T, diagnostic_sample_step) + 1 : 0
    SvN_horizontal = calculate_SvN_horizontal ? zeros(Float64, n_diagnostic_samples) : Float64[]
    SvN_vertical = calculate_SvN_vertical ? zeros(Float64, n_diagnostic_samples) : Float64[]
    IPR_vals = calculate_IPR ? zeros(Float64, T) : Float64[]  # IPR is cheap, store every step
    purity_vals = calculate_purity ? zeros(Float64, T) : Float64[]  # Purity is cheap, store every step
    neg_horizontal = calculate_negativity_horizontal ? zeros(Float64, n_diagnostic_samples) : Float64[]
    neg_vertical = calculate_negativity_vertical ? zeros(Float64, n_diagnostic_samples) : Float64[]
    diagnostic_steps = Int[]
    
    # Validate integrator symbol - prevent silent failures
    valid_integrators = [:exact_cpu, :exact_precomp_cpu, :trotter_cpu, :krylov_cpu, :krylov_bitwise_cpu, :chebyshev_cpu, :rk4_sparse]
    if integrator ∉ valid_integrators
        @warn """
        Invalid integrator symbol: $integrator
        Valid CPU DM integrators: $valid_integrators
        
        Common mistakes:
          :trotter     → should be :trotter_cpu
          :exact       → should be :exact_cpu
          :trotter_gpu → GPU integrator, use GPUDensityMatrix instead
        
        Defaulting to :exact_cpu...
        """
        integrator = :exact_cpu
    end
    
    # Precompute for different integrators
    U_exact, U_adj = nothing, nothing
    trotter_gates = FastTrotterGate[]
    
    if integrator == :exact_cpu || integrator == :exact_precomp_cpu
        println("[DM Bitwise] Precomputing exact unitary exp(-iHt)...")
        H_dense = Matrix(H_sparse)
        U_exact = exp(-1im * H_dense * T_evol)
        U_adj = U_exact'
    elseif integrator == :trotter_cpu
        dt = T_evol / n_substeps
        trotter_gates = precompute_trotter_gates_bitwise_cpu(H_params, dt)
    end
    
    println("[DM Bitwise] Starting evolution (T=$T steps, integrator=$integrator)...")
    if any_diagnostics
        println("[DM Bitwise] Computing diagnostics every $diagnostic_sample_step steps")
    end
    current_rho = copy(rho0)
    p = Progress(T, 0.5, "DM Bitwise: ")
    
    diag_idx = 0  # Counter for diagnostic samples
    
    for k in 1:T
        next!(p)
        
        # --- A. INPUT ENCODING ---
        # All modes expect padded_u with L zeros prepended by caller
        # Forward-order window extraction (matches GPU implementation):
        #   :batch_causal    -> [u_{k-L+1}, ..., u_k]  = padded_u[k+q] for q=1:L
        #   :batch           -> same as :batch_causal (legacy alias)
        #   :broadcast       -> u_k for all qubits
        u_win = zeros(Float64, L)
        if encoding == :batch_causal || encoding == :batch
            # Causal: u_win = [u_{k-L+1}, ..., u_k]
            for q in 1:L
                u_win[q] = padded_u[k + q]
            end
        else  # :broadcast
            u_win .= padded_u[k + L]  # all qubits get u_k
        end


        
        # --- B. RESET: Encode input state and tensor with traced-out reservoir ---
        psi_in = encode_input_ry_psi(u_win)
        rho_in = psi_in * psi_in'  # |psi_in><psi_in|
        
        # Apply reset based on protocol
        if reset_type == :traceout_rail_1 || reset_type == :traceout_rail_2
            # Partial trace over input rail, tensor with fresh input
            next_rho = qrc_reset_rho(current_rho, rho_in, L, n_rails, reset_type)
        else
            next_rho = current_rho
        end
        
        # --- C. EVOLUTION ---
        if integrator == :exact_cpu || integrator == :exact_precomp_cpu
            # ρ(t) = U ρ_0 U†
            next_rho = U_exact * next_rho * U_adj
            
        elseif integrator == :trotter_cpu
            # Trotter for DM: apply U to columns, U† to rows
            conj_gates = [FastTrotterGate(g.indices, conj.(g.U)) for g in trotter_gates]
            
            for _ in 1:n_substeps
                # Apply U to each column
                for j in 1:dim
                    col = view(next_rho, :, j)
                    apply_fast_trotter_step_cpu!(col, trotter_gates, N)
                end
                # Apply U† to each row
                apply_fast_trotter_gates_to_matrix_rows_cpu!(next_rho, conj_gates, N)
            end
        else
            error("Unsupported integrator: $integrator. Supported: :exact_cpu, :trotter_cpu")
        end
        
        current_rho = next_rho
        
        # --- D. MEASUREMENT (Bitwise) ---
        features[k, :] = measure_all_observables_density_matrix_cpu(current_rho, L, n_rails)
        
        # --- E. DIAGNOSTICS ---
        if any_diagnostics
            # IPR and Purity are cheap - compute every step if enabled
            if calculate_IPR
                IPR_vals[k] = inverse_participation_ratio(current_rho)
            end
            if calculate_purity
                purity_vals[k] = purity(current_rho)
            end
            
            # Expensive diagnostics - sample periodically
            if k % diagnostic_sample_step == 0 || k == 1
                diag_idx += 1
                push!(diagnostic_steps, k)
                
                if calculate_SvN_horizontal
                    SvN_horizontal[diag_idx] = entanglement_entropy_horizontal(current_rho, L, n_rails)
                end
                if calculate_SvN_vertical
                    SvN_vertical[diag_idx] = entanglement_entropy_vertical(current_rho, L, n_rails)
                end
                if calculate_negativity_horizontal
                    neg_horizontal[diag_idx] = negativity_horizontal(current_rho, L, n_rails)
                end
                if calculate_negativity_vertical
                    neg_vertical[diag_idx] = negativity_vertical(current_rho, L, n_rails)
                end
            end
        end
    end
    
    # Return based on whether diagnostics were computed
    if any_diagnostics
        # Trim arrays to actual size
        SvN_horizontal = calculate_SvN_horizontal ? SvN_horizontal[1:diag_idx] : Float64[]
        SvN_vertical = calculate_SvN_vertical ? SvN_vertical[1:diag_idx] : Float64[]
        neg_horizontal = calculate_negativity_horizontal ? neg_horizontal[1:diag_idx] : Float64[]
        neg_vertical = calculate_negativity_vertical ? neg_vertical[1:diag_idx] : Float64[]
        
        diagnostics = (
            SvN_horizontal = SvN_horizontal,
            SvN_vertical = SvN_vertical,
            IPR = IPR_vals,
            purity = purity_vals,
            neg_horizontal = neg_horizontal,
            neg_vertical = neg_vertical,
            diagnostic_steps = diagnostic_steps
        )
        return features, diagnostics
    else
        return features
    end
end

export run_dm_simulation

end # module


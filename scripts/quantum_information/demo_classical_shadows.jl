#=
================================================================================
    demo_classical_shadows.jl - Classical Shadows Demo
================================================================================
REFERENCE: Huang, Kueng, Preskill, Nature Physics 16, 1050 (2020)

This demo compares two classical shadow measurement groups:

PAULI SHADOWS (:pauli)
----------------------
- Random X/Y/Z single-qubit measurements
- Inverse channel factor: 3 per qubit → 3^N total
- Best for: k-local observables, simple implementation
- Sample complexity: O(3^k / ε²) for k-local observable

LOCAL CLIFFORD SHADOWS (:local_clifford)
-----------------------------------------
- Random single-qubit Cliffords (24-element group per qubit)
- Inverse channel factor: 3 per qubit → 3^N total (same as Pauli)
- Best for: k-local observables with better Bloch sphere coverage
- Sample complexity: O(3^k / ε²) (same as Pauli, but lower variance)

PLOTS GENERATED
---------------
For each configuration (state, N), generates:
  - Individual group plots: observables, correlators, HS distance vs N_shadows
  - Comparison plots: all groups on the same panels
  - Timing comparison: reconstruction time vs N for all groups

NOTE: The reconstructed ρ* is NOT a valid density matrix (not positive semidefinite).
      Observable estimates are unbiased, but purity/fidelity can exceed physical bounds.
================================================================================
=#


using LinearAlgebra
using Statistics
using Printf
using Plots
using LaTeXStrings
using Random
using ProgressMeter
using Dates

Random.seed!(42)

# ==============================================================================
# MODULE LOADING
# ==============================================================================

include("../utils/cpu/cpuQuantumChannelGates.jl")
include("../utils/cpu/cpuQuantumStateMeasurements.jl")
include("../utils/cpu/cpuClassicalShadows.jl")
include("../utils/cpu/cpuQuantumChannelKrausOperators.jl")
include("../utils/cpu/cpuQuantumStatePartialTrace.jl")
include("../utils/cpu/cpuQuantumStatePreparation.jl")
include("../utils/cpu/cpuQuantumStateObservables.jl")

using .CPUQuantumChannelGates
using .CPUQuantumStateMeasurements
using .CPUClassicalShadows
using .CPUQuantumChannelKraus
using .CPUQuantumStatePreparation
using .CPUQuantumStatePartialTrace
using .CPUQuantumStateObservables

println("="^70)
println("  CLASSICAL SHADOWS DEMO")
println("="^70)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

const PRECISION = 3
 
const N_QUBITS_LIST = [2, 3, 4, 5, 6]  # Keep N≤5 for global Clifford speed
const N_SHADOWS_LIST = [100, 1000, 10000, 100000, 1000000]  # Reasonable shadow counts


const STATE_TYPES = [
                    :w, 
                    :ghz
                    ]  # Both test states
const RECONSTRUCTION_METHODS = [
                                :bitwise
                                ]  # Reconstruction algorithm
# Shadow measurement groups:
# - :pauli - Random X/Y/Z single-qubit measurements (factor 3 per qubit)
# - :local_clifford - Random single-qubit Cliffords (equivalent to Pauli for local observables)
const SHADOW_GROUPS = [
                        :pauli, 
                        :local_clifford,
                        :global_clifford
                    ]


const DEPOL_NOISE = 0.1  # Depolarizing noise probability
const SKIP_OBSERVABLES = false  # Enable observables to verify reconstruction
const N_MAX_FULL_TOMOGRAPHY = 14  # Skip full ρ reconstruction for N > this (O(4^N) memory)
const N_MAX_SAVE_DENSITY_MATRIX = 6  # Save ρ matrices to file for N <= this



const OUTPUT_DIR = joinpath(@__DIR__, "demo_classical_shadows")
const DATA_DIR = joinpath(OUTPUT_DIR, "data")
const FIG_DIR = joinpath(OUTPUT_DIR, "figures")

mkpath(DATA_DIR)
mkpath(FIG_DIR)
mkpath(DATA_DIR)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Format time in hh:mm:ss.ms format
function format_time(seconds::Float64)
    h = floor(Int, seconds / 3600)
    m = floor(Int, (seconds % 3600) / 60)
    s = floor(Int, seconds % 60)
    ms = floor(Int, (seconds * 1000) % 1000)
    @sprintf("%02d:%02d:%02d.%03d", h, m, s, ms)
end

# Save timing data to file with thread count
function save_timing_data(N::Int, state_name::String, M_list::Vector{Int}, times::Vector{Float64})
    filename = joinpath(DATA_DIR, "evaluation_time_N$(N)_$(state_name).txt")
    open(filename, "w") do f
        println(f, "# Evaluation Time Data")
        println(f, "# N_qubits: $N")
        println(f, "# State: $state_name")
        println(f, "# Julia threads: $(Threads.nthreads())")
        println(f, "# Date: $(Dates.now())")
        println(f, "#")
        println(f, "# N_shadows, time_seconds, time_formatted")
        for (M, t) in zip(M_list, times)
            println(f, "$M, $(round(t, digits=4)), $(format_time(t))")
        end
    end
end

# Save comparison of target and reconstructed density matrices to a single file
# Format: 6 matrix sections for easy visual comparison
function save_density_matrix_comparison(ρ_target::Matrix{ComplexF64}, ρ_reconstructed::Matrix{ComplexF64}, 
                                         filename::String; label::String="", reconstruction_time::Float64=0.0)
    dim = size(ρ_target, 1)
    Δρ = ρ_target - ρ_reconstructed
    
    # Format a single value with explicit sign for proper column alignment
    pos_fmt = Printf.Format("+%." * string(PRECISION) * "f")
    neg_fmt = Printf.Format("%." * string(PRECISION) * "f")
    
    function fmt_val(x::Float64)
        if x >= 0
            return Printf.format(pos_fmt, x)
        else
            return Printf.format(neg_fmt, x)  # negative sign included
        end
    end
    
    function write_matrix(f, matrix::Matrix{Float64}, title::String)
        println(f, "# === $title ===")
        for i in 1:dim
            row_vals = [fmt_val(matrix[i,j]) for j in 1:dim]
            println(f, join(row_vals, "  "))
        end
        println(f, "")
    end
    
    # Compute metrics
    hs_distance = sqrt(real(tr(Δρ' * Δρ)))  # Hilbert-Schmidt distance: ||ρ_target - ρ_shadows||_HS
    sum_abs_diff = sum(abs.(Δρ))             # Sum of absolute element-wise differences
    
    open(filename, "w") do f
        println(f, "# Density Matrix Comparison")
        println(f, "# Label: $label")
        println(f, "# Dimension: $dim x $dim")
        println(f, "# Precision: $PRECISION digits")
        println(f, "# Date: $(Dates.now())")
        println(f, "#")
        println(f, "# Hilbert-Schmidt distance: $hs_distance")
        println(f, "# Sum of abs element-wise differences: $sum_abs_diff")
        println(f, "# Reconstruction time: $(reconstruction_time) s")
        println(f, "#")
        println(f, "")
        
        # Section 1: Re(target)
        write_matrix(f, real.(ρ_target), "Re(ρ_target)")
        
        # Section 2: Re(reconstructed)
        write_matrix(f, real.(ρ_reconstructed), "Re(ρ_reconstructed)")
        
        # Section 3: Im(target)
        write_matrix(f, imag.(ρ_target), "Im(ρ_target)")
        
        # Section 4: Im(reconstructed)
        write_matrix(f, imag.(ρ_reconstructed), "Im(ρ_reconstructed)")
        
        # Section 5: |Re(ρ_target - ρ_reconstructed)|
        write_matrix(f, abs.(real.(Δρ)), "|Re(ρ_target - ρ_reconstructed)|")
        
        # Section 6: |Im(ρ_target - ρ_reconstructed)|
        write_matrix(f, abs.(imag.(Δρ)), "|Im(ρ_target - ρ_reconstructed)|")
    end
end

# ==============================================================================
# STATE PREPARATION
# ==============================================================================

create_ghz_state(N) = (ψ = zeros(ComplexF64, 1<<N); ψ[1]=ψ[end]=1/sqrt(2); ψ)
create_w_state(N) = (ψ = zeros(ComplexF64, 1<<N); [ψ[(1<<(k-1))+1]=1/sqrt(N) for k in 1:N]; ψ)
function create_dicke_state(N; k=N÷2)
    ψ = zeros(ComplexF64, 1<<N)
    c = 0
    for i in 0:(1<<N)-1
        if count_ones(i) == k; ψ[i+1] = 1.0; c += 1; end
    end
    ψ ./= sqrt(c)
end

function create_haar_state(N)
    ψ = randn(ComplexF64, 1<<N)
    ψ ./= norm(ψ)
end

# Returns: (state, name, latex_title, is_mixed)
# For mixed states, 'state' is a density matrix ρ
function create_state(stype::Symbol, N)
    if stype == :ghz
        ψ = create_ghz_state(N)
        return ψ, "GHZ", L"|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle^{\otimes N} + |1\rangle^{\otimes N})", false
    elseif stype == :w
        ψ = create_w_state(N)
        return ψ, "W", L"|\psi\rangle = \frac{1}{\sqrt{N}}\sum_{k=1}^{N}|1_k\rangle", false
    elseif stype == :dicke
        ψ = create_dicke_state(N)
        k = N ÷ 2
        return ψ, "Dicke", L"|\psi\rangle = |D_{N,%$k}\rangle", false
    elseif stype == :haar
        ψ = create_haar_state(N)
        return ψ, "Haar", "Random", false
    elseif stype == :w_noisy
        ψ = create_w_state(N)
        ρ = ψ * ψ'
        apply_channel_depolarizing!(ρ, DEPOL_NOISE, collect(1:N), N)
        return ρ, "W_noisy", "W + Depol($(DEPOL_NOISE))", true
    end
    error("Unknown: $stype")
end

# ==============================================================================
# SHADOW OBSERVABLE ESTIMATION
# ==============================================================================

"""
Compute local observables ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩ with STATISTICAL UNCERTAINTY.
Uses classical shadows formula for mean and bootstrap for std.
Returns ((X, Y, Z), (X_std, Y_std, Z_std))
"""
function shadow_mean_local_with_std(snapshots::Vector{PauliSnapshot}, N::Int)
    M = length(snapshots)
    
    # Build Pauli strings for qubit 1
    ps_X = zeros(Int, N); ps_X[1] = 1
    ps_Y = zeros(Int, N); ps_Y[1] = 2
    ps_Z = zeros(Int, N); ps_Z[1] = 3
    
    # Compute means using the correct classical shadows formula
    X_mean = get_expectation_value_from_shadows(snapshots, ps_X, N)
    Y_mean = get_expectation_value_from_shadows(snapshots, ps_Y, N)
    Z_mean = get_expectation_value_from_shadows(snapshots, ps_Z, N)
    
    # Bootstrap resampling for std
    n_bootstrap = 50
    X_boot = zeros(Float64, n_bootstrap)
    Y_boot = zeros(Float64, n_bootstrap)
    Z_boot = zeros(Float64, n_bootstrap)
    
    for b in 1:n_bootstrap
        indices = rand(1:M, M)
        resampled = snapshots[indices]
        X_boot[b] = get_expectation_value_from_shadows(resampled, ps_X, N)
        Y_boot[b] = get_expectation_value_from_shadows(resampled, ps_Y, N)
        Z_boot[b] = get_expectation_value_from_shadows(resampled, ps_Z, N)
    end
    
    stds = (std(X_boot), std(Y_boot), std(Z_boot))
    return ((X_mean, Y_mean, Z_mean), stds)
end

# Overload for CliffordSnapshot - compute from per-shadow density matrix
function shadow_mean_local_with_std(snapshots::Vector{CliffordSnapshot}, N::Int)
    M = length(snapshots)
    X_per_shadow = zeros(Float64, M)
    Y_per_shadow = zeros(Float64, M)
    Z_per_shadow = zeros(Float64, M)
    
    for (m, snap) in enumerate(snapshots)
        ρ_m = single_snapshot_to_dm(snap, N)
        X_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:X], [1], N))
        Y_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:Y], [1], N))
        Z_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:Z], [1], N))
    end
    
    means = (mean(X_per_shadow), mean(Y_per_shadow), mean(Z_per_shadow))
    stds = (std(X_per_shadow) / sqrt(M), std(Y_per_shadow) / sqrt(M), std(Z_per_shadow) / sqrt(M))
    return means, stds
end

# Overload for GlobalCliffordSnapshot - compute from per-shadow density matrix
function shadow_mean_local_with_std(snapshots::Vector{GlobalCliffordSnapshot}, N::Int)
    M = length(snapshots)
    dim = 1 << N
    inv_factor = Float64(dim + 1)
    
    X_per_shadow = zeros(Float64, M)
    Y_per_shadow = zeros(Float64, M)
    Z_per_shadow = zeros(Float64, M)
    
    for (m, snap) in enumerate(snapshots)
        U = build_global_clifford_unitary(snap.circuit)
        b_int = 0
        for j in 1:N
            b_int += snap.outcomes[j] << (j-1)
        end
        b_state = zeros(ComplexF64, dim)
        b_state[b_int + 1] = 1.0
        rotated = U' * b_state
        ρ_m = inv_factor * (rotated * rotated') - I(dim)
        
        X_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:X], [1], N))
        Y_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:Y], [1], N))
        Z_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:Z], [1], N))
    end
    
    means = (mean(X_per_shadow), mean(Y_per_shadow), mean(Z_per_shadow))
    stds = (std(X_per_shadow) / sqrt(M), std(Y_per_shadow) / sqrt(M), std(Z_per_shadow) / sqrt(M))
    return means, stds
end

function shadow_mean_local(snapshots::Vector{Snapshot}, N::Int)
    obs = estimate_local_observables(snapshots, N)
    return obs.X[1], obs.Y[1], obs.Z[1]  # Just qubit 1
end



"""
Compute correlators ⟨X₁X₂⟩, ⟨Y₁Y₂⟩, ⟨Z₁Z₂⟩ with STATISTICAL UNCERTAINTY.
Uses classical shadows formula for mean and bootstrap for std.
Returns ((XX, YY, ZZ), (XX_std, YY_std, ZZ_std))
"""
function shadow_mean_correlators(snapshots::Vector{PauliSnapshot}, N::Int)
    if N < 2; return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0); end
    
    M = length(snapshots)
    
    # Build Pauli strings for pair (1, 2)
    ps_XX = zeros(Int, N); ps_XX[1] = 1; ps_XX[2] = 1
    ps_YY = zeros(Int, N); ps_YY[1] = 2; ps_YY[2] = 2
    ps_ZZ = zeros(Int, N); ps_ZZ[1] = 3; ps_ZZ[2] = 3
    
    # Compute means using the correct classical shadows formula
    XX_mean = get_expectation_value_from_shadows(snapshots, ps_XX, N)
    YY_mean = get_expectation_value_from_shadows(snapshots, ps_YY, N)
    ZZ_mean = get_expectation_value_from_shadows(snapshots, ps_ZZ, N)
    
    # Bootstrap resampling for std
    n_bootstrap = 50
    XX_boot = zeros(Float64, n_bootstrap)
    YY_boot = zeros(Float64, n_bootstrap)
    ZZ_boot = zeros(Float64, n_bootstrap)
    
    for b in 1:n_bootstrap
        indices = rand(1:M, M)
        resampled = snapshots[indices]
        XX_boot[b] = get_expectation_value_from_shadows(resampled, ps_XX, N)
        YY_boot[b] = get_expectation_value_from_shadows(resampled, ps_YY, N)
        ZZ_boot[b] = get_expectation_value_from_shadows(resampled, ps_ZZ, N)
    end
    
    stds = (std(XX_boot), std(YY_boot), std(ZZ_boot))
    return ((XX_mean, YY_mean, ZZ_mean), stds)
end




# Overload for CliffordSnapshot - compute from per-shadow reconstructed ρ
function shadow_mean_correlators(snapshots::Vector{CliffordSnapshot}, N::Int)
    if N < 2; return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0); end
    
    M = length(snapshots)
    
    # Compute per-shadow estimates for pair (1, 2)
    XX_per_shadow = zeros(Float64, M)
    YY_per_shadow = zeros(Float64, M)
    ZZ_per_shadow = zeros(Float64, M)
    
    for (m, snap) in enumerate(snapshots)
        ρ_m = single_snapshot_to_dm(snap, N)
        XX_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:X, :X], [1, 2], N))
        YY_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:Y, :Y], [1, 2], N))
        ZZ_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:Z, :Z], [1, 2], N))
    end
    
    means = (mean(XX_per_shadow), mean(YY_per_shadow), mean(ZZ_per_shadow))
    stds = (std(XX_per_shadow) / sqrt(M), std(YY_per_shadow) / sqrt(M), std(ZZ_per_shadow) / sqrt(M))
    return means, stds
end

# Helper: Convert single CliffordSnapshot to density matrix
function single_snapshot_to_dm(snap::CliffordSnapshot, N::Int)
    # Build single-snapshot shadow: ⊗_j [3 C_j† |b_j⟩⟨b_j| C_j - I]
    ρ = single_qubit_clifford_shadow(snap.clifford_indices[N], snap.outcomes[N])
    for j in (N-1):-1:1
        ρ = kron(ρ, single_qubit_clifford_shadow(snap.clifford_indices[j], snap.outcomes[j]))
    end
    return ρ
end

# Overload for GlobalCliffordSnapshot - compute from per-shadow reconstructed ρ
function shadow_mean_correlators(snapshots::Vector{GlobalCliffordSnapshot}, N::Int)
    if N < 2; return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0); end
    
    M = length(snapshots)
    dim = 1 << N
    inv_factor = Float64(dim + 1)  # 2^N + 1
    
    # Compute per-shadow estimates for pair (1, 2)
    XX_per_shadow = zeros(Float64, M)
    YY_per_shadow = zeros(Float64, M)
    ZZ_per_shadow = zeros(Float64, M)
    
    for (m, snap) in enumerate(snapshots)
        # Build single-snapshot shadow: (2^N + 1) C† |b⟩⟨b| C - I
        U = build_global_clifford_unitary(snap.circuit)
        b_int = 0
        for j in 1:N
            b_int += snap.outcomes[j] << (j-1)
        end
        b_state = zeros(ComplexF64, dim)
        b_state[b_int + 1] = 1.0
        rotated = U' * b_state
        ρ_m = inv_factor * (rotated * rotated') - I(dim)
        
        XX_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:X, :X], [1, 2], N))
        YY_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:Y, :Y], [1, 2], N))
        ZZ_per_shadow[m] = real(get_expectation_pauli_string(ρ_m, [:Z, :Z], [1, 2], N))
    end
    
    means = (mean(XX_per_shadow), mean(YY_per_shadow), mean(ZZ_per_shadow))
    stds = (std(XX_per_shadow) / sqrt(M), std(YY_per_shadow) / sqrt(M), std(ZZ_per_shadow) / sqrt(M))
    return means, stds
end
# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

function run_analysis(reconstruction_method::Symbol, state_type::Symbol, N::Int, M_list::Vector{Int}; 
                      shadow_group::Symbol=:pauli)
    state, state_name, latex_title, is_mixed = create_state(state_type, N)
    group_str = string(shadow_group)
    
    # Target density matrix
    if is_mixed
        ρ_target = state
    else
        ψ = state
        ρ_target = ψ * ψ'
    end
    
    # Target observables for specific qubit/pair
    target_X = real(get_expectation_pauli_string(ρ_target, [:X], [1], N))  # Qubit 1
    target_Y = real(get_expectation_pauli_string(ρ_target, [:Y], [1], N))
    target_Z = real(get_expectation_pauli_string(ρ_target, [:Z], [1], N))
    target_XX = N >= 2 ? real(get_expectation_pauli_string(ρ_target, [:X, :X], [1, 2], N)) : 0.0  # Pair 1-2
    target_YY = N >= 2 ? real(get_expectation_pauli_string(ρ_target, [:Y, :Y], [1, 2], N)) : 0.0
    target_ZZ = N >= 2 ? real(get_expectation_pauli_string(ρ_target, [:Z, :Z], [1, 2], N)) : 0.0
    target_purity = real(tr(ρ_target^2))
    
    println("\n  [$reconstruction_method | $group_str | $state_name, N=$N] $latex_title")
    @printf("    Target: ⟨X₁⟩=%.3f, ⟨Y₁⟩=%.3f, ⟨Z₁⟩=%.3f, Purity=%.4f\n", 
            target_X, target_Y, target_Z, target_purity)
    @printf("            ⟨X₁X₂⟩=%.3f, ⟨Y₁Y₂⟩=%.3f, ⟨Z₁Z₂⟩=%.3f\n", target_XX, target_YY, target_ZZ)
    
    n_M = length(M_list)
    
    # Results storage
    reconstruction = (X=zeros(n_M), Y=zeros(n_M), Z=zeros(n_M), 
                      XX=zeros(n_M), YY=zeros(n_M), ZZ=zeros(n_M), hs_dist=zeros(n_M))
    shadow = (X=zeros(n_M), Y=zeros(n_M), Z=zeros(n_M),
              XX=zeros(n_M), YY=zeros(n_M), ZZ=zeros(n_M),
              X_std=zeros(n_M), Y_std=zeros(n_M), Z_std=zeros(n_M),
              XX_std=zeros(n_M), YY_std=zeros(n_M), ZZ_std=zeros(n_M))
    reconstruction_times = zeros(n_M)
    ρ_reconstructed = Matrix{ComplexF64}(undef, 2^N, 2^N)  # Pre-declare for scoping
    
    @showprogress desc="  N_shadows ($group_str): " for (i, M) in enumerate(M_list)
        # Create config with specified measurement group
        config = ShadowConfig(n_qubits=N, n_shots=M, measurement_group=shadow_group)
        
        # Collect shadows using config
        snapshots = is_mixed ? collect_shadows(ρ_target, config) : collect_shadows(state, config)
        
        # Reconstruct density matrix using the specified method
        reconstruction_times[i] = @elapsed begin
            if shadow_group == :local_clifford
                ρ_reconstructed = reconstruct_density_matrix_clifford(snapshots, N)
            elseif shadow_group == :global_clifford
                ρ_reconstructed = reconstruct_density_matrix_global_clifford(snapshots, N)
            elseif reconstruction_method == :kron
                ρ_reconstructed = reconstruct_density_matrix_shadows_kron(snapshots, N)
            else  # :bitwise (default for :pauli)
                ρ_reconstructed = reconstruct_density_matrix_shadows_bitwise(snapshots, N)
            end
        end

        
        if !SKIP_OBSERVABLES
            # Observables from reconstructed density matrix (qubit 1 and pair 1-2)
            reconstruction.X[i] = real(get_expectation_pauli_string(ρ_reconstructed, [:X], [1], N))
            reconstruction.Y[i] = real(get_expectation_pauli_string(ρ_reconstructed, [:Y], [1], N))
            reconstruction.Z[i] = real(get_expectation_pauli_string(ρ_reconstructed, [:Z], [1], N))
            reconstruction.XX[i] = N >= 2 ? real(get_expectation_pauli_string(ρ_reconstructed, [:X, :X], [1, 2], N)) : 0.0
            reconstruction.YY[i] = N >= 2 ? real(get_expectation_pauli_string(ρ_reconstructed, [:Y, :Y], [1, 2], N)) : 0.0
            reconstruction.ZZ[i] = N >= 2 ? real(get_expectation_pauli_string(ρ_reconstructed, [:Z, :Z], [1, 2], N)) : 0.0
        end
        
        # Hilbert-Schmidt distance
        Δρ = ρ_reconstructed - ρ_target
        reconstruction.hs_dist[i] = sqrt(real(tr(Δρ' * Δρ)))
        
        # Shadow estimation (qubit 1 and pair 1-2)
        if !SKIP_OBSERVABLES
            # Shadow estimation with std - use qubit 1 values
            obs = estimate_local_observables(snapshots, N)
            shadow.X[i], shadow.Y[i], shadow.Z[i] = obs.X[1], obs.Y[1], obs.Z[1]  # Just qubit 1
            # Bootstrap std for qubit 1 observables
            local_means, local_stds = shadow_mean_local_with_std(snapshots, N)
            shadow.X_std[i], shadow.Y_std[i], shadow.Z_std[i] = local_stds
            
            corr_means, corr_stds = shadow_mean_correlators(snapshots, N)
            shadow.XX[i], shadow.YY[i], shadow.ZZ[i] = corr_means
            shadow.XX_std[i], shadow.YY_std[i], shadow.ZZ_std[i] = corr_stds
        end

        
        # Print table row
        t_str = format_time(reconstruction_times[i])
        @printf("    N_s=%7d | t_reconstruction=%s (%.3fs) | HS=%.4f | ⟨Z⟩=%.2f±%.2f\n",
                M, t_str, reconstruction_times[i], reconstruction.hs_dist[i], 
                shadow.Z[i], shadow.Z_std[i])
        
        # Save density matrix comparison for small N
        if N <= N_MAX_SAVE_DENSITY_MATRIX
            N_str = @sprintf("N%02d", N)  # e.g., N03, N10
            M_str = M >= 1000 ? @sprintf("%.0e", M) : string(M)  # e.g., 1e+05 for 100000
            comp_file = joinpath(DATA_DIR, "rho_comparison_$(reconstruction_method)_$(state_name)_$(N_str)_Nshadows$(M_str).txt")
            save_density_matrix_comparison(ρ_target, ρ_reconstructed, comp_file;
                                           label="$state_name, N=$N, Nshadows=$M, method=$reconstruction_method",
                                           reconstruction_time=reconstruction_times[i])
            println("    -> Saved: rho_comparison_$(reconstruction_method)_$(state_name)_$(N_str)_Nshadows$(M_str).txt")
        end
    end
    
    # Save timing data with method name
    save_timing_data(N, "$(state_name)_$(reconstruction_method)_$(shadow_group)", M_list, reconstruction_times)
    
    target = (X=target_X, Y=target_Y, Z=target_Z, XX=target_XX, YY=target_YY, ZZ=target_ZZ,
              purity=target_purity)
    return Dict(:reconstruction_method=>reconstruction_method, :state=>state_name, :N=>N, 
                :M_list=>M_list, :latex=>latex_title, :target=>target, 
                :reconstruction=>reconstruction, :shadow=>shadow, :times=>reconstruction_times,
                :shadow_group=>shadow_group)
end


# ==============================================================================
# PLOTTING
# ==============================================================================

function generate_plot(results::Dict)
    reconstruction_method = results[:reconstruction_method]
    state_name = results[:state]
    latex_title = results[:latex]
    N = results[:N]
    M_list = results[:M_list]
    t = results[:target]
    r = results[:reconstruction]
    s = results[:shadow]
    shadow_group = get(results, :shadow_group, :pauli)  # Default for backward compat
    # Nice formatted group names for titles
    group_display = Dict(
        :pauli => "Group: Pauli",
        :local_clifford => "Group: Local Clifford", 
        :global_clifford => "Group: Global Clifford"
    )
    group_str = get(group_display, shadow_group, string(shadow_group))
    # Filename-safe group names (no spaces)
    group_filename = string(shadow_group)

    
    ms, lw = 8, 2.5
    
    # Compute xticks for powers of 10
    x_min_pow = floor(Int, log10(minimum(M_list)))
    x_max_pow = ceil(Int, log10(maximum(M_list)))
    xticks_vals = [10.0^k for k in x_min_pow:x_max_pow]
    
    function make_subplot(reconstruction_data, estimation_data, target_value, ylabel, title; estimation_error=nothing)
        p = plot(M_list, reconstruction_data, label=L"\rho_{\mathrm{shadow}}", marker=:circle, markersize=ms,
                 linewidth=lw, linestyle=:solid, color=:red,
                 xlabel=L"N_{\mathrm{shadows}}", ylabel=ylabel, title=title, xscale=:log10,
                 legend=:topright, legendfontsize=7, xticks=xticks_vals, minorticks=true)
        if estimation_error !== nothing
            plot!(p, M_list, estimation_data, ribbon=estimation_error, label=L"\mathrm{Estimation}\pm\sigma", 
                  marker=:diamond, markersize=ms, linewidth=lw, linestyle=:dot, 
                  color=:blue, fillalpha=0.2)
        else
            plot!(p, M_list, estimation_data, label="Estimation", marker=:diamond, markersize=ms,
                  linewidth=lw, linestyle=:dot, color=:blue)
        end
        hline!(p, [target_value], linestyle=:dash, linewidth=2.5, color=:black, label="Target")
        return p
    end
    
    # Row 1: Local observables (qubit 1)
    p1 = make_subplot(r.X, s.X, t.X, L"\langle X_1 \rangle", L"\langle X_1 \rangle", estimation_error=s.X_std)
    p2 = make_subplot(r.Y, s.Y, t.Y, L"\langle Y_1 \rangle", L"\langle Y_1 \rangle", estimation_error=s.Y_std)
    p3 = make_subplot(r.Z, s.Z, t.Z, L"\langle Z_1 \rangle", L"\langle Z_1 \rangle", estimation_error=s.Z_std)
    
    # Row 2: Correlators (qubits 1-2)
    p4 = make_subplot(r.XX, s.XX, t.XX, L"\langle X_1 X_2 \rangle", L"\langle X_1 X_2 \rangle", estimation_error=s.XX_std)
    p5 = make_subplot(r.YY, s.YY, t.YY, L"\langle Y_1 Y_2 \rangle", L"\langle Y_1 Y_2 \rangle", estimation_error=s.YY_std)
    p6 = make_subplot(r.ZZ, s.ZZ, t.ZZ, L"\langle Z_1 Z_2 \rangle", L"\langle Z_1 Z_2 \rangle", estimation_error=s.ZZ_std)
    
    # Row 3: Hilbert-Schmidt distance
    p7 = plot(M_list, r.hs_dist, label=L"||\rho^* - \rho||_{HS}", 
              marker=:circle, markersize=ms, linewidth=lw, color=:blue,
              xlabel=L"N_{\mathrm{shadows}}", ylabel=L"||\rho^* - \rho||_{HS}", 
              title="Hilbert-Schmidt Distance",
              xscale=:log10, yscale=:log10, legend=:topright,
              minorticks=true, minorgrid=true)
    
    # Layout with reconstruction method AND shadow group in title
    layout = @layout [a b c; d e f; g{0.33h}]
    p_all = plot(p1, p2, p3, p4, p5, p6, p7, layout=layout, size=(1400, 1000), dpi=150,
                 plot_title="$(group_str) shadows | $(reconstruction_method) | $state_name (N=$N): $latex_title",
                 left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    
    fig_name = "fig_classical_shadows_$(group_filename)_$(reconstruction_method)_$(state_name)_N$(N).png"

    savefig(p_all, joinpath(FIG_DIR, fig_name))
    println("    -> Figure: $fig_name")
end

"""
Generate comparison plot with BOTH Pauli and Clifford shadows on same panels.
Allows direct visual comparison of reconstruction quality between groups.
"""
function generate_comparison_plot(results_pauli::Dict, results_clifford::Dict)
    # Extract common info (should be same for both)
    reconstruction_method = results_pauli[:reconstruction_method]
    state_name = results_pauli[:state]
    latex_title = results_pauli[:latex]
    N = results_pauli[:N]
    M_list = results_pauli[:M_list]
    t = results_pauli[:target]
    
    # Results from both groups
    r_pauli = results_pauli[:reconstruction]
    r_cliff = results_clifford[:reconstruction]
    
    ms, lw = 6, 2.0
    
    # Compute xticks for powers of 10
    x_min_pow = floor(Int, log10(minimum(M_list)))
    x_max_pow = ceil(Int, log10(maximum(M_list)))
    xticks_vals = [10.0^k for k in x_min_pow:x_max_pow]
    
    function make_comparison_subplot(pauli_data, cliff_data, target_value, ylabel, title)
        p = plot(M_list, pauli_data, label="Pauli", marker=:circle, markersize=ms,
                 linewidth=lw, linestyle=:solid, color=:blue,
                 xlabel=L"N_{\mathrm{shadows}}", ylabel=ylabel, title=title, xscale=:log10,
                 legend=:topright, legendfontsize=7, xticks=xticks_vals, minorticks=true)
        plot!(p, M_list, cliff_data, label="Clifford", marker=:square, markersize=ms,
              linewidth=lw, linestyle=:dash, color=:red)
        hline!(p, [target_value], linestyle=:dash, linewidth=2.5, color=:black, label="Target")
        return p
    end
    
    # Row 1: Local observables (qubit 1)
    p1 = make_comparison_subplot(r_pauli.X, r_cliff.X, t.X, L"\langle X_1 \rangle", L"\langle X_1 \rangle")
    p2 = make_comparison_subplot(r_pauli.Y, r_cliff.Y, t.Y, L"\langle Y_1 \rangle", L"\langle Y_1 \rangle")
    p3 = make_comparison_subplot(r_pauli.Z, r_cliff.Z, t.Z, L"\langle Z_1 \rangle", L"\langle Z_1 \rangle")
    
    # Row 2: Correlators (qubits 1-2)
    p4 = make_comparison_subplot(r_pauli.XX, r_cliff.XX, t.XX, L"\langle X_1 X_2 \rangle", L"\langle X_1 X_2 \rangle")
    p5 = make_comparison_subplot(r_pauli.YY, r_cliff.YY, t.YY, L"\langle Y_1 Y_2 \rangle", L"\langle Y_1 Y_2 \rangle")
    p6 = make_comparison_subplot(r_pauli.ZZ, r_cliff.ZZ, t.ZZ, L"\langle Z_1 Z_2 \rangle", L"\langle Z_1 Z_2 \rangle")
    
    # Row 3: HS distance comparison
    p7 = plot(M_list, r_pauli.hs_dist, label="Pauli", marker=:circle, markersize=ms,
              linewidth=lw, linestyle=:solid, color=:blue,
              xlabel=L"N_{\mathrm{shadows}}", ylabel=L"||\rho^* - \rho||_{HS}", 
              title="Hilbert-Schmidt Distance",
              xscale=:log10, yscale=:log10, legend=:topright,
              minorticks=true, minorgrid=true)
    plot!(p7, M_list, r_cliff.hs_dist, label="Clifford", marker=:square, markersize=ms,
          linewidth=lw, linestyle=:dash, color=:red)
    
    # Layout
    layout = @layout [a b c; d e f; g{0.33h}]
    p_all = plot(p1, p2, p3, p4, p5, p6, p7, layout=layout, size=(1400, 1000), dpi=150,
                 plot_title="Pauli vs Clifford | $(reconstruction_method) | $state_name (N=$N): $latex_title",
                 left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    
    fig_name = "fig_classical_shadows_comparison_$(reconstruction_method)_$(state_name)_N$(N).png"
    savefig(p_all, joinpath(FIG_DIR, fig_name))
    println("    -> Comparison Figure: $fig_name")
end

"""
Generate comparison plot with ALL THREE shadow groups on same panels.
"""
function generate_all_groups_comparison(results_dict::Dict)
    # results_dict has keys :pauli, :local_clifford, :global_clifford
    # Each value is the results dict from run_analysis
    
    r_pauli = results_dict[:pauli]
    r_local = results_dict[:local_clifford]
    r_global = results_dict[:global_clifford]
    
    reconstruction_method = r_pauli[:reconstruction_method]
    state_name = r_pauli[:state]
    latex_title = r_pauli[:latex]
    N = r_pauli[:N]
    M_list = r_pauli[:M_list]
    t = r_pauli[:target]
    
    # Get observables from reconstruction
    rp = r_pauli[:reconstruction]
    rl = r_local[:reconstruction]
    rg = r_global[:reconstruction]
    
    # Shadow estimation with stds
    sp = r_pauli[:shadow]
    sl = r_local[:shadow]
    sg = r_global[:shadow]
    
    ms, lw = 6, 2.0
    
    x_min_pow = floor(Int, log10(minimum(M_list)))
    x_max_pow = ceil(Int, log10(maximum(M_list)))
    xticks_vals = [10.0^k for k in x_min_pow:x_max_pow]
    
    colors = Dict(:pauli => :blue, :local_clifford => :green, :global_clifford => :red)
    markers = Dict(:pauli => :circle, :local_clifford => :square, :global_clifford => :diamond)
    labels = Dict(:pauli => "Pauli", :local_clifford => "Local Clifford", :global_clifford => "Global Clifford")
    
    function make_3group_subplot(pauli_data, local_data, global_data, target_value, ylabel, title;
                                 pauli_std=nothing, local_std=nothing, global_std=nothing)
        p = plot(M_list, pauli_data, label=labels[:pauli], marker=markers[:pauli], markersize=ms,
                 linewidth=lw, color=colors[:pauli],
                 xlabel=L"N_{\mathrm{shadows}}", ylabel=ylabel, title=title, xscale=:log10,
                 legend=:topright, legendfontsize=6, xticks=xticks_vals, minorticks=true)
        if pauli_std !== nothing
            plot!(p, M_list, pauli_data, ribbon=pauli_std, fillalpha=0.15, label="", color=colors[:pauli])
        end
        
        plot!(p, M_list, local_data, label=labels[:local_clifford], marker=markers[:local_clifford], 
              markersize=ms, linewidth=lw, color=colors[:local_clifford])
        if local_std !== nothing
            plot!(p, M_list, local_data, ribbon=local_std, fillalpha=0.15, label="", color=colors[:local_clifford])
        end
        
        plot!(p, M_list, global_data, label=labels[:global_clifford], marker=markers[:global_clifford],
              markersize=ms, linewidth=lw, color=colors[:global_clifford])
        if global_std !== nothing
            plot!(p, M_list, global_data, ribbon=global_std, fillalpha=0.15, label="", color=colors[:global_clifford])
        end
        
        hline!(p, [target_value], linestyle=:dash, linewidth=2.5, color=:black, label="Target")
        return p
    end
    
    # Row 1: Local observables (qubit 1) with std
    p1 = make_3group_subplot(sp.X, sl.X, sg.X, t.X, L"\langle X_1 \rangle", L"\langle X_1 \rangle",
                             pauli_std=sp.X_std, local_std=sl.X_std, global_std=sg.X_std)
    p2 = make_3group_subplot(sp.Y, sl.Y, sg.Y, t.Y, L"\langle Y_1 \rangle", L"\langle Y_1 \rangle",
                             pauli_std=sp.Y_std, local_std=sl.Y_std, global_std=sg.Y_std)
    p3 = make_3group_subplot(sp.Z, sl.Z, sg.Z, t.Z, L"\langle Z_1 \rangle", L"\langle Z_1 \rangle",
                             pauli_std=sp.Z_std, local_std=sl.Z_std, global_std=sg.Z_std)
    
    # Row 2: Correlators (qubits 1-2) with std
    p4 = make_3group_subplot(sp.XX, sl.XX, sg.XX, t.XX, L"\langle X_1 X_2 \rangle", L"\langle X_1 X_2 \rangle",
                             pauli_std=sp.XX_std, local_std=sl.XX_std, global_std=sg.XX_std)
    p5 = make_3group_subplot(sp.YY, sl.YY, sg.YY, t.YY, L"\langle Y_1 Y_2 \rangle", L"\langle Y_1 Y_2 \rangle",
                             pauli_std=sp.YY_std, local_std=sl.YY_std, global_std=sg.YY_std)
    p6 = make_3group_subplot(sp.ZZ, sl.ZZ, sg.ZZ, t.ZZ, L"\langle Z_1 Z_2 \rangle", L"\langle Z_1 Z_2 \rangle",
                             pauli_std=sp.ZZ_std, local_std=sl.ZZ_std, global_std=sg.ZZ_std)
    
    # Row 3: HS distance comparison
    p7 = plot(M_list, rp.hs_dist, label=labels[:pauli], marker=markers[:pauli], markersize=ms,
              linewidth=lw, color=colors[:pauli],
              xlabel=L"N_{\mathrm{shadows}}", ylabel=L"||\rho^* - \rho||_{HS}",
              title="Hilbert-Schmidt Distance", xscale=:log10, yscale=:log10,
              legend=:topright, legendfontsize=6, xticks=xticks_vals, minorticks=true, minorgrid=true)
    plot!(p7, M_list, rl.hs_dist, label=labels[:local_clifford], marker=markers[:local_clifford],
          markersize=ms, linewidth=lw, color=colors[:local_clifford])
    plot!(p7, M_list, rg.hs_dist, label=labels[:global_clifford], marker=markers[:global_clifford],
          markersize=ms, linewidth=lw, color=colors[:global_clifford])
    
    layout = @layout [a b c; d e f; g{0.33h}]
    p_all = plot(p1, p2, p3, p4, p5, p6, p7, layout=layout, size=(1400, 1000), dpi=150,
                 plot_title="All Groups Comparison | $state_name (N=$N): $latex_title",
                 left_margin=10Plots.mm, bottom_margin=5Plots.mm)
    
    fig_name = "fig_classical_shadows_all_groups_$(state_name)_N$(N).png"
    savefig(p_all, joinpath(FIG_DIR, fig_name))
    println("    -> All Groups Comparison: $fig_name")
end


# ==============================================================================
# MAIN
# ==============================================================================


function main()
    # JIT warmup with small N=4 to pre-compile all functions
    println("  [JIT Warmup] Compiling with N=4, 1 shadow...")
    warmup_t = @elapsed begin
        ψ_warmup = create_ghz_state(4)
        snaps_warmup = collect_shadows(ψ_warmup, 4, 1)
        _ = reconstruct_density_matrix_shadows_bitwise(snaps_warmup, 4)
        _ = estimate_local_observables(snaps_warmup, 4)
        # Also warmup local Clifford shadows
        config_local = ShadowConfig(n_qubits=4, n_shots=1, measurement_group=:local_clifford)
        snaps_local = collect_shadows(ψ_warmup, config_local)
        _ = reconstruct_density_matrix_clifford(snaps_local, 4)
    end

    @printf("  [JIT Warmup] Done in %.2fs\n\n", warmup_t)
    
    println("  Config: groups=$SHADOW_GROUPS, reconstruction=$RECONSTRUCTION_METHODS, N=$N_QUBITS_LIST, N_shadows=$N_SHADOWS_LIST, States=$STATE_TYPES\n")
    
    # Grid search: outermost=group, then reconstruction, then state, then N
    # Collect timing data for comparison plots keyed by (group, method, N)
    all_timing_data = Dict{Tuple{Symbol, Symbol, Int}, Vector{Float64}}()
    # Store all results for comparison plots keyed by (group, method, state, N)
    all_results = Dict{Tuple{Symbol, Symbol, Symbol, Int}, Dict}()
    
    for (shadow_group, reconstruction_method, state_type, N) in Iterators.product(SHADOW_GROUPS, RECONSTRUCTION_METHODS, STATE_TYPES, N_QUBITS_LIST)
        results = run_analysis(reconstruction_method, state_type, N, N_SHADOWS_LIST; shadow_group=shadow_group)
        generate_plot(results)
        
        # Store results for comparison plots
        all_results[(shadow_group, reconstruction_method, state_type, N)] = results
        
        # Store timing for each (group, method, N) combination - only first state type to avoid duplication
        if state_type == first(STATE_TYPES)
            all_timing_data[(shadow_group, reconstruction_method, N)] = results[:times]
        end
    end

    # Generate comparison plots (all 3 groups on same plot)
    println("\n  --- Generating All Groups Comparison Plots ---")
    for (reconstruction_method, state_type, N) in Iterators.product(RECONSTRUCTION_METHODS, STATE_TYPES, N_QUBITS_LIST)
        pauli_key = (:pauli, reconstruction_method, state_type, N)
        local_key = (:local_clifford, reconstruction_method, state_type, N)
        global_key = (:global_clifford, reconstruction_method, state_type, N)
        
        if haskey(all_results, pauli_key) && haskey(all_results, local_key) && haskey(all_results, global_key)
            results_dict = Dict(
                :pauli => all_results[pauli_key],
                :local_clifford => all_results[local_key],
                :global_clifford => all_results[global_key]
            )
            generate_all_groups_comparison(results_dict)
        end
    end


    
    # Generate time scaling plots for each (group, method) combination
    for (shadow_group, reconstruction_method) in Iterators.product(SHADOW_GROUPS, RECONSTRUCTION_METHODS)
        method_timing = Dict{Int, Vector{Float64}}()
        for N in N_QUBITS_LIST
            key = (shadow_group, reconstruction_method, N)
            if haskey(all_timing_data, key)
                method_timing[N] = all_timing_data[key]
            end
        end
        if !isempty(method_timing)
            generate_timing_plot(method_timing, reconstruction_method; shadow_group=shadow_group)
        end
    end
    
    # Save timing comparison to txt file (for both groups)
    for shadow_group in SHADOW_GROUPS
        timing_file = joinpath(DATA_DIR, "timing_$(shadow_group)_vs_N.txt")
        open(timing_file, "w") do io
            println(io, "# Reconstruction timing comparison ($shadow_group shadows)")
            println(io, "# N\ttime[s]")
            for N in N_QUBITS_LIST
                key = (shadow_group, :bitwise, N)
                t = haskey(all_timing_data, key) ? all_timing_data[key][1] : NaN
                @printf(io, "%d\t%.6f\n", N, t)
            end
        end
        println("    -> Timing data: $timing_file")
    end
    
    # Generate time vs N comparison plot for each group
    for shadow_group in SHADOW_GROUPS
        # Filter timing data for this group
        group_timing = Dict{Tuple{Symbol, Int}, Vector{Float64}}()
        for ((g, method, N), times) in all_timing_data
            if g == shadow_group
                group_timing[(method, N)] = times
            end
        end
        generate_timing_vs_N_comparison(group_timing, N_SHADOWS_LIST[1]; shadow_group=shadow_group)
    end
    
    # Generate combined timing comparison plot (all groups on same figure)
    generate_combined_timing_plot(all_timing_data, N_QUBITS_LIST, N_SHADOWS_LIST)
    
    println("\n" * "="^70)
    println("  DONE - Outputs: demo_classical_shadows/figures/")
    println("="^70)
end



# Time scaling plot: time (log10) vs N_shadows for each N_qubits
function generate_timing_plot(timing_data::Dict{Int, Vector{Float64}}, reconstruction_method::Symbol; 
                              shadow_group::Symbol=:pauli)
    M_list = N_SHADOWS_LIST
    ms, lw = 8, 2.5
    group_str = string(shadow_group)
    
    p = plot(xlabel=L"N_{\mathrm{shadows}}", ylabel=L"\mathrm{Time\,[s]}", 
             title="Timing: $(group_str) shadows ($(reconstruction_method))",
             xscale=:log10, yscale=:log10, legend=:topleft,
             minorticks=true, minorgrid=true)
    
    colors = [:blue, :red, :green, :purple, :orange, :cyan, :magenta, :brown]
    for (idx, N) in enumerate(sort(collect(keys(timing_data))))
        color = colors[mod1(idx, length(colors))]
        plot!(p, M_list, timing_data[N], label="N=$N", 
              marker=:circle, markersize=ms, linewidth=lw, color=color)
    end
    
    fig_name = "fig_time_$(group_str)_$(reconstruction_method)_vs_N.png"
    savefig(p, joinpath(FIG_DIR, fig_name))
    println("    -> Figure: $fig_name")
end

# Time vs N plot (log10 y-axis)
function generate_timing_vs_N_comparison(all_timing_data::Dict{Tuple{Symbol, Int}, Vector{Float64}}, 
                                         n_shadows::Int; shadow_group::Symbol=:pauli)
    ms, lw = 8, 2.5
    group_str = string(shadow_group)
    
    p = plot(xlabel=L"N", ylabel=L"\mathrm{Time\,[s]}", 
             title="$(group_str) Shadow Reconstruction Time ($n_shadows shadows)",
             yscale=:log10, legend=:topleft,
             minorticks=true, minorgrid=true)
    
    # Extract timing for each method
    for (method, color, lstyle) in [(:kron, :blue, :solid), (:bitwise, :red, :dash)]
        N_vals = Int[]
        t_vals = Float64[]
        for N in sort(N_QUBITS_LIST)
            key = (method, N)
            if haskey(all_timing_data, key) && !isempty(all_timing_data[key])
                push!(N_vals, N)
                push!(t_vals, all_timing_data[key][1])  # First shadow count
            end
        end
        if !isempty(N_vals)
            plot!(p, N_vals, t_vals, label=string(method), 
                  marker=:circle, markersize=ms, linewidth=lw, 
                  color=color, linestyle=lstyle)
        end
    end
    
    fig_name = "fig_time_vs_N_$(group_str)_$(n_shadows)shadows.png"
    savefig(p, joinpath(FIG_DIR, fig_name))
    println("    -> Figure: $fig_name")
end

"""
Generate combined timing plot comparing all shadow groups on the same figure.
"""
function generate_combined_timing_plot(all_timing_data::Dict, N_QUBITS_LIST::Vector{Int}, N_SHADOWS_LIST::Vector{Int})
    ms, lw = 8, 2.5
    
    colors = Dict(:pauli => :blue, :local_clifford => :green, :global_clifford => :red)
    markers = Dict(:pauli => :circle, :local_clifford => :square, :global_clifford => :diamond)
    labels = Dict(:pauli => "Pauli", :local_clifford => "Local Clifford", :global_clifford => "Global Clifford")
    
    # Plot for first shadow count
    n_shadows = N_SHADOWS_LIST[1]
    
    p = plot(xlabel="N (qubits)", ylabel=L"\mathrm{Time\,[s]}", 
             title="Reconstruction Time Comparison ($(n_shadows) shadows)",
             yscale=:log10, legend=:topleft,
             minorticks=true, minorgrid=true)
    
    for group in [:pauli, :local_clifford, :global_clifford]
        N_vals = Int[]
        t_vals = Float64[]
        for N in sort(N_QUBITS_LIST)
            key = (group, :bitwise, N)
            if haskey(all_timing_data, key) && !isempty(all_timing_data[key])
                push!(N_vals, N)
                push!(t_vals, all_timing_data[key][1])  # First shadow count
            end
        end
        if !isempty(N_vals)
            plot!(p, N_vals, t_vals, label=labels[group], 
                  marker=markers[group], markersize=ms, linewidth=lw, 
                  color=colors[group])
        end
    end
    
    fig_name = "fig_timing_all_groups_comparison.png"
    savefig(p, joinpath(FIG_DIR, fig_name))
    println("    -> Combined Timing: $fig_name")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

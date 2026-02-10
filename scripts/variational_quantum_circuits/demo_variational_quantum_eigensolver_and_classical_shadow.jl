#!/usr/bin/env julia
#=
================================================================================
    demo_variational_quantum_eigensolver_and_classical_shadow.jl
================================================================================

TITLE:
    Variational Quantum Eigensolver (VQE) + Classical Shadow Tomography Demo

DESCRIPTION:
    This demo integrates two core quantum computing primitives:

    1. VARIATIONAL QUANTUM EIGENSOLVER (VQE)
       - Hybrid quantum-classical algorithm to find ground states
       - Uses parameterized quantum circuits (ansatzes) optimized via gradient descent
       - Supports multiple optimizers: SPSA (stochastic) and Enzyme autodiff (exact)

    2. CLASSICAL SHADOW TOMOGRAPHY
       - Efficient method to estimate many observables from few measurements
       - Uses random Pauli measurements (X, Y, Z rotations before Z-basis)
       - Scales as O(log(M)/ε²) for M observables to accuracy ε

    The combination simulates a realistic near-term quantum experiment workflow:
    prepare a state via VQE, then characterize it via classical shadows.

PHYSICS BACKGROUND:
    The target Hamiltonians are XXZ spin chains with transverse field:

       H = Σᵢ [Jₓ Xᵢ Xᵢ₊₁ + Jᵧ Yᵢ Yᵢ₊₁ + Jᵤ Zᵢ Zᵢ₊₁] + hₓ Σᵢ Xᵢ

    Configurations:
    - XXZ:  Jₓ = Jᵧ = -1, Jᵤ = -0.5, hₓ = 1  (anisotropic antiferromagnet)
    - XY:   Jₓ = Jᵧ = -1, Jᵤ = 0,    hₓ = 1  (planar model)

ALGORITHM WORKFLOW:
    For each configuration in the grid search:

    1. GROUND STATE REFERENCE
       - Compute exact E_gs and |ψ_gs⟩ via Lanczos algorithm
       - Calculate reference observables ⟨O⟩_gs for comparison

    2. VQE TRAINING
       - Build parametrized ansatz: HEA (hardware-efficient) or brick
       - Initialize parameters θ ~ N(0, 0.1)
       - Optimize via Adam with either:
         • spsa_adam:     SPSA stochastic gradient (2 cost evaluations/step)
         • autograd_adam: Enzyme autodiff (exact gradient, more expensive)
       - Track energy E(θ) and fidelity F = |⟨ψ_gs|ψ(θ)⟩|²

    3. OBSERVABLE ESTIMATION
       - Compute exact observables on VQE state: ⟨O⟩_vqe
       - Estimate via classical shadows: ⟨O⟩_shadow ± σ
       - Three-way comparison: GS exact vs VQE exact vs Shadow estimate

    4. ANALYSIS
       - Error: |⟨O⟩_shadow - ⟨O⟩_vqe| (shadow accuracy vs VQE "truth")
       - Scaling: errors should decay as 1/√M with shadow count M

OBSERVABLES ESTIMATED:
    1-body:  ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩       (single-qubit magnetization)
    2-body:  ⟨X₁X₂⟩, ⟨Y₁Y₂⟩, ⟨Z₁Z₂⟩  (nearest-neighbor correlations)
    3-body:  ⟨X₁X₂X₃⟩, ⟨Y₁Y₂Y₃⟩, ⟨Z₁Z₂Z₃⟩  (3-body correlations)

GRID SEARCH PARAMETERS:
    - Hamiltonians:   XXZ, XY (different coupling regimes)
    - System sizes:   N = 8 qubits (configurable via N_VALUES)
    - Ansatzes:       :hea (hardware-efficient), :brick (brick layout)
    - VQE layers:     L = 2, 4, 6 (circuit depth)
    - Optimizers:     :spsa_adam, :autograd_adam (gradient estimation method)
    - Shadow counts:  M = 100, 1K, 10K, 100K, 1M (measurement budget)

    Total configs = |Ham| × |N| × |Ansatz| × |L| × |Optimizer| = 2×1×2×3×2 = 24

OUTPUT:
    Figures:
    - energy_convergence_*.png:  VQE training curves per Hamiltonian
    - observable_comparison_*.png: Shadow vs VQE vs GS observables

    Data:
    - vqe_shadow_analysis.csv:  Full results table with all observables

DEPENDENCIES:
    - cpuQuantumStateLanczos:       Lanczos ground state solver
    - cpuQuantumStateObservables:   Exact observable computation (expect_local, get_expectation_pauli_string)
    - cpuClassicalShadows:          Shadow collection and estimation
    - cpuVQCEnzymeWrapper:          Enzyme autodiff for exact gradients
    - Optimisers.jl:                Adam optimizer

AUTHOR:
    VQE + Classical Shadows Integration Demo
    Date: 2026
================================================================================
=#

using LinearAlgebra
using Random
using Printf
using Plots
using LaTeXStrings
using Statistics
using Optimisers

# ==============================================================================
# PATHS & INCLUDES
# ==============================================================================
const OUTPUT_DIR = joinpath(@__DIR__, "demo_variational_quantum_eigensolver_and_classical_shadow")
mkpath(OUTPUT_DIR)
mkpath(joinpath(OUTPUT_DIR, "figures"))
mkpath(joinpath(OUTPUT_DIR, "data"))

const UTILS_CPU = joinpath(@__DIR__, "..", "utils", "cpu")
include(joinpath(UTILS_CPU, "cpuQuantumChannelGates.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateMeasurements.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateLanczos.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateCharacteristic.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateObservables.jl"))
include(joinpath(UTILS_CPU, "cpuVariationalQuantumCircuitCostFunctions.jl"))
include(joinpath(UTILS_CPU, "cpuClassicalShadows.jl"))
include(joinpath(UTILS_CPU, "cpuVQCEnzymeWrapper.jl"))

using .CPUQuantumChannelGates
using .CPUQuantumStatePartialTrace: partial_trace
using .CPUQuantumStatePreparation: normalize_state!
using .CPUQuantumStateLanczos: ground_state_xxz
using .CPUQuantumStateCharacteristic: von_neumann_entropy
using .CPUQuantumStateObservables: expect_local, expect_corr, get_expectation_pauli_string
using .CPUVariationalQuantumCircuitCostFunctions: fidelity
using .CPUClassicalShadows
using .CPUVQCEnzymeWrapper: make_cost_pure, gradient_enzyme, build_enzyme_wrapper

println("=" ^ 70)
println("  VQE + CLASSICAL SHADOWS DEMO")
println("=" ^ 70)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# System sizes
const N_VALUES = [6]

# VQE layers
const L_VALUES = [2, 4]

# Training
const N_EPOCHS = 500
const LR = 0.01

# Optimizers: :spsa_adam (stochastic gradient with exact cost) or :shadow_spsa (hardware-like with shadow cost)
const OPTIMIZERS = [:spsa_adam]

# Ansatzes
const ANSATZ_TYPES = [:hea, :brick]
const ROTATION_GATES = [:ry, :rz]
const ENTANGLER_GATE = :cz

# Classical shadow configurations
const N_SHADOWS_LIST = [1000, 10000, 100000]

# Hamiltonians: (name, Jx, Jy, Jz, hx)
const HAMILTONIAN_CONFIGS = [
    ("XXZ",  -1.0, -1.0, -0.5, 1.0),
    ("XY",   -1.0, -1.0,  0.0, 1.0),
]

# Grid search: all parameter combinations
const GRID_PARAMS = collect(Iterators.product(
    HAMILTONIAN_CONFIGS, N_VALUES, ANSATZ_TYPES, L_VALUES, OPTIMIZERS
))

println("\nConfig:")
println("  N = $N_VALUES, L = $L_VALUES, epochs = $N_EPOCHS")
println("  Optimizers = $OPTIMIZERS")
println("  N_shadows = $N_SHADOWS_LIST")
println("  Ansatzes = $ANSATZ_TYPES")
println("  Hamiltonians = $(length(HAMILTONIAN_CONFIGS))")
println("  Total VQE configs = $(length(GRID_PARAMS))")

# ==============================================================================
# ANSATZ BUILDER
# ==============================================================================

function build_ansatz_spec(N, n_layers, ansatz)
    gates = Tuple{Symbol, Any, Int}[]
    pidx = 0

    for layer in 1:n_layers
        # Rotation layer
        for q in 1:N
            for rot in ROTATION_GATES
                pidx += 1
                push!(gates, (rot, q, pidx))
            end
        end

        # Entangling layer
        if ansatz == :brick
            if layer % 2 == 1
                for q in 1:2:(N-1); push!(gates, (ENTANGLER_GATE, (q, q+1), 0)); end
            else
                for q in 2:2:(N-1); push!(gates, (ENTANGLER_GATE, (q, q+1), 0)); end
            end
        elseif ansatz == :hea
            for q in 1:(N-1); push!(gates, (ENTANGLER_GATE, (q, q+1), 0)); end
        elseif ansatz == :full
            for i in 1:N, j in (i+1):N
                push!(gates, (ENTANGLER_GATE, (i, j), 0))
            end
        end
    end

    # Final rotation layer
    for q in 1:N
        for rot in ROTATION_GATES
            pidx += 1
            push!(gates, (rot, q, pidx))
        end
    end
    return pidx, gates
end

# ==============================================================================
# STATE OPERATIONS
# ==============================================================================

function apply_ansatz_psi!(ψ, θ, gates, N)
    for (g, t, p) in gates
        g == :ry && apply_ry_psi!(ψ, t, θ[p], N)
        g == :rz && apply_rz_psi!(ψ, t, θ[p], N)
        g == :cz && apply_cz_psi!(ψ, t[1], t[2], N)
    end
end

function prepare_ansatz_state(θ, gates, N)
    ψ = zeros(ComplexF64, 2^N)
    ψ[1] = 1.0
    apply_ansatz_psi!(ψ, θ, gates, N)
    return ψ
end

# ==============================================================================
# ENERGY COMPUTATION
# ==============================================================================

function compute_energy(ψ, N, Jx, Jy, Jz, hx)
    dim = 2^N
    E = 0.0

    # Local X field
    for i in 1:N
        for k in 0:(dim-1)
            j = xor(k, 1 << (i-1))
            E += hx * real(conj(ψ[k+1]) * ψ[j+1])
        end
    end

    # Two-body terms
    for i in 1:(N-1)
        j = i + 1
        for k in 0:(dim-1)
            bi = (k >> (i-1)) & 1
            bj = (k >> (j-1)) & 1

            # ZZ
            E += Jz * (1-2*bi) * (1-2*bj) * abs2(ψ[k+1])

            # XX
            k_xx = xor(xor(k, 1 << (i-1)), 1 << (j-1))
            E += Jx * real(conj(ψ[k+1]) * ψ[k_xx+1])

            # YY
            phase_i = bi == 0 ? im : -im
            phase_j = bj == 0 ? im : -im
            E += Jy * real(conj(ψ[k+1]) * phase_i * phase_j * ψ[k_xx+1])
        end
    end

    return E
end

# ==============================================================================
# EXACT OBSERVABLES (using cpuQuantumStateObservables module)
# ==============================================================================

"""Compute exact ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩ on qubit 1 using module functions."""
function exact_local_observables(ψ, N)
    X1 = expect_local(ψ, 1, N, :x)
    Y1 = expect_local(ψ, 1, N, :y)
    Z1 = expect_local(ψ, 1, N, :z)
    return X1, Y1, Z1
end

"""Compute exact ⟨X₁X₂⟩, ⟨Y₁Y₂⟩, ⟨Z₁Z₂⟩ using module functions."""
function exact_two_body(ψ, N)
    XX = expect_corr(ψ, 1, 2, N, :xx)
    YY = expect_corr(ψ, 1, 2, N, :yy)
    ZZ = expect_corr(ψ, 1, 2, N, :zz)
    return XX, YY, ZZ
end

"""Compute exact ⟨X₁X₂X₃⟩, ⟨Y₁Y₂Y₃⟩, ⟨Z₁Z₂Z₃⟩ using module Pauli string function."""
function exact_three_body(ψ, N)
    if N < 3; return 0.0, 0.0, 0.0; end

    XXX = get_expectation_pauli_string(ψ, [:X, :X, :X], [1, 2, 3], N)
    YYY = get_expectation_pauli_string(ψ, [:Y, :Y, :Y], [1, 2, 3], N)
    ZZZ = get_expectation_pauli_string(ψ, [:Z, :Z, :Z], [1, 2, 3], N)

    return XXX, YYY, ZZZ
end

# ==============================================================================
# PAULI OPERATOR HELPERS (for exact observable computation)
# ==============================================================================

const σI = ComplexF64[1 0; 0 1]
const σX = ComplexF64[0 1; 1 0]
const σY = ComplexF64[0 -im; im 0]
const σZ = ComplexF64[1 0; 0 -1]

"""Build single-qubit Pauli X operator on qubit q (1-indexed) for N qubits."""
function pauliX(q::Int, N::Int)
    ops = fill(σI, N)
    ops[q] = σX
    return reduce(kron, ops)
end

function pauliY(q::Int, N::Int)
    ops = fill(σI, N)
    ops[q] = σY
    return reduce(kron, ops)
end

function pauliZ(q::Int, N::Int)
    ops = fill(σI, N)
    ops[q] = σZ
    return reduce(kron, ops)
end

"""Build two-qubit correlator operators."""
pauliXX(q1::Int, q2::Int, N::Int) = pauliX(q1, N) * pauliX(q2, N)
pauliYY(q1::Int, q2::Int, N::Int) = pauliY(q1, N) * pauliY(q2, N)
pauliZZ(q1::Int, q2::Int, N::Int) = pauliZ(q1, N) * pauliZ(q2, N)

"""Measure expectation value ⟨ψ|O|ψ⟩ exactly."""
function measure_observable_direct(ψ::Vector{ComplexF64}, O::Matrix{ComplexF64})
    return dot(ψ, O * ψ)
end

# ==============================================================================
# SHADOW OBSERVABLE ESTIMATION
# ==============================================================================

"""Estimate ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩ from Pauli shadows."""
function shadow_local_observables(snapshots::Vector{PauliSnapshot}, N::Int)
    M = length(snapshots)
    X_vals = zeros(Float64, M)
    Y_vals = zeros(Float64, M)
    Z_vals = zeros(Float64, M)

    for (m, snap) in enumerate(snapshots)
        # Qubit 1 (index 1)
        b = snap.bases[1]  # 1=X, 2=Y, 3=Z
        s = snap.outcomes[1]  # 0 or 1
        val = 3.0 * (1 - 2*s)  # Channel inversion factor 3

        if b == 1; X_vals[m] = val
        elseif b == 2; Y_vals[m] = val
        else; Z_vals[m] = val
        end
    end

    return (mean(X_vals), mean(Y_vals), mean(Z_vals),
            std(X_vals)/sqrt(M), std(Y_vals)/sqrt(M), std(Z_vals)/sqrt(M))
end

"""Estimate ⟨X₁X₂⟩, ⟨Y₁Y₂⟩, ⟨Z₁Z₂⟩ from Pauli shadows.

IMPORTANT: For unbiased estimation, we must average over ALL M snapshots:
- HIT (both qubits in correct basis): contribute 9 × (±1)
- MISS: contribute 0
The factor of 9 = 3² compensates for P(hit) = 1/9.
"""
function shadow_two_body(snapshots::Vector{PauliSnapshot}, N::Int)
    M = length(snapshots)

    # Allocate arrays for ALL M snapshots (with 0 for misses)
    XX_vals = zeros(Float64, M)
    YY_vals = zeros(Float64, M)
    ZZ_vals = zeros(Float64, M)

    for (m, snap) in enumerate(snapshots)
        b1, b2 = snap.bases[1], snap.bases[2]
        s1, s2 = snap.outcomes[1], snap.outcomes[2]

        # HIT: both bases match → contribute 9 × product of outcomes
        # MISS: contribute 0 (already initialized)
        if b1 == 1 && b2 == 1  # XX
            XX_vals[m] = 9.0 * (1-2*s1) * (1-2*s2)
        elseif b1 == 2 && b2 == 2  # YY
            YY_vals[m] = 9.0 * (1-2*s1) * (1-2*s2)
        elseif b1 == 3 && b2 == 3  # ZZ
            ZZ_vals[m] = 9.0 * (1-2*s1) * (1-2*s2)
        end
        # else: MISS, vals[m] stays 0
    end

    # Average over ALL M snapshots
    XX_mean = mean(XX_vals)
    YY_mean = mean(YY_vals)
    ZZ_mean = mean(ZZ_vals)
    XX_std = std(XX_vals) / sqrt(M)
    YY_std = std(YY_vals) / sqrt(M)
    ZZ_std = std(ZZ_vals) / sqrt(M)

    return (XX_mean, YY_mean, ZZ_mean, XX_std, YY_std, ZZ_std)
end

"""Estimate ⟨X₁X₂X₃⟩, ⟨Y₁Y₂Y₃⟩, ⟨Z₁Z₂Z₃⟩ from Pauli shadows.

IMPORTANT: For unbiased estimation, average over ALL M snapshots:
- HIT (all 3 qubits in correct basis): contribute 27 × (±1)
- MISS: contribute 0
The factor of 27 = 3³ compensates for P(hit) = 1/27.
"""
function shadow_three_body(snapshots::Vector{PauliSnapshot}, N::Int)
    if N < 3; return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0); end

    M = length(snapshots)

    # Allocate arrays for ALL M snapshots (with 0 for misses)
    XXX_vals = zeros(Float64, M)
    YYY_vals = zeros(Float64, M)
    ZZZ_vals = zeros(Float64, M)

    n_XXX, n_YYY, n_ZZZ = 0, 0, 0  # Count hits for warning

    for (m, snap) in enumerate(snapshots)
        b1, b2, b3 = snap.bases[1], snap.bases[2], snap.bases[3]
        s1, s2, s3 = snap.outcomes[1], snap.outcomes[2], snap.outcomes[3]

        # HIT: all 3 bases match → contribute 27 × product
        if b1 == 1 && b2 == 1 && b3 == 1  # XXX
            XXX_vals[m] = 27.0 * (1-2*s1) * (1-2*s2) * (1-2*s3)
            n_XXX += 1
        elseif b1 == 2 && b2 == 2 && b3 == 2  # YYY
            YYY_vals[m] = 27.0 * (1-2*s1) * (1-2*s2) * (1-2*s3)
            n_YYY += 1
        elseif b1 == 3 && b2 == 3 && b3 == 3  # ZZZ
            ZZZ_vals[m] = 27.0 * (1-2*s1) * (1-2*s2) * (1-2*s3)
            n_ZZZ += 1
        end
        # else: MISS, vals[m] stays 0
    end

    # Warn if effective count is too low for reliable statistics
    min_samples = 10
    if n_XXX < min_samples || n_YYY < min_samples || n_ZZZ < min_samples
        @warn "Low effective sample count for 3-body (M=$M): XXX=$n_XXX, YYY=$n_YYY, ZZZ=$n_ZZZ"
    end

    # Average over ALL M snapshots
    XXX_mean = mean(XXX_vals)
    YYY_mean = mean(YYY_vals)
    ZZZ_mean = mean(ZZZ_vals)
    XXX_std = std(XXX_vals) / sqrt(M)
    YYY_std = std(YYY_vals) / sqrt(M)
    ZZZ_std = std(ZZZ_vals) / sqrt(M)

    return (XXX_mean, YYY_mean, ZZZ_mean, XXX_std, YYY_std, ZZZ_std)
end

"""
    shadow_energy_estimate(snapshots, N, ham_config) -> (mean_E, std_E)

Estimate the Hamiltonian expectation ⟨H⟩ purely from classical shadow data.

Algorithm: "Hit or Miss" term-by-term estimation
    1. For each snapshot, loop over all Hamiltonian terms
    2. If the random basis matches the term's Pauli operator → HIT, contribute
    3. If not → MISS, contribute 0
    4. Average over all M snapshots

The power of shadows: one snapshot can estimate MULTIPLE non-overlapping terms.
Example: bases [X,X,Z,Z,Y,Y] simultaneously estimates ⟨X₁X₂⟩, ⟨Z₃Z₄⟩, ⟨Y₅Y₆⟩

Returns: (mean_energy, standard_error)
"""
function shadow_energy_estimate(snapshots::Vector{PauliSnapshot}, N::Int, ham_config)
    name, Jx, Jy, Jz, hx = ham_config
    M = length(snapshots)
    energy_per_shot = zeros(Float64, M)

    for (m, snap) in enumerate(snapshots)
        E_single_shot = 0.0

        # --- LOCAL FIELDS: hₓ Σᵢ Xᵢ ---
        for i in 1:N
            # Check: Did we measure qubit i in X basis?
            if snap.bases[i] == 1
                # HIT! Inverse shadow factor = 3, outcome: s=0→+1, s=1→-1
                val = 3.0 * (1 - 2*snap.outcomes[i])
                E_single_shot += hx * val
            end
            # MISS: bases[i] ≠ X → contributes 0
        end

        # --- TWO-BODY TERMS: Jₓ XᵢXᵢ₊₁ + Jᵧ YᵢYᵢ₊₁ + Jᵤ ZᵢZᵢ₊₁ ---
        for i in 1:(N-1)
            j = i + 1
            b1, b2 = snap.bases[i], snap.bases[j]

            # Check: Did both qubits measure in the same basis matching a term?
            coeff = 0.0
            if b1 == 1 && b2 == 1      # Measured XX → estimates Jₓ⟨XᵢXⱼ⟩
                coeff = Jx
            elseif b1 == 2 && b2 == 2  # Measured YY → estimates Jᵧ⟨YᵢYⱼ⟩
                coeff = Jy
            elseif b1 == 3 && b2 == 3  # Measured ZZ → estimates Jᵤ⟨ZᵢZⱼ⟩
                coeff = Jz
            end
            # MISS: mismatched bases → coeff=0, contributes 0

            if coeff != 0.0
                # HIT! Inverse shadow factor = 9 (3²), parity: s1⊕s2
                sign_val = (1 - 2*snap.outcomes[i]) * (1 - 2*snap.outcomes[j])
                E_single_shot += coeff * 9.0 * sign_val
            end
        end

        energy_per_shot[m] = E_single_shot
    end

    # Rigorous statistics: mean and standard error
    mean_E = mean(energy_per_shot)
    std_E = std(energy_per_shot) / sqrt(M)
    return mean_E, std_E
end

# ==============================================================================
# SPSA OPTIMIZER
# ==============================================================================

mutable struct SPSA
    c::Float64
    γ::Float64
    k::Int
end
SPSA() = SPSA(0.15, 0.101, 0)

function spsa_grad!(s::SPSA, cost_fn, θ)
    s.k += 1
    c_k = s.c / s.k^s.γ
    Δ = 2.0 .* (rand(length(θ)) .> 0.5) .- 1.0
    return (cost_fn(θ .+ c_k.*Δ) - cost_fn(θ .- c_k.*Δ)) ./ (2*c_k.*Δ)
end

# ==============================================================================
# VQE TRAINING
# ==============================================================================

function train_vqe(N, n_layers, ansatz, ham_config, optimizer; n_epochs=500, lr=0.01, M_shadow=100)
    name, Jx, Jy, Jz, hx = ham_config

    # Exact ground state
    E_gs, ψ_exact = ground_state_xxz(N, Jx, Jy, Jz, hx)

    # Build ansatz
    n_params, gates = build_ansatz_spec(N, n_layers, ansatz)
    θ = 0.1 * randn(n_params)

    # Cost function: exact (fast) or shadow (hardware-like)
    use_shadow_cost = (optimizer == :shadow_spsa)

    if use_shadow_cost
        # Hardware-like: estimate energy from M_shadow samples
        cost_fn = θ_t -> begin
            ψ = prepare_ansatz_state(θ_t, gates, N)
            config = ShadowConfig(n_qubits=N, n_shots=M_shadow, measurement_group=:pauli)
            snapshots = collect_shadows(ψ, config)
            E_shadow, _ = shadow_energy_estimate(snapshots, N, ham_config)
            return E_shadow
        end
    else
        # Exact: direct ⟨ψ|H|ψ⟩ calculation
        cost_fn = θ_t -> begin
            ψ = prepare_ansatz_state(θ_t, gates, N)
            compute_energy(ψ, N, Jx, Jy, Jz, hx)
        end
    end

    # Optimizer setup
    spsa = SPSA()
    adam_opt = Optimisers.setup(Adam(lr), θ)

    energies = Float64[]
    fidelities = Float64[]

    for ep in 1:n_epochs
        if optimizer == :spsa_adam || optimizer == :shadow_spsa
            # SPSA gradient + Adam momentum with gradient clipping
            g = spsa_grad!(spsa, cost_fn, θ)
            # Clip gradient to max norm of 1.0 for stability
            g_norm = norm(g)
            if g_norm > 1.0
                g = g .* (1.0 / g_norm)
            end
            adam_opt, θ = Optimisers.update(adam_opt, θ, g)
        else  # :autograd_adam - Enzyme autodiff
            g = gradient_enzyme(cost_fn, θ)
            adam_opt, θ = Optimisers.update(adam_opt, θ, g)
        end

        # Track progress with EXACT energy (even if training with shadows)
        ψ = prepare_ansatz_state(θ, gates, N)
        E = compute_energy(ψ, N, Jx, Jy, Jz, hx)
        ψ_norm = ψ / norm(ψ)
        ψ_exact_norm = ψ_exact / norm(ψ_exact)
        F = abs2(dot(ψ_exact_norm, ψ_norm))

        push!(energies, E)
        push!(fidelities, F)

        if ep % 100 == 0
            opt_str = optimizer == :spsa_adam ? "spsa" : "autodiff"
            @printf("    [%s|L=%d|%s] ep=%d: E=%.4f (ΔE=%.2e), F=%.4f\n",
                    ansatz, n_layers, opt_str, ep, E, E-E_gs, F)
        end
    end

    # Return optimized state
    ψ_final = prepare_ansatz_state(θ, gates, N)
    return ψ_final, energies, fidelities, E_gs, ψ_exact
end

"""
Train VQE and track shadow-based observable estimates at each epoch.
Returns rich data for plotting observables vs epochs.
"""
function train_vqe_with_shadow_tracking(N, n_layers, ansatz, ham_config, optimizer;
                                        n_epochs=500, lr=0.01, M_shadows=[100, 1000, 10000],
                                        track_every=10)
    name, Jx, Jy, Jz, hx = ham_config

    # Exact ground state
    E_gs, ψ_exact = ground_state_xxz(N, Jx, Jy, Jz, hx)

    # Build ansatz
    n_params, gates = build_ansatz_spec(N, n_layers, ansatz)
    θ = 0.1 * randn(n_params)

    # Exact cost function
    cost_fn = θ_t -> begin
        ψ = prepare_ansatz_state(θ_t, gates, N)
        compute_energy(ψ, N, Jx, Jy, Jz, hx)
    end

    # Optimizer setup
    spsa = SPSA()
    adam_opt = Optimisers.setup(Adam(lr), θ)

    # Storage for each shadow count M
    shadow_data = Dict{Int, NamedTuple}()
    for M in M_shadows
        shadow_data[M] = (
            epochs = Int[],
            E_mean = Float64[], E_std = Float64[],
            X1_mean = Float64[], X1_std = Float64[],
            Y1_mean = Float64[], Y1_std = Float64[],
            Z1_mean = Float64[], Z1_std = Float64[],
            XX_mean = Float64[], XX_std = Float64[],
            YY_mean = Float64[], YY_std = Float64[],
            ZZ_mean = Float64[], ZZ_std = Float64[],
        )
    end

    # Also track exact values
    epochs_tracked = Int[]
    exact_E = Float64[]
    exact_F = Float64[]
    exact_X1 = Float64[]
    exact_Y1 = Float64[]
    exact_Z1 = Float64[]
    exact_XX = Float64[]
    exact_YY = Float64[]
    exact_ZZ = Float64[]

    for ep in 1:n_epochs
        # Training step
        if optimizer == :spsa_adam || optimizer == :shadow_spsa
            g = spsa_grad!(spsa, cost_fn, θ)
            g_norm = norm(g)
            if g_norm > 1.0
                g = g .* (1.0 / g_norm)
            end
            adam_opt, θ = Optimisers.update(adam_opt, θ, g)
        else
            g = gradient_enzyme(cost_fn, θ)
            adam_opt, θ = Optimisers.update(adam_opt, θ, g)
        end

        # Track at specified intervals
        if ep % track_every == 0 || ep == 1
            ψ = prepare_ansatz_state(θ, gates, N)

            # Exact values
            E_exact_ep = compute_energy(ψ, N, Jx, Jy, Jz, hx)
            ψ_norm = ψ / norm(ψ)
            ψ_exact_norm = ψ_exact / norm(ψ_exact)
            F = abs2(dot(ψ_exact_norm, ψ_norm))

            # Track exact observables
            push!(epochs_tracked, ep)
            push!(exact_E, E_exact_ep)
            push!(exact_F, F)
            push!(exact_X1, real(measure_observable_direct(ψ, pauliX(1, N))))
            push!(exact_Y1, real(measure_observable_direct(ψ, pauliY(1, N))))
            push!(exact_Z1, real(measure_observable_direct(ψ, pauliZ(1, N))))
            push!(exact_XX, real(measure_observable_direct(ψ, pauliXX(1, 2, N))))
            push!(exact_YY, real(measure_observable_direct(ψ, pauliYY(1, 2, N))))
            push!(exact_ZZ, real(measure_observable_direct(ψ, pauliZZ(1, 2, N))))

            # Shadow estimates for each M (parallel)
            # Use thread-safe: collect per M first, then merge
            shadow_results = Dict{Int, NamedTuple}()
            Threads.@threads for i in 1:length(M_shadows)
                M = M_shadows[i]
                config = ShadowConfig(n_qubits=N, n_shots=M, measurement_group=:pauli)
                snapshots = collect_shadows(ψ, config)

                # Energy
                E_sh, E_err = shadow_energy_estimate(snapshots, N, ham_config)

                # Local observables
                X1, Y1, Z1, X1_err, Y1_err, Z1_err = shadow_local_observables(snapshots, N)

                # 2-body correlators
                XX, YY, ZZ, XX_err, YY_err, ZZ_err = shadow_two_body(snapshots, N)

                shadow_results[M] = (
                    E_sh=E_sh, E_err=E_err,
                    X1=X1, Y1=Y1, Z1=Z1, X1_err=X1_err, Y1_err=Y1_err, Z1_err=Z1_err,
                    XX=XX, YY=YY, ZZ=ZZ, XX_err=XX_err, YY_err=YY_err, ZZ_err=ZZ_err
                )
            end

            # Merge into shadow_data (sequential to avoid race)
            for M in M_shadows
                res = shadow_results[M]
                sd = shadow_data[M]
                push!(sd.epochs, ep)
                push!(sd.E_mean, res.E_sh); push!(sd.E_std, res.E_err)
                push!(sd.X1_mean, res.X1); push!(sd.X1_std, res.X1_err)
                push!(sd.Y1_mean, res.Y1); push!(sd.Y1_std, res.Y1_err)
                push!(sd.Z1_mean, res.Z1); push!(sd.Z1_std, res.Z1_err)
                push!(sd.XX_mean, res.XX); push!(sd.XX_std, res.XX_err)
                push!(sd.YY_mean, res.YY); push!(sd.YY_std, res.YY_err)
                push!(sd.ZZ_mean, res.ZZ); push!(sd.ZZ_std, res.ZZ_err)
            end

            if ep % 100 == 0
                @printf("    ep=%d: E=%.4f (ΔE=%.2e), F=%.4f\n", ep, E_exact_ep, E_exact_ep-E_gs, F)
            end
        end
    end

    # Ground state observables
    gs_obs = (
        E = E_gs,
        X1 = real(measure_observable_direct(ψ_exact, pauliX(1, N))),
        Y1 = real(measure_observable_direct(ψ_exact, pauliY(1, N))),
        Z1 = real(measure_observable_direct(ψ_exact, pauliZ(1, N))),
        XX = real(measure_observable_direct(ψ_exact, pauliXX(1, 2, N))),
        YY = real(measure_observable_direct(ψ_exact, pauliYY(1, 2, N))),
        ZZ = real(measure_observable_direct(ψ_exact, pauliZZ(1, 2, N))),
    )

    exact_data = (epochs=epochs_tracked, E=exact_E, F=exact_F,
                  X1=exact_X1, Y1=exact_Y1, Z1=exact_Z1,
                  XX=exact_XX, YY=exact_YY, ZZ=exact_ZZ)

    ψ_final = prepare_ansatz_state(θ, gates, N)
    return ψ_final, shadow_data, exact_data, gs_obs
end

# ==============================================================================
# MAIN
# ==============================================================================

function run_analysis()
    all_results = []
    all_training_data = []  # Store energy/fidelity curves

    # Cache GS observables by (ham_name, N) to avoid recomputation
    gs_cache = Dict{Tuple{String,Int}, NamedTuple}()

    n_configs = length(GRID_PARAMS)
    println("\n  Running $n_configs VQE configurations...")

    for (idx, params) in enumerate(GRID_PARAMS)
        ham_config, N, ansatz, n_layers, optimizer = params
        name, Jx, Jy, Jz, hx = ham_config

        println("\n" * "-" ^ 60)
        @printf("  [%d/%d] %s | N=%d | %s | L=%d | %s\n",
                idx, n_configs, name, N, ansatz, n_layers, optimizer)
        println("-" ^ 60)

        # Compute/retrieve ground state observables
        cache_key = (name, N)
        if !haskey(gs_cache, cache_key)
            E_gs_ref, ψ_gs = ground_state_xxz(N, Jx, Jy, Jz, hx)
            X1_gs, Y1_gs, Z1_gs = exact_local_observables(ψ_gs, N)
            XX_gs, YY_gs, ZZ_gs = exact_two_body(ψ_gs, N)
            XXX_gs, YYY_gs, ZZZ_gs = exact_three_body(ψ_gs, N)
            gs_cache[cache_key] = (
                E_gs=E_gs_ref,
                X1=X1_gs, Y1=Y1_gs, Z1=Z1_gs,
                XX=XX_gs, YY=YY_gs, ZZ=ZZ_gs,
                XXX=XXX_gs, YYY=YYY_gs, ZZZ=ZZZ_gs
            )
            @printf("  GS: E=%.4f, ⟨Z₁⟩=%.4f, ⟨ZZ⟩=%.4f, ⟨ZZZ⟩=%.4f\n",
                    E_gs_ref, Z1_gs, ZZ_gs, ZZZ_gs)
        end
        gs = gs_cache[cache_key]

        # Seed for reproducibility
        Random.seed!(42 + N*1000 + n_layers*100 + hash(ansatz) + hash(optimizer))

        # Train VQE with specified optimizer
        println("  Training VQE...")
        ψ_vqe, energies, fidelities, E_gs, ψ_exact = train_vqe(
            N, n_layers, ansatz, ham_config, optimizer; n_epochs=N_EPOCHS, lr=LR)

        final_F = fidelities[end]
        final_E = energies[end]
        @printf("  Final: E=%.4f (ΔE=%.2e), F=%.4f\n",
                final_E, final_E-E_gs, final_F)

        # Store training history
        push!(all_training_data, (
            hamiltonian=name, N=N, ansatz=ansatz, layers=n_layers, optimizer=optimizer,
            energies=energies, fidelities=fidelities, E_gs=E_gs
        ))

        # Exact observables on VQE state
        X1_vqe, Y1_vqe, Z1_vqe = exact_local_observables(ψ_vqe, N)
        XX_vqe, YY_vqe, ZZ_vqe = exact_two_body(ψ_vqe, N)
        XXX_vqe, YYY_vqe, ZZZ_vqe = exact_three_body(ψ_vqe, N)

        @printf("  ⟨Z⟩: GS=%.4f | VQE=%.4f (Δ=%.2e)\n",
                gs.Z1, Z1_vqe, abs(gs.Z1 - Z1_vqe))

        # Classical shadow estimation
        println("  Classical Shadows (Pauli):")
        for M in N_SHADOWS_LIST
            config = ShadowConfig(n_qubits=N, n_shots=M, measurement_group=:pauli)
            snapshots = collect_shadows(ψ_vqe, config)

            # Estimate observables (bitwise, no matrix reconstruction)
            X1_sh, Y1_sh, Z1_sh, X1_err, Y1_err, Z1_err = shadow_local_observables(snapshots, N)
            XX_sh, YY_sh, ZZ_sh, XX_err, YY_err, ZZ_err = shadow_two_body(snapshots, N)
            XXX_sh, YYY_sh, ZZZ_sh, XXX_err, YYY_err, ZZZ_err = shadow_three_body(snapshots, N)

            # Estimate total energy from shadows (key metric!)
            E_shadow, E_shadow_err = shadow_energy_estimate(snapshots, N, ham_config)

            # Print shadow vs exact comparison
            @printf("    M=%7d: ⟨Z₁⟩=%.3f±%.3f (exact=%.3f) | E=%.2f±%.2f (VQE=%.2f, GS=%.2f)\n",
                    M, Z1_sh, Z1_err, Z1_vqe, E_shadow, E_shadow_err, final_E, E_gs)

            # Store results with all 3 references
            push!(all_results, (
                hamiltonian=name, N=N, ansatz=ansatz, layers=n_layers,
                optimizer=optimizer, n_shadows=M,
                fidelity=final_F, energy=final_E, E_gs=E_gs,
                E_shadow=E_shadow, E_shadow_err=E_shadow_err,  # NEW: shadow energy
                # Ground state exact observables
                X1_gs=gs.X1, Y1_gs=gs.Y1, Z1_gs=gs.Z1,
                XX_gs=gs.XX, YY_gs=gs.YY, ZZ_gs=gs.ZZ,
                XXX_gs=gs.XXX, YYY_gs=gs.YYY, ZZZ_gs=gs.ZZZ,
                # VQE state exact observables
                X1_vqe=X1_vqe, Y1_vqe=Y1_vqe, Z1_vqe=Z1_vqe,
                XX_vqe=XX_vqe, YY_vqe=YY_vqe, ZZ_vqe=ZZ_vqe,
                XXX_vqe=XXX_vqe, YYY_vqe=YYY_vqe, ZZZ_vqe=ZZZ_vqe,
                # Shadow estimates
                X1_shadow=X1_sh, Y1_shadow=Y1_sh, Z1_shadow=Z1_sh,
                X1_err=X1_err, Y1_err=Y1_err, Z1_err=Z1_err,
                XX_shadow=XX_sh, YY_shadow=YY_sh, ZZ_shadow=ZZ_sh,
                XX_err=XX_err, YY_err=YY_err, ZZ_err=ZZ_err,
                XXX_shadow=XXX_sh, YYY_shadow=YYY_sh, ZZZ_shadow=ZZZ_sh,
                XXX_err=XXX_err, YYY_err=YYY_err, ZZZ_err=ZZZ_err
            ))
        end
    end

    # === Run shadow tracking demo for one configuration ===
    println("\n" * "-" ^ 60)
    println("  SHADOW TRACKING DEMO (observables vs epochs)")
    println("-" ^ 60)

    # Pick first Hamiltonian configuration
    ham_config = HAMILTONIAN_CONFIGS[1]
    N = N_VALUES[1]
    ansatz = :hea
    n_layers = L_VALUES[1]

    println("  Config: $(ham_config[1]), N=$N, $ansatz, L=$n_layers")
    println("  Tracking shadows at each epoch for M=$(N_SHADOWS_LIST)...")

    ψ_final, shadow_data, exact_data, gs_obs = train_vqe_with_shadow_tracking(
        N, n_layers, ansatz, ham_config, :spsa_adam;
        n_epochs=N_EPOCHS, lr=LR, M_shadows=N_SHADOWS_LIST, track_every=10
    )

    # Generate the shadow convergence plot
    plot_shadow_convergence(shadow_data, exact_data, gs_obs, ham_config;
                            N=N, L=n_layers, ansatz=ansatz, optimizer=:spsa_adam)

    return all_results, all_training_data
end

# ==============================================================================
# PLOTTING
# ==============================================================================

# Colors for different M values
# Default colors for shadow M values (fallback to palette for unknown M)
const M_COLORS_DEFAULT = Dict(100 => :coral, 1000 => :dodgerblue, 10000 => :limegreen, 100000 => :purple)
const PALETTE = [:coral, :dodgerblue, :limegreen, :purple, :orange, :teal, :crimson]

function get_m_color(M::Int, idx::Int)
    return get(M_COLORS_DEFAULT, M, PALETTE[mod1(idx, length(PALETTE))])
end

"""
Plot shadow observable convergence during training with shaded std regions.
Title includes full Hamiltonian form with parameters.
"""
function plot_shadow_convergence(shadow_data, exact_data, gs_obs, ham_config;
                                  N::Int=6, L::Int=2, ansatz::Symbol=:hea,
                                  optimizer::Symbol=:spsa_adam,
                                  name_prefix="", save_dir=OUTPUT_DIR)
    ham_name, Jx, Jy, Jz, hx = ham_config
    epochs = exact_data.epochs

    M_list = sort(collect(keys(shadow_data)))

    # Build informative title
    # Hamiltonian form: H = Jx∑XX + Jy∑YY + Jz∑ZZ + hx∑X
    ham_latex = L"H = J_x \sum X_i X_j + J_y \sum Y_i Y_j + J_z \sum Z_i Z_j + h_x \sum X_i"
    params_str = "Jx=$(Jx), Jy=$(Jy), Jz=$(Jz), hx=$(hx)"
    ansatz_str = uppercase(String(ansatz))
    opt_str = optimizer == :spsa_adam ? "SPSA+Adam" : String(optimizer)

    # Create 3x3 grid: Energy, Fidelity, X1/Y1/Z1, XX/YY/ZZ
    p = plot(layout=(3, 3), size=(1600, 1200), margin=5Plots.mm,
             left_margin=10Plots.mm, bottom_margin=8Plots.mm)

    # Create M index mapping for consistent colors
    M_indices = Dict(M => i for (i, M) in enumerate(M_list))

    # Helper function for shaded ribbon plot
    function plot_with_ribbon!(sp, x, y_mean, y_std, M; kwargs...)
        c = get_m_color(M, M_indices[M])
        # Shaded region
        plot!(p[sp], x, y_mean, ribbon=y_std, fillalpha=0.2, color=c,
              lw=0, label="")
        # Mean line
        plot!(p[sp], x, y_mean, lw=2, color=c, label="M=$M"; kwargs...)
    end

    # === Plot 1: Energy ===
    plot!(p[1], epochs, exact_data.E, lw=3, color=:black, ls=:solid, label="⟨O⟩")
    hline!(p[1], [gs_obs.E], lw=2, color=:red, ls=:dash, label="GS")
    for M in M_list
        sd = shadow_data[M]
        plot_with_ribbon!(1, sd.epochs, sd.E_mean, sd.E_std, M)
    end
    plot!(p[1], xlabel="Epoch", ylabel=L"\langle H \rangle",
          title="Energy", legend=:topright)

    # === Plot 2: Fidelity ===
    plot!(p[2], epochs, exact_data.F, lw=3, color=:black, label="Fidelity")
    hline!(p[2], [1.0], lw=2, color=:red, ls=:dash, label="Target")
    plot!(p[2], xlabel="Epoch", ylabel=L"|⟨ψ_{GS}|ψ⟩|^2",
          title="Fidelity", legend=:bottomright)

    # === Plot 3: X1 ===
    plot!(p[3], epochs, exact_data.X1, lw=3, color=:black, label="⟨O⟩")
    hline!(p[3], [gs_obs.X1], lw=2, color=:red, ls=:dash, label="GS")
    for M in M_list
        sd = shadow_data[M]
        plot_with_ribbon!(3, sd.epochs, sd.X1_mean, sd.X1_std, M)
    end
    plot!(p[3], xlabel="Epoch", ylabel=L"\langle X_1 \rangle",
          title=L"⟨X_1⟩", legend=:best)

    # === Plot 4: Y1 ===
    plot!(p[4], epochs, exact_data.Y1, lw=3, color=:black, label="⟨O⟩")
    hline!(p[4], [gs_obs.Y1], lw=2, color=:red, ls=:dash, label="GS")
    for M in M_list
        sd = shadow_data[M]
        plot_with_ribbon!(4, sd.epochs, sd.Y1_mean, sd.Y1_std, M)
    end
    plot!(p[4], xlabel="Epoch", ylabel=L"\langle Y_1 \rangle",
          title=L"⟨Y_1⟩", legend=:best)

    # === Plot 5: Z1 ===
    plot!(p[5], epochs, exact_data.Z1, lw=3, color=:black, label="⟨O⟩")
    hline!(p[5], [gs_obs.Z1], lw=2, color=:red, ls=:dash, label="GS")
    for M in M_list
        sd = shadow_data[M]
        plot_with_ribbon!(5, sd.epochs, sd.Z1_mean, sd.Z1_std, M)
    end
    plot!(p[5], xlabel="Epoch", ylabel=L"\langle Z_1 \rangle",
          title=L"⟨Z_1⟩", legend=:best)

    # === Plot 6: XX ===
    plot!(p[6], epochs, exact_data.XX, lw=3, color=:black, label="⟨O⟩")
    hline!(p[6], [gs_obs.XX], lw=2, color=:red, ls=:dash, label="GS")
    for M in M_list
        sd = shadow_data[M]
        plot_with_ribbon!(6, sd.epochs, sd.XX_mean, sd.XX_std, M)
    end
    plot!(p[6], xlabel="Epoch", ylabel=L"\langle X_1 X_2 \rangle",
          title=L"⟨X_1 X_2⟩", legend=:best)

    # === Plot 7: YY ===
    plot!(p[7], epochs, exact_data.YY, lw=3, color=:black, label="⟨O⟩")
    hline!(p[7], [gs_obs.YY], lw=2, color=:red, ls=:dash, label="GS")
    for M in M_list
        sd = shadow_data[M]
        plot_with_ribbon!(7, sd.epochs, sd.YY_mean, sd.YY_std, M)
    end
    plot!(p[7], xlabel="Epoch", ylabel=L"\langle Y_1 Y_2 \rangle",
          title=L"⟨Y_1 Y_2⟩", legend=:best)

    # === Plot 8: ZZ ===
    plot!(p[8], epochs, exact_data.ZZ, lw=3, color=:black, label="⟨O⟩")
    hline!(p[8], [gs_obs.ZZ], lw=2, color=:red, ls=:dash, label="GS")
    for M in M_list
        sd = shadow_data[M]
        plot_with_ribbon!(8, sd.epochs, sd.ZZ_mean, sd.ZZ_std, M)
    end
    plot!(p[8], xlabel="Epoch", ylabel=L"\langle Z_1 Z_2 \rangle",
          title=L"⟨Z_1 Z_2⟩", legend=:best)

    # === Plot 9: Legend panel ===
    plot!(p[9], title="Legend", axis=false, grid=false, legend=:topleft)
    plot!(p[9], [0], [0], lw=3, color=:black, label="⟨O⟩ (from |ψ⟩)")
    plot!(p[9], [0], [0], lw=2, color=:red, ls=:dash, label="Ground State")
    for (i, M) in enumerate(sort(M_list))
        plot!(p[9], [0], [0], lw=2, color=get_m_color(M, i), label="Shadow M=$M")
    end

    # Super title: Hamiltonian form + Parameters | N, L, Ansatz, Optimizer
    title_line1 = "H = Jₓ∑XᵢXⱼ + Jᵧ∑YᵢYⱼ + Jᵤ∑ZᵢZⱼ + hₓ∑Xᵢ"
    title_line2 = "$(params_str) | N=$(N), L=$(L), $(ansatz_str), $(opt_str)"
    full_title = "$title_line1\n$title_line2"

    plot!(p, plot_title=full_title, plot_titlefontsize=12)

    fname = "fig_shadow_training_convergence_$(lowercase(ham_name))_N$(N)_$(ansatz)_L$(L).png"
    savefig(p, joinpath(save_dir, "figures", fname))
    println("  → Saved: $fname")

    return p
end

function generate_plots(results, training_data)
    println("\n" * "=" ^ 60)
    println("  GENERATING PLOTS")
    println("=" ^ 60)

    # Group training data by (hamiltonian, N, optimizer)
    training_by_key = Dict()
    for t in training_data
        key = (t.hamiltonian, t.N, t.optimizer)
        if !haskey(training_by_key, key)
            training_by_key[key] = []
        end
        push!(training_by_key[key], t)
    end

    # Group results by (hamiltonian, N, ansatz, layers, optimizer)
    groups = Dict()
    for r in results
        key = (r.hamiltonian, r.N, r.ansatz, r.layers, r.optimizer)
        if !haskey(groups, key)
            groups[key] = []
        end
        push!(groups[key], r)
    end

    colors = [:blue, :red, :green, :purple, :orange, :cyan, :magenta, :brown]

    # === Plot 1: Energy convergence per (Hamiltonian, N, Optimizer) ===
    for ((ham, N, optimizer), curves) in training_by_key
        E_gs = curves[1].E_gs
        opt_str = optimizer == :spsa_adam ? "SPSA" : "Autodiff"

        p = plot(size=(1000, 500), layout=(1, 2),
                 margin=5Plots.mm, left_margin=12Plots.mm, bottom_margin=8Plots.mm)

        for (i, t) in enumerate(curves)
            lbl = "$(t.ansatz) L=$(t.layers)"
            plot!(p[1], t.energies, lw=2, color=colors[mod1(i, length(colors))],
                  label=lbl)
            plot!(p[2], t.fidelities, lw=2, color=colors[mod1(i, length(colors))],
                  label=lbl)
        end
        hline!(p[1], [E_gs], lw=2, ls=:dash, color=:black,
               label=latexstring("E_{\\mathrm{GS}}=$(@sprintf("%.2f", E_gs))"))
        hline!(p[2], [1.0], lw=1, ls=:dot, color=:black, label="")

        # LaTeX titles and labels
        plot!(p[1], xlabel=latexstring("\\mathrm{Epoch}"),
              ylabel=latexstring("E(\\theta)"),
              title=latexstring("\\mathrm{Energy\\ Convergence}"))
        plot!(p[2], xlabel=latexstring("\\mathrm{Epoch}"),
              ylabel=latexstring("|\\langle\\psi_{\\mathrm{GS}}|\\psi(\\theta)\\rangle|^2"),
              title=latexstring("\\mathrm{Fidelity\\ Convergence}"))

        # Add super-title with all parameters
        plot!(p, plot_title=latexstring("\\mathrm{VQE\\ Training:}\\ $(ham),\\ N=$(N),\\ \\mathrm{Optimizer}=$(opt_str)"),
              plot_titlefontsize=12)

        # Parameter-encoded filename
        fname = "fig_vqe_training_ham_$(lowercase(ham))_N$(N)_opt_$(optimizer).png"
        savefig(p, joinpath(OUTPUT_DIR, "figures", fname))
        println("  → Saved: $fname")
    end

    # === Plot 2: 3-way comparison (Shadow vs VQE vs GS) for each config ===
    for (key, data) in groups
        ham, N, ansatz, layers, optimizer = key
        sort!(data, by=x->x.n_shadows)

        M_vals = [d.n_shadows for d in data]
        opt_str = optimizer == :spsa_adam ? "SPSA" : "Autodiff"
        fidelity = data[1].fidelity

        # Dynamic axis selection: XY model orders in X-Y plane, XXZ in Z
        # Plot the dominant correlations based on Hamiltonian
        # Use proper LaTeX subscripts with braces
        if ham == "XY"
            # XY model: plot X observables (Z correlations are near zero)
            O1_label, O2_label, O3_label = "X_{1}", "X_{1}X_{2}", "X_{1}X_{2}X_{3}"
            O1_gs, O1_vqe = data[1].X1_gs, data[1].X1_vqe
            O1_shadow = [d.X1_shadow for d in data]
            O1_err = [d.X1_err for d in data]
            O2_gs, O2_vqe = data[1].XX_gs, data[1].XX_vqe
            O2_shadow = [d.XX_shadow for d in data]
            O2_err = [d.XX_err for d in data]
            O3_gs, O3_vqe = data[1].XXX_gs, data[1].XXX_vqe
            O3_shadow = [d.XXX_shadow for d in data]
            O3_err = [d.XXX_err for d in data]
        else
            # XXZ model: plot Z observables (strong correlations)
            O1_label, O2_label, O3_label = "Z_{1}", "Z_{1}Z_{2}", "Z_{1}Z_{2}Z_{3}"
            O1_gs, O1_vqe = data[1].Z1_gs, data[1].Z1_vqe
            O1_shadow = [d.Z1_shadow for d in data]
            O1_err = [d.Z1_err for d in data]
            O2_gs, O2_vqe = data[1].ZZ_gs, data[1].ZZ_vqe
            O2_shadow = [d.ZZ_shadow for d in data]
            O2_err = [d.ZZ_err for d in data]
            O3_gs, O3_vqe = data[1].ZZZ_gs, data[1].ZZZ_vqe
            O3_shadow = [d.ZZZ_shadow for d in data]
            O3_err = [d.ZZZ_err for d in data]
        end

        # Create 2x3 grid plot
        p = plot(layout=(2, 3), size=(1400, 700),
                 margin=5Plots.mm, left_margin=12Plots.mm, bottom_margin=8Plots.mm)

        # Row 1: Observables with error bars
        # 1-body
        scatter!(p[1], M_vals, O1_shadow, yerr=O1_err, ms=8, color=:blue,
                 marker=:circle, label="Shadow", xscale=:log10)
        hline!(p[1], [O1_vqe], lw=2, color=:green, ls=:solid,
               label=latexstring("\\mathrm{VQE}=$(@sprintf("%.3f", O1_vqe))"))
        hline!(p[1], [O1_gs], lw=2, color=:red, ls=:dash,
               label=latexstring("\\mathrm{GS}=$(@sprintf("%.3f", O1_gs))"))
        plot!(p[1], xlabel=latexstring("M\\ (\\mathrm{shadows})"),
              ylabel=latexstring("\\langle $(O1_label) \\rangle"),
              title=latexstring("\\langle $(O1_label) \\rangle"))

        # 2-body
        scatter!(p[2], M_vals, O2_shadow, yerr=O2_err, ms=8, color=:blue,
                 marker=:circle, label="Shadow", xscale=:log10)
        hline!(p[2], [O2_vqe], lw=2, color=:green, ls=:solid,
               label=latexstring("\\mathrm{VQE}=$(@sprintf("%.3f", O2_vqe))"))
        hline!(p[2], [O2_gs], lw=2, color=:red, ls=:dash,
               label=latexstring("\\mathrm{GS}=$(@sprintf("%.3f", O2_gs))"))
        plot!(p[2], xlabel=latexstring("M\\ (\\mathrm{shadows})"),
              ylabel=latexstring("\\langle $(O2_label) \\rangle"),
              title=latexstring("\\langle $(O2_label) \\rangle"))

        # 3-body
        scatter!(p[3], M_vals, O3_shadow, yerr=O3_err, ms=8, color=:blue,
                 marker=:circle, label="Shadow", xscale=:log10)
        hline!(p[3], [O3_vqe], lw=2, color=:green, ls=:solid,
               label=latexstring("\\mathrm{VQE}=$(@sprintf("%.3f", O3_vqe))"))
        hline!(p[3], [O3_gs], lw=2, color=:red, ls=:dash,
               label=latexstring("\\mathrm{GS}=$(@sprintf("%.3f", O3_gs))"))
        plot!(p[3], xlabel=latexstring("M\\ (\\mathrm{shadows})"),
              ylabel=latexstring("\\langle $(O3_label) \\rangle"),
              title=latexstring("\\langle $(O3_label) \\rangle"))

        # Row 2: Error vs M (log-log) with anchored 1/√M reference
        O1_abs_err = abs.(O1_shadow .- O1_vqe)
        O2_abs_err = abs.(O2_shadow .- O2_vqe)
        O3_abs_err = abs.(O3_shadow .- O3_vqe)

        # Anchor the theoretical 1/√M line to the first data point
        C1 = (O1_abs_err[1] + 1e-10) * sqrt(M_vals[1])
        C2 = (O2_abs_err[1] + 1e-10) * sqrt(M_vals[1])
        C3 = (O3_abs_err[1] + 1e-10) * sqrt(M_vals[1])

        scatter!(p[4], M_vals, O1_abs_err .+ 1e-10, ms=8, color=:blue,
                 xscale=:log10, yscale=:log10, label="")
        plot!(p[4], M_vals, C1 ./ sqrt.(M_vals), lw=2, ls=:dash, color=:black,
              label=latexstring("\\propto 1/\\sqrt{M}"))
        plot!(p[4], xlabel=latexstring("M\\ (\\mathrm{shadows})"),
              ylabel=latexstring("|\\mathrm{Shadow} - \\mathrm{VQE}|"),
              title=latexstring("|\\langle $(O1_label) \\rangle - \\mathrm{VQE}|"))

        scatter!(p[5], M_vals, O2_abs_err .+ 1e-10, ms=8, color=:blue,
                 xscale=:log10, yscale=:log10, label="")
        plot!(p[5], M_vals, C2 ./ sqrt.(M_vals), lw=2, ls=:dash, color=:black,
              label=latexstring("\\propto 1/\\sqrt{M}"))
        plot!(p[5], xlabel=latexstring("M\\ (\\mathrm{shadows})"),
              ylabel=latexstring("|\\mathrm{Shadow} - \\mathrm{VQE}|"),
              title=latexstring("|\\langle $(O2_label) \\rangle - \\mathrm{VQE}|"))

        scatter!(p[6], M_vals, O3_abs_err .+ 1e-10, ms=8, color=:blue,
                 xscale=:log10, yscale=:log10, label="")
        plot!(p[6], M_vals, C3 ./ sqrt.(M_vals), lw=2, ls=:dash, color=:black,
              label=latexstring("\\propto 1/\\sqrt{M}"))
        plot!(p[6], xlabel=latexstring("M\\ (\\mathrm{shadows})"),
              ylabel=latexstring("|\\mathrm{Shadow} - \\mathrm{VQE}|"),
              title=latexstring("|\\langle $(O3_label) \\rangle - \\mathrm{VQE}|"))

        # Super-title with all parameters
        axis_note = ham == "XY" ? "X-basis" : "Z-basis"
        plot!(p, plot_title=latexstring("$(ham),\\ N=$(N),\\ $(ansatz),\\ L=$(layers),\\ $(opt_str),\\ F=$(@sprintf("%.3f", fidelity))\\ ($(axis_note))"),
              plot_titlefontsize=12)

        # Parameter-encoded filename
        fname = "fig_shadow_comparison_ham_$(lowercase(ham))_N$(N)_ansatz_$(ansatz)_L$(layers)_opt_$(optimizer).png"
        savefig(p, joinpath(OUTPUT_DIR, "figures", fname))
        println("  → Saved: $fname")

        # === Plot 3: Energy estimation convergence vs M ===
        E_shadow_vals = [d.E_shadow for d in data]
        E_shadow_errs = [d.E_shadow_err for d in data]
        E_vqe = data[1].energy
        E_gs = data[1].E_gs

        p_energy = plot(size=(800, 500), margin=8Plots.mm, left_margin=12Plots.mm)

        # Shadow energy estimates with error bars
        scatter!(p_energy, M_vals, E_shadow_vals, yerr=E_shadow_errs,
                 ms=10, color=:blue, marker=:circle, label="Shadow Estimate", xscale=:log10)

        # Reference lines
        hline!([E_vqe], lw=2, color=:green, ls=:solid,
               label=latexstring("E_{\\mathrm{VQE}} = $(@sprintf("%.2f", E_vqe))"))
        hline!([E_gs], lw=2, color=:red, ls=:dash,
               label=latexstring("E_{\\mathrm{GS}} = $(@sprintf("%.2f", E_gs))"))

        plot!(xlabel=latexstring("M\\ (\\mathrm{shadows})"),
              ylabel=latexstring("\\langle H \\rangle"),
              title=latexstring("$(ham),\\ N=$(N),\\ $(ansatz),\\ L=$(layers),\\ $(opt_str)"),
              legend=:topright)

        fname_energy = "fig_energy_vs_shadows_ham_$(lowercase(ham))_N$(N)_ansatz_$(ansatz)_L$(layers)_opt_$(optimizer).png"
        savefig(p_energy, joinpath(OUTPUT_DIR, "figures", fname_energy))
        println("  → Saved: $fname_energy")

        # === Plot 4: Energy error scaling (log-log) ===
        E_abs_err = abs.(E_shadow_vals .- E_vqe)
        C_E = (E_abs_err[1] + 1e-10) * sqrt(M_vals[1])

        p_err = plot(size=(600, 400), margin=8Plots.mm)
        scatter!(p_err, M_vals, E_abs_err .+ 1e-10, ms=10, color=:blue,
                 xscale=:log10, yscale=:log10, label="")
        plot!(p_err, M_vals, C_E ./ sqrt.(M_vals), lw=2, ls=:dash, color=:black,
              label=latexstring("\\propto 1/\\sqrt{M}"))
        plot!(p_err, M_vals, E_shadow_errs, lw=2, color=:orange, label="Std Error")
        plot!(xlabel=latexstring("M\\ (\\mathrm{shadows})"),
              ylabel=latexstring("|E_{\\mathrm{shadow}} - E_{\\mathrm{VQE}}|"),
              title=latexstring("$(ham),\\ N=$(N),\\ $(ansatz),\\ L=$(layers),\\ $(opt_str)"),
              legend=:bottomleft)

        fname_err = "fig_energy_error_scaling_ham_$(lowercase(ham))_N$(N)_ansatz_$(ansatz)_L$(layers)_opt_$(optimizer).png"
        savefig(p_err, joinpath(OUTPUT_DIR, "figures", fname_err))
        println("  → Saved: $fname_err")
    end
end

# ==============================================================================
# SAVE DATA
# ==============================================================================

function save_data(results)
    fname = joinpath(OUTPUT_DIR, "data", "vqe_shadow_results.txt")
    open(fname, "w") do f
        println(f, "# VQE + Classical Shadow Results")
        println(f, "# Columns: hamiltonian,N,ansatz,layers,optimizer,n_shadows,fidelity,energy,E_gs,E_shadow,E_shadow_err,")
        println(f, "#          Z1_gs,Z1_vqe,Z1_shadow,Z1_err,ZZ_gs,ZZ_vqe,ZZ_shadow,ZZ_err,ZZZ_gs,ZZZ_vqe,ZZZ_shadow,ZZZ_err")
        for r in results
            @printf(f, "%s,%d,%s,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    r.hamiltonian, r.N, r.ansatz, r.layers, r.optimizer, r.n_shadows,
                    r.fidelity, r.energy, r.E_gs, r.E_shadow, r.E_shadow_err,
                    r.Z1_gs, r.Z1_vqe, r.Z1_shadow, r.Z1_err,
                    r.ZZ_gs, r.ZZ_vqe, r.ZZ_shadow, r.ZZ_err,
                    r.ZZZ_gs, r.ZZZ_vqe, r.ZZZ_shadow, r.ZZZ_err)
        end
    end
    println("  → Data saved: vqe_shadow_results.txt")
end

# ==============================================================================
# MAIN
# ==============================================================================

function main()
    # Redirect output to file
    output_file = joinpath(OUTPUT_DIR, "output.txt")
    open(output_file, "w") do io
        # Redirect both stdout and stderr
        redirect_stdout(io) do
            redirect_stderr(io) do
                results, training_data = run_analysis()
                generate_plots(results, training_data)
                save_data(results)

                println("\n" * "=" ^ 60)
                println("  VQE + CLASSICAL SHADOWS DEMO COMPLETE")
                println("=" ^ 60)
                println("  Output: $(basename(OUTPUT_DIR))/")
            end
        end
    end

    # Also print to console (relative path)
    println("Demo complete! Output saved to: $(basename(OUTPUT_DIR))/output.txt")
end

main()

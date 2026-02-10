#!/usr/bin/env julia
#=
================================================================================
    demo_variational_quantum_eigensolver.jl - VQE for Ground State Preparation
================================================================================

DESCRIPTION:
    Demonstrates Variational Quantum Eigensolver (VQE) for finding ground states
    of spin Hamiltonians. The variational circuit is trained to minimize the
    energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩.

VQE ARCHITECTURE:
    |0...0⟩ → [Ansatz U(θ)] → |ψ(θ)⟩ → measure ⟨H⟩

    - Ansatz: Parameterized variational circuit U(θ)
    - Energy: E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
    - Goal: Minimize E(θ) → E_GS (ground state energy)

HAMILTONIAN CONVENTION:
    H = Jx∑XX + Jy∑YY + Jz∑ZZ + hx∑X

    Parameters control the sign directly (no hidden minus signs):
    - Jx = -1: Antiferromagnetic XX coupling (spins prefer anti-alignment)
    - Jx = +1: Ferromagnetic XX coupling (spins prefer alignment)

    Supported models:
    - XXZ: Jx = Jy ≠ Jz (anisotropic Heisenberg with transverse field)
    - TFIM: Jx = Jy = 0 (transverse field Ising model)
    - Heisenberg: Jx = Jy = Jz (isotropic)

METRICS TRACKED (9 panels, 3x3 grid):
    Row 1: Core VQE metrics
    - Energy E(θ) vs epochs (with exact E_GS reference line)
    - S_vN at bipartite split vs epochs (entanglement entropy)
    - Fidelity F = |⟨ψ_exact|ψ(θ)⟩|² vs epochs

    Row 2: Local observables (mean per qubit)
    - ⟨X⟩ = (1/N) Σᵢ ⟨Xᵢ⟩
    - ⟨Y⟩ = (1/N) Σᵢ ⟨Yᵢ⟩
    - ⟨Z⟩ = (1/N) Σᵢ ⟨Zᵢ⟩

    Row 3: Two-body correlators (mean per nearest-neighbor bond)
    - ⟨XX⟩ = (1/(N-1)) Σᵢ ⟨XᵢXᵢ₊₁⟩
    - ⟨YY⟩ = (1/(N-1)) Σᵢ ⟨YᵢYᵢ₊₁⟩
    - ⟨ZZ⟩ = (1/(N-1)) Σᵢ ⟨ZᵢZᵢ₊₁⟩

PARAMETER SWEEPS:
    - N: Number of qubits [4, 6, 8, ...]
    - L: Number of variational layers [2, 4, 6, ...]
    - Ansatz: Circuit topology [:hea, :full, :brick, ...]
    - Hamiltonians: XXZ, TFIM, etc.

ANSATZ TYPES:
    :hea       - Hardware Efficient Ansatz (nearest-neighbor CZ)
    :full      - All-to-all CZ connectivity (for long-range correlations)
    :brick     - Alternating brick pattern (even-odd layers)
    :chain     - Linear chain connectivity
    :ring      - Periodic boundary conditions

OPTIMIZER:
    SPSA + Adam hybrid:
    - SPSA: Simultaneous Perturbation Stochastic Approximation for gradients
    - Adam: Adaptive momentum for parameter updates

THEORY - VARIATIONAL PRINCIPLE:
    The quantum variational principle guarantees:
        E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ ≥ E_GS

    for any normalized trial state |ψ(θ)⟩. Equality holds when |ψ(θ)⟩ = |ψ_GS⟩.

    The ansatz expressibility determines how closely VQE can approach E_GS.
    Critical ground states (e.g., TFIM at J=h) may require deep circuits
    or all-to-all connectivity to capture long-range correlations.

OUTPUT:
    - Training curves: 3x3 grid (E/S/F, X/Y/Z, XX/YY/ZZ) vs epochs
    - PNG plots saved to demo_variational_quantum_eigensolver/ folder
    - CSV data files with full training history

================================================================================
=#

using LinearAlgebra
using Random
using Printf
using Plots
using Statistics
using Optimisers

const OUTPUT_DIR = joinpath(@__DIR__, "demo_variational_quantum_eigensolver")
mkpath(OUTPUT_DIR)

const UTILS_CPU = joinpath(@__DIR__, "..", "utils", "cpu")
include(joinpath(UTILS_CPU, "cpuQuantumChannelGates.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateMeasurements.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateLanczos.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateCharacteristic.jl"))
include(joinpath(UTILS_CPU, "cpuVariationalQuantumCircuitCostFunctions.jl"))

using .CPUQuantumChannelGates
using .CPUQuantumStatePartialTrace: partial_trace
using .CPUQuantumStatePreparation: normalize_state!
using .CPUQuantumStateLanczos: ground_state_xxz
using .CPUQuantumStateCharacteristic: von_neumann_entropy
using .CPUVariationalQuantumCircuitCostFunctions: fidelity

println("=" ^ 70)
println("  VARIATIONAL QUANTUM EIGENSOLVER - Ground State Preparation")
println("=" ^ 70)

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM SIZE & TRAINING
# ─────────────────────────────────────────────────────────────────────────────
const N_VALUES = [
    6,
    8,
    10,
    12,
]

const L_VALUES = [
    1,
    2,
    3,
    4,
    6,
    8
]

const N_EPOCHS = 2000

const LR = 0.005

# ─────────────────────────────────────────────────────────────────────────────
# ANSATZ CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
const ANSATZ_TYPES = [
    :hea,        # Hardware Efficient Ansatz (nearest-neighbor)
    :full,       # All-to-all connectivity
    :brick,      # Alternating brick pattern
]

const ROTATION_GATES = [:ry, :rz]

const ENTANGLER_GATES = [
    :cz,
]

# ─────────────────────────────────────────────────────────────────────────────
# HAMILTONIAN CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (name, Jx, Jy, Jz, hx, hy, hz)
# Hamiltonian: H = Jx∑XX + Jy∑YY + Jz∑ZZ + hx∑X
# Use Jx=-1 for antiferromagnetic, Jx=+1 for ferromagnetic
const HAMILTONIAN_CONFIGS = [
    ("XXZ",  -1.0, -1.0, -0.5, 1.0, 0.0, 0.0),  # XXZ antiferromagnetic + transverse field
    ("XY",   -1.0, -1.0,  0.0, 1.0, 0.0, 0.0),  # XY model (XX+YY) + transverse field
    ("TFIM",  0.0,  0.0, -1.0, 1.0, 0.0, 0.0),  # Transverse Field Ising
]

println("\nConfig: N=$N_VALUES, L=$L_VALUES, $N_EPOCHS epochs")
println("Hamiltonians: $(length(HAMILTONIAN_CONFIGS)) configs")
println("Ansatzes: $ANSATZ_TYPES")

# ==============================================================================
# ANSATZ BUILDER
# ==============================================================================

function build_ansatz_spec(N, n_layers, ansatz, entangler_gate)
    gates = Tuple{Symbol, Any, Int}[]
    pidx = 0

    for layer in 1:n_layers
        # Rotation layer: RY-RZ on each qubit
        for q in 1:N
            for rot in ROTATION_GATES
                pidx += 1
                push!(gates, (rot, q, pidx))
            end
        end

        # Entangling layer
        if ansatz == :brick
            if layer % 2 == 1
                for q in 1:2:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
            else
                for q in 2:2:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
            end
        elseif ansatz == :chain
            for q in 1:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
        elseif ansatz == :ring
            for q in 1:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
            push!(gates, (entangler_gate, (N, 1), 0))
        elseif ansatz == :full
            for i in 1:N, j in (i+1):N
                push!(gates, (entangler_gate, (i, j), 0))
            end
        elseif ansatz == :hea
            for q in 1:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
        else
            error("Unknown ansatz: $ansatz")
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
# PURE STATE OPERATIONS
# ==============================================================================

function apply_ansatz_psi!(ψ, θ, gates, N)
    for (g, t, p) in gates
        g == :rx && apply_rx_psi!(ψ, t, θ[p], N)
        g == :ry && apply_ry_psi!(ψ, t, θ[p], N)
        g == :rz && apply_rz_psi!(ψ, t, θ[p], N)
        g == :cz && apply_cz_psi!(ψ, t[1], t[2], N)
    end
end

function prepare_ansatz_state(θ, gates, N)
    ψ = zeros(ComplexF64, 2^N)
    ψ[1] = 1.0  # Start from |0...0⟩
    apply_ansatz_psi!(ψ, θ, gates, N)
    return ψ
end

# ==============================================================================
# HAMILTONIAN ENERGY COMPUTATION (Matrix-Free)
# ==============================================================================

"""
Compute ⟨ψ|H|ψ⟩ for XXZ Hamiltonian:
H = Jx∑XX + Jy∑YY + Jz∑ZZ + hx∑X
Use Jx=-1 for antiferromagnetic, Jx=+1 for ferromagnetic.
Uses matrix-free bitwise operations.
"""
function compute_energy(ψ, N, Jx, Jy, Jz, hx)
    dim = 2^N
    E = 0.0

    # Local X field: hx * Σ_i X_i
    for i in 1:N
        for k in 0:(dim-1)
            j = xor(k, 1 << (i-1))  # Flip bit i
            E += hx * real(conj(ψ[k+1]) * ψ[j+1])
        end
    end

    # Two-body terms: Jx XX + Jy YY + Jz ZZ
    for i in 1:(N-1)
        j = i + 1  # Nearest-neighbor
        for k in 0:(dim-1)
            bi = (k >> (i-1)) & 1
            bj = (k >> (j-1)) & 1

            # ZZ term: diagonal
            zi = 1 - 2*bi
            zj = 1 - 2*bj
            E += Jz * zi * zj * abs2(ψ[k+1])

            # XX term: flips both bits
            k_xx = xor(xor(k, 1 << (i-1)), 1 << (j-1))
            E += Jx * real(conj(ψ[k+1]) * ψ[k_xx+1])

            # YY term: flips both bits with phase
            # σy|0⟩ = i|1⟩, σy|1⟩ = -i|0⟩
            phase_i = bi == 0 ? im : -im
            phase_j = bj == 0 ? im : -im
            phase = phase_i * phase_j  # Total phase
            E += Jy * real(conj(ψ[k+1]) * phase * ψ[k_xx+1])
        end
    end

    return E
end

# ==============================================================================
# BIPARTITE ENTROPY
# ==============================================================================

function bipartite_entropy(ψ, N)
    # Split at N/2
    k = N ÷ 2
    subsystem_B = collect((k+1):N)  # Trace out second half
    ρ_A = partial_trace(ψ, subsystem_B, N)
    return von_neumann_entropy(ρ_A)
end

# ==============================================================================
# LOCAL OBSERVABLES
# ==============================================================================

"""Compute mean ⟨X⟩, ⟨Y⟩, ⟨Z⟩ averaged over all qubits."""
function compute_local_observables(ψ, N)
    dim = 2^N
    X_mean, Y_mean, Z_mean = 0.0, 0.0, 0.0

    for i in 1:N
        X_i, Y_i, Z_i = 0.0, 0.0, 0.0
        for k in 0:(dim-1)
            bi = (k >> (i-1)) & 1
            # Z_i
            Z_i += (1 - 2*bi) * abs2(ψ[k+1])
            # X_i: flip bit i
            j = xor(k, 1 << (i-1))
            X_i += real(conj(ψ[k+1]) * ψ[j+1])
            # Y_i: flip bit i with phase
            phase = bi == 0 ? im : -im
            Y_i += real(conj(ψ[k+1]) * phase * ψ[j+1])
        end
        X_mean += X_i
        Y_mean += Y_i
        Z_mean += Z_i
    end

    return X_mean/N, Y_mean/N, Z_mean/N
end

"""Compute mean ⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩ averaged over all nearest-neighbor bonds."""
function compute_correlators(ψ, N)
    dim = 2^N
    n_bonds = N - 1
    XX_mean, YY_mean, ZZ_mean = 0.0, 0.0, 0.0

    for i in 1:(N-1)
        j = i + 1  # Nearest neighbor
        XX_ij, YY_ij, ZZ_ij = 0.0, 0.0, 0.0

        for k in 0:(dim-1)
            bi = (k >> (i-1)) & 1
            bj = (k >> (j-1)) & 1

            # ZZ
            ZZ_ij += (1-2*bi) * (1-2*bj) * abs2(ψ[k+1])

            # XX: flip both bits
            k_xx = xor(xor(k, 1 << (i-1)), 1 << (j-1))
            XX_ij += real(conj(ψ[k+1]) * ψ[k_xx+1])

            # YY: flip both bits with phase
            phase_i = bi == 0 ? im : -im
            phase_j = bj == 0 ? im : -im
            YY_ij += real(conj(ψ[k+1]) * phase_i * phase_j * ψ[k_xx+1])
        end

        XX_mean += XX_ij
        YY_mean += YY_ij
        ZZ_mean += ZZ_ij
    end

    return XX_mean/n_bonds, YY_mean/n_bonds, ZZ_mean/n_bonds
end

# ==============================================================================
# VQE EVALUATION
# ==============================================================================

function evaluate_vqe(θ, gates, N, Jx, Jy, Jz, hx, ψ_exact)
    ψ = prepare_ansatz_state(θ, gates, N)

    # Energy
    E = compute_energy(ψ, N, Jx, Jy, Jz, hx)

    # Bipartite entropy
    S = bipartite_entropy(ψ, N)

    # Fidelity with exact ground state
    ψ_norm = ψ / norm(ψ)
    ψ_exact_norm = ψ_exact / norm(ψ_exact)
    F = abs2(dot(ψ_exact_norm, ψ_norm))

    # Local observables: ⟨X⟩, ⟨Y⟩, ⟨Z⟩
    X_mean, Y_mean, Z_mean = compute_local_observables(ψ, N)

    # Correlators: ⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩
    XX_mean, YY_mean, ZZ_mean = compute_correlators(ψ, N)

    return E, S, F, X_mean, Y_mean, Z_mean, XX_mean, YY_mean, ZZ_mean, ψ
end

# ==============================================================================
# SPSA OPTIMIZER
# ==============================================================================

mutable struct SPSA
    a::Float64
    c::Float64
    A::Float64
    α::Float64
    γ::Float64
    k::Int
end
SPSA() = SPSA(0.1, 0.15, 10.0, 0.602, 0.101, 0)

function spsa_grad!(s::SPSA, cost_fn, θ)
    s.k += 1
    c_k = s.c / s.k^s.γ
    Δ = 2.0 .* (rand(length(θ)) .> 0.5) .- 1.0
    return (cost_fn(θ .+ c_k.*Δ) - cost_fn(θ .- c_k.*Δ)) ./ (2*c_k.*Δ)
end

# ==============================================================================
# TRAINING
# ==============================================================================

function train_vqe(N, n_layers, ansatz, entangler_gate, ham_config; n_epochs=500, lr=0.02)
    name, Jx, Jy, Jz, hx, hy, hz = ham_config

    # Get exact ground state
    E_gs, ψ_exact = ground_state_xxz(N, Jx, Jy, Jz, hx)
    S_gs = bipartite_entropy(ψ_exact, N)

    # Ground state observables for reference
    X_gs, Y_gs, Z_gs = compute_local_observables(ψ_exact, N)
    XX_gs, YY_gs, ZZ_gs = compute_correlators(ψ_exact, N)

    # Build ansatz
    n_params, gates = build_ansatz_spec(N, n_layers, ansatz, entangler_gate)
    θ = 0.1 * randn(n_params)

    # Optimizer
    spsa = SPSA()
    opt = Optimisers.setup(Adam(lr), θ)

    # Training history
    energies, entropies, fidelities = Float64[], Float64[], Float64[]
    X_hist, Y_hist, Z_hist = Float64[], Float64[], Float64[]
    XX_hist, YY_hist, ZZ_hist = Float64[], Float64[], Float64[]

    total_time = @elapsed for ep in 1:n_epochs
        cost = θ_t -> begin
            ψ = prepare_ansatz_state(θ_t, gates, N)
            compute_energy(ψ, N, Jx, Jy, Jz, hx)
        end

        g = spsa_grad!(spsa, cost, θ)
        opt, θ = Optimisers.update(opt, θ, g)

        E, S, F, X, Y, Z, XX, YY, ZZ, _ = evaluate_vqe(θ, gates, N, Jx, Jy, Jz, hx, ψ_exact)
        push!(energies, E); push!(entropies, S); push!(fidelities, F)
        push!(X_hist, X); push!(Y_hist, Y); push!(Z_hist, Z)
        push!(XX_hist, XX); push!(YY_hist, YY); push!(ZZ_hist, ZZ)

        if ep % 50 == 0
            ΔE = E - E_gs
            @printf("    [%s|L=%d] ep=%d: E = %.6f (ΔE = %.2e), S = %.4f, F = %.4f\n",
                    ansatz, n_layers, ep, E, ΔE, S, F)
        end
    end

    @printf("    [%s|L=%d] Training time: %.2fs (%.2f ms/epoch)\n",
            ansatz, n_layers, total_time, 1000*total_time/n_epochs)

    return (energies=energies, entropies=entropies, fidelities=fidelities,
            X_hist=X_hist, Y_hist=Y_hist, Z_hist=Z_hist,
            XX_hist=XX_hist, YY_hist=YY_hist, ZZ_hist=ZZ_hist,
            E_gs=E_gs, S_gs=S_gs, X_gs=X_gs, Y_gs=Y_gs, Z_gs=Z_gs,
            XX_gs=XX_gs, YY_gs=YY_gs, ZZ_gs=ZZ_gs, time=total_time)
end

# ==============================================================================
# MAIN LOOP
# ==============================================================================

colors = [:blue, :red, :green, :purple, :orange, :cyan, :magenta, :brown]

for (ham_config) in HAMILTONIAN_CONFIGS
    name, Jx, Jy, Jz, hx, hy, hz = ham_config
    println("\n" * "=" ^ 70)
    println("  HAMILTONIAN: $name (Jx=$Jx, Jy=$Jy, Jz=$Jz, hx=$hx)")
    println("=" ^ 70)

for ansatz in ANSATZ_TYPES
    println("\n  ANSATZ: $ansatz")

for entangler_gate in ENTANGLER_GATES
    println("    Entangler: $entangler_gate")

for N in N_VALUES
    println("\n    N = $N QUBITS")

    # Get exact ground state info for reference lines
    E_gs_ref, ψ_exact_ref = ground_state_xxz(N, Jx, Jy, Jz, hx)
    S_gs_ref = bipartite_entropy(ψ_exact_ref, N)
    X_gs_ref, Y_gs_ref, Z_gs_ref = compute_local_observables(ψ_exact_ref, N)
    XX_gs_ref, YY_gs_ref, ZZ_gs_ref = compute_correlators(ψ_exact_ref, N)
    println("      E_GS = $(@sprintf("%.6f", E_gs_ref)), S_GS = $(@sprintf("%.4f", S_gs_ref))")

    # Storage for all L values
    results = Dict{Int, NamedTuple}()

    for (i, n_layers) in enumerate(L_VALUES)
        println("      L = $n_layers")
        Random.seed!(42 + N*1000 + n_layers*100 + hash(ansatz))

        res = train_vqe(N, n_layers, ansatz, entangler_gate, ham_config;
                        n_epochs=N_EPOCHS, lr=LR)
        results[n_layers] = res
    end

    # Create plot: 3 rows x 3 columns
    # Row 1: Energy, Entropy, Fidelity
    # Row 2: <X>, <Y>, <Z>
    # Row 3: <XX>, <YY>, <ZZ>
    rot_str = join([uppercase(string(r)[2]) for r in ROTATION_GATES], "")
    ent_str = uppercase(string(entangler_gate))

    # Title line 1: Model name + ansatz + N
    # Title line 2: General Hamiltonian + parameters
    title_line1 = "VQE: $name | $ansatz | N=$N"
    title_line2 = "H = Jxx∑XX + Jyy∑YY + Jzz∑ZZ + hx∑X  |  Jxx=$Jx, Jyy=$Jy, Jzz=$Jz, hx=$hx"

    plt = plot(layout=(3, 3), size=(1500, 1200),
               plot_title="$title_line1\n$title_line2",
               margin=5Plots.mm, left_margin=12Plots.mm, bottom_margin=8Plots.mm,
               titlefontsize=12, guidefontsize=11, tickfontsize=9, legendfontsize=9)

    # Row 1: Energy, Entropy, Fidelity
    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        time_str = @sprintf("%.1fs", data.time)
        final_E = @sprintf("%.4f", data.energies[end])
        plot!(plt[1], data.energies, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers ($time_str) → $final_E",
              xlabel="Epoch", ylabel="Energy E(θ)",
              title="Energy", legend=:topright)
    end
    hline!(plt[1], [E_gs_ref], lw=2, color=:black, ls=:dash, label="GS=$(@sprintf("%.2f", E_gs_ref))")

    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        final_S = @sprintf("%.4f", data.entropies[end])
        plot!(plt[2], data.entropies, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers → $final_S",
              xlabel="Epoch", ylabel="Sᵥₙ",
              title="Entanglement Entropy", legend=:topright)
    end
    hline!(plt[2], [S_gs_ref], lw=2, color=:black, ls=:dash, label="GS=$(@sprintf("%.3f", S_gs_ref))")

    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        final_F = @sprintf("%.4f", data.fidelities[end])
        plot!(plt[3], data.fidelities, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers → $final_F",
              xlabel="Epoch", ylabel="Fidelity F",
              title="Fidelity with GS", legend=:bottomright)
    end
    hline!(plt[3], [1.0], lw=1, color=:black, ls=:dot, label="")

    # Row 2: Local observables <X>, <Y>, <Z>
    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        final_X = @sprintf("%.4f", data.X_hist[end])
        plot!(plt[4], data.X_hist, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers → $final_X",
              xlabel="Epoch", ylabel="⟨X⟩",
              title="Mean Local ⟨X⟩", legend=:topright)
    end
    hline!(plt[4], [X_gs_ref], lw=2, color=:black, ls=:dash, label="GS=$(@sprintf("%.3f", X_gs_ref))")

    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        final_Y = @sprintf("%.4f", data.Y_hist[end])
        plot!(plt[5], data.Y_hist, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers → $final_Y",
              xlabel="Epoch", ylabel="⟨Y⟩",
              title="Mean Local ⟨Y⟩", legend=:topright)
    end
    hline!(plt[5], [Y_gs_ref], lw=2, color=:black, ls=:dash, label="GS=$(@sprintf("%.3f", Y_gs_ref))")

    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        final_Z = @sprintf("%.4f", data.Z_hist[end])
        plot!(plt[6], data.Z_hist, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers → $final_Z",
              xlabel="Epoch", ylabel="⟨Z⟩",
              title="Mean Local ⟨Z⟩", legend=:topright)
    end
    hline!(plt[6], [Z_gs_ref], lw=2, color=:black, ls=:dash, label="GS=$(@sprintf("%.3f", Z_gs_ref))")

    # Row 3: Correlators <XX>, <YY>, <ZZ>
    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        final_XX = @sprintf("%.4f", data.XX_hist[end])
        plot!(plt[7], data.XX_hist, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers → $final_XX",
              xlabel="Epoch", ylabel="⟨XX⟩",
              title="Mean NN ⟨XX⟩", legend=:topright)
    end
    hline!(plt[7], [XX_gs_ref], lw=2, color=:black, ls=:dash, label="GS=$(@sprintf("%.3f", XX_gs_ref))")

    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        final_YY = @sprintf("%.4f", data.YY_hist[end])
        plot!(plt[8], data.YY_hist, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers → $final_YY",
              xlabel="Epoch", ylabel="⟨YY⟩",
              title="Mean NN ⟨YY⟩", legend=:topright)
    end
    hline!(plt[8], [YY_gs_ref], lw=2, color=:black, ls=:dash, label="GS=$(@sprintf("%.3f", YY_gs_ref))")

    for (i, n_layers) in enumerate(L_VALUES)
        data = results[n_layers]
        final_ZZ = @sprintf("%.4f", data.ZZ_hist[end])
        plot!(plt[9], data.ZZ_hist, lw=2, color=colors[mod1(i, length(colors))],
              label="L=$n_layers → $final_ZZ",
              xlabel="Epoch", ylabel="⟨ZZ⟩",
              title="Mean NN ⟨ZZ⟩", legend=:topright)
    end
    hline!(plt[9], [ZZ_gs_ref], lw=2, color=:black, ls=:dash, label="GS=$(@sprintf("%.3f", ZZ_gs_ref))")

    # Save plot
    fname = lowercase(name)
    base_fname = "vqe_$(fname)_$(ansatz)_N$(N)_R$(rot_str)_$(ent_str)"
    savefig(plt, joinpath(OUTPUT_DIR, "fig_$base_fname.png"))
    println("      → Saved: fig_$base_fname.png")

    # Save data
    data_dir = joinpath(OUTPUT_DIR, "data")
    mkpath(data_dir)
    for n_layers in L_VALUES
        data = results[n_layers]
        data_fname = joinpath(data_dir, "$(base_fname)_L$(n_layers).txt")
        open(data_fname, "w") do f
            println(f, "# VQE results: $(name), N=$N, L=$n_layers, ansatz=$ansatz")
            println(f, "# E_GS=$(data.E_gs), S_GS=$(data.S_gs)")
            println(f, "# X_GS=$(data.X_gs), Y_GS=$(data.Y_gs), Z_GS=$(data.Z_gs)")
            println(f, "# XX_GS=$(data.XX_gs), YY_GS=$(data.YY_gs), ZZ_GS=$(data.ZZ_gs)")
            println(f, "epoch,energy,entropy,fidelity,X,Y,Z,XX,YY,ZZ")
            for ep in 1:length(data.energies)
                @printf(f, "%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                        ep, data.energies[ep], data.entropies[ep], data.fidelities[ep],
                        data.X_hist[ep], data.Y_hist[ep], data.Z_hist[ep],
                        data.XX_hist[ep], data.YY_hist[ep], data.ZZ_hist[ep])
            end
        end
    end
    println("      → Data saved to data/ subfolder")

end  # N
end  # entangler_gate
end  # ansatz
end  # ham_config

println("\n" * "=" ^ 70)
println("  VQE DEMO COMPLETE")
println("=" ^ 70)
println("\nDone!")

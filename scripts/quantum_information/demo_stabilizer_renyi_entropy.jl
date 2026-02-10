#!/usr/bin/env julia
#=
╔══════════════════════════════════════════════════════════════════════════════╗
║              STABILIZER RÉNYI ENTROPY (SRE) BENCHMARK                        ║
║              Measuring "Magic" in Quantum States                             ║
║                                                                              ║
║              Two Algorithms: Brute Force O(8^N) vs FWHT O(N·4^N)             ║
╚══════════════════════════════════════════════════════════════════════════════╝

# WHAT IS THIS DEMO?
═══════════════════════════════════════════════════════════════════════════════

This script benchmarks the computation of the Stabilizer Rényi Entropy M₂,
which quantifies the "nonstabilizerness" or "magic" of a quantum state.

We use TWO ALGORITHMS and compare their accuracy and performance:

  1. BRUTE FORCE (cpuStabilizerRenyiEntropy.jl)
     - Complexity: O(8^N) = O(4^N Paulis × 2^N per expectation)
     - Accurate but slow, practical limit: N ≤ 12

  2. FAST WALSH-HADAMARD TRANSFORM (cpuStabilizerRenyiEntropyFastWalshHadamardTransform.jl)
     - Complexity: O(N·4^N) using FWHT decomposition (EXACT sum over all 4^N Paulis)
     - Reference: Sierant, Vallès-Muns & Garcia-Saez (2026), arXiv:2601.07824
     - Enables computation up to N ~ 16-18 (vs N ~ 10-12 for brute force)

We compare results on two types of states:
  • |+...+⟩ = H⊗N|0...0⟩   : A STABILIZER state (no magic, M₂ = 0)
  • T⊗N|+...+⟩             : A MAGICAL state (M₂ > 0)


# WHY T⊗N|+...+⟩ ?
═══════════════════════════════════════════════════════════════════════════════

The T gate (π/8 rotation) is the simplest source of "magic" in quantum computing:

    T = diag(1, e^{iπ/4})

When applied to the |+⟩ = (|0⟩ + |1⟩)/√2 state:

    T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2

This creates one of the fundamental "magic states" used in fault-tolerant QC.


# WHY PRODUCT STATES?
═══════════════════════════════════════════════════════════════════════════════

The state T⊗N|+...+⟩ is a PRODUCT STATE of N copies of T|+⟩:

    T⊗N|+...+⟩ = T|+⟩ ⊗ T|+⟩ ⊗ ... ⊗ T|+⟩   (N times)

This is special because SRE has the ADDITIVITY property:

    M₂(|ψ⟩ ⊗ |φ⟩) = M₂(|ψ⟩) + M₂(|φ⟩)

Therefore:
    M₂(T⊗N|+...+⟩) = N × M₂(T|+⟩) ≈ 0.415 × N

This provides a perfect test case with KNOWN, LINEAR scaling!


# EXPECTED RESULTS TABLE
═══════════════════════════════════════════════════════════════════════════════

    ┌───────┬─────────────────┬────────────────┬─────────────────┐
    │   N   │   4^N Paulis    │  M₂(|+...+⟩)   │  M₂(T⊗N|+...+⟩) │
    ├───────┼─────────────────┼────────────────┼─────────────────┤
    │   2   │            16   │      0.000     │       0.830     │
    │   4   │           256   │      0.000     │       1.660     │
    │   6   │         4,096   │      0.000     │       2.490     │
    │   8   │        65,536   │      0.000     │       3.320     │
    │  10   │     1,048,576   │      0.000     │       4.150     │
    │  12   │    16,777,216   │      0.000     │       4.980     │
    │  14   │   268,435,456   │      0.000     │       5.811     │
    └───────┴─────────────────┴────────────────┴─────────────────┘

    Key: M₂(T⊗N|+⟩) ≈ 0.4150 × N  (linear scaling!)


# COMPUTATIONAL COMPLEXITY COMPARISON
═══════════════════════════════════════════════════════════════════════════════

    BRUTE FORCE: O(8^N)                    FWHT: O(N·4^N)
    ──────────────────────                 ─────────────────
    • N=10:  ~1.5 seconds                  • N=10:  ~0.04s     (~35× faster)
    • N=12:  ~90 seconds                   • N=12:  ~0.6s      (~150× faster)
    • N=14:  ~1.5 hours                    • N=14:  ~8s        (~600× faster)
    • N=16:  IMPOSSIBLE                    • N=16:  ~150s

    Speedup grows as O(2^N) since 8^N / (N·4^N) = 2^N / N


# USAGE
═══════════════════════════════════════════════════════════════════════════════

    julia -t auto demo_stabilizer_renyi_entropy.jl

    Outputs:
    • figures/fig_sre_M2_plus_vs_T_plus.png  : M₂ vs N plot
    • figures/fig_sre_time_vs_N.png         : Timing plot
    • data/summary_output.txt               : Tabulated results

=#

using CairoMakie
using CairoMakie
using Dates
using LinearAlgebra
using Pkg
using Printf
using Random





# Plotting
try
catch
    Pkg.add("CairoMakie")
end

# Include modules
using SmoQ.CPUStabilizerRenyiEntropyBruteForce
using SmoQ.CPUStabilizerRenyiEntropyFastWalshHadamardTransform
using SmoQ.CPUQuantumChannelGates

# Aliases for the two methods
const magic_brute = CPUStabilizerRenyiEntropyBruteForce.magic    # O(8^N) brute force
const magic_fwht = CPUStabilizerRenyiEntropyFastWalshHadamardTransform.magic_fwht  # O(N·2^N) FWHT

# ============================================================================
# Output Directories
# ============================================================================

const OUTPUT_DIR = joinpath(@__DIR__, "demo_stabilizer_renyi_entropy")
const FIGURES_DIR = joinpath(OUTPUT_DIR, "figures")
const DATA_DIR = joinpath(OUTPUT_DIR, "data")

mkpath(FIGURES_DIR)
mkpath(DATA_DIR)

# ============================================================================
# State Preparation
# ============================================================================

"""Create GHZ state: (|00...0⟩ + |11...1⟩)/√2"""
function state_ghz(N::Int)
    ψ = zeros(ComplexF64, 2^N)
    ψ[1] = 1/sqrt(2)
    ψ[end] = 1/sqrt(2)
    return ψ
end

"""Create |+...+⟩ = H⊗N|0...0⟩ - uniform superposition"""
function state_plus(N::Int)
    return ones(ComplexF64, 2^N) / sqrt(2^N)
end

"""
Create T⊗N|+...+⟩ - apply T gate to each qubit of plus state.

This is a product of T|+⟩ magic states, so by additivity:
    M₂(T⊗N|+...+⟩) = N × M₂(T|+⟩) ≈ 0.415 × N
"""
function state_T_plus(N::Int)
    ψ = state_plus(N)
    for k in 1:N
        apply_t_psi!(ψ, k, N)
    end
    return ψ
end

"""
Create star graph state: |Star_N⟩

A star graph has N qubits: 1 central node (qubit 1) and N-1 leaf nodes.
All leaves are connected to the center via CZ gates.

    Topology:
           2
           |
       3 — 1 — 4     (qubit 1 is center)
           |
           5

Construction:
    1. Start with |+...+⟩ (all qubits in |+⟩)
    2. Apply CZ(1,k) for k = 2, 3, ..., N

This is a STABILIZER state (M₂ = 0).
"""
function state_star_graph(N::Int)
    N < 2 && error("Star graph requires N ≥ 2")
    ψ = state_plus(N)
    # Apply CZ between center (qubit 1) and each leaf (qubits 2 to N)
    for k in 2:N
        apply_cz_psi!(ψ, 1, k, N)
    end
    return ψ
end

"""
Create star graph with T on ALL nodes: T⊗N|Star_N⟩

Apply T gate to every qubit (center + all leaves).
This is an ENTANGLED magical state.

Unlike the product state T⊗N|+...+⟩, the star graph has entanglement,
so the magic scaling may differ from the simple additive case.
"""
function state_star_T_all(N::Int)
    N < 2 && error("Star+T requires N ≥ 2")
    ψ = state_star_graph(N)
    # Apply T to ALL qubits
    for k in 1:N
        apply_t_psi!(ψ, k, N)
    end
    return ψ
end

"""
Create star graph with T ONLY on center: T₁|Star_N⟩

Apply T gate only to the central qubit (qubit 1).
The leaves remain in their original stabilizer state.

This tests how magic from one qubit propagates through entanglement.
"""
function state_star_T_center(N::Int)
    N < 2 && error("Star+T requires N ≥ 2")
    ψ = state_star_graph(N)
    # Apply T only to center (qubit 1)
    apply_t_psi!(ψ, 1, N)
    return ψ
end

"""Create Haar-random state"""
function state_haar_random(N::Int; seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    ψ = randn(ComplexF64, 2^N)
    ψ ./= norm(ψ)
    return ψ
end

# Helper functions
mean(x) = sum(x) / length(x)
std(x) = length(x) > 1 ? sqrt(sum((xi - mean(x))^2 for xi in x) / (length(x) - 1)) : 0.0

# ============================================================================
# Main Benchmark
# ============================================================================

function run_benchmark()
    # Output buffer for saving to file
    output_lines = String[]

    function log(s::String="")
        push!(output_lines, s)
        println(s)
    end

    # Print banner
    log()
    log("╔" * "═"^78 * "╗")
    log("║" * " "^15 * "STABILIZER RÉNYI ENTROPY (SRE) BENCHMARK" * " "^22 * "║")
    log("║" * " "^15 * "Measuring 'Magic' in Quantum States" * " "^26 * "║")
    log("╚" * "═"^78 * "╝")
    log()

    log("┌─────────────────────────────────────────────────────────────────────────────┐")
    log("│  PHYSICS BACKGROUND                                                         │")
    log("├─────────────────────────────────────────────────────────────────────────────┤")
    log("│  • T gate = diag(1, e^{iπ/4}) — SINGLE-qubit non-Clifford gate              │")
    log("│  • T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2 — 'magic state' for universal QC           │")
    log("│  • M₂ (SRE) = 0 for stabilizer states, M₂ > 0 for magical states            │")
    log("│                                                                             │")
    log("│  TEST STATES:                                                               │")
    log("│  1. T⊗N|+...+⟩  — Product state, ADDITIVE magic: M₂ = 0.415 × N             │")
    log("│  2. Star graph  — Entangled (CZ between center and leaves)                  │")
    log("│     + T on ALL nodes    (T⊗N|Star⟩)                                         │")
    log("│     + T on center ONLY  (T₁|Star⟩)                                          │")
    log("└─────────────────────────────────────────────────────────────────────────────┘")
    log()

    log("Configuration:")
    log("  • Threads: $(Threads.nthreads())")
    log("  • N range: 2 → 14  (4^14 = 268M Pauli strings at N=14)")
    log("  • Complexity: O(8^N) per state")
    log()

    # Results storage
    N_vals = 2:10
    results = Dict{String, Dict{Int, NamedTuple}}()
    for state in ["plus", "T_plus", "star", "star_T_all", "star_T_center"]
        results[state] = Dict{Int, NamedTuple}()
    end

    # Warmup
    log("[Warmup] JIT compiling with N=4...")
    magic(state_T_plus(4), 4)
    magic(state_star_T_all(4), 4)
    log("[Warmup] Done.")
    log()

    # =========================================================================
    # BENCHMARK 1: Product States (T|+)
    # =========================================================================
    log("═"^80)
    log("BENCHMARK 1: PRODUCT STATES — T⊗N|+...+⟩")
    log("═"^80)
    log()
    log("  Expected: M₂(T⊗N|+⟩) = N × M₂(T|+⟩) ≈ 0.415 × N  (ADDITIVE)")
    log()
    log("  ┌──────┬──────────────┬────────────┬────────────┬───────────┐")
    log("  │   N  │  4^N Paulis  │  M₂(|+⟩)   │  M₂(T|+⟩)  │   time    │")
    log("  ├──────┼──────────────┼────────────┼────────────┼───────────┤")

    N_completed = Int[]
    for N in N_vals
        paulis = 4^N

        # |+...+⟩ state (stabilizer)
        t_plus = @elapsed M2_plus = magic(state_plus(N), N)
        results["plus"][N] = (M2=M2_plus, time=t_plus, paulis=paulis)

        # T⊗N|+...+⟩ state (magical)
        t_T_plus = @elapsed M2_T_plus = magic(state_T_plus(N), N)
        results["T_plus"][N] = (M2=M2_T_plus, time=t_T_plus, paulis=paulis)

        push!(N_completed, N)

        fmt_t = t_T_plus < 60 ? @sprintf("%.2fs", t_T_plus) : @sprintf("%.1fm", t_T_plus/60)
        line = @sprintf("  │  %2d  │ %12d │   %+.4f   │   %+.4f   │ %9s │",
                        N, paulis, M2_plus, M2_T_plus, fmt_t)
        log(line)

        if t_T_plus > 7200
            log("  └──────┴──────────────┴────────────┴────────────┴───────────┘")
            log("  ⚠ Stopping: time limit reached")
            break
        end
    end
    log("  └──────┴──────────────┴────────────┴────────────┴───────────┘")

    # Linear fit
    M2_vals = [results["T_plus"][N].M2 for N in N_completed]
    slope_product = sum(N_completed .* M2_vals) / sum(N_completed.^2)
    log()
    log("  → Linear fit: M₂(T⊗N|+⟩) ≈ $(round(slope_product, digits=4)) × N")
    log("  → Theory:     M₂(T|+⟩)  = 0.4150")
    log()

    # =========================================================================
    # BENCHMARK 2: Star Graph States
    # =========================================================================
    log("═"^80)
    log("BENCHMARK 2: STAR GRAPH STATES — Entangled with CZ edges")
    log("═"^80)
    log()
    log("  Star topology: qubit 1 = center, qubits 2..N = leaves")
    log("         2                                    ")
    log("         │                                    ")
    log("     3 — 1 — 4    CZ gates: (1,2), (1,3), ... (1,N)")
    log("         │                                    ")
    log("         5                                    ")
    log()
    log("  Comparing:")
    log("    • |Star⟩ alone (stabilizer, M₂ = 0)")
    log("    • T⊗N|Star⟩ (T on ALL qubits)")
    log("    • T₁|Star⟩  (T on center ONLY)")
    log()
    log("  ┌──────┬────────────┬────────────┬────────────┬───────────┐")
    log("  │   N  │ M₂(|Star⟩) │ M₂(T⊗N|S⟩) │ M₂(T₁|S⟩)  │   time    │")
    log("  ├──────┼────────────┼────────────┼────────────┼───────────┤")

    for N in N_completed
        paulis = 4^N

        # Star graph (stabilizer)
        t_star = @elapsed M2_star = magic(state_star_graph(N), N)
        results["star"][N] = (M2=M2_star, time=t_star, paulis=paulis)

        # Star + T on all
        t_all = @elapsed M2_all = magic(state_star_T_all(N), N)
        results["star_T_all"][N] = (M2=M2_all, time=t_all, paulis=paulis)

        # Star + T on center only
        t_center = @elapsed M2_center = magic(state_star_T_center(N), N)
        results["star_T_center"][N] = (M2=M2_center, time=t_center, paulis=paulis)

        total_time = t_star + t_all + t_center
        fmt_t = total_time < 60 ? @sprintf("%.2fs", total_time) : @sprintf("%.1fm", total_time/60)
        line = @sprintf("  │  %2d  │   %+.4f   │   %+.4f   │   %+.4f   │ %9s │",
                        N, M2_star, M2_all, M2_center, fmt_t)
        log(line)

        if total_time > 7200
            log("  └──────┴────────────┴────────────┴────────────┴───────────┘")
            log("  ⚠ Stopping: time limit reached")
            break
        end
    end
    log("  └──────┴────────────┴────────────┴────────────┴───────────┘")

    # Analysis
    M2_all_vals = [results["star_T_all"][N].M2 for N in N_completed]
    M2_center_vals = [results["star_T_center"][N].M2 for N in N_completed]
    slope_all = sum(N_completed .* M2_all_vals) / sum(N_completed.^2)

    log()
    log("  → T on ALL nodes:    M₂ ≈ $(round(slope_all, digits=4)) × N  (similar to product)")
    log("  → T on center ONLY:  M₂ ≈ $(round(mean(M2_center_vals), digits=4))   (const? or weak N-dep?)")
    log()

    # =========================================================================
    # Summary
    # =========================================================================
    log("═"^80)
    log("SUMMARY")
    log("═"^80)
    log()
    log("  ┌────────────────────────┬────────────────────────────────────────┐")
    log("  │  State                 │  Magic Scaling                         │")
    log("  ├────────────────────────┼────────────────────────────────────────┤")
    log(@sprintf("  │  T⊗N|+...+⟩ (product)  │  M₂ ≈ %.4f × N  (additive)             │", slope_product))
    log(@sprintf("  │  T⊗N|Star⟩ (entangled) │  M₂ ≈ %.4f × N                         │", slope_all))
    log(@sprintf("  │  T₁|Star⟩ (1 T gate)   │  M₂ ≈ %.4f      (subextensive)         │", mean(M2_center_vals)))
    log("  └────────────────────────┴────────────────────────────────────────┘")
    log()

    # =========================================================================
    # BENCHMARK 3: Method Comparison - Brute Force vs FWHT
    # =========================================================================
    log("═"^80)
    log("BENCHMARK 3: ALGORITHM COMPARISON — Brute Force O(8^N) vs FWHT O(N·2^N)")
    log("═"^80)
    log()
    log("  Reference: Sierant, Vallès-Muns & Garcia-Saez (2025), arXiv:2601.07824")
    log()

    # Accuracy comparison
    log("  ┌──────┬────────────┬────────────┬────────────┐")
    log("  │   N  │ M₂(brute)  │ M₂(FWHT)   │   |diff|   │")
    log("  ├──────┼────────────┼────────────┼────────────┤")

    for N in 2:min(10, maximum(N_completed))
        ψ = state_T_plus(N)
        M2_b = magic_brute(ψ, N)
        M2_f = magic_fwht(ψ, N)
        diff = abs(M2_b - M2_f)
        log(@sprintf("  │  %2d  │  %+.5f  │  %+.5f  │  %.2e  │", N, M2_b, M2_f, diff))
    end
    log("  └──────┴────────────┴────────────┴────────────┘")
    log()

    # Timing comparison
    log("  ┌──────┬────────────────┬────────────────┬────────────┐")
    log("  │   N  │  t(brute) [s]  │  t(FWHT) [s]   │  speedup   │")
    log("  ├──────┼────────────────┼────────────────┼────────────┤")

    for N in 2:min(12, maximum(N_completed))
        ψ = state_T_plus(N)

        t_brute = @elapsed magic_brute(ψ, N)
        t_fwht = @elapsed magic_fwht(ψ, N)
        speedup = t_brute / max(t_fwht, 1e-9)

        log(@sprintf("  │  %2d  │    %10.4f    │    %10.6f    │   %6.1f×   │",
                     N, t_brute, t_fwht, speedup))
    end
    log("  └──────┴────────────────┴────────────────┴────────────┘")
    log()

    # FWHT extended range (beyond brute force limits)
    log("  FWHT SCALABILITY: Beyond brute force limits (N > 12)")
    log("  ┌──────┬────────────────┬────────────┐")
    log("  │   N  │  t(FWHT) [s]   │   M₂(T|+⟩) │")
    log("  ├──────┼────────────────┼────────────┤")

    for N in 13:18
        ψ = state_T_plus(N)
        t_fwht = @elapsed M2 = magic_fwht(ψ, N)
        log(@sprintf("  │  %2d  │    %10.4f    │  %+.4f   │", N, t_fwht, M2))

        if t_fwht > 60
            break
        end
    end
    log("  └──────┴────────────────┴────────────┘")
    log()

    # Save console output to file
    output_file = joinpath(DATA_DIR, "console_output.txt")
    open(output_file, "w") do f
        for line in output_lines
            println(f, line)
        end
    end
    println("  → Console output saved to: data/console_output.txt")

    return N_completed, results
end

# ============================================================================
# Plot Generation
# ============================================================================

function generate_plots(N_vals, results)
    println("  Generating plots...")

    # Extract data
    M2_plus = [results["plus"][N].M2 for N in N_vals]
    M2_T_plus = [results["T_plus"][N].M2 for N in N_vals]
    M2_star = [results["star"][N].M2 for N in N_vals]
    M2_star_T_all = [results["star_T_all"][N].M2 for N in N_vals]
    M2_star_T_center = [results["star_T_center"][N].M2 for N in N_vals]
    times_T_plus = [results["T_plus"][N].time for N in N_vals]

    # -------------------------------------------------------------------------
    # Plot 1: M₂ vs N - Product State
    # -------------------------------------------------------------------------
    fig1 = Figure(size=(800, 500))
    ax1 = Axis(fig1[1, 1],
        xlabel = "Number of qubits N",
        ylabel = "M₂ (Stabilizer Rényi Entropy)",
        title = "Product State: |+...+⟩ vs T⊗N|+...+⟩",
        xticks = collect(N_vals)
    )

    scatter!(ax1, N_vals, M2_plus, markersize=12, color=:blue, label="|+...+⟩ (stabilizer)")
    lines!(ax1, N_vals, M2_plus, color=:blue, linewidth=2)

    scatter!(ax1, N_vals, M2_T_plus, markersize=12, color=:red, label="T⊗N|+...+⟩ (magic)")
    lines!(ax1, N_vals, M2_T_plus, color=:red, linewidth=2)

    # Linear fit
    α = sum(N_vals .* M2_T_plus) / sum(Float64.(N_vals).^2)
    lines!(ax1, N_vals, α .* N_vals, color=:red, linestyle=:dash, linewidth=1,
           label="Linear fit: M₂ ≈ $(round(α, digits=3))N")

    axislegend(ax1, position=:lt)
    save(joinpath(FIGURES_DIR, "fig_sre_product_states.png"), fig1, px_per_unit=2)
    println("    → Saved: fig_sre_product_states.png")

    # -------------------------------------------------------------------------
    # Plot 2: M₂ vs N - Star Graph States
    # -------------------------------------------------------------------------
    fig2 = Figure(size=(900, 500))
    ax2 = Axis(fig2[1, 1],
        xlabel = "Number of qubits N",
        ylabel = "M₂ (Stabilizer Rényi Entropy)",
        title = "Star Graph: T on All Nodes vs T on Center Only",
        xticks = collect(N_vals)
    )

    scatter!(ax2, N_vals, M2_star, markersize=10, color=:gray, label="|Star⟩ (stabilizer)")
    lines!(ax2, N_vals, M2_star, color=:gray, linewidth=2)

    scatter!(ax2, N_vals, M2_star_T_all, markersize=12, color=:red, label="T⊗N|Star⟩ (T on ALL)")
    lines!(ax2, N_vals, M2_star_T_all, color=:red, linewidth=2)

    scatter!(ax2, N_vals, M2_star_T_center, markersize=12, color=:purple, label="T₁|Star⟩ (T on center)")
    lines!(ax2, N_vals, M2_star_T_center, color=:purple, linewidth=2)

    axislegend(ax2, position=:lt)
    save(joinpath(FIGURES_DIR, "fig_sre_star_graph.png"), fig2, px_per_unit=2)
    println("    → Saved: fig_sre_star_graph.png")

    # -------------------------------------------------------------------------
    # Plot 3: Time vs N (log scale)
    # -------------------------------------------------------------------------
    fig3 = Figure(size=(800, 500))
    ax3 = Axis(fig3[1, 1],
        xlabel = "Number of qubits N",
        ylabel = "Computation time (s)",
        title = "SRE (M₂) Computation Time - O(8^N) Scaling",
        yscale = log10,
        xticks = collect(N_vals)
    )

    scatter!(ax3, N_vals, times_T_plus, markersize=12, color=:red, label="T⊗N|+...+⟩")
    lines!(ax3, N_vals, times_T_plus, color=:red, linewidth=2)

    # O(8^N) fit
    if times_T_plus[1] > 0
        c = times_T_plus[1] / 8^N_vals[1]
        theoretical = [c * 8^N for N in N_vals]
        lines!(ax3, N_vals, theoretical, color=:gray, linestyle=:dash, linewidth=2,
               label="O(8^N) fit")
    end

    axislegend(ax3, position=:lt)
    text!(ax3, 0.95, 0.05, text="$(Threads.nthreads()) threads",
          align=(:right, :bottom), space=:relative, fontsize=10)

    save(joinpath(FIGURES_DIR, "fig_sre_timing.png"), fig3, px_per_unit=2)
    println("    → Saved: fig_sre_timing.png")

    # Note: Timing comparison plot skipped - data already shown in console output (Benchmark 3)
    # Recomputing FWHT for N=11-18 would take several minutes.

    return fig1, fig2, fig3
end

# ============================================================================
# Save Results
# ============================================================================

function save_results(N_vals, results)
    println("  Saving detailed results...")

    filepath = joinpath(DATA_DIR, "summary_data.txt")
    open(filepath, "w") do f
        println(f, "# Stabilizer Rényi Entropy (SRE) Benchmark")
        println(f, "# Date: $(Dates.now())")
        println(f, "# Threads: $(Threads.nthreads())")
        println(f, "#")
        println(f, "# States tested:")
        println(f, "# - |+...+⟩: Product of |+⟩ states (stabilizer)")
        println(f, "# - T⊗N|+...+⟩: T gate on each qubit of |+...+⟩")
        println(f, "# - |Star⟩: Star graph (CZ from center to all leaves)")
        println(f, "# - T⊗N|Star⟩: T gate on ALL qubits of star graph")
        println(f, "# - T₁|Star⟩: T gate on CENTER qubit only")
        println(f, "#")
        println(f, "# N    M2_plus    M2_T_plus   M2_star    M2_star_T_all   M2_star_T_center   time[s]")

        for N in N_vals
            @printf(f, "%2d   %+.5f   %+.5f   %+.5f   %+.5f        %+.5f          %.3f\n",
                    N,
                    results["plus"][N].M2,
                    results["T_plus"][N].M2,
                    results["star"][N].M2,
                    results["star_T_all"][N].M2,
                    results["star_T_center"][N].M2,
                    results["T_plus"][N].time)
        end
    end
    println("    → Saved: summary_data.txt")
end

# ============================================================================
# Main
# ============================================================================

function main()
    # Run benchmark (outputs console_output.txt automatically)
    N_vals, results = run_benchmark()

    # Generate plots
    generate_plots(N_vals, results)

    # Save data
    save_results(N_vals, results)

    println()
    println("═"^80)
    println("  COMPLETE - All outputs saved to: demo_stabilizer_renyi_entropy/")
    println("═"^80)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

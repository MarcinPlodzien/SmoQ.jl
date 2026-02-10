# Date: 2026
#
#!/usr/bin/env julia
#=
################################################################################
################################################################################
##                                                                            ##
##   DEMO: PURE STATE TIME EVOLUTION                            ##
##   Matrix-Free Trotter Implementation with Observables & Entanglement       ##
##                                                                            ##
################################################################################
################################################################################

OVERVIEW
========
This script demonstrates the MATRIX-FREE BITWISE implementation for quantum
time evolution on a 1D spin chain. It showcases the entire simulation pipeline:

  1. State Preparation     : |ψ₀⟩ = |00...0⟩ using make_product_ket (bitwise)
  2. Trotter Evolution     : U(dt) = Π_gates exp(-iH_local·dt) via FastTrotterGate
  3. Exact Evolution       : U = exp(-iHt) for small N (baseline comparison)
  4. Observable Measurement: ⟨X⟩, ⟨Y⟩, ⟨Z⟩, ⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩ (O(2^N) bitwise)
  5. Entanglement Metrics  : S_vN (SVD), Negativity (SVD) - all O(2^N)

HAMILTONIAN
===========
H = Σᵢ Jxx·(σˣᵢ⊗σˣᵢ₊₁) + Σᵢ hx·σˣᵢ

This is the XX model with transverse field:
  - Jxx = 1.0 : Nearest-neighbor XX coupling
  - hx  = 1.0 : Transverse magnetic field (X direction)

The model is integrable and exactly solvable, making it ideal for validating
Trotter accuracy against exact diagonalization.

TROTTER DECOMPOSITION
=====================
For each time step dt, we apply a sequence of local unitaries:
  - (N-1) two-qubit XX gates: exp(-i·Jxx·XX·dt) for bonds (1,2), (2,3), ..., (N-1,N)
  - N single-qubit X gates:   exp(-i·hx·X·dt) for sites 1, 2, ..., N

Total gates per step = (N-1) + N = 2N-1
Each gate is applied using BITWISE MATRIX-FREE operations - no full matrices stored!

OBSERVABLES (All Matrix-Free, O(2^N))
=====================================
Local Pauli expectations:
  ⟨Zₖ⟩ = Σᵢ |ψᵢ|² × (-1)^{bit_k(i)}           -- Full loop, diagonal
  ⟨Xₖ⟩ = 2 × Re(Σᵢ:bₖ=0 ψᵢ* × ψᵢ₊ₛₜₑₚ)       -- Half loop, bit-flip pairs
  ⟨Yₖ⟩ = 2 × Im(Σᵢ:bₖ=0 ψᵢ₊ₛₜₑₚ × ψᵢ*)       -- Half loop, bit-flip pairs

Two-body correlators:
  ⟨ZᵢZⱼ⟩ = Σₛ |ψₛ|² × (-1)^{bᵢ(s)⊕bⱼ(s)}    -- Full loop, XOR parity
  ⟨XᵢXⱼ⟩ = 2 × Re(Σₛ:bᵢ=bⱼ=0 ψ₀₀* ψ₁₁ + ψ₀₁* ψ₁₀)  -- Quarter loop
  ⟨YᵢYⱼ⟩ = -2 × Re(ψ₀₀* ψ₁₁) + 2 × Re(ψ₀₁* ψ₁₀)    -- Quarter loop

ENTANGLEMENT (Matrix-Free via SVD, O(2^N))
==========================================
Entanglement Entropy (von Neumann):
  S = -Σᵢ pᵢ log(pᵢ)  where pᵢ = σᵢ² from SVD of reshaped ψ

Negativity (for pure states):
  N = ((Σσᵢ)² - 1) / 2  -- computed directly from singular values!
  No density matrix construction required (O(2^N) instead of O(4^N))

PARAMETERS
==========
  - t_max = 10.0        : Total evolution time
  - dt = 0.05           : Trotter time step (200 total steps)
  - N_time_shots = 20   : Number of measurement snapshots
  - System sizes        : N = 2, 4, 6, ..., 30 qubits
  - max_N_exact = 12    : Maximum N for exact evolution comparison

OUTPUT FILES
============
For each system size N:
  - observables_trotter_N{N}.csv  : Time series of all observables (Trotter)
  - observables_exact_N{N}.csv   : Time series of all observables (exact, N≤12)
  - entropy_SvN_N{N}.csv          : Entanglement entropy vs time
  - negativity_N{N}.csv           : Negativity vs time
  - fidelity_trotter_vs_exact_N{N}.csv : |⟨ψ_exact|ψ_trotter⟩|² (N≤12)
  - observable_deviation_N{N}.csv : RMSE between exact/trotter observables

================================================================================
=#

using LinearAlgebra
using Statistics
using Printf
using Random
using Dates

# ==============================================================================
#                       STARTUP BANNER
# ==============================================================================
n_threads = Threads.nthreads()
println("=" ^ 80)
println("  DEMO: PURE STATE TIME EVOLUTION")
println("  Matrix-Free Bitwise Trotter Implementation")
println("  Running on $n_threads threads")
println("=" ^ 80)

# Setup paths
SCRIPT_DIR = @__DIR__
WORKSPACE = dirname(SCRIPT_DIR)
UTILS_CPU = joinpath(WORKSPACE, "utils", "cpu")

# Include modules (in correct dependency order)
include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateObservables.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateCharacteristic.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumChannelUnitaryEvolutionTrotter.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumChannelUnitaryEvolutionExact.jl"))
include(joinpath(UTILS_CPU, "cpuHamiltonianBuilder.jl"))

using .CPUQuantumStatePartialTrace
using .CPUQuantumStatePreparation
using .CPUQuantumStateObservables
using .CPUQuantumStateCharacteristic
using .CPUQuantumChannelUnitaryEvolutionTrotter
using .CPUQuantumChannelUnitaryEvolutionExact
using .CPUHamiltonianBuilder
using SparseArrays
using Plots

# ==============================================================================
#                       SIMULATION PARAMETERS
# ==============================================================================

# Output directory structure (inside scripts/)
const DEMO_DIR = joinpath(SCRIPT_DIR, "demo_pure_state_evolution")
const DATA_DIR = joinpath(DEMO_DIR, "data")
const FIGURES_DIR = joinpath(DEMO_DIR, "figures")
const T_max = 5  # One full period
const dt = 0.02
const N_time_shots = 100
const TEST_SIZES = collect(1:1:24)

# Hamiltonian parameters: XXZ + transverse X field
# H = Σ(Jxx σxσx + Jyy σyσy + Jzz σzσz) + hx·Σσx
const Jxx = 1.0   # XX coupling
const Jyy = 1.0   # YY coupling
const Jzz = 0.5   # ZZ coupling (anisotropy)
const hx = 1.0    # Transverse X field

# Max N for exact evolution comparison
const max_N_exact = 0  # Set to 0 for pure Trotter timing (no exact comparison)

# Toggle entanglement calculations (S_vN, Negativity) - disable for faster benchmarks
const COMPUTE_ENTANGLEMENT = false

# ==============================================================================
#                       HELPER FUNCTIONS
# ==============================================================================

"""Create initial |0...0⟩ state using codebase's make_product_ket."""
make_zero_state(N::Int) = make_product_ket(fill(:zero, N))
"""Precompute Trotter gates for XXZ + transverse X Hamiltonian on 1D chain.
H = Σᵢ(Jxx XᵢXᵢ₊₁ + Jyy YᵢYᵢ₊₁ + Jzz ZᵢZᵢ₊₁) + hx·Σᵢσxᵢ

True first-order Trotter: exp(-iHdt) ≈ Π exp(-iJxx·XX·dt) × exp(-iJyy·YY·dt) × exp(-iJzz·ZZ·dt) × exp(-ihx·X·dt)
Gates per step: 3(N-1) two-qubit + N single-qubit = 4N-3 total
"""
function build_xxz_transverse_gates(N::Int, dt::Float64; Jxx=1.0, Jyy=1.0, Jzz=0.5, hx=0.5)
    gates = FastTrotterGate[]

    # Pauli matrices
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]

    # Two-qubit Pauli-Pauli operators
    XX = kron(σx, σx)
    YY = kron(σy, σy)
    ZZ = kron(σz, σz)

    # Apply separate Rxx, Ryy, Rzz gates for each bond (true Trotter decomposition)
    for i in 1:(N-1)
        # exp(-i · Jxx · XX · dt)
        U_xx = exp(-im * Jxx * XX * dt)
        push!(gates, FastTrotterGate([i, i+1], U_xx))

        # exp(-i · Jyy · YY · dt)
        U_yy = exp(-im * Jyy * YY * dt)
        push!(gates, FastTrotterGate([i, i+1], U_yy))

        # exp(-i · Jzz · ZZ · dt)
        U_zz = exp(-im * Jzz * ZZ * dt)
        push!(gates, FastTrotterGate([i, i+1], U_zz))
    end

    # Single-qubit X field: exp(-i · hx · X · dt)
    for i in 1:N
        U_x = exp(-im * hx * σx * dt)
        push!(gates, FastTrotterGate([i], U_x))
    end

    return gates
end

"""Measure all local observables (X, Y, Z) for sites 1..N."""
function measure_local_observables(ψ::Vector{ComplexF64}, N::Int)
    X_vals = Float64[]
    Y_vals = Float64[]
    Z_vals = Float64[]

    for k in 1:N
        push!(X_vals, expect_local(ψ, k, N, :x))
        push!(Y_vals, expect_local(ψ, k, N, :y))
        push!(Z_vals, expect_local(ψ, k, N, :z))
    end

    return X_vals, Y_vals, Z_vals
end

"""Measure nearest-neighbor correlators (XX, YY, ZZ)."""
function measure_nn_correlators(ψ::Vector{ComplexF64}, N::Int)
    XX_vals = Float64[]
    YY_vals = Float64[]
    ZZ_vals = Float64[]

    for i in 1:(N-1)
        push!(XX_vals, expect_corr(ψ, i, i+1, N, :xx))
        push!(YY_vals, expect_corr(ψ, i, i+1, N, :yy))
        push!(ZZ_vals, expect_corr(ψ, i, i+1, N, :zz))
    end

    return XX_vals, YY_vals, ZZ_vals
end

"""Compute Inverse Participation Ratio: IPR = Σᵢ |ψᵢ|⁴.
Higher IPR = more localized state. For uniform superposition IPR = 1/2^N."""
compute_ipr(ψ::Vector{ComplexF64}) = sum(abs2.(ψ).^2)

"""Compute state norm ||ψ||."""
compute_norm(ψ::Vector{ComplexF64}) = norm(ψ)

"""Build header string for observables CSV."""
function build_observables_header(N::Int)
    cols = ["time"]

    # Local X
    for i in 1:N
        push!(cols, "X_$i")
    end
    # Local Y
    for i in 1:N
        push!(cols, "Y_$i")
    end
    # Local Z
    for i in 1:N
        push!(cols, "Z_$i")
    end
    # NN correlators XX
    for i in 1:(N-1)
        push!(cols, "XX_$(i)$(i+1)")
    end
    # NN correlators YY
    for i in 1:(N-1)
        push!(cols, "YY_$(i)$(i+1)")
    end
    # NN correlators ZZ
    for i in 1:(N-1)
        push!(cols, "ZZ_$(i)$(i+1)")
    end

    return join(cols, ",")
end

"""Build sparse Hamiltonian for XXZ + transverse X on 1D chain using codebase HamiltonianBuilder.
H = Σᵢ(Jxx XᵢXᵢ₊₁ + Jyy YᵢYᵢ₊₁ + Jzz ZᵢZᵢ₊₁) + hx·Σᵢσxᵢ
"""
function build_xxz_transverse_hamiltonian(N::Int; Jxx=1.0, Jyy=1.0, Jzz=0.5, hx=0.5)
    # For 1D chain: Nx=N, Ny=1 (single rail)
    params = build_hamiltonian_parameters(N, 1;
        J_x_direction=(Jxx, Jyy, Jzz),    # XXZ coupling
        J_y_direction=(0.0, 0.0, 0.0),    # No y-direction bonds in 1D
        h_field=(hx, 0.0, 0.0))           # Transverse X field
    return construct_sparse_hamiltonian(params)
end

"""Compute fidelity |⟨ψ_exact|ψ_trotter⟩|²."""
function compute_fidelity(ψ_exact::Vector{ComplexF64}, ψ_trotter::Vector{ComplexF64})
    overlap = dot(ψ_exact, ψ_trotter)
    return abs2(overlap)
end

"""Compute observable deviation between exact and Trotter."""
function compute_observable_deviation(obs_exact::Vector{Float64}, obs_trotter::Vector{Float64})
    if length(obs_exact) != length(obs_trotter)
        return NaN
    end
    return sqrt(sum((obs_exact .- obs_trotter).^2) / length(obs_exact))
end

"""Format time as hh:mm:ss:ms."""
function format_time(seconds::Float64)
    hours = floor(Int, seconds / 3600)
    mins = floor(Int, (seconds % 3600) / 60)
    secs = floor(Int, seconds % 60)
    ms = floor(Int, (seconds * 1000) % 1000)
    return @sprintf("%02d:%02d:%02d:%03d", hours, mins, secs, ms)
end

"""
Generate 3×3 plot grid for a given system size N.
Layout:
  Row 1: Sum of X, Sum of Y, Sum of Z (local observables)
  Row 2: Sum of XX, Sum of YY, Sum of ZZ (correlators)
  Row 3: Fidelity (if N≤max_N_exact), IPR, Norm
  Row 4: S_vN (entropy), Negativity, (placeholder if no entanglement)

Suptitle shows: N qubits, T_max, dt, #gates, elapsed time
"""
function generate_plot_grid(N::Int, obs_trotter::Vector, obs_exact::Vector,
                            svn_data::Vector, neg_data::Vector, ipr_data::Vector, norm_data::Vector,
                            fidelity_data::Vector, do_exact::Bool, n_gates::Int, elapsed::Float64)

    # Format elapsed time as hh:mm:ss:ms
    elapsed_str = format_time(elapsed)

    # Extract time points from first column of obs_trotter
    times = [row[1] for row in obs_trotter]

    # Parse observables: columns are [time, X_1..X_N, Y_1..Y_N, Z_1..Z_N, XX_12..XX, YY.., ZZ..]
    # Total X = sum of X_i for i=1..N → columns 2:(N+1)
    # Total Y = sum of Y_i for i=1..N → columns (N+2):(2N+1)
    # Total Z = sum of Z_i for i=1..N → columns (2N+2):(3N+1)

    total_X_trotter = [sum(row[2:N+1]) for row in obs_trotter]
    total_Y_trotter = [sum(row[N+2:2N+1]) for row in obs_trotter]
    total_Z_trotter = [sum(row[2N+2:3N+1]) for row in obs_trotter]

    # Correlator column indices
    start_XX = 3N + 2
    end_XX = start_XX + (N-1) - 1
    start_YY = end_XX + 1
    end_YY = start_YY + (N-1) - 1
    start_ZZ = end_YY + 1
    end_ZZ = start_ZZ + (N-1) - 1

    total_XX_trotter = [sum(row[start_XX:end_XX]) for row in obs_trotter]
    total_YY_trotter = [sum(row[start_YY:end_YY]) for row in obs_trotter]
    total_ZZ_trotter = [sum(row[start_ZZ:end_ZZ]) for row in obs_trotter]

    # If exact data available, extract same observables
    if do_exact && length(obs_exact) > 0
        total_X_exact = [sum(row[2:N+1]) for row in obs_exact]
        total_Y_exact = [sum(row[N+2:2N+1]) for row in obs_exact]
        total_Z_exact = [sum(row[2N+2:3N+1]) for row in obs_exact]
        total_XX_exact = [sum(row[start_XX:end_XX]) for row in obs_exact]
        total_YY_exact = [sum(row[start_YY:end_YY]) for row in obs_exact]
        total_ZZ_exact = [sum(row[start_ZZ:end_ZZ]) for row in obs_exact]
    end

    # Extract entanglement data (may be empty if COMPUTE_ENTANGLEMENT=false)
    svn_times = length(svn_data) > 0 ? [t for (t, _) in svn_data] : times
    svn_vals = length(svn_data) > 0 ? [s for (_, s) in svn_data] : fill(NaN, length(times))
    neg_vals = length(neg_data) > 0 ? [n for (_, n) in neg_data] : fill(NaN, length(times))

    # Extract IPR and norm data
    ipr_times = [t for (t, _) in ipr_data]
    ipr_vals = [v for (_, v) in ipr_data]
    norm_vals = [v for (_, v) in norm_data]

    # Create plot - Row 1: X, Y, Z
    if do_exact && length(obs_exact) > 0
        # Plot both exact (solid) and Trotter (dashed)
        p1 = plot(times, total_X_exact, label="Exact", xlabel="t", ylabel="Σᵢ Xᵢ", lw=2, color=:blue)
        plot!(p1, times, total_X_trotter, label="Trotter", lw=2, ls=:dash, color=:red)

        p2 = plot(times, total_Y_exact, label="Exact", xlabel="t", ylabel="Σᵢ Yᵢ", lw=2, color=:blue)
        plot!(p2, times, total_Y_trotter, label="Trotter", lw=2, ls=:dash, color=:red)

        p3 = plot(times, total_Z_exact, label="Exact", xlabel="t", ylabel="Σᵢ Zᵢ", lw=2, color=:blue)
        plot!(p3, times, total_Z_trotter, label="Trotter", lw=2, ls=:dash, color=:red)

        # Row 2: XX, YY, ZZ
        p4 = plot(times, total_XX_exact, label="Exact", xlabel="t", ylabel="Σᵢ XᵢXᵢ₊₁", lw=2, color=:blue)
        plot!(p4, times, total_XX_trotter, label="Trotter", lw=2, ls=:dash, color=:red)

        p5 = plot(times, total_YY_exact, label="Exact", xlabel="t", ylabel="Σᵢ YᵢYᵢ₊₁", lw=2, color=:blue)
        plot!(p5, times, total_YY_trotter, label="Trotter", lw=2, ls=:dash, color=:red)

        p6 = plot(times, total_ZZ_exact, label="Exact", xlabel="t", ylabel="Σᵢ ZᵢZᵢ₊₁", lw=2, color=:blue)
        plot!(p6, times, total_ZZ_trotter, label="Trotter", lw=2, ls=:dash, color=:red)
    else
        # Trotter only
        p1 = plot(times, total_X_trotter, label="Trotter", xlabel="t", ylabel="Σᵢ Xᵢ", lw=2, color=:red)
        p2 = plot(times, total_Y_trotter, label="Trotter", xlabel="t", ylabel="Σᵢ Yᵢ", lw=2, color=:red)
        p3 = plot(times, total_Z_trotter, label="Trotter", xlabel="t", ylabel="Σᵢ Zᵢ", lw=2, color=:red)
        p4 = plot(times, total_XX_trotter, label="Trotter", xlabel="t", ylabel="Σᵢ XᵢXᵢ₊₁", lw=2, color=:red)
        p5 = plot(times, total_YY_trotter, label="Trotter", xlabel="t", ylabel="Σᵢ YᵢYᵢ₊₁", lw=2, color=:red)
        p6 = plot(times, total_ZZ_trotter, label="Trotter", xlabel="t", ylabel="Σᵢ ZᵢZᵢ₊₁", lw=2, color=:red)
    end

    # Row 3: Fidelity, IPR, Norm
    if do_exact && length(fidelity_data) > 0
        fid_times = [t for (t, _) in fidelity_data]
        fid_vals = [f for (_, f) in fidelity_data]
        p7 = plot(fid_times, fid_vals, label="Fidelity", xlabel="t", ylabel="|⟨ψₑ|ψₜ⟩|²",
                  lw=2, color=:green, ylims=(minimum(fid_vals)-0.01, 1.01))
    else
        p7 = plot([0, T_max], [NaN, NaN], label="N/A", xlabel="t", ylabel="|⟨ψₑ|ψₜ⟩|²",
                  title="(N > max_N_exact)")
    end

    ipr_min = 1.0 / (1 << N)  # 1/2^N
    p8 = plot(ipr_times, log2.(ipr_vals), label="log₂(IPR)", xlabel="t", ylabel="log₂(IPR)", lw=2, color=:orange)
    hline!(p8, [log2(ipr_min)], label="1/2^N", lw=1, ls=:dot, color=:black)
    p9 = plot(ipr_times, norm_vals, label="Norm", xlabel="t", ylabel="||ψ||", lw=2, color=:teal)

    # Row 4: S_vN, Negativity, placeholder
    if length(svn_data) > 0
        p10 = plot(svn_times, svn_vals, label="S_vN", xlabel="t", ylabel="S_vN", lw=2, color=:purple)
        p11 = plot(svn_times, neg_vals, label="Negativity", xlabel="t", ylabel="Negativity", lw=2, color=:magenta)
    else
        p10 = plot([0, T_max], [NaN, NaN], label="S_vN", xlabel="t", ylabel="S_vN", title="(disabled)")
        p11 = plot([0, T_max], [NaN, NaN], label="Neg", xlabel="t", ylabel="Negativity", title="(disabled)")
    end
    p12 = plot([0, T_max], [NaN, NaN], xlabel="t", ylabel="", label="", title="")  # Placeholder

    # Combine into 4×3 layout with suptitle
    suptitle = "N=$N | H=XXZ+hx | T_max=$T_max | dt=$dt | time=$elapsed_str"
    fig = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,
               layout=(4, 3), size=(1200, 1100),
               plot_title=suptitle, plot_titlefontsize=11)

    # Save to figures directory
    fig_path = joinpath(FIGURES_DIR, "N$(N)_evolution.png")
    savefig(fig, fig_path)

    return fig_path
end

# ==============================================================================
#                       MAIN SIMULATION LOOP
# ==============================================================================

println("\n  Simulation Parameters:")
println("    - t_max = $T_max, dt = $dt")
println("    - N_time_shots = $N_time_shots")
println("    - Hamiltonian: H = Jxx·XX + Jyy·YY + Jzz·ZZ + hx·X (Jxx=$Jxx, hx=$hx)")
println("    - System sizes: $TEST_SIZES")
println("    - Results directory: $DATA_DIR")
println()

# Create output directories
mkpath(DATA_DIR)
mkpath(FIGURES_DIR)

# Summary data
summary_data = []

# Time points for measurements
n_steps = floor(Int, T_max / dt)
step_interval = max(1, n_steps ÷ N_time_shots)
time_points = [i * step_interval * dt for i in 0:N_time_shots if i * step_interval * dt <= T_max]

println("  ╔══════════════════════════════════════════════════════════════════════════════╗")
println("  ║  TIME EVOLUTION BENCHMARK - Matrix-Free Bitwise Implementation              ║")
println("  ╠══════════════════════════════════════════════════════════════════════════════╣")
println("  ║  - Trotter steps:    $(n_steps) (t_max=$T_max, dt=$dt)                       ")
println("  ║  - Gates per step:   (N-1)(Rxx+Ryy+Rzz) + N Rx = (4N-3) per step           ")
println("  ║  - Measurement pts:  $N_time_shots snapshots                                 ")
println("  ║  - Exact comparison: N ≤ $max_N_exact (computing |⟨ψ_exact|ψ_trotter⟩|²)     ")
println("  ╠══════════════════════════════════════════════════════════════════════════════╣")
println("  ║  Using codebase:                                                             ║")
println("  ║    - expect_local, expect_corr (bitwise O(2^N))                              ║")
println("  ║    - entanglement_entropy_schmidt (SVD)                                      ║")
println("  ║    - apply_fast_trotter_step_cpu! (bitwise gates)                            ║")
println("  ║    - construct_sparse_hamiltonian + exp(-iHt) for exact                      ║")
println("  ╚══════════════════════════════════════════════════════════════════════════════╝")
println()

println("  ┌──────┬─────────┬──────────────────────────────────────────────────────────────┐")
println("  │  N   │ #Gates  │  Status                                                      │")
println("  ├──────┼─────────┼──────────────────────────────────────────────────────────────┤")

for N in TEST_SIZES
    t_start = time()

    # Memory check
    state_size_gb = (1 << N) * 16 / 1e9
    n_gates = 4*N - 3  # (N-1) Rxx + (N-1) Ryy + (N-1) Rzz + N Rx gates per Trotter step

    if state_size_gb > 32
        @printf("  │ %4d │  %5d  │  SKIPPED: %.1f GB required                           │\n", N, n_gates, state_size_gb)
        push!(summary_data, (N, NaN, "SKIPPED", NaN, NaN))
        continue
    end

    @printf("  │ %4d │  %5d  │  Running...                                              │\r", N, n_gates)

    # Initialize Trotter state
    ψ_trotter = make_zero_state(N)
    gates = build_xxz_transverse_gates(N, dt; Jxx=Jxx, Jyy=Jyy, Jzz=Jzz, hx=hx)

    # Initialize exact evolution if N is small enough
    do_exact = (N <= max_N_exact)
    ψ_exact = nothing
    U_exact = nothing
    if do_exact
        ψ_exact = copy(ψ_trotter)
        H = build_xxz_transverse_hamiltonian(N; Jxx=Jxx, Jyy=Jyy, Jzz=Jzz, hx=hx)
        U_exact, _ = precompute_exact_propagator_cpu(H, dt)
    end

    # Storage for results
    obs_data_trotter = []
    obs_data_exact = []
    svn_data = []
    neg_data = []
    ipr_data = []   # IPR tracking
    norm_data = []  # Norm tracking
    fidelity_data = []
    obs_deviation_data = []

    # Initial measurements
    X_vals, Y_vals, Z_vals = measure_local_observables(ψ_trotter, N)
    XX_vals, YY_vals, ZZ_vals = measure_nn_correlators(ψ_trotter, N)

    # IPR and norm (always computed - cheap)
    push!(ipr_data, (0.0, compute_ipr(ψ_trotter)))
    push!(norm_data, (0.0, compute_norm(ψ_trotter)))

    # Entanglement metrics (optional - expensive for large N)
    if COMPUTE_ENTANGLEMENT
        S_vN = entanglement_entropy_1d_chain(ψ_trotter, N)
        neg = negativity_1d_chain(ψ_trotter, N)
        push!(svn_data, (0.0, S_vN))
        push!(neg_data, (0.0, neg))
    end

    obs_trotter = vcat(X_vals, Y_vals, Z_vals, XX_vals, YY_vals, ZZ_vals)
    push!(obs_data_trotter, vcat([0.0], obs_trotter))

    if do_exact
        push!(obs_data_exact, vcat([0.0], obs_trotter))  # Same at t=0
        push!(fidelity_data, (0.0, 1.0))  # Perfect fidelity at start
        push!(obs_deviation_data, (0.0, 0.0))  # No deviation at start
    end

    # Time evolution with progress display
    current_time = 0.0
    step_count = 0
    next_shot_idx = 1

    while current_time < T_max - dt/2
        # Evolve Trotter one step
        apply_fast_trotter_step_cpu!(ψ_trotter, gates, N)

        # Evolve exact one step (if applicable)
        if do_exact
            evolve_exact_psi_cpu!(ψ_exact, U_exact)
        end

        current_time += dt
        step_count += 1

        # Check if measurement needed
        if next_shot_idx <= length(time_points) && step_count * dt >= time_points[next_shot_idx]
            # Trotter observables
            X_vals, Y_vals, Z_vals = measure_local_observables(ψ_trotter, N)
            XX_vals, YY_vals, ZZ_vals = measure_nn_correlators(ψ_trotter, N)

            obs_trotter = vcat(X_vals, Y_vals, Z_vals, XX_vals, YY_vals, ZZ_vals)
            push!(obs_data_trotter, vcat([current_time], obs_trotter))

            # IPR and norm (always computed - cheap)
            push!(ipr_data, (current_time, compute_ipr(ψ_trotter)))
            push!(norm_data, (current_time, compute_norm(ψ_trotter)))

            # Entanglement metrics (optional)
            if COMPUTE_ENTANGLEMENT
                S_vN = entanglement_entropy_1d_chain(ψ_trotter, N)
                neg = negativity_1d_chain(ψ_trotter, N)
                push!(svn_data, (current_time, S_vN))
                push!(neg_data, (current_time, neg))
            end

            # Exact comparison (if applicable)
            if do_exact
                X_ex, Y_ex, Z_ex = measure_local_observables(ψ_exact, N)
                XX_ex, YY_ex, ZZ_ex = measure_nn_correlators(ψ_exact, N)
                obs_exact = vcat(X_ex, Y_ex, Z_ex, XX_ex, YY_ex, ZZ_ex)
                push!(obs_data_exact, vcat([current_time], obs_exact))

                # Fidelity and deviation
                fid = compute_fidelity(ψ_exact, ψ_trotter)
                dev = compute_observable_deviation(obs_exact, obs_trotter)
                push!(fidelity_data, (current_time, fid))
                push!(obs_deviation_data, (current_time, dev))
            end

            next_shot_idx += 1
        end

        # Progress bar (for large N)
        if N >= 24 && step_count % 20 == 0
            progress = current_time / T_max * 100
            @printf("  │ %4d │  Progress: %5.1f%%                                                  │\r", N, progress)
        end
    end

    elapsed = time() - t_start

    # Final fidelity for summary
    final_fidelity = do_exact && length(fidelity_data) > 0 ? fidelity_data[end][2] : NaN
    final_deviation = do_exact && length(obs_deviation_data) > 0 ? obs_deviation_data[end][2] : NaN

    # Write results
    obs_file = joinpath(DATA_DIR, "observables_trotter_N$(N).csv")
    svn_file = joinpath(DATA_DIR, "entropy_SvN_N$(N).csv")
    neg_file = joinpath(DATA_DIR, "negativity_N$(N).csv")

    # Write Trotter observables
    open(obs_file, "w") do f
        println(f, build_observables_header(N))
        for row in obs_data_trotter
            println(f, join([@sprintf("%.8f", v) for v in row], ","))
        end
    end

    # Write exact observables (if applicable)
    if do_exact
        obs_exact_file = joinpath(DATA_DIR, "observables_exact_N$(N).csv")
        open(obs_exact_file, "w") do f
            println(f, build_observables_header(N))
            for row in obs_data_exact
                println(f, join([@sprintf("%.8f", v) for v in row], ","))
            end
        end

        # Write fidelity
        fid_file = joinpath(DATA_DIR, "fidelity_trotter_vs_exact_N$(N).csv")
        open(fid_file, "w") do f
            println(f, "time,fidelity")
            for (t, fid) in fidelity_data
                println(f, @sprintf("%.8f,%.12f", t, fid))
            end
        end

        # Write observable deviation
        dev_file = joinpath(DATA_DIR, "observable_deviation_N$(N).csv")
        open(dev_file, "w") do f
            println(f, "time,rmse_deviation")
            for (t, dev) in obs_deviation_data
                println(f, @sprintf("%.8f,%.12f", t, dev))
            end
        end
    end

    # Write entropy
    open(svn_file, "w") do f
        println(f, "time,S_vN")
        for (t, s) in svn_data
            println(f, @sprintf("%.8f,%.8f", t, s))
        end
    end

    # Write negativity
    open(neg_file, "w") do f
        println(f, "time,negativity")
        for (t, n) in neg_data
            if isnan(n)
                println(f, @sprintf("%.8f,NaN", t))
            else
                println(f, @sprintf("%.8f,%.8f", t, n))
            end
        end
    end

    # Generate 4×3 plot for this N (includes both exact and Trotter curves for N≤max_N_exact)
    fig_path = generate_plot_grid(N, obs_data_trotter, obs_data_exact, svn_data, neg_data, ipr_data, norm_data, fidelity_data, do_exact, n_gates, elapsed)

    push!(summary_data, (N, elapsed, "OK", final_fidelity, final_deviation, n_gates))
    time_str = format_time(elapsed)
    if do_exact
        @printf("  │ %4d │  %5d  │  DONE (%s) F=%.6f                           │\n", N, n_gates, time_str, final_fidelity)
    else
        @printf("  │ %4d │  %5d  │  DONE (%s)                                       │\n", N, n_gates, time_str)
    end

    GC.gc()
end

println("  └──────┴─────────┴──────────────────────────────────────────────────────────────┘")

# ==============================================================================
#                       SUMMARY TABLE
# ==============================================================================

println("\n")
println("█" ^ 80)
println("█  SUMMARY: Time Evolution Results                                            █")
println("█" ^ 80)

println("\n┌──────┬──────────────┬──────────────┬──────────────┐")
println("  │  N   │     Time     │   Fidelity   │    Status    │")
println("  ├──────┼──────────────┼──────────────┼──────────────┤")

for (N, elapsed, status, fidelity, deviation) in summary_data
    if status == "SKIPPED"
        @printf("  │ %4d │      ---     │      ---     │   SKIPPED    │\n", N)
    elseif isnan(fidelity)
        time_str = format_time(elapsed)
        @printf("  │ %4d │ %s │      ---     │      OK      │\n", N, time_str)
    else
        time_str = format_time(elapsed)
        @printf("  │ %4d │ %s │   %.6f   │      OK      │\n", N, time_str, fidelity)
    end
end

println("  └──────┴──────────────┴──────────────┴──────────────┘")

# Write summary
summary_file = joinpath(DATA_DIR, "summary.txt")
open(summary_file, "w") do f
    println(f, "=" ^ 80)
    println(f, "PURE STATE TIME EVOLUTION - SUMMARY")
    println(f, "=" ^ 80)
    println(f, "")
    println(f, "Using MATRIX-FREE BITWISE implementations:")
    println(f, "  - Trotter: apply_fast_trotter_step_cpu! (bitwise gates)")
    println(f, "  - Observables: expect_local, expect_corr (O(2^N) bitwise)")
    println(f, "  - Entropy: entanglement_entropy_schmidt (SVD)")
    println(f, "  - Exact (N≤$max_N_exact): construct_sparse_hamiltonian + exp(-iHt)")
    println(f, "")
    println(f, "Parameters:")
    println(f, "  t_max = $T_max")
    println(f, "  dt = $dt")
    println(f, "  N_time_shots = $N_time_shots")
    println(f, "  Hamiltonian = XX + transverse X (Jxx=$Jxx, hx=$hx)")
    println(f, "")
    println(f, "-" ^ 80)
    @printf(f, "%-8s %-15s %-15s %-15s %-10s\n", "N", "Time (s)", "Fidelity(t=T)", "Obs RMSE", "Status")
    println(f, "-" ^ 80)
    for (N, elapsed, status, fidelity, deviation) in summary_data
        if status == "SKIPPED"
            @printf(f, "%-8d %-15s %-15s %-15s %-10s\n", N, "---", "---", "---", status)
        elseif isnan(fidelity)
            @printf(f, "%-8d %-15.3f %-15s %-15s %-10s\n", N, elapsed, "---", "---", status)
        else
            @printf(f, "%-8d %-15.3f %-15.10f %-15.10f %-10s\n", N, elapsed, fidelity, deviation, status)
        end
    end
    println(f, "-" ^ 80)
end

println("\n  Results saved to: $DATA_DIR")
println("  Summary file: $summary_file")

# ==============================================================================
#                       TIMING ANALYSIS & PLOT
# ==============================================================================

# Extract timing data (skip warmup/first run artifacts)
timing_data = [(N, elapsed) for (N, elapsed, status, _, _, _) in summary_data if status == "OK"]

if length(timing_data) > 0
    # Save timing data to txt file
    timing_file = joinpath(DATA_DIR, "timing_vs_N.txt")
    open(timing_file, "w") do f
        println(f, "# Pure State Time Evolution - Timing Data")
        println(f, "# Date: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
        println(f, "# Threads: $n_threads")
        println(f, "# T_max=$T_max, dt=$dt, N_time_shots=$N_time_shots")
        println(f, "# Hamiltonian: XXZ (Jxx=$Jxx, Jyy=$Jyy, Jzz=$Jzz) + hx=$hx")
        println(f, "#")
        println(f, "# N, Time(s), log10(Time)")
        for (N, elapsed) in timing_data
            log_time = elapsed > 0 ? log10(elapsed) : -Inf
            @printf(f, "%d, %.6f, %.6f\n", N, elapsed, log_time)
        end
    end
    println("  Timing data: $timing_file")

    # Create timing plot: log10(time) vs N
    Ns = [N for (N, _) in timing_data]
    times = [t for (_, t) in timing_data]
    log_times = [t > 0 ? log10(t) : -3 for t in times]

    timing_plot = plot(Ns, log_times,
                       xlabel="N (qubits)", ylabel="log₁₀(time [s])",
                       label="Trotter evolution", lw=2, marker=:circle, ms=6,
                       title="Pure State Evolution Time vs System Size\n($n_threads threads, T=$T_max, dt=$dt)",
                       legend=:topleft)

    # Save timing plot
    timing_fig_path = joinpath(FIGURES_DIR, "timing_vs_N.png")
    savefig(timing_plot, timing_fig_path)
    println("  Timing plot: $timing_fig_path")
end

println()

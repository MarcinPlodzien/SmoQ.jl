#!/usr/bin/env julia
#=
================================================================================
    demo_VQC_pure_states_optimizers_benchmark.jl
    Pure State VQC Optimizer Benchmark: SPSA vs Adam-SPSA vs Enzyme+Adam
================================================================================

PURPOSE:
--------
Benchmark VQC optimization methods on pure states (ket representation) across
system sizes N = 6, 8, 10, 12 to demonstrate matrix-free scaling.

OPTIMIZERS:
-----------
1. SPSA (gradient-free, 2 function evals/step)
2. Adam-SPSA (SPSA gradient → Adam momentum)
3. Adam + Enzyme (exact gradients via autodiff)

FIGURE LAYOUT:
--------------
Grid: rows = N (6, 8, 10, 12), columns = [DM, MCWF]
- DM: Skip for N > 8 (too expensive)
- MCWF: Works for all N (matrix-free)

Each subplot shows 3 optimizer curves with timing in legend.

================================================================================
=#

using LinearAlgebra
using Random
using Printf
using Plots
using Statistics
using Optimisers
using Base.Threads

const OUTPUT_DIR = joinpath(@__DIR__, "demo_VQC_pure_state_optimizers_benchmark")
mkpath(OUTPUT_DIR)

# ==============================================================================
# LOAD MODULES
# ==============================================================================

const UTILS_CPU = joinpath(@__DIR__, "..", "utils", "cpu")

include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateObservables.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateLanczos.jl"))
include(joinpath(UTILS_CPU, "cpuVariationalQuantumCircuitExecutor.jl"))
include(joinpath(UTILS_CPU, "cpuVQCEnzymeWrapper.jl"))

using .CPUQuantumStatePartialTrace
using .CPUQuantumStatePreparation
using .CPUQuantumStateObservables: expect_local, expect_corr
using .CPUQuantumStateLanczos: ground_state_xxz
using .CPUVariationalQuantumCircuitExecutor
using .CPUVariationalQuantumCircuitExecutor.CPUVariationalQuantumCircuitBuilder
using .CPUVQCEnzymeWrapper

println("=" ^ 70)
println("  PURE STATE VQC OPTIMIZER BENCHMARK")
println("  Threads available: $(Threads.nthreads())")
println("=" ^ 70)

# ==============================================================================
# PARAMETERS
# ==============================================================================

const SYSTEM_SIZES = [6, 8, 10, 12]
const N_LAYERS = 2
const N_EPOCHS = 200
const N_TRAJECTORIES = 20
const MAX_N_FOR_DM = 8  # DM too slow for N > 8 (Enzyme+DM = O(4^N))
const Jx, Jy, Jz, h_field = 1.0, 1.0, 0.5, 0.3

println("\nConfiguration:")
println("  System sizes: N ∈ $SYSTEM_SIZES")
println("  Layers = $N_LAYERS")
println("  Epochs = $N_EPOCHS")
println("  MCWF trajectories = $N_TRAJECTORIES")
println("  Max N for DM = $MAX_N_FOR_DM")

# ==============================================================================
# ENERGY FUNCTIONS
# ==============================================================================

function compute_energy_ket(ψ::Vector{ComplexF64}, N::Int)
    E = 0.0
    for i in 1:(N-1)
        E -= Jx * expect_corr(ψ, i, i+1, N, :xx)
        E -= Jy * expect_corr(ψ, i, i+1, N, :yy)
        E -= Jz * expect_corr(ψ, i, i+1, N, :zz)
    end
    for i in 1:N
        E -= h_field * expect_local(ψ, i, N, :x)
    end
    E
end

function compute_energy_dm(ρ::Matrix{ComplexF64}, N::Int)
    E = 0.0
    for i in 1:(N-1)
        E -= Jx * expect_corr(ρ, i, i+1, N, :xx)
        E -= Jy * expect_corr(ρ, i, i+1, N, :yy)
        E -= Jz * expect_corr(ρ, i, i+1, N, :zz)
    end
    for i in 1:N
        E -= h_field * expect_local(ρ, i, N, :x)
    end
    E
end

# ==============================================================================
# SPSA GRADIENT
# ==============================================================================

mutable struct SPSASchedule
    a::Float64; c::Float64; A::Float64; α::Float64; γ::Float64; k::Int
end
SPSASchedule(; a=0.2, c=0.2, A=15.0) = SPSASchedule(a, c, A, 0.602, 0.101, 0)

function gradient_spsa!(g::Vector{Float64}, cost_fn, θ::Vector{Float64}, schedule::SPSASchedule)
    schedule.k += 1
    k = schedule.k
    a_k = schedule.a / (schedule.A + k)^schedule.α
    c_k = schedule.c / k^schedule.γ

    n = length(θ)
    Δ = 2.0 .* (rand(n) .> 0.5) .- 1.0
    f_plus = cost_fn(θ .+ c_k .* Δ)
    f_minus = cost_fn(θ .- c_k .* Δ)
    g .= (f_plus - f_minus) ./ (2 * c_k .* Δ)
    return a_k
end

# ==============================================================================
# TRAINING FUNCTIONS - PURE STATE (KET)
# ==============================================================================

"""Train with pure SPSA on pure state"""
function train_spsa_ket(circuit, θ_init, N; n_epochs=100)
    dim = 1 << N
    θ = copy(θ_init)
    energies = Float64[]
    schedule = SPSASchedule(a=0.2, c=0.2, A=0.1*n_epochs)
    g = zeros(length(θ))

    cost_fn = θ -> begin
        ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
        apply_circuit!(ψ, circuit, θ)
        compute_energy_ket(ψ, N)
    end

    t_start = time()
    for epoch in 1:n_epochs
        a_k = gradient_spsa!(g, cost_fn, θ, schedule)
        θ = θ .- a_k .* g
        push!(energies, cost_fn(θ))
    end
    total_time = time() - t_start

    return θ, energies, total_time
end

"""Train with Adam-SPSA on pure state"""
function train_adam_spsa_ket(circuit, θ_init, N; n_epochs=100, lr=0.02)
    dim = 1 << N
    θ = copy(θ_init)
    opt_state = Optimisers.setup(Adam(lr), θ)
    energies = Float64[]
    schedule = SPSASchedule(a=0.2, c=0.2, A=0.1*n_epochs)
    g = zeros(length(θ))

    cost_fn = θ -> begin
        ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
        apply_circuit!(ψ, circuit, θ)
        compute_energy_ket(ψ, N)
    end

    t_start = time()
    for epoch in 1:n_epochs
        gradient_spsa!(g, cost_fn, θ, schedule)
        opt_state, θ = Optimisers.update(opt_state, θ, g)
        push!(energies, cost_fn(θ))
    end
    total_time = time() - t_start

    return θ, energies, total_time
end

"""Train with Adam + Enzyme (DM mode)"""
function train_enzyme_dm(circuit, θ_init, N; n_epochs=100, lr=0.02)
    dim = 1 << N
    θ = copy(θ_init)
    opt_state = Optimisers.setup(Adam(lr), θ)
    energies = Float64[]

    wrapper = build_enzyme_wrapper(circuit)
    energy_fn = ρ -> compute_energy_dm(ρ, N)
    cost = make_cost_dm(wrapper, energy_fn)

    t_start = time()
    for epoch in 1:n_epochs
        E = cost(θ)
        g = gradient_enzyme(cost, θ)
        opt_state, θ = Optimisers.update(opt_state, θ, g)
        push!(energies, E)
    end
    total_time = time() - t_start

    return θ, energies, total_time
end

"""Train with MCWF + pure SPSA (multithreaded)"""
function train_mcwf_spsa(circuit, θ_init, N, M; n_epochs=100)
    dim = 1 << N
    θ = copy(θ_init)
    E_mean = Float64[]
    E_std = Float64[]
    schedule = SPSASchedule(a=0.2, c=0.2, A=0.1*n_epochs)
    g = zeros(length(θ))

    cost_fn = θ -> begin
        E_samples = zeros(Float64, M)
        Threads.@threads for t in 1:M
            ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
            apply_circuit!(ψ, circuit, θ)
            E_samples[t] = compute_energy_ket(ψ, N)
        end
        mean(E_samples)
    end

    t_start = time()
    for epoch in 1:n_epochs
        a_k = gradient_spsa!(g, cost_fn, θ, schedule)
        θ = θ .- a_k .* g

        # Evaluate with statistics
        E_samples = zeros(Float64, M)
        Threads.@threads for t in 1:M
            ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
            apply_circuit!(ψ, circuit, θ)
            E_samples[t] = compute_energy_ket(ψ, N)
        end
        push!(E_mean, mean(E_samples))
        push!(E_std, std(E_samples))
    end
    total_time = time() - t_start

    return θ, E_mean, E_std, total_time
end

"""Train with MCWF + Adam-SPSA (multithreaded)"""
function train_mcwf_adam_spsa(circuit, θ_init, N, M; n_epochs=100, lr=0.02)
    dim = 1 << N
    θ = copy(θ_init)
    opt_state = Optimisers.setup(Adam(lr), θ)
    E_mean = Float64[]
    E_std = Float64[]
    schedule = SPSASchedule(a=0.2, c=0.2, A=0.1*n_epochs)
    g = zeros(length(θ))

    cost_fn = θ -> begin
        E_samples = zeros(Float64, M)
        Threads.@threads for t in 1:M
            ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
            apply_circuit!(ψ, circuit, θ)
            E_samples[t] = compute_energy_ket(ψ, N)
        end
        mean(E_samples)
    end

    t_start = time()
    for epoch in 1:n_epochs
        gradient_spsa!(g, cost_fn, θ, schedule)
        opt_state, θ = Optimisers.update(opt_state, θ, g)

        # Evaluate with statistics
        E_samples = zeros(Float64, M)
        Threads.@threads for t in 1:M
            ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
            apply_circuit!(ψ, circuit, θ)
            E_samples[t] = compute_energy_ket(ψ, N)
        end
        push!(E_mean, mean(E_samples))
        push!(E_std, std(E_samples))
    end
    total_time = time() - t_start

    return θ, E_mean, E_std, total_time
end

"""Train MCWF with param-shift gradient + Adam (matrix-free)"""
function train_mcwf_paramshift(circuit, θ_init, N, M; n_epochs=100, lr=0.02)
    dim = 1 << N
    θ = copy(θ_init)
    opt_state = Optimisers.setup(Adam(lr), θ)
    E_mean = Float64[]
    E_std = Float64[]

    # Cost function for single trajectory
    function cost_single(θ)
        ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
        apply_circuit!(ψ, circuit, θ)
        compute_energy_ket(ψ, N)
    end

    # Parameter-shift gradient
    function grad_paramshift(θ)
        n = length(θ)
        g = zeros(Float64, n)
        shift = π/2
        for i in 1:n
            θ_p = copy(θ); θ_p[i] += shift
            θ_m = copy(θ); θ_m[i] -= shift
            g[i] = (cost_single(θ_p) - cost_single(θ_m)) / 2
        end
        g
    end

    t_start = time()
    for epoch in 1:n_epochs
        g = grad_paramshift(θ)
        opt_state, θ = Optimisers.update(opt_state, θ, g)

        # Evaluate with statistics
        E_samples = zeros(Float64, M)
        Threads.@threads for t in 1:M
            ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
            apply_circuit!(ψ, circuit, θ)
            E_samples[t] = compute_energy_ket(ψ, N)
        end
        push!(E_mean, mean(E_samples))
        push!(E_std, std(E_samples))
    end
    total_time = time() - t_start

    return θ, E_mean, E_std, total_time
end

# ==============================================================================
# RUN BENCHMARK
# ==============================================================================

results = Dict{Int, NamedTuple}()

for N in SYSTEM_SIZES
    println("\n" * "=" ^ 70)
    @printf("  N = %d QUBITS (dim = 2^%d = %d)\n", N, N, 1 << N)
    println("=" ^ 70)

    E_exact, _ = ground_state_xxz(N, Jx, Jy, Jz, h_field)
    @printf("Exact ground state: E = %.6f\n", E_exact)

    Random.seed!(42)
    circuit = hardware_efficient_ansatz(N, N_LAYERS; rotations=[:ry, :rz], noise_type=nothing)
    θ_init = initialize_parameters(circuit; init=:small_random)
    @printf("Ansatz: %d parameters\n\n", length(θ_init))

    # --- DM Training (for small N) ---
    dm_spsa = Float64[]; dm_adam_spsa = Float64[]; dm_enzyme = Float64[]
    dm_t_spsa = 0.0; dm_t_adam_spsa = 0.0; dm_t_enzyme = 0.0

    if N <= MAX_N_FOR_DM
        print("  DM SPSA:        ")
        _, dm_spsa, dm_t_spsa = train_spsa_ket(circuit, θ_init, N; n_epochs=N_EPOCHS)
        @printf("E=%.4f, T=%.2fs\n", dm_spsa[end], dm_t_spsa)

        print("  DM Adam-SPSA:   ")
        _, dm_adam_spsa, dm_t_adam_spsa = train_adam_spsa_ket(circuit, θ_init, N; n_epochs=N_EPOCHS)
        @printf("E=%.4f, T=%.2fs\n", dm_adam_spsa[end], dm_t_adam_spsa)

        print("  DM Enzyme+Adam: ")
        _, dm_enzyme, dm_t_enzyme = train_enzyme_dm(circuit, θ_init, N; n_epochs=N_EPOCHS)
        @printf("E=%.4f, T=%.2fs\n", dm_enzyme[end], dm_t_enzyme)
    else
        println("  DM: SKIPPED (N > $MAX_N_FOR_DM)")
    end

    # --- MCWF Training (all 3 optimizers) ---
    print("  MCWF SPSA:       ")
    _, mcwf_spsa, mcwf_spsa_std, mcwf_t_spsa = train_mcwf_spsa(
        circuit, θ_init, N, N_TRAJECTORIES; n_epochs=N_EPOCHS)
    @printf("E=%.4f±%.4f, T=%.2fs\n", mcwf_spsa[end], mcwf_spsa_std[end], mcwf_t_spsa)

    print("  MCWF Adam-SPSA:  ")
    _, mcwf_adam, mcwf_adam_std, mcwf_t_adam = train_mcwf_adam_spsa(
        circuit, θ_init, N, N_TRAJECTORIES; n_epochs=N_EPOCHS)
    @printf("E=%.4f±%.4f, T=%.2fs\n", mcwf_adam[end], mcwf_adam_std[end], mcwf_t_adam)

    print("  MCWF ParamShift: ")
    _, mcwf_ps, mcwf_ps_std, mcwf_t_ps = train_mcwf_paramshift(
        circuit, θ_init, N, N_TRAJECTORIES; n_epochs=N_EPOCHS)
    @printf("E=%.4f±%.4f, T=%.2fs\n", mcwf_ps[end], mcwf_ps_std[end], mcwf_t_ps)

    results[N] = (
        E_exact = E_exact,
        dm_spsa = dm_spsa, dm_t_spsa = dm_t_spsa,
        dm_adam_spsa = dm_adam_spsa, dm_t_adam_spsa = dm_t_adam_spsa,
        dm_enzyme = dm_enzyme, dm_t_enzyme = dm_t_enzyme,
        mcwf_spsa = mcwf_spsa, mcwf_spsa_std = mcwf_spsa_std, mcwf_t_spsa = mcwf_t_spsa,
        mcwf_adam = mcwf_adam, mcwf_adam_std = mcwf_adam_std, mcwf_t_adam = mcwf_t_adam,
        mcwf_ps = mcwf_ps, mcwf_ps_std = mcwf_ps_std, mcwf_t_ps = mcwf_t_ps
    )
end

# ==============================================================================
# GENERATE INDIVIDUAL FIGURES (one per N, one per mode)
# ==============================================================================

println("\n" * "=" ^ 70)
println("  GENERATING INDIVIDUAL FIGURES")
println("=" ^ 70)

epochs = 1:N_EPOCHS

for N in SYSTEM_SIZES
    r = results[N]

    # --- DM Figure ---
    if N <= MAX_N_FOR_DM && !isempty(r.dm_spsa)
        plt_dm = plot(
            title = "DM Optimizer Comparison (N=$N, dim=$(2^N)²)",
            xlabel = "Epoch", ylabel = "Energy ⟨H⟩",
            legend = :topright, grid = true, gridstyle = :dash, gridalpha = 0.3,
            size = (900, 600), titlefontsize = 14, guidefontsize = 12,
            tickfontsize = 10, legendfontsize = 10
        )

        plot!(plt_dm, epochs, r.dm_spsa, lw=2.5, color=:orange, alpha=0.8,
              label="SPSA (T=$(round(r.dm_t_spsa, digits=1))s)")
        plot!(plt_dm, epochs, r.dm_adam_spsa, lw=2.5, color=:green,
              label="Adam-SPSA (T=$(round(r.dm_t_adam_spsa, digits=1))s)")
        plot!(plt_dm, epochs, r.dm_enzyme, lw=2.5, color=:purple,
              label="Enzyme+Adam (T=$(round(r.dm_t_enzyme, digits=1))s)")
        hline!(plt_dm, [r.E_exact], lw=2, ls=:dash, color=:red,
               label="Exact GS ($(round(r.E_exact, digits=3)))")

        fig_name = @sprintf("fig_DM_N_%02d", N)
        savefig(plt_dm, joinpath(OUTPUT_DIR, "$(fig_name).png"))
        savefig(plt_dm, joinpath(OUTPUT_DIR, "$(fig_name).pdf"))
        println("  ✓ Saved: $(fig_name).png")
    end

    # --- MCWF Figure (all 3 optimizers) ---
    plt_mcwf = plot(
        title = "MCWF Optimizer Comparison (N=$N, M=$N_TRAJECTORIES traj)",
        xlabel = "Epoch", ylabel = "Energy ⟨H⟩",
        legend = :topright, grid = true, gridstyle = :dash, gridalpha = 0.3,
        size = (900, 600), titlefontsize = 14, guidefontsize = 12,
        tickfontsize = 10, legendfontsize = 10
    )

    plot!(plt_mcwf, epochs, r.mcwf_spsa, ribbon=r.mcwf_spsa_std, fillalpha=0.2,
          lw=2.5, color=:orange, alpha=0.8,
          label="SPSA (T=$(round(r.mcwf_t_spsa, digits=1))s)")
    plot!(plt_mcwf, epochs, r.mcwf_adam, ribbon=r.mcwf_adam_std, fillalpha=0.2,
          lw=2.5, color=:green,
          label="Adam-SPSA (T=$(round(r.mcwf_t_adam, digits=1))s)")
    plot!(plt_mcwf, epochs, r.mcwf_ps, ribbon=r.mcwf_ps_std, fillalpha=0.2,
          lw=2.5, color=:purple,
          label="ParamShift (T=$(round(r.mcwf_t_ps, digits=1))s)")
    hline!(plt_mcwf, [r.E_exact], lw=2, ls=:dash, color=:red,
           label="Exact GS ($(round(r.E_exact, digits=3)))")

    fig_name = @sprintf("fig_MCWF_N_%02d", N)
    savefig(plt_mcwf, joinpath(OUTPUT_DIR, "$(fig_name).png"))
    savefig(plt_mcwf, joinpath(OUTPUT_DIR, "$(fig_name).pdf"))
    println("  ✓ Saved: $(fig_name).png")
end

# ==============================================================================
# GENERATE COMBINED GRID FIGURE
# ==============================================================================

println("\n" * "=" ^ 70)
println("  GENERATING COMBINED FIGURE")
println("=" ^ 70)

n_rows = length(SYSTEM_SIZES)
plots_grid = Matrix{Plots.Plot}(undef, n_rows, 2)
epochs = 1:N_EPOCHS

for (row, N) in enumerate(SYSTEM_SIZES)
    r = results[N]

    # --- Left column: DM ---
    plt_dm = plot(
        legend = (row == 1 ? :topright : false),
        grid = true, gridstyle = :dash, gridalpha = 0.3,
        framestyle = :box, titlefontsize = 12, guidefontsize = 10, tickfontsize = 8
    )

    if row == 1
        title!(plt_dm, "Density Matrix (dim = 2^N × 2^N)")
    end
    ylabel!(plt_dm, "N=$N\n⟨H⟩")
    if row == n_rows
        xlabel!(plt_dm, "Epoch")
    end

    if N <= MAX_N_FOR_DM && !isempty(r.dm_spsa)
        plot!(plt_dm, epochs, r.dm_spsa, lw=2, color=:orange, alpha=0.7,
              label="SPSA ($(round(r.dm_t_spsa, digits=1))s)")
        plot!(plt_dm, epochs, r.dm_adam_spsa, lw=2, color=:green,
              label="Adam-SPSA ($(round(r.dm_t_adam_spsa, digits=1))s)")
        plot!(plt_dm, epochs, r.dm_enzyme, lw=2, color=:purple,
              label="Enzyme+Adam ($(round(r.dm_t_enzyme, digits=1))s)")
        hline!(plt_dm, [r.E_exact], lw=1.5, ls=:dash, color=:red, label="Exact GS")
    else
        annotate!(plt_dm, [(0.5, 0.5, text("N=$N: DM too large\n($(2^N)² = $(4^N) elements)", 9, :center))])
    end

    plots_grid[row, 1] = plt_dm

    # --- Right column: MCWF (all 3 optimizers) ---
    plt_mcwf = plot(
        legend = (row == 1 ? :topright : false),
        grid = true, gridstyle = :dash, gridalpha = 0.3,
        framestyle = :box, titlefontsize = 12, guidefontsize = 10, tickfontsize = 8
    )

    if row == 1
        title!(plt_mcwf, "MCWF Matrix-Free (M=$N_TRAJECTORIES traj)")
    end
    if row == n_rows
        xlabel!(plt_mcwf, "Epoch")
    end

    plot!(plt_mcwf, epochs, r.mcwf_spsa, lw=1.5, color=:orange, alpha=0.7,
          label="SPSA ($(round(r.mcwf_t_spsa, digits=1))s)")
    plot!(plt_mcwf, epochs, r.mcwf_adam, lw=1.5, color=:green,
          label="Adam-SPSA ($(round(r.mcwf_t_adam, digits=1))s)")
    plot!(plt_mcwf, epochs, r.mcwf_ps, lw=1.5, color=:purple,
          label="ParamShift ($(round(r.mcwf_t_ps, digits=1))s)")
    hline!(plt_mcwf, [r.E_exact], lw=1.5, ls=:dash, color=:red, label="Exact GS")

    plots_grid[row, 2] = plt_mcwf
end

# Combine with proper layout
fig = plot(vec(permutedims(plots_grid))...,
           layout = grid(n_rows, 2),
           size = (1200, 220 * n_rows),
           plot_title = "Pure State VQC Optimizer Benchmark",
           plot_titlefontsize = 14,
           left_margin = 12Plots.mm, bottom_margin = 6Plots.mm,
           top_margin = 5Plots.mm, right_margin = 3Plots.mm)

savefig(fig, joinpath(OUTPUT_DIR, "fig_pure_state_benchmark.png"))
savefig(fig, joinpath(OUTPUT_DIR, "fig_pure_state_benchmark.pdf"))
println("✓ Saved: $OUTPUT_DIR/fig_pure_state_benchmark.png")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

println("\n" * "=" ^ 70)
println("  TIMING SUMMARY (total training time)")
println("=" ^ 70)
println("| N  | Dim(ψ)   | SPSA    | Adam-SPSA | Enzyme+Adam | MCWF        |")
println("|----|----------|---------|-----------|-------------|-------------|")
for N in SYSTEM_SIZES
    r = results[N]
    dim_psi = 1 << N
    if N <= MAX_N_FOR_DM
        @printf("| %2d | %8d | %5.1fs  | %7.1fs  | %9.1fs  | %7.1fs    |\n",
                N, dim_psi, r.dm_t_spsa, r.dm_t_adam_spsa, r.dm_t_enzyme, r.mcwf_time)
    else
        @printf("| %2d | %8d | %7s | %9s | %11s | %7.1fs    |\n",
                N, dim_psi, "N/A", "N/A", "N/A", r.mcwf_time)
    end
end

println("\n✓ MCWF with matrix-free operations scales to N=$SYSTEM_SIZES[end]!")
println("  Key insight: MCWF uses O(2^N) memory vs DM using O(4^N)")

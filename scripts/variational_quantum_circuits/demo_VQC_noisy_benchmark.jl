#!/usr/bin/env julia
#=
================================================================================
    demo_VQC_noisy_benchmark.jl
    VQC Optimizer Benchmark with Noise Models
================================================================================

PURPOSE:
--------
Benchmark VQC optimization across (N, representation, noise_model, p) using
grid search with Iterators.product. Saves figures incrementally.

FIGURE NAMING:
--------------
fig_N{N}_{REPR}_{NOISE}_{p}.png
Example: fig_N08_MCWF_depolarizing_p0.05.png

================================================================================
=#

using LinearAlgebra
using Random
using Printf
using Plots
using Statistics
using Optimisers
using Base.Threads
using Base.Iterators: product

const OUTPUT_DIR = joinpath(@__DIR__, "demo_VQC_noisy_benchmark")
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
println("  VQC NOISY BENCHMARK (Grid Search)")
println("  Threads available: $(Threads.nthreads())")
println("=" ^ 70)

# ==============================================================================
# GRID SEARCH PARAMETERS
# ==============================================================================

const SYSTEM_SIZES = [4, 6, 8, 10, 12]
const REPRESENTATIONS = [
                        #   :DM,
                            :MCWF
                        ]
const NOISE_MODELS = [
                      # nothing,
                        #:depolarizing,
                        :amplitude_damping
                    ]
const NOISE_PROBS = [0.05]
const N_LAYERS_RANGE = [1, 2, 3]
const N_EPOCHS = 500
const N_TRAJECTORIES = 1000
const MAX_N_FOR_DM = 8
const Jx, Jy, Jz, h_field = 1.0, 1.0, 0.5, 0.3
const HARDWARE = "CPU"

# Generate all configurations: (N, repr, noise, p, n_layers)
configs = collect(product(SYSTEM_SIZES, REPRESENTATIONS, NOISE_MODELS, NOISE_PROBS, N_LAYERS_RANGE))
# Filter: skip DM for N > MAX_N_FOR_DM, skip noise=nothing with p>0
configs = filter(configs) do (N, repr, noise, p, n_layers)
    if repr == :DM && N > MAX_N_FOR_DM
        return false
    end
    if noise === nothing && p > 0
        return false
    end
    return true
end

println("\nGrid Search Configuration:")
println("  Hardware: $HARDWARE")
println("  System sizes: $SYSTEM_SIZES")
println("  Representations: $REPRESENTATIONS")
println("  Noise models: $NOISE_MODELS")
println("  Noise probs: $NOISE_PROBS")
println("  Layers: $N_LAYERS_RANGE")
println("  Total configs: $(length(configs))")

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
# TRAINING FUNCTIONS
# ==============================================================================

"""Train with SPSA on pure state (ket)"""
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

"""Train MCWF with SPSA (multithreaded, with noise variance)"""
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

"""Train MCWF with Adam+SPSA (multithreaded, with noise variance)"""
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

"""Train MCWF with Enzyme+Adam (multithreaded, with noise variance)"""
function train_mcwf_enzyme(circuit, θ_init, N, M; n_epochs=100, lr=0.02)
    dim = 1 << N
    θ = copy(θ_init)
    opt_state = Optimisers.setup(Adam(lr), θ)
    E_mean = Float64[]
    E_std = Float64[]

    wrapper = build_enzyme_wrapper(circuit)
    energy_fn = ρ -> compute_energy_dm(ρ, N)
    cost = make_cost_dm(wrapper, energy_fn)

    t_start = time()
    for epoch in 1:n_epochs
        E = cost(θ)
        g = gradient_enzyme(cost, θ)
        opt_state, θ = Optimisers.update(opt_state, θ, g)

        # Evaluate with statistics using MCWF sampling
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
# RUN BENCHMARK (Grid Search)
# ==============================================================================

println("\n" * "=" ^ 70)
println("  RUNNING BENCHMARK")
println("=" ^ 70)

for (i, (N, repr, noise, p, n_layers)) in enumerate(configs)
    noise_str = noise === nothing ? "none" : string(noise)
    p_str = @sprintf("p%.2f", p)

    println("\n[$i/$(length(configs))] N=$N, repr=$repr, L=$n_layers, noise=$noise_str, p=$p")

    # Compute exact ground state
    E_exact, _ = ground_state_xxz(N, Jx, Jy, Jz, h_field)

    # Create circuit with noise
    Random.seed!(42)
    circuit = hardware_efficient_ansatz(N, n_layers;
        rotations=[:ry, :rz], noise_type=noise, noise_p=p)
    θ_init = initialize_parameters(circuit; init=:small_random)
    n_params = length(θ_init)

    epochs = 1:N_EPOCHS

    # Title info
    ansatz_info = "HEA, L=$n_layers, N_θ=$n_params"

    if repr == :DM
        # DM training with 2 optimizers (SPSA variants only - fast & suitable for noisy)
        print("  SPSA:        ")
        _, E_spsa, t_spsa = train_spsa_ket(circuit, θ_init, N; n_epochs=N_EPOCHS)
        @printf("E=%.4f, T=%.1fs\n", E_spsa[end], t_spsa)

        print("  Adam+SPSA:   ")
        _, E_adam, t_adam = train_adam_spsa_ket(circuit, θ_init, N; n_epochs=N_EPOCHS)
        @printf("E=%.4f, T=%.1fs\n", E_adam[end], t_adam)

        # Create figure
        plt = plot(
            title = "$HARDWARE DM (N=$N, $ansatz_info, $noise_str, p=$p)",
            xlabel = "Epoch", ylabel = "Energy ⟨H⟩",
            legend = :topright, grid = true, gridstyle = :dash, gridalpha = 0.3,
            size = (900, 600), titlefontsize = 14, guidefontsize = 12,
            tickfontsize = 10, legendfontsize = 10
        )
        plot!(plt, epochs, E_spsa, lw=2.5, color=:orange, alpha=0.8,
              label="SPSA (T=$(round(t_spsa, digits=1))s)")
        plot!(plt, epochs, E_adam, lw=2.5, color=:green,
              label="Adam+SPSA (T=$(round(t_adam, digits=1))s)")
        hline!(plt, [E_exact], lw=2, ls=:dash, color=:red,
               label="Exact GS ($(round(E_exact, digits=3)))")

    else  # MCWF
        # MCWF training with 2 optimizers (SPSA variants only)
        print("  SPSA:        ")
        _, E_spsa_mean, E_spsa_std, t_spsa = train_mcwf_spsa(
            circuit, θ_init, N, N_TRAJECTORIES; n_epochs=N_EPOCHS)
        @printf("E=%.4f±%.4f, T=%.1fs\n", E_spsa_mean[end], E_spsa_std[end], t_spsa)

        print("  Adam+SPSA:   ")
        _, E_adam_mean, E_adam_std, t_adam = train_mcwf_adam_spsa(
            circuit, θ_init, N, N_TRAJECTORIES; n_epochs=N_EPOCHS)
        @printf("E=%.4f±%.4f, T=%.1fs\n", E_adam_mean[end], E_adam_std[end], t_adam)

        # Create figure
        plt = plot(
            title = "$HARDWARE MCWF (N=$N, M=$N_TRAJECTORIES, $ansatz_info, $noise_str, p=$p)",
            xlabel = "Epoch", ylabel = "Energy ⟨H⟩",
            legend = :topright, grid = true, gridstyle = :dash, gridalpha = 0.3,
            size = (900, 600), titlefontsize = 14, guidefontsize = 12,
            tickfontsize = 10, legendfontsize = 10
        )
        plot!(plt, epochs, E_spsa_mean, ribbon=E_spsa_std, fillalpha=0.15,
              lw=2.5, color=:orange, label="SPSA (T=$(round(t_spsa, digits=1))s)")
        plot!(plt, epochs, E_adam_mean, ribbon=E_adam_std, fillalpha=0.15,
              lw=2.5, color=:green, label="Adam+SPSA (T=$(round(t_adam, digits=1))s)")
        hline!(plt, [E_exact], lw=2, ls=:dash, color=:red,
               label="Exact GS ($(round(E_exact, digits=3)))")
    end

    # Save figure immediately (PNG and PDF)
    fig_name = @sprintf("fig_%s_N%02d_L%d_%s_%s_%s", HARDWARE, N, n_layers, repr, noise_str, p_str)
    savefig(plt, joinpath(OUTPUT_DIR, "$(fig_name).png"))
    savefig(plt, joinpath(OUTPUT_DIR, "$(fig_name).pdf"))
    println("  ✓ Saved: $(fig_name).png & .pdf")
end

println("\n" * "=" ^ 70)
println("  BENCHMARK COMPLETE!")
println("  All figures saved to: $OUTPUT_DIR")
println("=" ^ 70)

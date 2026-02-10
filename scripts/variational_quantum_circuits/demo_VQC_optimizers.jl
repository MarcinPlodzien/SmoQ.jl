#!/usr/bin/env julia
#=
================================================================================
    test_VQC_optimizers.jl - Comprehensive VQC Optimizer Comparison
================================================================================

PURPOSE:
--------
Test and compare all available VQC optimization methods:
1. Adam + Parameter-Shift gradients (exact, O(2n) evals per gradient)
2. Adam + Enzyme gradients (exact, O(1) backprop)
3. SPSA (gradient-free, O(2) evals per gradient)
4. Adam-SPSA (hybrid: SPSA gradient → Adam momentum)

FEATURES:
---------
- Multithreaded MCWF trajectories
- Timing benchmarks
- Convergence comparison plots

================================================================================
=#

using LinearAlgebra
using Random
using Printf
using Plots
using Statistics
using Optimisers
using Base.Threads

const OUTPUT_DIR = joinpath(@__DIR__, "demo_VQC_optimizers")
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
include(joinpath(UTILS_CPU, "cpuVariationalQuantumCircuitOptimizers.jl"))
include(joinpath(UTILS_CPU, "cpuVQCEnzymeWrapper.jl"))

using .CPUQuantumStatePartialTrace
using .CPUQuantumStatePreparation
using .CPUQuantumStateObservables: expect_local, expect_corr
using .CPUQuantumStateLanczos: ground_state_xxz
using .CPUVariationalQuantumCircuitExecutor
using .CPUVariationalQuantumCircuitExecutor.CPUVariationalQuantumCircuitBuilder
using .CPUVariationalQuantumCircuitOptimizers: SPSAOptimizer, AdamOptimizer, optimize!
using .CPUVQCEnzymeWrapper

println("=" ^ 70)
println("  VQC OPTIMIZER COMPARISON TEST")
println("  Threads available: $(Threads.nthreads())")
println("=" ^ 70)

# ==============================================================================
# PARAMETERS
# ==============================================================================

const N_QUBITS = 4
const N_LAYERS = 2
const N_EPOCHS = 500    # More epochs for SPSA to converge
const N_TRAJECTORIES = 20
const Jx, Jy, Jz, h_field = 1.0, 1.0, 0.5, 0.3

println("\nConfiguration:")
println("  N = $N_QUBITS qubits")
println("  Layers = $N_LAYERS")
println("  Epochs = $N_EPOCHS")
println("  MCWF trajectories = $N_TRAJECTORIES")

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
# GRADIENT METHODS
# ==============================================================================

"""Parameter-shift gradient (exact for Pauli rotations)"""
function gradient_param_shift(cost_fn, θ::Vector{Float64})
    n = length(θ)
    grad = zeros(Float64, n)
    shift = π/2
    @inbounds for i in 1:n
        θ_p, θ_m = copy(θ), copy(θ)
        θ_p[i] += shift
        θ_m[i] -= shift
        grad[i] = (cost_fn(θ_p) - cost_fn(θ_m)) / 2
    end
    grad
end

"""SPSA gradient estimate with proper Spall (1998) scheduling

Optimal parameters from Spall (1998):
- α = 0.602 (step decay exponent)
- γ = 0.101 (perturbation decay exponent)
- A ≈ 10% of max iterations
- a = tuned for problem scale (larger for bigger problems)
- c = perturbation size (0.1-0.2 typical)
"""
mutable struct SPSASchedule
    a::Float64; c::Float64; A::Float64; α::Float64; γ::Float64; k::Int
end

# Better SPSA defaults for VQE: smaller a, larger c for stability
# a=0.2 (conservative step), c=0.2 (decent perturbation), A=50 (stability)
SPSASchedule(; a=0.2, c=0.2, A=50.0) = SPSASchedule(a, c, A, 0.602, 0.101, 0)

function gradient_spsa(cost_fn, θ::Vector{Float64}, schedule::SPSASchedule)
    schedule.k += 1
    k = schedule.k
    a_k = schedule.a / (schedule.A + k)^schedule.α
    c_k = schedule.c / k^schedule.γ

    n = length(θ)
    Δ = 2.0 .* (rand(n) .> 0.5) .- 1.0
    f_plus = cost_fn(θ .+ c_k .* Δ)
    f_minus = cost_fn(θ .- c_k .* Δ)
    g = (f_plus - f_minus) ./ (2 * c_k .* Δ)
    return g, a_k
end

# ==============================================================================
# MCWF WITH MULTITHREADING
# ==============================================================================

"""Run M MCWF trajectories in parallel and return (mean, std)"""
function mcwf_energy_parallel(circuit, θ, N, M; base_seed=0)
    dim = 1 << N
    E_samples = zeros(Float64, M)

    Threads.@threads for t in 1:M
        ψ = zeros(ComplexF64, dim)
        ψ[1] = 1.0
        Random.seed!(base_seed + t)
        apply_circuit!(ψ, circuit, θ)
        E_samples[t] = compute_energy_ket(ψ, N)
    end

    return mean(E_samples), std(E_samples)
end

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

"""Train with Adam + Parameter-Shift"""
function train_adam_paramshift(circuit, θ_init, N; n_epochs=100, lr=0.02)
    dim = 1 << N
    θ = copy(θ_init)
    opt_state = Optimisers.setup(Adam(lr), θ)
    energies = Float64[]
    times = Float64[]

    cost_fn = θ -> begin
        ρ = zeros(ComplexF64, dim, dim); ρ[1,1] = 1.0
        apply_circuit!(ρ, circuit, θ)
        compute_energy_dm(ρ, N)
    end

    for epoch in 1:n_epochs
        t0 = time()
        E = cost_fn(θ)
        g = gradient_param_shift(cost_fn, θ)
        opt_state, θ = Optimisers.update(opt_state, θ, g)
        push!(energies, E)
        push!(times, time() - t0)
    end

    return θ, energies, mean(times)
end

"""Train with Adam + Enzyme (via wrapper)"""
function train_adam_enzyme(circuit, θ_init, N; n_epochs=100, lr=0.02)
    dim = 1 << N
    θ = copy(θ_init)
    opt_state = Optimisers.setup(Adam(lr), θ)
    energies = Float64[]
    times = Float64[]

    # Build Enzyme wrapper
    wrapper = build_enzyme_wrapper(circuit)

    energy_fn = ρ -> compute_energy_dm(ρ, N)
    cost = make_cost_dm(wrapper, energy_fn)

    for epoch in 1:n_epochs
        t0 = time()
        E = cost(θ)
        g = gradient_enzyme(cost, θ)
        opt_state, θ = Optimisers.update(opt_state, θ, g)
        push!(energies, E)
        push!(times, time() - t0)
    end

    return θ, energies, mean(times)
end

"""Train with pure SPSA (no Adam momentum)"""
function train_spsa(circuit, θ_init, N; n_epochs=100)
    dim = 1 << N
    θ = copy(θ_init)
    energies = Float64[]
    times = Float64[]
    # Tuned SPSA hyperparams:
    # a=0.2 (conservative step size), c=0.2 (perturbation), A=10% of epochs
    schedule = SPSASchedule(a=0.2, c=0.2, A=0.1*n_epochs)

    cost_fn = θ -> begin
        ρ = zeros(ComplexF64, dim, dim); ρ[1,1] = 1.0
        apply_circuit!(ρ, circuit, θ)
        compute_energy_dm(ρ, N)
    end

    for epoch in 1:n_epochs
        t0 = time()
        E = cost_fn(θ)
        g, a_k = gradient_spsa(cost_fn, θ, schedule)
        θ = θ .- a_k .* g
        push!(energies, E)
        push!(times, time() - t0)
    end

    return θ, energies, mean(times)
end

"""Train with Adam-SPSA hybrid (SPSA gradient → Adam momentum)"""
function train_adam_spsa(circuit, θ_init, N; n_epochs=100, lr=0.02)
    dim = 1 << N
    θ = copy(θ_init)
    opt_state = Optimisers.setup(Adam(lr), θ)
    energies = Float64[]
    times = Float64[]
    # Same SPSA hyperparams for fair comparison
    schedule = SPSASchedule(a=0.2, c=0.2, A=0.1*n_epochs)

    cost_fn = θ -> begin
        ρ = zeros(ComplexF64, dim, dim); ρ[1,1] = 1.0
        apply_circuit!(ρ, circuit, θ)
        compute_energy_dm(ρ, N)
    end

    for epoch in 1:n_epochs
        t0 = time()
        E = cost_fn(θ)
        # Use SPSA gradient, but feed to Adam
        g, _ = gradient_spsa(cost_fn, θ, schedule)
        opt_state, θ = Optimisers.update(opt_state, θ, g)
        push!(energies, E)
        push!(times, time() - t0)
    end

    return θ, energies, mean(times)
end

# ==============================================================================
# RUN COMPARISON
# ==============================================================================

E_exact, _ = ground_state_xxz(N_QUBITS, Jx, Jy, Jz, h_field)
@printf("\nExact ground state energy: E = %.6f\n", E_exact)

Random.seed!(42)
circuit = hardware_efficient_ansatz(N_QUBITS, N_LAYERS;
                                     rotations=[:ry, :rz], noise_type=nothing)
θ_init = initialize_parameters(circuit; init=:small_random)
@printf("Ansatz: %d parameters\n\n", length(θ_init))

println("Running optimizers...")
println("-" ^ 50)

# 1. Adam + Enzyme
print("Adam + Enzyme:      ")
θ1, E1, t1 = train_adam_enzyme(circuit, θ_init, N_QUBITS; n_epochs=N_EPOCHS)
@printf("E = %7.4f, time/epoch = %.2fms\n", E1[end], t1*1000)

# 2. Pure SPSA
print("SPSA:               ")
θ2, E2, t2 = train_spsa(circuit, θ_init, N_QUBITS; n_epochs=N_EPOCHS)
@printf("E = %7.4f, time/epoch = %.2fms\n", E2[end], t2*1000)

# 3. Adam-SPSA hybrid
print("Adam-SPSA:          ")
θ3, E3, t3 = train_adam_spsa(circuit, θ_init, N_QUBITS; n_epochs=N_EPOCHS)
@printf("E = %7.4f, time/epoch = %.2fms\n", E3[end], t3*1000)

println("-" ^ 50)

# ==============================================================================
# PLOT COMPARISON
# ==============================================================================

epochs = 1:N_EPOCHS

plt = plot(
    title = "VQC Optimizer Comparison (N=$N_QUBITS, $N_LAYERS layers)",
    xlabel = "Epoch",
    ylabel = "Energy ⟨H⟩",
    legend = :topright,
    grid = true,
    gridstyle = :dash,
    gridalpha = 0.3,
    size = (900, 600),
    titlefontsize = 14,
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 10
)

plot!(plt, epochs, E1, lw=2.5, label="Adam + Enzyme ($(round(t1*1000, digits=1))ms/ep)", color=:purple)
plot!(plt, epochs, E2, lw=2.5, label="SPSA ($(round(t2*1000, digits=1))ms/ep)", color=:orange, alpha=0.8)
plot!(plt, epochs, E3, lw=2.5, label="Adam-SPSA ($(round(t3*1000, digits=1))ms/ep)", color=:green)
hline!(plt, [E_exact], lw=2, ls=:dash, color=:red, label="Exact GS ($(round(E_exact, digits=3)))")

fig_name = @sprintf("fig_optimizer_comparison_N_%02d", N_QUBITS)
savefig(plt, joinpath(OUTPUT_DIR, "$(fig_name).png"))
savefig(plt, joinpath(OUTPUT_DIR, "$(fig_name).pdf"))

println("\nPlot saved to: $OUTPUT_DIR/$(fig_name).png")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

println("\n" * "=" ^ 70)
println("  SUMMARY")
println("=" ^ 70)
println("""
| Optimizer        | Final E   | Time/Epoch |
|------------------|-----------|------------|
| Adam+Enzyme      | $(round(E1[end], digits=4)) | $(round(t1*1000, digits=1))ms   |
| SPSA             | $(round(E2[end], digits=4)) | $(round(t2*1000, digits=2))ms   |
| Adam-SPSA        | $(round(E3[end], digits=4)) | $(round(t3*1000, digits=2))ms   |

Exact GS: $(round(E_exact, digits=4))
""")

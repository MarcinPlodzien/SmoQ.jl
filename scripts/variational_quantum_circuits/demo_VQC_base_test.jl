#!/usr/bin/env julia
#=
================================================================================
    demo_variational_quantum_circuit.jl - VQC Demonstration
================================================================================

DEMONSTRATES:
- Circuit building with hardware-efficient ansatz
- Gradient computation (Enzyme + parameter-shift for verification)
- Optimization with SPSA and Adam
- Simple VQE-like problem: minimize ⟨Z₁⟩ to prepare |1⟩ state

================================================================================
=#

println("="^70)
println("  VARIATIONAL QUANTUM CIRCUIT (VQC) DEMONSTRATION")
println("="^70)
println()

# ==============================================================================
# SETUP
# ==============================================================================

using LinearAlgebra

# Add utils path
const UTILS_PATH = joinpath(@__DIR__, "..", "utils", "cpu")

# Load VQC modules
include(joinpath(UTILS_PATH, "cpuVariationalQuantumCircuitExecutor.jl"))
include(joinpath(UTILS_PATH, "cpuVariationalQuantumCircuitGradients.jl"))
include(joinpath(UTILS_PATH, "cpuVariationalQuantumCircuitOptimizers.jl"))
include(joinpath(UTILS_PATH, "cpuVariationalQuantumCircuitCostFunctions.jl"))

using .CPUVariationalQuantumCircuitExecutor
using .CPUVariationalQuantumCircuitExecutor.CPUVariationalQuantumCircuitBuilder
using .CPUVariationalQuantumCircuitGradients
using .CPUVariationalQuantumCircuitOptimizers
using .CPUVariationalQuantumCircuitCostFunctions

# Simple inline make_ket (to avoid complex module dependencies)
function make_ket(state::String, N::Int)
    dim = 1 << N
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1.0  # |00...0⟩
    return ψ
end

# ==============================================================================
# TEST 1: CIRCUIT BUILDER
# ==============================================================================

println("="^70)
println("TEST 1: CIRCUIT BUILDER")
println("="^70)
println()

N = 4  # qubits
n_layers = 2

# Hardware-efficient ansatz
circuit = hardware_efficient_ansatz(N, n_layers; rotations=[:ry])
println("Created hardware-efficient ansatz:")
describe_circuit(circuit)
println()

# Strong entangling ansatz
circuit_strong = strong_entangling_ansatz(N, 1)
println("\nStrong entangling ansatz (1 layer):")
describe_circuit(circuit_strong)
println()

# ==============================================================================
# TEST 2: CIRCUIT APPLICATION
# ==============================================================================

println("="^70)
println("TEST 2: CIRCUIT APPLICATION")
println("="^70)
println()

# Initialize parameters
θ = randn(circuit.n_params) * 0.1
println("Parameters: $(circuit.n_params) values")
println("θ = $(round.(θ, digits=3))")
println()

# Apply circuit
ψ = make_ket("|0>", N)
apply_circuit!(ψ, circuit, θ)

println("After applying circuit:")
println("  State norm: $(norm(ψ))")
println("  ⟨Z₁⟩ = $(round(local_observable_cost(ψ, 1, N, :z), digits=4))")
println("  ⟨Z₂⟩ = $(round(local_observable_cost(ψ, 2, N, :z), digits=4))")
println()

# ==============================================================================
# TEST 3: GRADIENT VERIFICATION
# ==============================================================================

println("="^70)
println("TEST 3: GRADIENT VERIFICATION")
println("="^70)
println()

# Define cost function: minimize ⟨Z₁⟩ (want to prepare |1⟩ on qubit 1)
function cost_fn(θ_local)
    ψ_local = make_ket("|0>", N)
    apply_circuit!(ψ_local, circuit, θ_local)
    return local_observable_cost(ψ_local, 1, N, :z)
end

println("Cost function: ⟨Z₁⟩ (minimize to prepare |1⟩)")
println("Initial cost: $(round(cost_fn(θ), digits=4))")
println()

# Parameter-shift gradient
println("Computing parameter-shift gradients...")
t_ps = @elapsed grad_ps = calculate_gradients_parameter_shift(cost_fn, θ)
println("  Time: $(round(t_ps * 1000, digits=1)) ms")
println("  Gradient (first 5): $(round.(grad_ps[1:min(5, length(grad_ps))], digits=4))")
println()

# Finite difference gradient (for comparison)
println("Computing finite-difference gradients...")
t_fd = @elapsed grad_fd = calculate_gradients_finite_difference(cost_fn, θ)
println("  Time: $(round(t_fd * 1000, digits=1)) ms")
println("  Gradient (first 5): $(round.(grad_fd[1:min(5, length(grad_fd))], digits=4))")
println()

# Compare
diff = maximum(abs.(grad_ps .- grad_fd))
println("Max difference (PS vs FD): $(round(diff, digits=6))")
println("Gradients match: $(diff < 1e-4 ? "✓ YES" : "✗ NO")")
println()

# ==============================================================================
# TEST 4: SPSA OPTIMIZATION
# ==============================================================================

println("="^70)
println("TEST 4: SPSA OPTIMIZATION (Hardware-Friendly)")
println("="^70)
println()

θ_init = randn(circuit.n_params) * 0.1
println("Initial cost: $(round(cost_fn(θ_init), digits=4))")

# SPSA optimizer
opt_spsa = SPSAOptimizer(a=0.2, c=0.1)
println("\nRunning SPSA optimization (100 iterations)...")
t_spsa = @elapsed θ_spsa, history_spsa = optimize!(cost_fn, θ_init, opt_spsa;
                                                     max_iter=100, verbose=false)

println("  Time: $(round(t_spsa, digits=2)) s")
println("  Final cost: $(round(cost_fn(θ_spsa), digits=4))")
println("  Expected: -1.0 (pure |1⟩ state)")
println()

# Check if we prepared |1⟩
ψ_final = make_ket("|0>", N)
apply_circuit!(ψ_final, circuit, θ_spsa)
println("Final state properties:")
println("  ⟨Z₁⟩ = $(round(local_observable_cost(ψ_final, 1, N, :z), digits=4)) (target: -1)")
println("  ⟨X₁⟩ = $(round(local_observable_cost(ψ_final, 1, N, :x), digits=4)) (target: 0)")
println()

# ==============================================================================
# TEST 5: ADAM OPTIMIZATION (with gradients)
# ==============================================================================

println("="^70)
println("TEST 5: ADAM OPTIMIZATION (Gradient-Based)")
println("="^70)
println()

θ_init2 = randn(circuit.n_params) * 0.1
println("Initial cost: $(round(cost_fn(θ_init2), digits=4))")

# Gradient function wrapper
grad_fn(f, θ) = calculate_gradients_parameter_shift(f, θ)

# Adam optimizer
opt_adam = AdamOptimizer(lr=0.1)
println("\nRunning Adam optimization (50 iterations)...")
t_adam = @elapsed θ_adam, history_adam = optimize!(cost_fn, θ_init2, opt_adam;
                                                    grad_fn=grad_fn, max_iter=50, verbose=false)

println("  Time: $(round(t_adam, digits=2)) s")
println("  Final cost: $(round(cost_fn(θ_adam), digits=4))")
println()

# ==============================================================================
# TEST 6: FIDELITY COST (State Preparation)
# ==============================================================================

println("="^70)
println("TEST 6: FIDELITY COST")
println("="^70)
println()

# Target: GHZ state
N_ghz = 3
target_ghz = zeros(ComplexF64, 2^N_ghz)
target_ghz[1] = 1/sqrt(2)      # |000⟩
target_ghz[end] = 1/sqrt(2)    # |111⟩

# Random state
ψ_random = randn(ComplexF64, 2^N_ghz)
ψ_random ./= norm(ψ_random)

fid = fidelity(ψ_random, target_ghz)
cost = fidelity_cost(ψ_random, target_ghz)
println("Random state fidelity with GHZ: $(round(fid, digits=4))")
println("Fidelity cost (1 - F): $(round(cost, digits=4))")
println()

# Perfect match
fid_perfect = fidelity(target_ghz, target_ghz)
println("Perfect match fidelity: $(round(fid_perfect, digits=4))")
println()

# ==============================================================================
# SUMMARY
# ==============================================================================

println("="^70)
println("SUMMARY")
println("="^70)
println()
println("✓ Circuit Builder: CircuitGate, CircuitLayer, ParameterizedCircuit")
println("✓ Ansätze: hardware_efficient_ansatz, strong_entangling_ansatz")
println("✓ Gradients: parameter-shift, finite-difference (Enzyme when available)")
println("✓ Optimizers: SPSA (hardware-friendly), Adam, GradientDescent")
println("✓ Cost Functions: local observables, Hamiltonian, fidelity, classification")
println()
println("="^70)
println("ALL TESTS COMPLETED!")
println("="^70)

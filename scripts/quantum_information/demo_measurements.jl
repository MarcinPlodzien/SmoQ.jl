# Date: 2026
#
#=
================================================================================
    demo_measurements.jl - Demonstration of Quantum Measurements & Resets
================================================================================
Tests all measurement functions from cpuQuantumStateMeasurements.jl:
  1. projective_measurement! (various bases)
  2. projective_measurement_all! (fast Z-basis)
  3. sample_state (no collapse)
  4. reset_state! (force to target)
================================================================================
=#

using LinearAlgebra
using Printf
using Statistics

# ==============================================================================
# LOAD MODULES
# ==============================================================================

include("../utils/cpu/cpuQuantumChannelGates.jl")
include("../utils/cpu/cpuQuantumStateMeasurements.jl")

using .CPUQuantumChannelGates
using .CPUQuantumStateMeasurements

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

"""Create GHZ state: (|00...0⟩ + |11...1⟩)/√2"""
function make_ghz_state(N::Int)
    dim = 1 << N
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1/√2          # |00...0⟩
    ψ[dim] = 1/√2        # |11...1⟩
    return ψ
end

"""Create product state |+⟩^⊗N"""
function make_plus_product_state(N::Int)
    dim = 1 << N
    ψ = fill(ComplexF64(1/√(2^N)), dim)
    return ψ
end

"""Extract bitstring from state index"""
function index_to_bitstring(idx::Int, N::Int)
    return [(idx >> (k-1)) & 1 for k in 1:N]
end

# ==============================================================================
# TEST 1: PROJECTIVE MEASUREMENT (Z-BASIS)
# ==============================================================================

function test_z_basis_measurement()
    println("\n" * "="^70)
    println("TEST 1: PROJECTIVE MEASUREMENT (Z-BASIS)")
    println("="^70)
    
    N = 3
    println("\nCreating 3-qubit GHZ state: (|000⟩ + |111⟩)/√2")
    ψ = make_ghz_state(N)
    println("Initial state norm: ", norm(ψ))
    
    # Measure qubit 1 in Z-basis
    println("\n--- Measuring qubit 1 in Z-basis ---")
    ψ_copy = copy(ψ)
    outcomes, ψ_copy = projective_measurement!(ψ_copy, [1], :z, N)
    println("Outcome: qubit 1 = ", outcomes[1])
    println("State norm after: ", norm(ψ_copy))
    
    # Find which basis state we collapsed to
    for i in 1:(1<<N)
        if abs2(ψ_copy[i]) > 0.99
            println("Collapsed to: |", join(index_to_bitstring(i-1, N)), "⟩")
        end
    end
    
    # Statistics over many runs
    println("\n--- Statistics (1000 measurements on GHZ state) ---")
    counts = Dict(0 => 0, 1 => 0)
    for _ in 1:1000
        ψ_test = make_ghz_state(N)
        outcomes, _ = projective_measurement!(ψ_test, [1], :z, N)
        counts[outcomes[1]] += 1
    end
    println("P(qubit 1 = 0) ≈ ", counts[0]/1000, " (expected: 0.5)")
    println("P(qubit 1 = 1) ≈ ", counts[1]/1000, " (expected: 0.5)")
end

# ==============================================================================
# TEST 2: PROJECTIVE MEASUREMENT (X-BASIS)
# ==============================================================================

function test_x_basis_measurement()
    println("\n" * "="^70)
    println("TEST 2: PROJECTIVE MEASUREMENT (X-BASIS)")
    println("="^70)
    
    N = 2
    
    # Test on |+⟩ state (should always give 0 in X-basis)
    println("\nCreating |+⟩ state on qubit 1...")
    ψ = zeros(ComplexF64, 2^N)
    ψ[1] = 1/√2  # |00⟩
    ψ[2] = 1/√2  # |10⟩ (little-endian: qubit 1 = 1)
    
    println("--- Measuring qubit 1 in X-basis (should always give 0 = |+⟩) ---")
    outcomes_plus = []
    for _ in 1:100
        ψ_test = copy(ψ)
        outcomes, _ = projective_measurement!(ψ_test, [1], :x, N)
        push!(outcomes_plus, outcomes[1])
    end
    println("All outcomes = 0 (|+⟩)?: ", all(outcomes_plus .== 0))
    
    # Test on |−⟩ state (should always give 1 in X-basis)
    println("\nCreating |−⟩ state on qubit 1...")
    ψ = zeros(ComplexF64, 2^N)
    ψ[1] = 1/√2   # |00⟩
    ψ[2] = -1/√2  # |10⟩
    
    println("--- Measuring qubit 1 in X-basis (should always give 1 = |−⟩) ---")
    outcomes_minus = []
    for _ in 1:100
        ψ_test = copy(ψ)
        outcomes, _ = projective_measurement!(ψ_test, [1], :x, N)
        push!(outcomes_minus, outcomes[1])
    end
    println("All outcomes = 1 (|−⟩)?: ", all(outcomes_minus .== 1))
end

# ==============================================================================
# TEST 3: PER-QUBIT BASIS MEASUREMENT
# ==============================================================================

function test_per_qubit_basis()
    println("\n" * "="^70)
    println("TEST 3: PER-QUBIT BASIS MEASUREMENT")
    println("="^70)
    
    N = 3
    println("\nCreating |+⟩|0⟩|1⟩ product state...")
    ψ = zeros(ComplexF64, 2^N)
    # |+⟩|0⟩|1⟩ = (|0⟩+|1⟩)/√2 ⊗ |0⟩ ⊗ |1⟩
    # Little-endian: qubit 1 = LSB, qubit 3 = MSB
    # |001⟩ (q1=+) + |011⟩ has q3=1, q2=0, q1=0 or 1
    # Actually: |100⟩ and |101⟩ in index form
    ψ[5] = 1/√2  # |100⟩ = q1=0, q2=0, q3=1
    ψ[6] = 1/√2  # |101⟩ = q1=1, q2=0, q3=1
    
    println("--- Measuring [q1:X, q2:Z, q3:Z] ---")
    println("Expected: q1 → 0 (|+⟩), q2 → 0 (|0⟩), q3 → 1 (|1⟩)")
    
    all_correct = true
    for _ in 1:100
        ψ_test = copy(ψ)
        outcomes, _ = projective_measurement!(ψ_test, [1, 2, 3], [:x, :z, :z], N)
        if outcomes != [0, 0, 1]
            all_correct = false
            println("Unexpected: ", outcomes)
        end
    end
    println("All 100 measurements correct: ", all_correct)
end

# ==============================================================================
# TEST 4: FAST ALL-QUBIT MEASUREMENT
# ==============================================================================

function test_fast_measurement()
    println("\n" * "="^70)
    println("TEST 4: FAST ALL-QUBIT MEASUREMENT (projective_measurement_all!)")
    println("="^70)
    
    N = 4
    println("\nCreating 4-qubit GHZ state...")
    ψ = make_ghz_state(N)
    
    println("--- 1000 fast measurements ---")
    results = Dict{Vector{Int}, Int}()
    for _ in 1:1000
        ψ_test = copy(ψ)
        bitstring, _ = projective_measurement_all!(ψ_test, N)
        key = bitstring
        results[key] = get(results, key, 0) + 1
    end
    
    println("Outcomes (should only be |0000⟩ and |1111⟩):")
    for (bitstring, count) in sort(collect(results), by=x->x[2], rev=true)
        @printf("  |%s⟩: %d times (%.1f%%)\n", join(bitstring), count, 100*count/1000)
    end
    
    # Performance comparison
    println("\n--- Performance comparison (N=12) ---")
    N_perf = 12
    ψ_perf = randn(ComplexF64, 2^N_perf); ψ_perf ./= norm(ψ_perf)
    
    t1 = @elapsed for _ in 1:100
        ψ_test = copy(ψ_perf)
        projective_measurement!(ψ_test, :z, N_perf)
    end
    
    t2 = @elapsed for _ in 1:100
        ψ_test = copy(ψ_perf)
        projective_measurement_all!(ψ_test, N_perf)
    end
    
    @printf("Qubit-by-qubit: %.3f s\n", t1)
    @printf("All at once:    %.3f s\n", t2)
    @printf("Speedup:        %.1f×\n", t1/t2)
end

# ==============================================================================
# TEST 5: SAMPLE STATE (NO COLLAPSE)
# ==============================================================================

function test_sample_state()
    println("\n" * "="^70)
    println("TEST 5: SAMPLE STATE (NO COLLAPSE)")
    println("="^70)
    
    N = 2
    println("\nCreating Bell state (|00⟩ + |11⟩)/√2...")
    ψ = make_ghz_state(N)
    ψ_original = copy(ψ)
    
    println("--- Sampling 1000 times (state should NOT change) ---")
    counts = Dict((0,0) => 0, (0,1) => 0, (1,0) => 0, (1,1) => 0)
    for _ in 1:1000
        outcomes = sample_state(ψ, [1, 2], :z, N)
        counts[(outcomes[1], outcomes[2])] += 1
    end
    
    println("Sample distribution:")
    for ((b1, b2), count) in sort(collect(counts))
        @printf("  |%d%d⟩: %d times (%.1f%%)\n", b1, b2, count, 100*count/1000)
    end
    println("\nExpected: |00⟩ and |11⟩ each ~50%, |01⟩ and |10⟩ each ~0%")
    
    # Verify state unchanged
    println("\nState unchanged after sampling: ", ψ ≈ ψ_original)
end

# ==============================================================================
# TEST 6: RESET STATE
# ==============================================================================

function test_reset_state()
    println("\n" * "="^70)
    println("TEST 6: RESET STATE")
    println("="^70)
    
    N = 3
    println("\nCreating random 3-qubit state...")
    ψ = randn(ComplexF64, 2^N); ψ ./= norm(ψ)
    
    # Reset to |0⟩
    println("\n--- Reset all qubits to |0⟩ ---")
    ψ_test = copy(ψ)
    reset_state!(ψ_test, :zero, N)
    println("State is |000⟩: ", abs2(ψ_test[1]) ≈ 1.0)
    
    # Reset specific qubits
    println("\n--- Reset from random state: qubit 1→|0⟩, qubit 2→|1⟩ ---")
    ψ = randn(ComplexF64, 2^N); ψ ./= norm(ψ)
    reset_state!(ψ, [1, 2], [:zero, :one], N)
    
    # Check marginal probabilities
    p_q1_zero = sum(abs2(ψ[i]) for i in 1:(2^N) if ((i-1) & 1) == 0)
    p_q2_one = sum(abs2(ψ[i]) for i in 1:(2^N) if ((i-1) >> 1) & 1 == 1)
    println("P(qubit 1 = 0) = ", round(p_q1_zero, digits=3), " (expected: 1.0)")
    println("P(qubit 2 = 1) = ", round(p_q2_one, digits=3), " (expected: 1.0)")
    
    # Reset to |+⟩
    println("\n--- Reset qubit 1 to |+⟩ ---")
    ψ = zeros(ComplexF64, 2^N); ψ[1] = 1.0  # Start from |000⟩
    reset_state!(ψ, [1], [:plus], N)
    
    # Measure in X-basis (should give 0 = |+⟩)
    outcomes_x = []
    for _ in 1:100
        ψ_test = copy(ψ)
        outcomes, _ = projective_measurement!(ψ_test, [1], :x, N)
        push!(outcomes_x, outcomes[1])
    end
    println("After reset to |+⟩, X-measurements all give 0: ", all(outcomes_x .== 0))
end

# ==============================================================================
# MAIN
# ==============================================================================

function main()
    println("\n" * "="^70)
    println("QUANTUM MEASUREMENTS & RESETS DEMONSTRATION")
    println("="^70)
    println("Testing cpuQuantumStateMeasurements.jl functions")
    
    test_z_basis_measurement()
    test_x_basis_measurement()
    test_per_qubit_basis()
    test_fast_measurement()
    test_sample_state()
    test_reset_state()
    
    println("\n" * "="^70)
    println("ALL TESTS COMPLETED!")
    println("="^70)
end

# Run
main()

# Date: 2026
#
#!/usr/bin/env julia
#=

# ==============================================================================
# LOAD MODULES
# ==============================================================================

using SmoQ.CPUQuantumChannelRandomUnitaries

================================================================================
    Test suite for cpuQuantumChannelRandomUnitaries.jl
================================================================================
=#

using LinearAlgebra




println("=" ^ 70)
println("  TESTING QUANTUM CHANNEL: RANDOM UNITARIES")
println("=" ^ 70)

# Setup path
SCRIPT_DIR = @__DIR__
WORKSPACE = dirname(SCRIPT_DIR)
UTILS_CPU = joinpath(WORKSPACE, "utils", "cpu")

# Include module

using SmoQ.CPUQuantumChannelRandomUnitaries: random_unitary, apply_gate!, apply_2qubit_gate!,
    apply_brickwall_layer!, random_brickwall!

# ==============================================================================
# TEST 1: random_unitary
# ==============================================================================
println("\n[TEST 1] random_unitary()")

# Single qubit
U1 = random_unitary(1)
@assert size(U1) == (2, 2) "Single-qubit should be 2x2"
@assert norm(U1' * U1 - I) < 1e-10 "U1 should be unitary"
@assert norm(U1 * U1' - I) < 1e-10 "U1 should be unitary (right)"
println("  [OK] Single-qubit Haar random unitary")

# Two qubits
U2 = random_unitary(2)
@assert size(U2) == (4, 4) "Two-qubit should be 4x4"
@assert norm(U2' * U2 - I) < 1e-10 "U2 should be unitary"
println("  [OK] Two-qubit Haar random unitary")

# Three qubits
U3 = random_unitary(3)
@assert size(U3) == (8, 8) "Three-qubit should be 8x8"
@assert norm(U3' * U3 - I) < 1e-10 "U3 should be unitary"
println("  [OK] Three-qubit Haar random unitary")

println("  [OK] All random_unitary tests passed")

# ==============================================================================
# TEST 2: apply_gate! (single qubit)
# ==============================================================================
println("\n[TEST 2] apply_gate!()")

N = 3
dim = 2^N

# Start with |000⟩
psi = zeros(ComplexF64, dim)
psi[1] = 1.0

# Apply X gate (NOT) to qubit 1
X = ComplexF64[0 1; 1 0]
apply_gate!(psi, X, 1, N)
@assert abs(psi[2] - 1.0) < 1e-12 "X on q1: |000⟩ → |001⟩"
println("  [OK] X gate on qubit 1")

# Reset and apply Hadamard to qubit 1
psi = zeros(ComplexF64, dim)
psi[1] = 1.0
H = ComplexF64[1 1; 1 -1] / sqrt(2)
apply_gate!(psi, H, 1, N)
@assert abs(psi[1] - 1/sqrt(2)) < 1e-12 "|000⟩ component"
@assert abs(psi[2] - 1/sqrt(2)) < 1e-12 "|001⟩ component"
println("  [OK] Hadamard gate on qubit 1")

# Apply H to qubit 3 (highest)
psi = zeros(ComplexF64, dim)
psi[1] = 1.0
apply_gate!(psi, H, 3, N)
@assert abs(psi[1] - 1/sqrt(2)) < 1e-12 "|000⟩ component"
@assert abs(psi[5] - 1/sqrt(2)) < 1e-12 "|100⟩ component (index 5)"
println("  [OK] Hadamard gate on qubit 3")

# State should remain normalized
@assert abs(norm(psi) - 1.0) < 1e-12 "State should be normalized"

println("  [OK] All apply_gate! tests passed")

# ==============================================================================
# TEST 3: apply_2qubit_gate!
# ==============================================================================
println("\n[TEST 3] apply_2qubit_gate!()")

N = 4
dim = 2^N

# Test with random 2-qubit unitary (preserves norm)
psi = zeros(ComplexF64, dim)
psi[1] = 1.0
U2 = random_unitary(2)
apply_2qubit_gate!(psi, U2, 1, 2, N)
@assert abs(norm(psi) - 1.0) < 1e-10 "Random 2-qubit gate preserves norm"
println("  [OK] Random 2-qubit gate on (1,2) preserves norm")

# Test on different qubit pairs
psi = zeros(ComplexF64, dim)
psi[1] = 1.0
apply_2qubit_gate!(psi, U2, 2, 3, N)
@assert abs(norm(psi) - 1.0) < 1e-10 "Random 2-qubit gate preserves norm"
println("  [OK] Random 2-qubit gate on (2,3) preserves norm")

# Test on non-adjacent qubits
psi = zeros(ComplexF64, dim)
psi[1] = 1.0
apply_2qubit_gate!(psi, U2, 1, 4, N)
@assert abs(norm(psi) - 1.0) < 1e-10 "Random 2-qubit gate preserves norm"
println("  [OK] Random 2-qubit gate on non-adjacent (1,4) preserves norm")

println("  [OK] All apply_2qubit_gate! tests passed")

# ==============================================================================
# TEST 4: Brick-wall circuit
# ==============================================================================
println("\n[TEST 4] Brick-wall circuit")

N = 6
dim = 2^N

# Start with |000000⟩
psi = zeros(ComplexF64, dim)
psi[1] = 1.0

# Generate random gates for one layer
gates_even = [random_unitary(2) for _ in 1:N÷2]
gates_odd = [random_unitary(2) for _ in 1:(N÷2 - 1)]

# Apply one layer
apply_brickwall_layer!(psi, gates_even, gates_odd, N)
@assert abs(norm(psi) - 1.0) < 1e-10 "Brick-wall layer preserves norm"
println("  [OK] Single brick-wall layer preserves norm")

# Apply full random brick-wall
psi = zeros(ComplexF64, dim)
psi[1] = 1.0
random_brickwall!(psi, 10, N)  # 10 layers
@assert abs(norm(psi) - 1.0) < 1e-10 "Random brick-wall preserves norm"
println("  [OK] Random brick-wall (10 layers) preserves norm")

# After many layers, state should be spread out (not concentrated)
entropy_proxy = sum(abs2.(psi) .^ 2)  # Purity in computational basis
@assert entropy_proxy < 0.5 "State should be spread after 10 layers"
println("  [OK] State becomes spread after random brick-wall")

println("  [OK] All brick-wall tests passed")

# ==============================================================================
# TEST 5: Performance sanity check
# ==============================================================================
println("\n[TEST 5] Performance sanity check")

N = 10
dim = 2^N
psi = zeros(ComplexF64, dim)
psi[1] = 1.0

# Time a few layers
t0 = time()
random_brickwall!(psi, 5, N)
t1 = time()
println("  5 layers on $N qubits: $(round((t1-t0)*1000, digits=2)) ms")

@assert abs(norm(psi) - 1.0) < 1e-10 "Still normalized"
println("  [OK] Performance test passed")

# ==============================================================================
# SUMMARY
# ==============================================================================
println("\n" * "=" ^ 70)
println("  ALL RANDOM UNITARIES TESTS PASSED [OK]")
println("=" ^ 70)

println("\n* cpuRandomUnitaries.jl API Summary:")
println("   Haar Random Unitaries:")
println("     random_unitary(n)           - n-qubit Haar random unitary (2^n × 2^n)")
println("   Gate Application (bitwise, matrix-free):")
println("     apply_gate!(psi, U, k, N)   - Single-qubit gate on qubit k")
println("     apply_2qubit_gate!(psi, U, q1, q2, N) - Two-qubit gate")
println("   Brick-Wall Circuits:")
println("     apply_brickwall_layer!(psi, even, odd, N) - One layer")
println("     random_brickwall!(psi, depth, N)    - Random circuit")

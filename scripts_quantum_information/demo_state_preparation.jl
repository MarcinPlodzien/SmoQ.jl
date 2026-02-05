# Date: 2026
#
#=
Test script for new state preparation and partial trace extensions
=#

println("=" ^ 70)
println("  TESTING STATE PREPARATION EXTENSIONS")
println("=" ^ 70)

using LinearAlgebra

# Setup paths
const PROJECT_ROOT = dirname(dirname(@__FILE__))
const UTILS_CPU = joinpath(PROJECT_ROOT, "utils", "cpu")

# Load modules in correct order
include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))

using .CPUQuantumStatePartialTrace: partial_trace, partial_trace_regions, find_connected_regions
using .CPUQuantumStatePreparation: make_ket, make_rho, normalize_state!, make_product_ket, make_initial_rho,
    tensor, tensor_product, make_maximally_mixed, tensor_product_ket, get_norm, get_trace,
    make_ghz, make_w, make_bell

# ==============================================================================
# TEST 1: make_ket string API
# ==============================================================================
println("\n[TEST 1] make_ket(string, N) API")

# Test equivalence with symbol-based API
@assert make_ket("|0>", 3) ≈ make_product_ket([:zero, :zero, :zero]) "make_ket |0> failed"
@assert make_ket("|1>", 2) ≈ make_product_ket([:one, :one]) "make_ket |1> failed"
@assert make_ket("|+>", 2) ≈ make_product_ket([:plus, :plus]) "make_ket |+> failed"
@assert make_ket("|->", 2) ≈ make_product_ket([:minus, :minus]) "make_ket |-> failed"

# Test with Dirac notation variations
@assert make_ket("|0⟩", 2) ≈ make_ket("|0>", 2) "Unicode Dirac notation failed"
@assert make_ket("0", 2) ≈ make_ket("|0>", 2) "Bare state string failed"

println("  [OK] Uniform state tests passed")

# ==============================================================================
# TEST 1b: Per-qubit state strings
# ==============================================================================
println("\n[TEST 1b] make_ket per-qubit specification (e.g., \"|0+0-1+>\")")

# Test per-qubit specification
psi_perqubit = make_ket("|01>")
@assert length(psi_perqubit) == 4 "Per-qubit |01> should give 2-qubit state"
@assert psi_perqubit ≈ make_product_ket([:zero, :one]) "|01> should equal [:zero, :one]"

# Test longer per-qubit string
psi_mixed = make_ket("|0+0-1+>")
@assert length(psi_mixed) == 64 "|0+0-1+> should give 6-qubit state"
@assert psi_mixed ≈ make_product_ket([:zero, :plus, :zero, :minus, :one, :plus]) "|0+0-1+> failed"

# Test with explicit N (should validate)
psi_explicit = make_ket("|0101>", 4)
@assert length(psi_explicit) == 16 "|0101> with N=4 should work"

# Test equivalence: |0000> == make_ket("|0>", 4)  
@assert make_ket("|0000>") ≈ make_ket("|0>", 4) "|0000> should equal make_ket(|0>, 4)"
@assert make_ket("|1111>") ≈ make_ket("|1>", 4) "|1111> should equal make_ket(|1>, 4)"
@assert make_ket("|++++>") ≈ make_ket("|+>", 4) "|++++> should equal make_ket(|+>, 4)"

# Test GHZ construction with per-qubit syntax
psi_GHZ_new = make_ket("|0000>") + make_ket("|1111>")
normalize_state!(psi_GHZ_new)
@assert abs(psi_GHZ_new[1] - 1/sqrt(2)) < 1e-12 "GHZ with per-qubit syntax failed"
@assert abs(psi_GHZ_new[16] - 1/sqrt(2)) < 1e-12 "GHZ with per-qubit syntax failed"

# Test Bell state with per-qubit syntax
psi_bell = make_ket("|00>") + make_ket("|11>")
normalize_state!(psi_bell)
@assert abs(psi_bell[1] - 1/sqrt(2)) < 1e-12 "Bell state |00> amplitude wrong"
@assert abs(psi_bell[4] - 1/sqrt(2)) < 1e-12 "Bell state |11> amplitude wrong"

println("  [OK] Per-qubit state string tests passed")

# ==============================================================================
# TEST 2: normalize_state! function
# ==============================================================================
println("\n[TEST 2] normalize_state!() function")

# Test pure state normalization
psi = make_ket("|0>", 3) + make_ket("|1>", 3)
@assert abs(norm(psi) - sqrt(2)) < 1e-12 "Pre-normalization norm wrong"
normalize_state!(psi)
@assert abs(norm(psi) - 1.0) < 1e-12 "normalize_state!(psi) failed"

# Test GHZ state structure
psi_GHZ = make_ket("|0>", 4) + make_ket("|1>", 4)
normalize_state!(psi_GHZ)
@assert abs(psi_GHZ[1] - 1/sqrt(2)) < 1e-12 "|0000⟩ coefficient wrong"
@assert abs(psi_GHZ[16] - 1/sqrt(2)) < 1e-12 "|1111⟩ coefficient wrong"
@assert sum(abs2, psi_GHZ[2:15]) < 1e-24 "Non-GHZ amplitudes should be zero"

# Test density matrix normalization
rho = ComplexF64[2.0 0; 0 2.0]
normalize_state!(rho)
@assert abs(tr(rho) - 1.0) < 1e-12 "normalize_state!(rho) failed"

println("  [OK] All normalize_state! tests passed")

# ==============================================================================
# TEST 2b: tensor() function and endian consistency
# ==============================================================================
println("\n[TEST 2b] tensor() function and endian consistency")

# Test basic tensor product equivalence with existing functions
psi_a = make_ket(:zero)
psi_b = make_ket(:one)
psi_tensor = tensor(psi_a, psi_b)
psi_explicit = tensor_product_ket(psi_a, psi_b, 1, 1)
@assert psi_tensor == psi_explicit "tensor() should match tensor_product_ket()"

# ENDIAN CONSISTENCY TEST:
# In little-endian convention:
#   - |01> means qubit 1 = 0 (LSB), qubit 2 = 1
#   - tensor(|0>, |1>) should give |01> where first arg is in lower bits
#   - Index = 0*1 + 1*2 = 2, so amplitude at index 3 (1-indexed)
psi_01 = tensor(make_ket(:zero), make_ket(:one))  # |0> (x) |1> = |01>
@assert length(psi_01) == 4 "tensor of two 1-qubit states should give 4 elements"
@assert abs(psi_01[3]) > 0.99 "tensor(|0>,|1>) should have amplitude at index 3 (|01> in little-endian)"

# Verify: |10> = tensor(|1>, |0>) should have amplitude at index 2 (1-indexed)
psi_10 = tensor(make_ket(:one), make_ket(:zero))  # |1> (x) |0> = |10>
@assert abs(psi_10[2]) > 0.99 "tensor(|1>,|0>) should have amplitude at index 2 (|10> in little-endian)"

# Verify against per-qubit make_ket (which we know is correct)
@assert psi_01 == make_ket("|01>") "tensor(|0>,|1>) should equal make_ket(|01>)"
@assert psi_10 == make_ket("|10>") "tensor(|1>,|0>) should equal make_ket(|10>)"

# Multi-qubit tensor test
psi_abc = tensor(tensor(make_ket(:zero), make_ket(:one)), make_ket(:plus))
@assert length(psi_abc) == 8 "3-qubit tensor should have 8 elements"

# Density matrix tensor
rho_a = make_rho(:zero)
rho_b = make_rho(:one)
rho_tensor = tensor(rho_a, rho_b)
@assert size(rho_tensor) == (4, 4) "tensor of two 1-qubit DMs should be 4x4"
@assert abs(tr(rho_tensor) - 1.0) < 1e-12 "tensor DM should have trace 1"

println("  [OK] tensor() and endian consistency tests passed")

# ==============================================================================
# TEST 2b2: tensor_product() for lists
# ==============================================================================
println("\n[TEST 2b2] tensor_product() for lists")

# Pure states list
psi_list = [make_ket(:zero), make_ket(:one), make_ket(:plus)]
psi_result = tensor_product(psi_list)
@assert length(psi_result) == 8 "3-qubit tensor should have 8 elements"
@assert psi_result isa Vector{ComplexF64} "All pure states should return Vector"

# Verify equivalence with nested tensor
psi_manual = tensor(tensor(make_ket(:zero), make_ket(:one)), make_ket(:plus))
@assert psi_result == psi_manual "tensor_product should equal nested tensor"

# Density matrices list
rho_list = [make_rho(:zero), make_rho(:one)]
rho_result = tensor_product(rho_list)
@assert size(rho_result) == (4, 4) "2-qubit DM tensor should be 4x4"
@assert rho_result isa Matrix{ComplexF64} "All DMs should return Matrix"

# Mixed list: pure state auto-converted to |psi><psi|
mixed_list = [make_rho(:zero), make_ket(:one), make_rho(:plus)]
mixed_result = tensor_product(mixed_list)
@assert size(mixed_result) == (8, 8) "3-qubit mixed tensor should be 8x8"
@assert mixed_result isa Matrix{ComplexF64} "Mixed inputs should return Matrix"
@assert abs(tr(mixed_result) - 1.0) < 1e-12 "Mixed tensor should have trace 1"

println("  [OK] tensor_product() for lists tests passed")

# ==============================================================================
# TEST 2c: make_maximally_mixed()
# ==============================================================================
println("\n[TEST 2c] make_maximally_mixed() function")

# Basic properties
rho_mixed = make_maximally_mixed(3)  # 3-qubit maximally mixed
@assert size(rho_mixed) == (8, 8) "3-qubit maximally mixed should be 8x8"
@assert abs(tr(rho_mixed) - 1.0) < 1e-12 "Trace should be 1"

# Purity should be 1/2^N = 1/8 for N=3
purity = real(tr(rho_mixed * rho_mixed))
@assert abs(purity - 1/8) < 1e-12 "Purity should be 1/2^N"

# Check it's diagonal
@assert typeof(rho_mixed) <: Diagonal "Should be Diagonal type for efficiency"

# Check all diagonal elements are equal
diag_vals = diag(rho_mixed)
@assert all(abs.(diag_vals .- 1/8) .< 1e-12) "All diagonal elements should be 1/2^N"

# Can convert to full matrix
rho_full = Matrix(rho_mixed)
@assert size(rho_full) == (8, 8) "Conversion to Matrix should work"

println("  [OK] make_maximally_mixed() tests passed")

# ==============================================================================
# TEST 2d: get_norm() and get_trace()
# ==============================================================================
println("\n[TEST 2d] get_norm() and get_trace() functions")

# Test get_norm on pure states
psi_test = make_ket("|0>", 3) + make_ket("|1>", 3)
@assert abs(get_norm(psi_test) - sqrt(2)) < 1e-12 "get_norm should return sqrt(2) for |0>+|1>"

psi_normalized = psi_test / get_norm(psi_test)
@assert abs(get_norm(psi_normalized) - 1.0) < 1e-12 "Normalized state should have norm 1"

# Verify consistency with LinearAlgebra.norm
@assert abs(get_norm(psi_test) - norm(psi_test)) < 1e-12 "get_norm should match LinearAlgebra.norm"

# Test get_trace on density matrices
rho_pure = make_rho("|0>", 3)
@assert abs(real(get_trace(rho_pure)) - 1.0) < 1e-12 "Pure state DM should have trace 1"

# Test get_trace on maximally mixed (Diagonal)
rho_diag = make_maximally_mixed(4)
@assert abs(real(get_trace(rho_diag)) - 1.0) < 1e-12 "Maximally mixed should have trace 1"

# Verify consistency with LinearAlgebra.tr
rho_test = make_rho("|+>", 2)
@assert abs(get_trace(rho_test) - tr(rho_test)) < 1e-12 "get_trace should match LinearAlgebra.tr"

println("  [OK] get_norm() and get_trace() tests passed")

# ==============================================================================
# TEST 2e: Entangled states (GHZ, W, Bell)
# ==============================================================================
println("\n[TEST 2e] Entangled states: make_ghz, make_w, make_bell")

# GHZ in Z basis (standard)
ghz_z = make_ghz(4)
@assert length(ghz_z) == 16 "4-qubit GHZ should have 16 elements"
@assert abs(ghz_z[1] - 1/sqrt(2)) < 1e-12 "|0000> amplitude should be 1/sqrt(2)"
@assert abs(ghz_z[16] - 1/sqrt(2)) < 1e-12 "|1111> amplitude should be 1/sqrt(2)"
@assert abs(get_norm(ghz_z) - 1.0) < 1e-12 "GHZ should be normalized"

# GHZ in X basis: (|++++> + |---->) / sqrt(2)
ghz_x = make_ghz(4, "x")
@assert abs(get_norm(ghz_x) - 1.0) < 1e-12 "GHZ-X should be normalized"
# In comp basis, |++++> and |----> should spread to all 16 states
@assert sum(abs2, ghz_x) > 0.99 "GHZ-X should have total probability 1"

# Per-qubit basis GHZ
ghz_mixed = make_ghz(3, "xyz")
@assert length(ghz_mixed) == 8 "3-qubit GHZ should have 8 elements"
@assert abs(get_norm(ghz_mixed) - 1.0) < 1e-12 "GHZ-xyz should be normalized"

# W state in Z basis
w_z = make_w(3)
@assert length(w_z) == 8 "3-qubit W should have 8 elements"
@assert abs(w_z[2] - 1/sqrt(3)) < 1e-12 "|001> amplitude should be 1/sqrt(3)"
@assert abs(w_z[3] - 1/sqrt(3)) < 1e-12 "|010> amplitude should be 1/sqrt(3)"
@assert abs(w_z[5] - 1/sqrt(3)) < 1e-12 "|100> amplitude should be 1/sqrt(3)"
@assert abs(get_norm(w_z) - 1.0) < 1e-12 "W should be normalized"

# W state in X basis
w_x = make_w(3, "x")
@assert abs(get_norm(w_x) - 1.0) < 1e-12 "W-X should be normalized"

# Bell states in Z basis
phi_plus = make_bell(:phi_plus)
@assert length(phi_plus) == 4 "Bell state should have 4 elements"
@assert abs(phi_plus[1] - 1/sqrt(2)) < 1e-12 "|00> amplitude"
@assert abs(phi_plus[4] - 1/sqrt(2)) < 1e-12 "|11> amplitude"
@assert abs(phi_plus[2]) < 1e-12 "|01> should be 0"
@assert abs(phi_plus[3]) < 1e-12 "|10> should be 0"

psi_minus = make_bell(:psi_minus)
# Little-endian: |01> = index 3 (0*1+1*2=2, 1-indexed=3), |10> = index 2 (1*1+0*2=1, 1-indexed=2)
@assert abs(psi_minus[3] - 1/sqrt(2)) < 1e-12 "psi_minus |01> amplitude (index 3)"
@assert abs(psi_minus[2] + 1/sqrt(2)) < 1e-12 "psi_minus |10> amplitude (index 2, negative)"

# Bell state in X basis
phi_plus_x = make_bell(:phi_plus, "x")
@assert abs(get_norm(phi_plus_x) - 1.0) < 1e-12 "Bell-X should be normalized"

# Per-qubit basis Bell state
phi_plus_xy = make_bell(:phi_plus, "xy")
@assert abs(get_norm(phi_plus_xy) - 1.0) < 1e-12 "Bell-xy should be normalized"

println("  [OK] Entangled state tests passed")

# ==============================================================================
# TEST 3: find_connected_regions helper
# ==============================================================================
println("\n[TEST 3] find_connected_regions helper")

regions = find_connected_regions([1,2,3,5,6,8,10])
@assert regions == [[1,2,3], [5,6], [8], [10]] "Connected regions detection failed"

regions2 = find_connected_regions([1,2,3])
@assert regions2 == [[1,2,3]] "Single region detection failed"

regions3 = find_connected_regions([1,3,5,7])
@assert regions3 == [[1], [3], [5], [7]] "All-isolated detection failed"

println("  [OK] All find_connected_regions tests passed")

# ==============================================================================
# TEST 4: partial_trace_regions (pure state)
# ==============================================================================
println("\n[TEST 4] partial_trace_regions (pure state)")

# Simple test: 4-qubit product state, trace out sites 2 and 4
psi_prod = make_ket("|0>", 4)  # |0000⟩
result = partial_trace_regions(psi_prod, [2, 4], 4)

# Expect two regions: [1], [3] - each should be |0⟩⟨0|
@assert length(result) == 2 "Should have 2 regions"
@assert size(result[1]) == (2, 2) "Region 1 should be 2x2"
@assert size(result[2]) == (2, 2) "Region 2 should be 2x2"
@assert abs(tr(result[1]) - 1.0) < 1e-12 "Region 1 trace should be 1"
@assert abs(tr(result[2]) - 1.0) < 1e-12 "Region 2 trace should be 1"
@assert abs(result[1][1,1] - 1.0) < 1e-12 "Region 1 should be |0⟩⟨0|"
@assert abs(result[2][1,1] - 1.0) < 1e-12 "Region 2 should be |0⟩⟨0|"

println("  [OK] partial_trace_regions (pure state) tests passed")

# ==============================================================================
# TEST 5: partial_trace_regions (density matrix)
# ==============================================================================
println("\n[TEST 5] partial_trace_regions (density matrix)")

# 6-qubit system, trace out sites 3 and 5
rho_6 = make_initial_rho(6)  # |000000⟩⟨000000|
result_dm = partial_trace_regions(rho_6, [3, 5], 6)

# Kept sites: [1,2,4,6] → regions: [1,2], [4], [6]
@assert length(result_dm) == 3 "Should have 3 regions"
@assert size(result_dm[1]) == (4, 4) "Region [1,2] should be 4x4"
@assert size(result_dm[2]) == (2, 2) "Region [4] should be 2x2"
@assert size(result_dm[3]) == (2, 2) "Region [6] should be 2x2"

for (i, ρ) in enumerate(result_dm)
    @assert abs(tr(ρ) - 1.0) < 1e-12 "Region $i trace should be 1"
end

println("  [OK] partial_trace_regions (density matrix) tests passed")

# ==============================================================================
# TEST 6: GHZ state entanglement check
# ==============================================================================
println("\n[TEST 6] GHZ state entanglement verification")

# 4-qubit GHZ state
psi_GHZ = make_ket("|0>", 4) + make_ket("|1>", 4)
normalize_state!(psi_GHZ)

# Trace out sites 3,4 to get 2-qubit subsystem on sites 1,2
rho_12 = partial_trace(psi_GHZ, [3,4], 4)

# For GHZ, tracing out half should give mixed state
purity = real(tr(rho_12 * rho_12))
@assert purity < 1.0 - 1e-6 "GHZ reduced state should be mixed"
@assert abs(purity - 0.5) < 1e-12 "GHZ purity should be 0.5"

println("  [OK] GHZ entanglement verification passed")
println("    Purity of 2-qubit reduced state: $purity (expected: 0.5)")

# ==============================================================================
# SUMMARY
# ==============================================================================
println("\n" * "=" ^ 70)
println("  ALL TESTS PASSED [OK]")
println("=" ^ 70)

println("\n* New API Summary:")
println("   State Construction:")
println("     make_ket(\"|0>\", N)           - N-qubit uniform product state")
println("     make_ket(\"|0+0-1+>\")         - Per-qubit state specification")
println("     make_rho(...), make_maximally_mixed(N)")
println("   Normalization:")
println("     normalize_state!(psi/rho)          - In-place normalization")
println("     get_norm(psi), get_trace(rho) - SIMD-optimized metrics")
println("   Tensor Products:")  
println("     tensor(a, b)                 - Two-state tensor")
println("     tensor_product([a, b, c])    - Multi-state tensor (mixed psi/rho)")
println("   Partial Trace:")
println("     partial_trace_regions(rho, trace_sites, N) - Disconnected regions")


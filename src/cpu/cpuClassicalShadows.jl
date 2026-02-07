# Date: 2026
#
#=
================================================================================
    cpuClassicalShadows.jl - Classical Shadows Quantum State Tomography
================================================================================

REFERENCE
---------
H.-Y. Huang, R. Kueng, J. Preskill
"Predicting many properties of a quantum system from very few measurements"
Nature Physics 16, 1050-1057 (2020)
https://www.nature.com/articles/s41567-020-0932-7

OVERVIEW
--------
Classical shadows enable efficient quantum state tomography using randomized
measurements. This module implements three measurement groups:

1. PAULI SHADOWS (Random X/Y/Z measurements)
2. LOCAL CLIFFORD SHADOWS (Random single-qubit Cliffords)
3. GLOBAL CLIFFORD SHADOWS (Random N-qubit entangling Cliffords)

================================================================================
                       GROUP REFERENCE GUIDE
================================================================================

╔═══════════════════════════════════════════════════════════════════════════════╗
║                        1. THE PAULI GROUP P_N                                 ║
║                 "The Skeleton" - Axes we measure                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Single Qubit (P₁): {I, X, Y, Z} (4 operators, 16 with phases ±1, ±i)
  I = [1 0; 0 1]    X = [0 1; 1 0]    Y = [0 -i; i 0]    Z = [1 0; 0 -1]

N-Qubits (P_N): All tensor products of {I,X,Y,Z}^⊗N
  Example: X ⊗ Z ⊗ I ⊗ Y  (N=4)
  Size: 4^N (ignoring global phases)

Role in Shadows: These are the OBSERVABLES we estimate.

╔═══════════════════════════════════════════════════════════════════════════════╗
║                   2. LOCAL CLIFFORD GROUP Cl₁^⊗N                              ║
║                "The Magnifying Glass" - Independent rotations                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Definition: Tensor product of N single-qubit Clifford groups (NO entanglement!)
  Cl₁⊗N = Cl₁ ⊗ Cl₁ ⊗ ... ⊗ Cl₁  (N copies)

Single-Qubit Clifford Group (Cl₁):
  Size: 24 elements (matrices that map Pauli axes to Pauli axes)
  Generators: H = (1/√2)[1 1; 1 -1], S = [1 0; 0 i]

The 24 Elements (Bloch Cube Rotations):
  ┌───────────────────┬───────┬─────────────────────────────────────────┐
  │ Class             │ Count │ Description                             │
  ├───────────────────┼───────┼─────────────────────────────────────────┤
  │ Identity          │   1   │ Does nothing: I                         │
  │ Pauli (180°)      │   3   │ π around X,Y,Z: X,Y,Z                   │
  │ Axis Exchange     │   6   │ ±π/2 around X,Y,Z: S,S†,SX,S†X,SY,S†Y   │
  │ Face Exchange     │   6   │ π/2 around axes like X+Y: H,HSH,etc     │
  │ Corner Swaps      │   8   │ 2π/3 around cube diagonals: HSH,etc     │
  ├───────────────────┼───────┼─────────────────────────────────────────┤
  │ TOTAL             │  24   │                                         │
  └───────────────────┴───────┴─────────────────────────────────────────┘

Size of Cl₁^⊗N: 24^N
  N=2: 576,  N=3: 13,824,  N=4: 331,776,  N=5: 7,962,624

Role in Shadows: Random local rotations = random Pauli basis (X,Y,Z) per qubit.

╔═══════════════════════════════════════════════════════════════════════════════╗
║                     3. GLOBAL CLIFFORD GROUP Cl_N                             ║
║               "The Wide-Angle Lens" - Entangling scramblers                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Definition: Unitaries on N qubits that normalize P_N (maps Paulis to Paulis).
  INCLUDES interactions between qubits (entanglement!)

Generators (3 types needed):
  1. Hadamard H_j (on any qubit j)
  2. Phase S_j (on any qubit j)
  3. CNOT(j,k) (between ANY pair of qubits) ← The differentiator!

Size: |Cl_N| = 2^(N²+2N) × ∏_{k=1}^N (4^k - 1)
  N=1:   24
  N=2:   11,520
  N=3:   92,897,280
  N=4:   12,128,668,876,800
  (Represented as Stabilizer Tableaus, size O(N²), NOT as 2^N × 2^N matrices!)

Role in Shadows: Measures GLOBAL properties (fidelity, entanglement).

╔═══════════════════════════════════════════════════════════════════════════════╗
║                          COMPARISON TABLE                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌────────────────────┬────────────────┬─────────────────┬─────────────────────┐
│ Feature            │ Pauli Group    │ Local Clifford  │ Global Clifford     │
├────────────────────┼────────────────┼─────────────────┼─────────────────────┤
│ Fundamental Unit   │ Pauli Operator │ Single-qubit U  │ N-qubit circuit     │
│ Generators         │ {X,Y,Z}        │ {H,S} per qubit │ {H,S,CNOT}          │
│ Entanglement?      │ N/A            │ NO (separable)  │ YES (multi-qubit)   │
│ Size (N qubits)    │ 4^N            │ 24^N            │ O(2^(N²)) HUGE      │
│ Simulation Cost    │ Trivial        │ O(N)            │ O(N²) Tableau       │
│ Best For           │ Observables    │ Local Hamiltons │ Fidelity, purity    │
└────────────────────┴────────────────┴─────────────────┴─────────────────────┘

================================================================================

                           PAULI SHADOWS
================================================================================

GROUP ELEMENTS (3 per qubit):
  Basis 1 (X measurement): Apply H before measuring in Z
  Basis 2 (Y measurement): Apply HS† before measuring in Z
  Basis 3 (Z measurement): Apply I (identity) - measure directly

Protocol for each snapshot m = 1, ..., M:
  1. Sample random Pauli basis U_m = ⊗_{j=1}^N u_j^(m) where u_j ∈ {H, HS†, I}
  2. Apply rotation: |ψ'⟩ = U_m |ψ⟩
  3. Measure in computational basis → bitstring b = (b_1, ..., b_N)
  4. Compute single-shot shadow: ρ_m = ⊗_{j=1}^N [3 u_j† |b_j⟩⟨b_j| u_j - I]

Reconstructed state: ρ* = (1/M) Σ_m ρ_m

Sample complexity for k-local observable: O(3^k / ε²)

================================================================================
                        LOCAL CLIFFORD SHADOWS
================================================================================

GROUP ELEMENTS (24 per qubit):
The single-qubit Clifford group is generated by H and S gates:
  H = (1/√2)[1  1; 1 -1]    (Hadamard)
  S = [1 0; 0 i]            (Phase gate)

The 24 elements can be enumerated as products of H and S:
  I, H, S, HS, SH, SS, HSS, SSH, SHS, HSH, HSSH, SHSH, SSHS,
  HSHS, SHSS, SSHH, HSHSH, SHSSH, SSHSH, HSSHSH, SHSSHS,
  SSHHSH, HSHSHS, SHSHSHS

These 24 rotations uniformly cover the Bloch sphere (vs only 3 for Pauli).

Protocol for each snapshot m = 1, ..., M:
  1. Sample random local Clifford C_m = ⊗_{j=1}^N C_j from the 24-element group
  2. Apply rotation: |ψ'⟩ = C_m |ψ⟩
  3. Measure in computational basis → bitstring b
  4. Compute single-shot shadow: ρ_m = ⊗_{j=1}^N [3 C_j† |b_j⟩⟨b_j| C_j - I]

Inverse channel factor: 3 per qubit (same as Pauli)
Sample complexity for k-local observable: O(3^k / ε²)

================================================================================
                        GLOBAL CLIFFORD SHADOWS
================================================================================

GROUP ELEMENTS:
The N-qubit Clifford group Cl(2^N) is generated by:
  - H_j (Hadamard on qubit j)
  - S_j (Phase on qubit j)
  - CNOT_{j,k} (CNOT from control j to target k)

Group size: |Cl(2^N)| = 2^(N² + 2N) ∏_{k=1}^N (4^k - 1)
  N=1:  |Cl(2)| = 24
  N=2:  |Cl(4)| = 11,520
  N=3:  |Cl(8)| = 92,897,280
  N=4:  |Cl(16)| = 12,128,668,876,800

RANDOM CLIFFORD CIRCUIT STRUCTURE:
We sample random Cliffords using layered circuits:
  1. Random single-qubit Clifford layer (H or S on each qubit)
  2. Random CNOT layer (each qubit is control for one random CNOT)
  3. Repeat for multiple layers

Protocol for each snapshot m = 1, ..., M:
  1. Sample random N-qubit Clifford C_m from Cl(2^N)
  2. Apply rotation: |ψ'⟩ = C_m |ψ⟩
  3. Measure in computational basis → bitstring b
  4. Compute single-shot shadow: ρ_m = (2^N + 1) C_m† |b⟩⟨b| C_m - I

INVERSE CHANNEL FACTOR LOOKUP TABLE:
  N=1:  2^1 + 1 = 3        (same as local Clifford)
  N=2:  2^2 + 1 = 5
  N=3:  2^3 + 1 = 9
  N=4:  2^4 + 1 = 17
  N=5:  2^5 + 1 = 33
  N=6:  2^6 + 1 = 65
  N=7:  2^7 + 1 = 129
  N=8:  2^8 + 1 = 257

Comparison: Local Clifford factor = 3^N
  N=2: Global=5,   Local=9     (1.8x smaller)
  N=3: Global=9,   Local=27    (3x smaller)
  N=4: Global=17,  Local=81    (4.8x smaller)
  N=5: Global=33,  Local=243   (7.4x smaller)

Key advantage: Captures GLOBAL properties (entanglement, fidelity) efficiently.
Key disadvantage: Circuit depth grows with N, harder to implement on hardware.

================================================================================
                           SAMPLE COMPLEXITY COMPARISON
================================================================================

For estimating an observable O with support on k qubits:

  PAULI / LOCAL CLIFFORD:  M ~ 3^k / ε²
  GLOBAL CLIFFORD:         M ~ 2^(2k) / ε²  (worse for local observables)

For Hilbert-Schmidt distance ||ρ* - ρ||_HS:

  PAULI / LOCAL CLIFFORD:  M ~ 3^N / ε²  (exponential in N)
  GLOBAL CLIFFORD:         M ~ 2^N / ε²  (exponentially better!)

Rule of thumb:
  - Use PAULI for k-local observables (fast, simple)
  - Use LOCAL CLIFFORD for k-local with better statistics (24x basis choices)
  - Use GLOBAL CLIFFORD for fidelity, purity, full ρ reconstruction

================================================================================
                    TABLEAU/STABILIZER SIMULATION (Efficient!)
================================================================================

For large N, we cannot store 2^N × 2^N matrices. Instead, we use the
HEISENBERG PICTURE: push the Observable through the circuit, not the state!

We calculate: tr(C†|b⟩⟨b|C · O) = tr(|b⟩⟨b| · COC†)

Instead of evolving |ψ⟩ → C|ψ⟩, we evolve O → COC†.

GATE TRANSFORMATION LOOKUP TABLE:
---------------------------------
These are the rules for how gates transform Pauli operators via conjugation.

| Gate applied      | X →      | Y →      | Z →      |
| ----------------- | -------- | -------- | -------- |
| Hadamard (H)      | Z        | -Y       | X        |
| Phase (S)         | Y        | -X       | Z        |
| CNOT(ctrl→tgt)    | X_c⊗X_t  | Y_c⊗X_t  | Z_c⊗I_t  |  (on control)
|                   | I_c⊗X_t  | Z_c⊗Y_t  | Z_c⊗Z_t  |  (on target)

STEP-BY-STEP EXAMPLE (2 qubits):
--------------------------------
Observable O = Z₁Z₂, Circuit C = H₂ · CNOT(1→2) · H₁, Measurement b = "01"

Step 0: Start with Observable String = "ZZ"

Step 1: Apply H₁ (Hadamard on qubit 1)
  - Rule: Z → X
  - String: "ZZ" → "XZ"

Step 2: Apply CNOT(1→2) (Control=1, Target=2)
  - Qubit 1 has X (control): X_c → X_c ⊗ X_t  (X spreads forward)
  - Qubit 2 has Z (target):  Z_t → Z_c ⊗ Z_t  (Z spreads backward)
  - Combined: "XZ" → "YY" (with phase tracking)

Step 3: Apply H₂ (Hadamard on qubit 2)
  - Rule: Y → -Y
  - String: "YY" → "-YY" (track minus sign)

Final: Transformed Observable O' = -Y₁Y₂

THE Z-CHECK:
------------
Compare O' = "YY" with measurement b = "01":
  - Does O' contain X or Y? YES → Estimate = 0
  - (Measuring in Z-basis gives no info about Y operators)

IF THE OBSERVABLE SURVIVED (only Z's):
--------------------------------------
If O' = "ZZ" and b = "01":
  - Qubit 1: Z with bit 0 → eigenvalue (-1)^0 = +1
  - Qubit 2: Z with bit 1 → eigenvalue (-1)^1 = -1
  - Product: (+1)×(-1) = -1
  - Estimate: (2^N + 1) × (-1) = -5 (for N=2)

WHY THIS IS FAST:
-----------------
  - Matrix method: O(4^N) memory, O(4^N × depth) time (impossible for N>20)
  - Tableau method: O(N) memory, O(N × depth) time (works for N=100+!)

================================================================================
=#



module CPUClassicalShadows

using LinearAlgebra
using Random
using Statistics
using Random: shuffle
using Base.Threads

# Note: Functions from other modules will be called with qualified names
# where necessary to avoid circular import issues

# Configuration
export ShadowConfig

# Pauli shadow snapshot and collection
export PauliSnapshot, Snapshot, sample_pauli_bases  # Snapshot is alias for PauliSnapshot
export classical_shadow_snapshot, collect_shadows

# Local Clifford shadow snapshot and collection
export CliffordSnapshot, SINGLE_QUBIT_CLIFFORDS
export clifford_shadow_snapshot, collect_clifford_shadows
export sample_clifford_index, sample_local_clifford_indices
export get_clifford_gate, build_local_clifford_unitary

# Global Clifford shadow snapshot and collection
export GlobalCliffordCircuit, GlobalCliffordSnapshot
export sample_random_global_clifford, build_global_clifford_unitary
export global_clifford_shadow_snapshot, collect_global_clifford_shadows

# Density matrix reconstruction
export reconstruct_density_matrix_shadows_kron, reconstruct_density_matrix_shadows_bitwise
export reconstruct_density_matrix_clifford, reconstruct_density_matrix_global_clifford
export single_qubit_shadow, single_qubit_clifford_shadow

# Observable estimation (the efficient part!)
export get_expectation_value_from_shadows
export estimate_local_observables, shadow_mean_correlators
export estimate_two_point_correlators
export estimate_three_body_correlators, shadow_mean_three_body_correlators



# Purity and fidelity
export estimate_purity_shadows
export estimate_fidelity_shadows

# ==============================================================================
# CONFIGURATION
# ==============================================================================

"""
    ShadowConfig

Configuration for classical shadows tomography.

# Fields
- `n_qubits::Int` - Number of qubits N
- `n_shots::Int` - Number of measurement shots (snapshots) M
- `measurement_group::Symbol` - Measurement ensemble (:pauli_group, :clifford_group)

# Measurement Groups
- `:pauli_group` (default) - Random single-qubit Pauli measurements (X, Y, Z)
  Optimal for local/few-body observables. Variance ~ 3^k / M for k-local P.

- `:clifford_group` - Random Clifford circuits (not yet implemented)
  Better for global observables but more expensive to implement.

# Example
```julia
config = ShadowConfig(
    n_qubits = 4,
    n_shots = 1000,
    measurement_group = :pauli  # or :local_clifford
)
snapshots = collect_shadows(ψ, config)
```
"""
struct ShadowConfig
    n_qubits::Int
    n_shots::Int
    measurement_group::Symbol

    function ShadowConfig(; n_qubits::Int, n_shots::Int, measurement_group::Symbol=:pauli)
        @assert n_qubits > 0 "n_qubits must be positive"
        @assert n_shots > 0 "n_shots must be positive"

        # Normalize aliases for backward compatibility
        valid_groups = [:pauli, :local_clifford, :global_clifford]
        aliases = Dict(
            :pauli_group => :pauli,
            :clifford_group => :local_clifford,
            :clifford => :local_clifford
        )
        if haskey(aliases, measurement_group)
            measurement_group = aliases[measurement_group]
        end
        @assert measurement_group ∈ valid_groups "measurement_group must be :pauli, :local_clifford, or :global_clifford"

        new(n_qubits, n_shots, measurement_group)
    end
end



# ==============================================================================
# SNAPSHOT DATA STRUCTURE
# ==============================================================================

"""
    PauliSnapshot

Structure holding one classical shadow measurement result.

# Fields
- `bases::Vector{Int}` - Measurement basis for each qubit (1=X, 2=Y, 3=Z)
- `outcomes::Vector{Int}` - Measurement outcomes (0 or 1 for each qubit)

# Classical Shadow Reconstruction
From each snapshot, the classical shadow is:
    ρ_m = ⊗_{j=1}^N [3 u_j† |s_j⟩⟨s_j| u_j - I]

where u_j is the rotation to basis j and s_j ∈ {0,1} is the outcome.

# Memory Efficiency
Each snapshot stores only 2N integers, not exponential in N!
M snapshots require O(M × N) storage.
"""
struct PauliSnapshot
    bases::Vector{Int}
    outcomes::Vector{Int}
end

# Backward compatibility alias
const Snapshot = PauliSnapshot


# ==============================================================================
# SINGLE-QUBIT CLIFFORD GROUP (24 ELEMENTS) - LOCAL CLIFFORD SHADOWS
# ==============================================================================
#
# The single-qubit Clifford group C₁ has 24 elements - the symmetry group of
# the Bloch sphere octahedron. It's generated by H (Hadamard) and S (phase gate).
#
# WARNING: Local Clifford shadows (tensor product C₁⊗C₁⊗...⊗C₁) do NOT form
# a 2-design on the full Hilbert space. For LOCAL observables (k=1,2 body),
# Pauli shadows are actually BETTER because:
# 1. Pauli shadows are tailored to Pauli basis measurements
# 2. Local Cliffords don't provide additional benefit for local observables
#
# GLOBAL Clifford shadows (full N-qubit Clifford group) would be optimal for
# global observables, but require O(2^N) gates - TODO for future implementation.

#
# Shadow inverse channel for local Clifford: (2+1)|b⟩⟨b| - I = 3|b⟩⟨b| - I
# Same as Pauli shadows! The benefit comes from more uniform observable coverage.

"""
    SINGLE_QUBIT_CLIFFORDS

All 24 single-qubit Clifford gates, generated by H and S.
These form a 2-design on the Bloch sphere and satisfy the twirl identity:
    (1/24) Σ_C Σ_b P(b|C,ρ) (3 C†|b⟩⟨b|C - I) = ρ
"""
const SINGLE_QUBIT_CLIFFORDS = let
    H = ComplexF64[1 1; 1 -1] / sqrt(2)
    S = ComplexF64[1 0; 0 im]

    # Generate all 24 distinct Cliffords by composing H and S
    function normalize_global_phase(M)
        idx = findfirst(x -> abs(x) > 0.1, M)
        if idx !== nothing
            phase = M[idx] / abs(M[idx])
            return M / phase
        end
        return M
    end

    function is_duplicate(new_C, existing_list)
        for existing in existing_list
            if isapprox(new_C, existing, atol=1e-8)
                return true
            end
        end
        return false
    end

    cliffs = Matrix{ComplexF64}[]
    push!(cliffs, Matrix{ComplexF64}(I, 2, 2))
    to_explore = [Matrix{ComplexF64}(I, 2, 2)]

    while true
        new_cliffs = Matrix{ComplexF64}[]
        for C in to_explore
            for G in [H, S]
                new_C = G * C
                new_C = normalize_global_phase(new_C)
                new_C = round.(new_C, digits=10)

                if !is_duplicate(new_C, cliffs)
                    push!(new_cliffs, new_C)
                    push!(cliffs, new_C)
                end
            end
        end
        if isempty(new_cliffs)
            break
        end
        to_explore = new_cliffs
    end

    @assert length(cliffs) == 24 "Expected 24 Cliffords, got $(length(cliffs))"
    cliffs
end



"""
    sample_clifford_index() -> Int

Sample uniformly from 1:24 (single-qubit Clifford group).
"""
@inline function sample_clifford_index()
    return rand(1:24)
end

"""
    sample_local_clifford_indices(N::Int) -> Vector{Int}

Sample N independent single-qubit Clifford indices for local Clifford shadow.
"""
function sample_local_clifford_indices(N::Int)
    return rand(1:24, N)
end

"""
    get_clifford_gate(idx::Int) -> Matrix{ComplexF64}

Get single-qubit Clifford gate by index (1-24).
"""
@inline function get_clifford_gate(idx::Int)
    return SINGLE_QUBIT_CLIFFORDS[idx]
end

"""
    build_local_clifford_unitary(indices::Vector{Int}, N::Int) -> Matrix{ComplexF64}

Build the full N-qubit local Clifford unitary C = C_N ⊗ ... ⊗ C_2 ⊗ C_1.
Note: Uses reversed kron order so qubit 1 is in lowest bit position,
matching the bit extraction convention in snapshot collection.
"""
function build_local_clifford_unitary(indices::Vector{Int}, N::Int)
    # Build kron(qN, kron(qN-1, ... kron(q2, q1))) so qubit 1 is in lowest bit
    C = get_clifford_gate(indices[N])
    for j in (N-1):-1:1
        C = kron(C, get_clifford_gate(indices[j]))
    end
    return C
end


# ==============================================================================
# CLIFFORD SHADOW SNAPSHOT
# ==============================================================================

"""
    CliffordSnapshot

Structure holding one local Clifford shadow measurement result.

# Fields
- `clifford_indices::Vector{Int}` - Clifford gate index (1-24) for each qubit
- `outcomes::Vector{Int}` - Measurement outcome per qubit (0 or 1)
"""
struct CliffordSnapshot
    clifford_indices::Vector{Int}
    outcomes::Vector{Int}
end

"""
    clifford_shadow_snapshot(ψ::Vector{ComplexF64}, N::Int) -> CliffordSnapshot

Perform one local Clifford shadow measurement on PURE STATE.

Protocol:
1. Sample random single-qubit Clifford C_j for each qubit j
2. Apply C = C_1 ⊗ ... ⊗ C_N to |ψ⟩
3. Measure in computational basis

The inverse channel is:  ρ_m = (2^N + 1) C† |b⟩⟨b| C - I
For local Cliffords, this simplifies to: ρ_m = ⊗_j (3 C_j† |b_j⟩⟨b_j| C_j - I)
"""
function clifford_shadow_snapshot(ψ::Vector{ComplexF64}, N::Int)
    dim = 1 << N

    # 1. Sample random local Clifford
    indices = sample_local_clifford_indices(N)

    # 2. Build and apply local Clifford
    C = build_local_clifford_unitary(indices, N)
    ψ_rotated = C * ψ

    # 3. Sample from |ψ_rotated|² distribution
    probs = abs2.(ψ_rotated)
    outcome_idx = CPUQuantumStateMeasurements.sample_from_probabilities(probs) - 1  # 0-based

    # 4. Extract bitstring
    outcomes = zeros(Int, N)
    @inbounds for k in 1:N
        outcomes[k] = (outcome_idx >> (k - 1)) & 1
    end

    return CliffordSnapshot(indices, outcomes)
end

"""
    clifford_shadow_snapshot(ρ::Matrix{ComplexF64}, N::Int) -> CliffordSnapshot

Perform one local Clifford shadow measurement on MIXED STATE.
"""
function clifford_shadow_snapshot(ρ::Matrix{ComplexF64}, N::Int)
    dim = 1 << N

    # 1. Sample random local Clifford
    indices = sample_local_clifford_indices(N)

    # 2. Build and apply local Clifford
    C = build_local_clifford_unitary(indices, N)
    ρ_rotated = C * ρ * C'

    # 3. Sample from diagonal probabilities
    probs = real.(diag(ρ_rotated))
    outcome_idx = CPUQuantumStateMeasurements.sample_from_probabilities(probs) - 1

    # 4. Extract bitstring
    outcomes = zeros(Int, N)
    @inbounds for k in 1:N
        outcomes[k] = (outcome_idx >> (k - 1)) & 1
    end

    return CliffordSnapshot(indices, outcomes)
end

"""
    collect_clifford_shadows(ψ::Vector{ComplexF64}, N::Int, M::Int) -> Vector{CliffordSnapshot}

Collect M local Clifford shadows from pure state ψ.
"""
function collect_clifford_shadows(ψ::Vector{ComplexF64}, N::Int, M::Int)
    snapshots = Vector{CliffordSnapshot}(undef, M)
    for m in 1:M
        snapshots[m] = clifford_shadow_snapshot(ψ, N)
    end
    return snapshots
end

"""
    collect_clifford_shadows(ρ::Matrix{ComplexF64}, N::Int, M::Int) -> Vector{CliffordSnapshot}

Collect M local Clifford shadows from mixed state ρ.
"""
function collect_clifford_shadows(ρ::Matrix{ComplexF64}, N::Int, M::Int)
    snapshots = Vector{CliffordSnapshot}(undef, M)
    for m in 1:M
        snapshots[m] = clifford_shadow_snapshot(ρ, N)
    end
    return snapshots
end

# ==============================================================================
# CLIFFORD SHADOW INVERSE CHANNEL
# ==============================================================================
#
# For local Clifford shadows, the inverse channel is:
#   ρ_m = ⊗_j [3 C_j† |b_j⟩⟨b_j| C_j - I]
#
# This is identical to Pauli shadows! The difference is in the measurement
# distribution - Cliffords provide more uniform coverage of the Bloch sphere.

"""
    single_qubit_clifford_shadow(clifford_idx::Int, outcome::Int) -> Matrix{ComplexF64}

Compute single-qubit shadow matrix for local Clifford measurement.

Returns: 3 · C† |b⟩⟨b| C - I
"""
function single_qubit_clifford_shadow(clifford_idx::Int, outcome::Int)
    C = get_clifford_gate(clifford_idx)
    ket = outcome == 0 ? ComplexF64[1, 0] : ComplexF64[0, 1]
    I2 = Matrix{ComplexF64}(I, 2, 2)

    # ρ_shadow = 3 · C† |b⟩⟨b| C - I
    proj = ket * ket'
    return 3 * (C' * proj * C) - I2
end

# Precompute all 48 Clifford shadow matrices (24 gates × 2 outcomes)
# This avoids allocations in the hot reconstruction loop!
const _CLIFFORD_SHADOW_MATRICES = let
    ket_0 = ComplexF64[1, 0]
    ket_1 = ComplexF64[0, 1]
    I2 = Matrix{ComplexF64}(I, 2, 2)

    # shadows[clifford_idx, outcome+1] where clifford_idx ∈ {1,...,24}, outcome ∈ {0,1}
    shadows = Array{Matrix{ComplexF64}}(undef, 24, 2)

    for c in 1:24
        C = SINGLE_QUBIT_CLIFFORDS[c]
        for b in 0:1
            ket = b == 0 ? ket_0 : ket_1
            proj = ket * ket'
            shadows[c, b+1] = 3 * (C' * proj * C) - I2
        end
    end
    shadows
end

"""
    reconstruct_density_matrix_clifford(snapshots::Vector{CliffordSnapshot}, N::Int) -> Matrix{ComplexF64}

Reconstruct density matrix from Clifford shadows using MATRIX-FREE BITWISE algorithm.
Computes each ρ[i,j] directly as product of single-qubit shadow matrix elements.

For large N, use observable estimation instead!
"""
function reconstruct_density_matrix_clifford(snapshots::Vector{CliffordSnapshot}, N::Int)
    M = length(snapshots)
    dim = 1 << N
    nthreads = Threads.nthreads()

    # Pre-extract snapshot data for thread safety
    cliff_data = [snap.clifford_indices for snap in snapshots]
    outcomes_data = [snap.outcomes for snap in snapshots]

    # Thread-local accumulator matrices
    ρ_local = [zeros(ComplexF64, dim, dim) for _ in 1:nthreads]

    # Parallel snapshot processing - ALLOCATION-FREE inner loop
    Threads.@threads for m in 1:M
        tid = mod1(Threads.threadid(), nthreads)
        cliffs = cliff_data[m]
        outcomes = outcomes_data[m]
        ρ_thread = ρ_local[tid]

        # Direct element-wise accumulation (no kron allocation!)
        @inbounds for j in 0:(dim-1)
            for i in 0:(dim-1)
                # Compute ρ_snap[i+1, j+1] = ∏_k shadow_k[bit_i, bit_j]
                val = ComplexF64(1.0)
                for k in 1:N
                    bit_i = (i >> (N - k)) & 1  # k-th qubit bit (kron ordering)
                    bit_j = (j >> (N - k)) & 1
                    val *= _CLIFFORD_SHADOW_MATRICES[cliffs[k], outcomes[k]+1][bit_i+1, bit_j+1]
                end
                ρ_thread[i+1, j+1] += val
            end
        end
    end

    # Reduction - combine thread-local results
    ρ = ρ_local[1]
    for t in 2:nthreads
        ρ .+= ρ_local[t]
    end

    ρ ./= M
    return ρ
end


# ==============================================================================
# GLOBAL CLIFFORD SHADOWS
# ==============================================================================
#
# Global Clifford shadows use entangling N-qubit Clifford circuits (H, S, CNOT).
# The inverse channel scaling is (2^N + 1) for the whole system, not 3 per qubit.
#
# Formula: ρ_shadow = (2^N + 1) C† |b⟩⟨b| C - I
#
# For small N (≤10), we use matrix representation directly.

"""
    GlobalCliffordCircuit

Represents an N-qubit Clifford circuit as a list of gates.

# Fields
- `n_qubits::Int` - Number of qubits
- `gates::Vector{Tuple}` - List of gates: (:H, q), (:S, q), (:CNOT, c, t)
"""
struct GlobalCliffordCircuit
    n_qubits::Int
    gates::Vector{Any}  # Tuple{Symbol, Vararg{Int}}
end

"""
    GlobalCliffordSnapshot

Snapshot from a global Clifford shadow measurement.

# Fields
- `circuit::GlobalCliffordCircuit` - The random Clifford circuit used
- `outcomes::Vector{Int}` - Measurement outcomes (0 or 1 per qubit)
"""
struct GlobalCliffordSnapshot
    circuit::GlobalCliffordCircuit
    outcomes::Vector{Int}
end

"""
    sample_random_global_clifford(N::Int; depth::Int=2N) -> GlobalCliffordCircuit

Generate a random N-qubit Clifford circuit by composing random layers.
"""
function sample_random_global_clifford(N::Int; depth::Int=2 * N)
    gates = Any[]

    for layer in 1:depth
        for q in 1:N
            r = rand(1:4)
            if r == 1
                push!(gates, (:H, q))
            elseif r == 2
                push!(gates, (:S, q))
            elseif r == 3
                push!(gates, (:H, q))
                push!(gates, (:S, q))
            end
        end

        if layer < depth && N >= 2
            qubits = shuffle(1:N)
            for i in 1:2:(N-1)
                push!(gates, (:CNOT, qubits[i], qubits[i+1]))
            end
        end
    end

    return GlobalCliffordCircuit(N, gates)
end

"""
    build_global_clifford_unitary(circuit::GlobalCliffordCircuit) -> Matrix{ComplexF64}

Build the full 2^N × 2^N unitary matrix for a global Clifford circuit.
WARNING: Only use for small N (≤10), as this is O(4^N) in memory.
"""
function build_global_clifford_unitary(circuit::GlobalCliffordCircuit)
    N = circuit.n_qubits
    dim = 1 << N

    U = Matrix{ComplexF64}(I, dim, dim)

    H = ComplexF64[1 1; 1 -1] / sqrt(2)
    S = ComplexF64[1 0; 0 im]
    I2 = Matrix{ComplexF64}(I, 2, 2)

    for gate in circuit.gates
        if gate[1] == :H
            q = gate[2]
            G = q == 1 ? H : I2
            for j in 2:N
                G = kron(G, j == q ? H : I2)
            end
            U = G * U
        elseif gate[1] == :S
            q = gate[2]
            G = q == 1 ? S : I2
            for j in 2:N
                G = kron(G, j == q ? S : I2)
            end
            U = G * U
        elseif gate[1] == :CNOT
            c, t = gate[2], gate[3]
            G = zeros(ComplexF64, dim, dim)
            for k in 0:(dim-1)
                control_bit = (k >> (c - 1)) & 1
                if control_bit == 0
                    G[k+1, k+1] = 1
                else
                    target_flipped = k ⊻ (1 << (t - 1))
                    G[target_flipped+1, k+1] = 1
                end
            end
            U = G * U
        end
    end

    return U
end

"""
    global_clifford_shadow_snapshot(ψ::Vector{ComplexF64}, N::Int) -> GlobalCliffordSnapshot

Perform one global Clifford shadow measurement on a pure state.
"""
function global_clifford_shadow_snapshot(ψ::Vector{ComplexF64}, N::Int)
    circuit = sample_random_global_clifford(N)
    U = build_global_clifford_unitary(circuit)
    ψ_rotated = U * ψ

    probs = abs2.(ψ_rotated)
    outcome_idx = CPUQuantumStateMeasurements.sample_from_probabilities(probs) - 1

    outcomes = zeros(Int, N)
    @inbounds for k in 1:N
        outcomes[k] = (outcome_idx >> (k - 1)) & 1
    end

    return GlobalCliffordSnapshot(circuit, outcomes)
end

"""
    global_clifford_shadow_snapshot(ρ::Matrix{ComplexF64}, N::Int) -> GlobalCliffordSnapshot

Perform one global Clifford shadow measurement on a mixed state.
"""
function global_clifford_shadow_snapshot(ρ::Matrix{ComplexF64}, N::Int)
    circuit = sample_random_global_clifford(N)
    U = build_global_clifford_unitary(circuit)
    ρ_rotated = U * ρ * U'

    probs = real.(diag(ρ_rotated))
    outcome_idx = CPUQuantumStateMeasurements.sample_from_probabilities(probs) - 1

    outcomes = zeros(Int, N)
    @inbounds for k in 1:N
        outcomes[k] = (outcome_idx >> (k - 1)) & 1
    end

    return GlobalCliffordSnapshot(circuit, outcomes)
end

"""
    collect_global_clifford_shadows(ψ::Vector{ComplexF64}, N::Int, M::Int) -> Vector{GlobalCliffordSnapshot}

Collect M global Clifford shadows from pure state ψ.
"""
function collect_global_clifford_shadows(ψ::Vector{ComplexF64}, N::Int, M::Int)
    snapshots = Vector{GlobalCliffordSnapshot}(undef, M)
    for m in 1:M
        snapshots[m] = global_clifford_shadow_snapshot(ψ, N)
    end
    return snapshots
end

"""
    collect_global_clifford_shadows(ρ::Matrix{ComplexF64}, N::Int, M::Int) -> Vector{GlobalCliffordSnapshot}

Collect M global Clifford shadows from mixed state ρ.
"""
function collect_global_clifford_shadows(ρ::Matrix{ComplexF64}, N::Int, M::Int)
    snapshots = Vector{GlobalCliffordSnapshot}(undef, M)
    for m in 1:M
        snapshots[m] = global_clifford_shadow_snapshot(ρ, N)
    end
    return snapshots
end

"""
    reconstruct_density_matrix_global_clifford(snapshots::Vector{GlobalCliffordSnapshot}, N::Int) -> Matrix{ComplexF64}

Reconstruct density matrix from global Clifford shadows.
Uses the global inverse channel: ρ_m = (2^N + 1) C† |b⟩⟨b| C - I
WARNING: Only use for small N (≤10) due to O(4^N) memory.
"""
function reconstruct_density_matrix_global_clifford(snapshots::Vector{GlobalCliffordSnapshot}, N::Int)
    M = length(snapshots)
    dim = 1 << N
    ρ = zeros(ComplexF64, dim, dim)

    factor = ComplexF64(dim + 1)
    I_N = Matrix{ComplexF64}(I, dim, dim)

    for snap in snapshots
        b = sum(snap.outcomes[k] << (k - 1) for k in 1:N)
        proj = zeros(ComplexF64, dim, dim)
        proj[b+1, b+1] = 1.0

        U = build_global_clifford_unitary(snap.circuit)
        ρ_m = factor * (U' * proj * U) - I_N

        ρ .+= ρ_m
    end

    return ρ / M
end


# ==============================================================================
# SNAPSHOT COLLECTION (Bitwise Implementation for Pauli)
# ==============================================================================


"""
    sample_pauli_bases(N::Int) -> Vector{Int}

Sample random Pauli measurement bases for N qubits.
Returns vector of N integers: 1=X, 2=Y, 3=Z (uniform random).
"""
function sample_pauli_bases(N::Int)

    return rand(1:3, N)
end

"""
    classical_shadow_snapshot(ψ::Vector{ComplexF64}, N::Int) -> Snapshot

Perform one classical shadow measurement on PURE STATE using bitwise rotations.

# For Mixed States
Use `classical_shadow_snapshot(ρ::Matrix{ComplexF64}, N::Int)` instead.
"""
function classical_shadow_snapshot(ψ::Vector{ComplexF64}, N::Int)
    ψ_work = copy(ψ)

    # 1. Sample random bases
    bases = sample_pauli_bases(N)

    # 2. Rotate to measurement basis (bitwise gates)
    for k in 1:N
        if bases[k] == 1      # X basis: apply H
            CPUQuantumChannelGates.apply_hadamard_psi!(ψ_work, k, N)
        elseif bases[k] == 2  # Y basis: apply S† then H
            CPUQuantumChannelGates.apply_sdagger_psi!(ψ_work, k, N)
            CPUQuantumChannelGates.apply_hadamard_psi!(ψ_work, k, N)
        end
        # Z basis (bases[k] == 3): no rotation needed
    end

    # 3. Sample from |ψ|² distribution
    dim = 1 << N
    probs = [abs2(ψ_work[i]) for i in 1:dim]
    outcome_idx = CPUQuantumStateMeasurements.sample_from_probabilities(probs) - 1  # 0-based for bit ops

    # 4. Extract bitstring (bitwise)
    outcomes = zeros(Int, N)
    @inbounds for k in 1:N
        outcomes[k] = (outcome_idx >> (k - 1)) & 1
    end

    return PauliSnapshot(bases, outcomes)
end

"""
    classical_shadow_snapshot(ρ::Matrix{ComplexF64}, N::Int) -> Snapshot

Perform one classical shadow measurement on MIXED STATE (density matrix).

For mixed states, we sample from diag(U ρ U†) where U is the Pauli rotation.
"""
function classical_shadow_snapshot(ρ::Matrix{ComplexF64}, N::Int)
    dim = 1 << N

    # 1. Sample random bases
    bases = sample_pauli_bases(N)

    # 2. Build rotation matrix U and compute U ρ U†
    # For efficiency, we build the full rotation and apply it
    U = build_pauli_rotation_matrix(bases, N)
    ρ_rotated = U * ρ * U'

    # 3. Sample from diagonal (measurement probabilities)
    probs = real.(diag(ρ_rotated))
    probs = max.(probs, 0.0)  # Numerical safety
    probs ./= sum(probs)      # Renormalize

    outcome_idx = CPUQuantumStateMeasurements.sample_from_probabilities(probs) - 1  # 0-based for bit ops

    # 4. Extract bitstring (bitwise)
    outcomes = zeros(Int, N)
    @inbounds for k in 1:N
        outcomes[k] = (outcome_idx >> (k - 1)) & 1
    end

    return PauliSnapshot(bases, outcomes)
end

"""
    build_pauli_rotation_matrix(bases::Vector{Int}, N::Int) -> Matrix{ComplexF64}

Build the full 2^N × 2^N rotation matrix for given Pauli bases.
Used for mixed state classical shadows.
"""
function build_pauli_rotation_matrix(bases::Vector{Int}, N::Int)
    # Single-qubit rotation matrices
    H = ComplexF64[1 1; 1 -1] / sqrt(2)
    S_dag = ComplexF64[1 0; 0 -im]
    I2 = ComplexF64[1 0; 0 1]

    # Build tensor product
    U = ones(ComplexF64, 1, 1)
    for k in 1:N
        if bases[k] == 1      # X basis: H
            U = kron(U, H)
        elseif bases[k] == 2  # Y basis: H S†
            U = kron(U, H * S_dag)
        else                  # Z basis: I
            U = kron(U, I2)
        end
    end

    return U
end

"""
    collect_shadows(ψ::Vector{ComplexF64}, N::Int, M::Int) -> Vector{PauliSnapshot}

Collect M classical shadow snapshots from pure state ψ.
"""
function collect_shadows(ψ::Vector{ComplexF64}, N::Int, M::Int)
    snapshots = Vector{PauliSnapshot}(undef, M)
    for m in 1:M
        snapshots[m] = classical_shadow_snapshot(ψ, N)
    end
    return snapshots
end

"""
    collect_shadows(ρ::Matrix{ComplexF64}, N::Int, M::Int) -> Vector{PauliSnapshot}

Collect M classical shadow snapshots from mixed state ρ (density matrix).
"""
function collect_shadows(ρ::Matrix{ComplexF64}, N::Int, M::Int)
    snapshots = Vector{PauliSnapshot}(undef, M)
    for m in 1:M
        snapshots[m] = classical_shadow_snapshot(ρ, N)
    end
    return snapshots
end

"""
    collect_shadows(state, config::ShadowConfig) -> Vector{PauliSnapshot} or Vector{CliffordSnapshot}

Collect snapshots using configuration. Works with both pure and mixed states.

# Arguments
- `state`: Pure state vector OR density matrix
- `config`: ShadowConfig with n_qubits, n_shots, measurement_group

# Measurement Groups
- `:pauli_group` (default): Random single-qubit Pauli rotations (X, Y, Z)
- `:clifford_group`: Random single-qubit Clifford gates (24 elements)

Returns Vector{PauliSnapshot} for Pauli, Vector{CliffordSnapshot} for local Clifford,
Vector{GlobalCliffordSnapshot} for global Clifford.
"""
function collect_shadows(state, config::ShadowConfig)
    if config.measurement_group == :local_clifford
        # Local Clifford shadows (equivalent to randomized Pauli)
        if state isa Vector
            return collect_clifford_shadows(state, config.n_qubits, config.n_shots)
        else
            return collect_clifford_shadows(state, config.n_qubits, config.n_shots)
        end
    elseif config.measurement_group == :global_clifford
        # Global Clifford shadows (entangling unitaries)
        if state isa Vector
            return collect_global_clifford_shadows(state, config.n_qubits, config.n_shots)
        else
            return collect_global_clifford_shadows(state, config.n_qubits, config.n_shots)
        end
    else
        # Default: Pauli shadows
        return collect_shadows(state, config.n_qubits, config.n_shots)
    end
end



# ==============================================================================
# DENSITY MATRIX RECONSTRUCTION
# ==============================================================================

"""
    single_qubit_shadow(basis::Int, outcome::Int) -> Matrix{ComplexF64}

Compute single-qubit classical shadow matrix: 3 u† |s⟩⟨s| u - I

# Arguments
- `basis`: 1=X, 2=Y, 3=Z
- `outcome`: 0 or 1

# Returns
2×2 complex matrix representing the local shadow.

# Mathematical Details
For Pauli measurements, the inverse channel M^{-1} factorizes:
    M^{-1}[ρ] = 3ρ - I

Applied to the post-measurement state |s⟩⟨s| rotated back:
- X, s=0: 3|+⟩⟨+| - I = (1/2)I + (3/2)X
- X, s=1: 3|-⟩⟨-| - I = (1/2)I - (3/2)X
- Y, s=0: 3|+i⟩⟨+i| - I = (1/2)I + (3/2)Y
- Y, s=1: 3|-i⟩⟨-i| - I = (1/2)I - (3/2)Y
- Z, s=0: 3|0⟩⟨0| - I = (1/2)I + (3/2)Z
- Z, s=1: 3|1⟩⟨1| - I = (1/2)I - (3/2)Z

General form: (1/2)I + (3/2)(1-2s)P where P is the measured Pauli.

# Precomputed Shadow Matrices (ALLOCATION-FREE!)
All 6 possible single-qubit shadows are computed once at module load.
Lookup table indexed by [basis, outcome+1] avoids any allocation.
"""

# Precompute all 6 possible single-qubit shadow matrices at module load
# This avoids all allocations in the hot loop!
const _SHADOW_MATRICES = let
    ket_0 = ComplexF64[1, 0]
    ket_1 = ComplexF64[0, 1]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    inv_sqrt2 = 1 / sqrt(2)

    # shadows[basis, outcome+1] where basis ∈ {1,2,3}, outcome ∈ {0,1}
    shadows = Array{Matrix{ComplexF64}}(undef, 3, 2)

    # X basis (basis=1): |+⟩, |-⟩
    ket_plus = inv_sqrt2 * (ket_0 + ket_1)
    ket_minus = inv_sqrt2 * (ket_0 - ket_1)
    shadows[1, 1] = 3 * (ket_plus * ket_plus') - I2    # outcome=0
    shadows[1, 2] = 3 * (ket_minus * ket_minus') - I2  # outcome=1

    # Y basis (basis=2): |+i⟩, |-i⟩
    ket_plus_i = inv_sqrt2 * ComplexF64[1, im]
    ket_minus_i = inv_sqrt2 * ComplexF64[1, -im]
    shadows[2, 1] = 3 * (ket_plus_i * ket_plus_i') - I2    # outcome=0
    shadows[2, 2] = 3 * (ket_minus_i * ket_minus_i') - I2  # outcome=1

    # Z basis (basis=3): |0⟩, |1⟩
    shadows[3, 1] = 3 * (ket_0 * ket_0') - I2  # outcome=0
    shadows[3, 2] = 3 * (ket_1 * ket_1') - I2  # outcome=1

    shadows
end

"""
    single_qubit_shadow(basis::Int, outcome::Int) -> Matrix{ComplexF64}

Return precomputed single-qubit classical shadow matrix (ALLOCATION-FREE lookup).
"""
@inline function single_qubit_shadow(basis::Int, outcome::Int)
    return _SHADOW_MATRICES[basis, outcome+1]
end

"""
    reconstruct_density_matrix_shadows(snapshots::Vector{PauliSnapshot}, N::Int) -> Matrix{ComplexF64}

Reconstruct N-qubit density matrix from classical shadows.

# Formula
    ρ* = (1/M) Σ_m ρ_m

where ρ_m = ⊗_{j=1}^N [3 u_j^(m)† |s_j^(m)⟩⟨s_j^(m)| u_j^(m) - I]

# Convergence
The error ‖ρ* - ρ‖_F scales as O(√(2^N / M)), so M ~ 2^N samples
are needed for full reconstruction with constant error.

# Warning
Full reconstruction requires explicit 2^N × 2^N matrix construction.
For large N, use observable estimation functions instead!

# When to Use
- Small systems (N ≤ 10) where full state is needed
- Benchmarking and validation
- Computing nonlinear functions of ρ (entanglement entropy, etc.)
"""
function reconstruct_density_matrix_shadows_kron(snapshots::Vector{PauliSnapshot}, N::Int)
    M = length(snapshots)
    dim = 1 << N
    nthreads = Threads.nthreads()

    # Thread-local accumulator matrices for parallel reduction
    ρ_local = [zeros(ComplexF64, dim, dim) for _ in 1:nthreads]

    # Parallel snapshot processing
    Threads.@threads for m in 1:M
        tid = mod1(Threads.threadid(), nthreads)
        snap = snapshots[m]

        # Build N-qubit shadow as tensor product
        ρ_snap = single_qubit_shadow(snap.bases[1], snap.outcomes[1])
        for k in 2:N
            ρ_k = single_qubit_shadow(snap.bases[k], snap.outcomes[k])
            ρ_snap = kron(ρ_snap, ρ_k)
        end
        ρ_local[tid] .+= ρ_snap
    end

    # Reduction - combine thread-local results
    ρ = ρ_local[1]
    for t in 2:nthreads
        ρ .+= ρ_local[t]
    end

    ρ ./= M
    return ρ
end

"""
    reconstruct_density_matrix_shadows_threaded(snapshots, N) -> Matrix{ComplexF64}

Threaded version of reconstruct_density_matrix_shadows that parallelizes over
snapshots for significant speedup on multi-core systems.

# Threading Strategy: Thread-Local Accumulation
========================================================

The key challenge with parallel accumulation is avoiding data races when multiple
threads try to update the same ρ matrix. We solve this using the "thread-local
accumulation" pattern:

    ┌──────────────────────────────────────────────────────────────┐
    │  PARALLEL PHASE: Each thread computes snapshots independently │
    ├──────────────────────────────────────────────────────────────┤
    │  Thread 1: ρ_local[1] += snapshot[1] + snapshot[5] + ...     │
    │  Thread 2: ρ_local[2] += snapshot[2] + snapshot[6] + ...     │
    │  Thread 3: ρ_local[3] += snapshot[3] + snapshot[7] + ...     │
    │  Thread 4: ρ_local[4] += snapshot[4] + snapshot[8] + ...     │
    └──────────────────────────────────────────────────────────────┘
                              ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  REDUCTION PHASE: Combine all thread-local results           │
    │       ρ = ρ_local[1] + ρ_local[2] + ρ_local[3] + ρ_local[4]  │
    └──────────────────────────────────────────────────────────────┘

# Why Thread-Local Buffers?

Alternative approaches and their problems:
1. **Atomic operations**: No native atomic complex addition in Julia
2. **Locks (mutex)**: Heavy contention, serializes execution → no speedup
3. **Thread-local buffers**: ✓ No synchronization needed during parallel phase!

# Memory Overhead

Each thread allocates its own 2^N × 2^N matrix:
- N=10: 16 threads × 8MB = 128MB (acceptable)
- N=12: 16 threads × 128MB = 2GB (still OK)
- N=14: 16 threads × 2GB = 32GB (problematic!)

This is why we have N_MAX_FULL_TOMOGRAPHY - for large N, skip reconstruction!

# Expected Speedup

Ideal speedup = number of threads (embarrassingly parallel workload)
Actual speedup ≈ 0.7-0.9 × nthreads due to:
- Memory bandwidth saturation
- Julia's threading overhead
- Reduction phase (serial)

For M=1,000,000 snapshots and N=10, expect ~8-12× speedup on 16 cores.
"""
function reconstruct_density_matrix_shadows_bitwise(snapshots::Vector{PauliSnapshot}, N::Int)
    M = length(snapshots)
    dim = 1 << N
    nthreads = Threads.nthreads()

    # Pre-extract snapshot data for thread safety
    bases_data = [snap.bases for snap in snapshots]
    outcomes_data = [snap.outcomes for snap in snapshots]

    # Thread-local accumulator matrices
    ρ_local = [zeros(ComplexF64, dim, dim) for _ in 1:nthreads]

    # Parallel snapshot processing - ALLOCATION-FREE inner loop
    # Instead of kron chain, compute each ρ[i,j] directly:
    #   ρ_snap[i,j] = ∏_{k=1}^N shadow_k[bit_i(k), bit_j(k)]
    # where bit_i(k) extracts the k-th bit of index i
    Threads.@threads for m in 1:M
        tid = mod1(Threads.threadid(), nthreads)
        bases = bases_data[m]
        outcomes = outcomes_data[m]
        ρ_thread = ρ_local[tid]

        # Direct element-wise accumulation (no kron allocation!)
        @inbounds for j in 0:(dim-1)
            for i in 0:(dim-1)
                # Compute ρ_snap[i+1, j+1] = ∏_k shadow_k[bit_i, bit_j]
                # kron(A,B) has A in high bits, so k=1 uses shift (N-1), k=N uses shift 0
                val = ComplexF64(1.0)
                for k in 1:N
                    bit_i = (i >> (N - k)) & 1  # k-th qubit bit (kron ordering)
                    bit_j = (j >> (N - k)) & 1
                    val *= _SHADOW_MATRICES[bases[k], outcomes[k]+1][bit_i+1, bit_j+1]
                end
                ρ_thread[i+1, j+1] += val
            end
        end
    end

    # Reduction - combine thread-local results
    ρ = ρ_local[1]
    for t in 2:nthreads
        ρ .+= ρ_local[t]
    end

    ρ ./= M
    return ρ
end

# ==============================================================================
# PAULI STRING OBSERVABLE ESTIMATION
# ==============================================================================
#
# This is the KEY ADVANTAGE of classical shadows: we can estimate ⟨P⟩ for any
# Pauli string P without reconstructing the full density matrix!
#
# THEORY
# ------
# For a k-local Pauli string P = P_1 ⊗ P_2 ⊗ ... ⊗ P_N (each P_j ∈ {I,X,Y,Z}),
# the classical shadows estimator is:
#
#     ⟨P⟩ ≈ (1/M) Σ_m f_m(P)
#
# where f_m(P) = 0 if any non-trivial P_j doesn't match the measurement basis,
# and otherwise f_m(P) = ∏_{j: P_j ≠ I} 3 × (1 - 2×s_j)
#
# VARIANCE BOUND (Huang et al.)
# -----------------------------
# For a k-local Pauli string (k non-identity terms):
#     Var[estimator] ≤ 3^k / M
#
# Sample complexity: M = O(3^k ε^{-2} log(1/δ)) for ε-accuracy with prob 1-δ.
# This is INDEPENDENT of N (the number of qubits)!
#
# EXAMPLES
# --------
# 1. Single-qubit Z₁: pauli_string = [3, 0, 0, 0, ...]  → k=1, M ~ 3/ε²
# 2. Two-qubit Z₁Z₂: pauli_string = [3, 3, 0, 0, ...]   → k=2, M ~ 9/ε²
# 3. Three-body:    pauli_string = [3, 3, 3, 0, ...]   → k=3, M ~ 27/ε²
#
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║              CLASSICAL SHADOWS ESTIMATION ALGORITHM                        ║
# ╠════════════════════════════════════════════════════════════════════════════╣
# ║                                                                            ║
# ║  PROBLEM: Estimate ⟨ψ|P|ψ⟩ for Pauli string P = P₁⊗P₂⊗...⊗Pₙ              ║
# ║  from M classical shadow snapshots, WITHOUT reconstructing ρ.              ║
# ║                                                                            ║
# ║  This is the "efficient" part of classical shadows - for k-local           ║
# ║  observables, only O(3^k/ε²) samples are needed, independent of N!         ║
# ╚════════════════════════════════════════════════════════════════════════════╝
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ THEORETICAL FOUNDATION (Huang et al. 2020)                                  │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  For random Pauli measurements, the shadow estimator for observable O is:  │
# │                                                                             │
# │    ω̂(O) = (1/M) Σₘ f_m(O)                                                  │
# │                                                                             │
# │  where f_m(O) is the contribution from snapshot m.                         │
# │                                                                             │
# │  KEY INSIGHT: For Pauli observables P, f_m(P) has a simple formula:        │
# │                                                                             │
# │    f_m(P) = ∏_{k: Pₖ≠I} { 3 × eₖ   if bₖ(m) = Pₖ                           │
# │                          { 0       otherwise                                │
# │                                                                             │
# │  where:                                                                     │
# │    - bₖ(m) is the measurement basis used on qubit k in snapshot m          │
# │    - eₖ = (1 - 2×sₖ(m)) ∈ {+1,-1} is the eigenvalue from outcome sₖ(m)     │
# │                                                                             │
# │  INTUITION: A snapshot only contributes if it measured in the SAME         │
# │  basis as each non-trivial Pauli. When it does contribute, the value       │
# │  is 3^k × (product of eigenvalues), where k = number of non-identity Ps.   │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ WHY THE FACTOR OF 3?                                                        │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  The factor of 3 comes from the INVERSION of the measurement channel.      │
# │                                                                             │
# │  For random Pauli measurements, the channel is:                            │
# │    M(ρ) = (1/3) Σ_{b∈{X,Y,Z}} Σ_{s∈{0,1}} ⟨s_b|ρ|s_b⟩ |s_b⟩⟨s_b|           │
# │                                                                             │
# │  The inverse is:                                                            │
# │    M⁻¹(|s_b⟩⟨s_b|) = 3|s_b⟩⟨s_b| - I                                       │
# │                                                                             │
# │  For each qubit where we want to estimate Pauli P_b (same as measurement   │
# │  basis b), we get: Tr(P_b × (3|s_b⟩⟨s_b| - I)) = 3×(1-2s) - 0 = 3×eₛ       │
# │                                                                             │
# │  Since qubits factorize, k non-trivial Paulis give 3^k × (product of e's). │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ VARIANCE AND SAMPLE COMPLEXITY                                              │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  For a k-local Pauli (k non-identity operators):                           │
# │                                                                             │
# │    Var[ω̂(P)] ≤ 3^k / M                                                     │
# │                                                                             │
# │  To achieve error ε with high probability: M = O(3^k / ε²)                 │
# │                                                                             │
# │  CRUCIAL: This is INDEPENDENT of N! A 1000-qubit system needs the same     │
# │  number of samples as a 4-qubit system to estimate the same P accurately.  │
# │                                                                             │
# │  Examples:                                                                  │
# │    - k=1 (single qubit): M ~ 3/ε² ≈ 300 for ε=0.1                          │
# │    - k=2 (two-body):     M ~ 9/ε² ≈ 900 for ε=0.1                          │
# │    - k=3 (three-body):   M ~ 27/ε² ≈ 2700 for ε=0.1                        │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ALGORITHM PSEUDOCODE:
#
#   1. Initialize total = 0
#   2. For each snapshot (bases, outcomes):
#      a. value = 1.0
#      b. For each qubit k where P_k ≠ I:
#         - If bases[k] ≠ P_k: value = 0, break
#         - Else: value *= 3 × (1 - 2×outcomes[k])
#      c. total += value
#   3. Return total / M
#
# ==============================================================================

"""
    get_expectation_value_from_shadows(snapshots::Vector{PauliSnapshot}, pauli_string::Vector{Int}, N::Int) -> Float64

Estimate expectation value ⟨P⟩ of a Pauli string operator using classical shadows.

This is the EFFICIENT method - complexity depends only on k (locality), not N (system size).
For k-local observables, need only O(3^k/ε²) samples regardless of N!

# Arguments
- `snapshots`: Vector of Snapshot objects from collect_shadows
- `pauli_string`: Vector of length N specifying Pauli operators:
    - 0 = I (identity)
    - 1 = X
    - 2 = Y
    - 3 = Z
- `N`: Number of qubits

# Returns
Estimated expectation value ⟨ψ|P|ψ⟩ ∈ [-1, 1] for normalized Pauli strings.

# Algorithm (Huang et al. 2020)
For each snapshot m with bases b_m and outcomes s_m:

1. Check if all non-trivial Paulis match the measurement basis
2. If any mismatch: contribution = 0 (snapshot doesn't contribute)
3. If all match: contribution = ∏_{j: P_j ≠ I} 3 × (1 - 2×s_j^(m))
4. Average over all snapshots

The factor of 3 comes from inverting the random Pauli measurement channel.

# Variance
For k non-identity Paulis: Var ≤ 3^k / M
→ For accurate estimation of k-local observable, need M ~ 3^k/ε² samples.

# Example
```julia
N = 4
snapshots = collect_shadows(ψ, N, 1000)

# Estimate ⟨Z₁Z₂⟩ (2-local correlator)
pauli_ZZ = [3, 3, 0, 0]  # Z on qubits 1,2, identity elsewhere
val = get_expectation_value_from_shadows(snapshots, pauli_ZZ, N)

# Estimate ⟨X₁⟩ (single-qubit)
pauli_X1 = [1, 0, 0, 0]
val = get_expectation_value_from_shadows(snapshots, pauli_X1, N)

# Estimate ⟨X₁Y₂Z₃⟩ (3-body term)
pauli_XYZ = [1, 2, 3, 0]
val = get_expectation_value_from_shadows(snapshots, pauli_XYZ, N)

# Three-body ZZZ correlator
pauli_ZZZ = [3, 3, 3, 0]
val = get_expectation_value_from_shadows(snapshots, pauli_ZZZ, N)
```

See also: [`estimate_local_observables`](@ref), [`estimate_two_point_correlators`](@ref)
"""
function get_expectation_value_from_shadows(snapshots::Vector{PauliSnapshot}, pauli_string::Vector{Int}, N::Int)
    @assert length(pauli_string) == N "Pauli string must have length N"
    M = length(snapshots)
    total = 0.0

    for snap in snapshots
        # ──────────────────────────────────────────────────────────────────────
        # Compute f_m(P) for this snapshot
        # Formula: f_m(P) = ∏_{k: Pₖ≠I} 3×(1-2×sₖ)  if all bases match, else 0
        # ──────────────────────────────────────────────────────────────────────
        value = 1.0
        valid = true

        for k in 1:N
            p_k = pauli_string[k]

            if p_k == 0  # Identity: contributes factor 1
                continue
            end

            b_k = snap.bases[k]  # Basis: 1=X, 2=Y, 3=Z
            s_k = snap.outcomes[k]  # Outcome: 0 or 1

            # ──────────────────────────────────────────────────────────────────
            # BASIS MATCH CHECK: Pauli operator must match measurement basis
            # P=1 (X) requires b=1 (X basis measurement)
            # P=2 (Y) requires b=2 (Y basis measurement)
            # P=3 (Z) requires b=3 (Z basis measurement)
            # ──────────────────────────────────────────────────────────────────
            if p_k != b_k
                # Mismatch: this snapshot doesn't contribute to this observable
                valid = false
                break
            end

            # ──────────────────────────────────────────────────────────────────
            # CONTRIBUTION: 3 × eigenvalue
            # Eigenvalue of Pauli P_b for outcome s is: (1 - 2s) ∈ {+1, -1}
            # - s=0 (measure +1 eigenstate): eigenvalue = +1
            # - s=1 (measure -1 eigenstate): eigenvalue = -1
            # The factor of 3 comes from channel inversion.
            # ──────────────────────────────────────────────────────────────────
            value *= 3.0 * (1 - 2 * s_k)
        end

        if valid
            total += value
        end
    end

    return total / M
end

"""
    estimate_local_observables(snapshots::Vector{PauliSnapshot}, N::Int) -> NamedTuple

Estimate all single-qubit Pauli expectations ⟨X_j⟩, ⟨Y_j⟩, ⟨Z_j⟩ from shadows.

# Returns NamedTuple with:
- `X::Vector{Float64}` - ⟨X_j⟩ for j = 1:N
- `Y::Vector{Float64}` - ⟨Y_j⟩ for j = 1:N
- `Z::Vector{Float64}` - ⟨Z_j⟩ for j = 1:N

# Variance
Each is a k=1 observable, so Var ≤ 3/M per observable.

# Example
```julia
obs = estimate_local_observables(snapshots, 4)
println("⟨Z₁⟩ = ", obs.Z[1])
println("⟨X₂⟩ = ", obs.X[2])
magnetization = mean(obs.Z)  # Average magnetization
```
"""
function estimate_local_observables(snapshots::Vector{PauliSnapshot}, N::Int)
    X_vals = zeros(Float64, N)
    Y_vals = zeros(Float64, N)
    Z_vals = zeros(Float64, N)

    for j in 1:N
        # Build Pauli strings
        pauli_X = zeros(Int, N)
        pauli_X[j] = 1
        pauli_Y = zeros(Int, N)
        pauli_Y[j] = 2
        pauli_Z = zeros(Int, N)
        pauli_Z[j] = 3

        X_vals[j] = get_expectation_value_from_shadows(snapshots, pauli_X, N)
        Y_vals[j] = get_expectation_value_from_shadows(snapshots, pauli_Y, N)
        Z_vals[j] = get_expectation_value_from_shadows(snapshots, pauli_Z, N)
    end

    return (X=X_vals, Y=Y_vals, Z=Z_vals)
end

"""
    shadow_mean_correlators(snapshots::Vector{PauliSnapshot}, N::Int)

Compute nearest-neighbor correlators ⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩ from Pauli shadows.
Returns (means, stds) where stds are the STATISTICAL UNCERTAINTY from shadow averaging.
The stds are computed from per-shadow estimates, NOT from variation across bonds.
"""
function shadow_mean_correlators(snapshots::Vector{PauliSnapshot}, N::Int)
    if N < 2
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    end

    M = length(snapshots)
    n_bonds = N - 1

    # Compute per-shadow estimates for each correlator type (averaged over bonds)
    XX_per_shadow = zeros(Float64, M)
    YY_per_shadow = zeros(Float64, M)
    ZZ_per_shadow = zeros(Float64, M)

    for (m, snap) in enumerate(snapshots)
        xx_sum, yy_sum, zz_sum = 0.0, 0.0, 0.0
        xx_count, yy_count, zz_count = 0, 0, 0

        for j in 1:n_bonds
            # For XX: need both qubits measured in X basis
            if snap.bases[j] == 1 && snap.bases[j+1] == 1  # Both X
                b1, b2 = (snap.outcomes >> (j - 1)) & 1, (snap.outcomes >> j) & 1
                xx_sum += 9.0 * (1 - 2 * b1) * (1 - 2 * b2)  # 3^2 factor
                xx_count += 1
            end
            # For YY: need both qubits measured in Y basis
            if snap.bases[j] == 2 && snap.bases[j+1] == 2  # Both Y
                b1, b2 = (snap.outcomes >> (j - 1)) & 1, (snap.outcomes >> j) & 1
                yy_sum += 9.0 * (1 - 2 * b1) * (1 - 2 * b2)
                yy_count += 1
            end
            # For ZZ: need both qubits measured in Z basis
            if snap.bases[j] == 3 && snap.bases[j+1] == 3  # Both Z
                b1, b2 = (snap.outcomes >> (j - 1)) & 1, (snap.outcomes >> j) & 1
                zz_sum += 9.0 * (1 - 2 * b1) * (1 - 2 * b2)
                zz_count += 1
            end
        end

        # Average over bonds for this shadow (or 0 if no matching bases)
        XX_per_shadow[m] = xx_count > 0 ? xx_sum / xx_count : 0.0
        YY_per_shadow[m] = yy_count > 0 ? yy_sum / yy_count : 0.0
        ZZ_per_shadow[m] = zz_count > 0 ? zz_sum / zz_count : 0.0
    end

    # Mean over shadows
    XX_mean = mean(XX_per_shadow)
    YY_mean = mean(YY_per_shadow)
    ZZ_mean = mean(ZZ_per_shadow)

    # Standard error from shadow-to-shadow variation
    XX_std = std(XX_per_shadow) / sqrt(M)
    YY_std = std(YY_per_shadow) / sqrt(M)
    ZZ_std = std(ZZ_per_shadow) / sqrt(M)

    return ((XX_mean, YY_mean, ZZ_mean), (XX_std, YY_std, ZZ_std))
end


"""

    estimate_two_point_correlators(snapshots::Vector{PauliSnapshot}, N::Int;
                                    pauli::Symbol=:Z) -> Matrix{Float64}

Estimate all two-point correlators ⟨P_i P_j⟩ for the specified Pauli.

# Arguments
- `snapshots`: Classical shadow snapshots
- `N`: Number of qubits
- `pauli`: Which Pauli operator (:X, :Y, or :Z, default :Z)

# Returns
N × N matrix C where C[i,j] = ⟨P_i P_j⟩

# Variance
Each is a k=2 observable, so Var ≤ 9/M per correlator.

# Example
```julia
# Get all ⟨Z_i Z_j⟩ correlators
C_ZZ = estimate_two_point_correlators(snapshots, N, pauli=:Z)

# Off-diagonal elements give connected correlations
# Diagonal gives ⟨Z_i²⟩ = 1 (always)
```
"""
function estimate_two_point_correlators(snapshots::Vector{PauliSnapshot}, N::Int;
    pauli::Symbol=:Z)
    p_code = pauli == :X ? 1 : (pauli == :Y ? 2 : 3)

    C = zeros(Float64, N, N)

    for i in 1:N
        for j in i:N
            pauli_string = zeros(Int, N)
            pauli_string[i] = p_code
            pauli_string[j] = p_code

            C[i, j] = get_expectation_value_from_shadows(snapshots, pauli_string, N)
            C[j, i] = C[i, j]  # Symmetric
        end
    end

    return C
end

"""
    estimate_three_body_correlators(snapshots::Vector{PauliSnapshot}, N::Int; pauli::Symbol=:Z) -> Vector{Float64}

Estimate all nearest-neighbor three-body correlators ⟨P_i P_{i+1} P_{i+2}⟩ from shadows.

Uses matrix-free O(N*M) implementation - no matrices constructed.

# Arguments
- `snapshots`: Classical shadow snapshots
- `N`: Number of qubits
- `pauli`: Which Pauli operator (:X, :Y, or :Z, default :Z)

# Returns
Vector of length N-2 where element i = ⟨P_i P_{i+1} P_{i+2}⟩

# Variance
Each is a k=3 observable, so Var ≤ 27/M per correlator.
"""
function estimate_three_body_correlators(snapshots::Vector{PauliSnapshot}, N::Int;
    pauli::Symbol=:Z)
    @assert N >= 3 "Need at least 3 qubits for three-body correlators"
    p_code = pauli == :X ? 1 : (pauli == :Y ? 2 : 3)

    n_correlators = N - 2
    C = zeros(Float64, n_correlators)

    for i in 1:n_correlators
        pauli_string = zeros(Int, N)
        pauli_string[i] = p_code
        pauli_string[i+1] = p_code
        pauli_string[i+2] = p_code

        C[i] = get_expectation_value_from_shadows(snapshots, pauli_string, N)
    end

    return C
end

"""
    shadow_mean_three_body_correlators(snapshots::Vector{PauliSnapshot}, N::Int) -> Tuple

Compute mean three-body correlators ⟨XXX⟩, ⟨YYY⟩, ⟨ZZZ⟩ averaged over all nearest-neighbor triplets.

# Returns
Tuple of (means, stds) where:
- means = (mean_XXX, mean_YYY, mean_ZZZ)
- stds = (std_XXX, std_YYY, std_ZZZ)
"""
function shadow_mean_three_body_correlators(snapshots::Vector{PauliSnapshot}, N::Int)
    if N < 3
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    end

    C_XX = estimate_three_body_correlators(snapshots, N; pauli=:X)
    C_YY = estimate_three_body_correlators(snapshots, N; pauli=:Y)
    C_ZZ = estimate_three_body_correlators(snapshots, N; pauli=:Z)

    means = (mean(C_XX), mean(C_YY), mean(C_ZZ))
    stds = (std(C_XX), std(C_YY), std(C_ZZ))

    return means, stds
end

# ==============================================================================
# PURITY AND FIDELITY ESTIMATION
# ==============================================================================

"""
    estimate_purity_shadows(snapshots::Vector{PauliSnapshot}, N::Int) -> Float64

Estimate purity tr(ρ²) using classical shadows WITHOUT full reconstruction.

# Algorithm (Huang et al. Theorem S7)
Uses pairs of snapshots to compute unbiased estimator:
    tr(ρ²) ≈ (2/M(M-1)) Σ_{m<m'} tr(ρ_m ρ_m')

For each pair of snapshots, tr(ρ_m ρ_m') = ∏_j tr(ρ_m^(j) ρ_m'^(j))
where single-qubit terms are computed analytically.

# Single-qubit trace formula
For matching bases: tr(ρ_j ρ_j') = 9 * δ_{s,s'} - 3
For different bases: tr(ρ_j ρ_j') = 1

# Variance
O(9^N / M²) for full purity, but works well for moderate N.

# Note
For large N, the product of local traces leads to exponentially small signal.
Consider using subsystem purity instead for scalability.
"""
function estimate_purity_shadows(snapshots::Vector{PauliSnapshot}, N::Int)
    M = length(snapshots)
    @assert M >= 2 "Need at least 2 snapshots for purity estimation"

    total = 0.0
    n_pairs = 0

    for m1 in 1:(M-1)
        for m2 in (m1+1):M
            # Compute tr(ρ_m1 ρ_m2) as product of local traces
            local_product = 1.0

            for j in 1:N
                b1, s1 = snapshots[m1].bases[j], snapshots[m1].outcomes[j]
                b2, s2 = snapshots[m2].bases[j], snapshots[m2].outcomes[j]

                if b1 == b2  # Same measurement basis
                    # tr(ρ_j ρ_j') = 9 * δ_{s,s'} - 3
                    local_product *= (s1 == s2 ? 6.0 : -3.0)
                else  # Different bases
                    # tr(ρ_j ρ_j') = 1 (shadows are orthogonal on average)
                    local_product *= 1.0
                end
            end

            total += local_product
            n_pairs += 1
        end
    end

    return total / n_pairs
end

"""
    estimate_fidelity_shadows(snapshots::Vector{PauliSnapshot}, ψ_target::Vector{ComplexF64}, N::Int) -> Float64

Estimate fidelity F = ⟨ψ|ρ*|ψ⟩ between reconstructed and target pure state.

# Algorithm
Computes (1/M) Σ_m ⟨ψ|ρ_m|ψ⟩ by explicitly constructing each ρ_m.

# Note
For large N, this is expensive because it builds 2^N × 2^N matrices.
Alternative: decompose |ψ⟩⟨ψ| in Pauli basis and estimate each term separately.
"""
function estimate_fidelity_shadows(snapshots::Vector{PauliSnapshot}, ψ_target::Vector{ComplexF64}, N::Int)
    M = length(snapshots)
    total = 0.0

    for snap in snapshots
        # Build ρ_m
        ρ_m = single_qubit_shadow(snap.bases[1], snap.outcomes[1])
        for k in 2:N
            ρ_k = single_qubit_shadow(snap.bases[k], snap.outcomes[k])
            ρ_m = kron(ρ_m, ρ_k)
        end

        # Compute ⟨ψ|ρ_m|ψ⟩
        total += real(dot(ψ_target, ρ_m * ψ_target))
    end

    return total / M
end

# ==============================================================================
# CLIFFORD SHADOW OBSERVABLE ESTIMATION
# ==============================================================================

"""
    estimate_local_observables(snapshots::Vector{CliffordSnapshot}, N::Int)

Estimate single-site observables ⟨X_j⟩, ⟨Y_j⟩, ⟨Z_j⟩ from Clifford shadows.
Uses density matrix reconstruction to extract observables.

# Returns
Named tuple (X, Y, Z) with each being Vector{Float64} of length N.
"""
function estimate_local_observables(snapshots::Vector{CliffordSnapshot}, N::Int)
    # For Clifford shadows, we compute observables from reconstructed ρ
    # This is less efficient than the Pauli median-of-means but provides std estimates
    M = length(snapshots)

    X_vals = zeros(Float64, N)
    Y_vals = zeros(Float64, N)
    Z_vals = zeros(Float64, N)

    # Reconstruct mean density matrix
    ρ = reconstruct_density_matrix_clifford(snapshots, N)

    # Extract local observables from ρ
    σ_x = ComplexF64[0 1; 1 0]
    σ_y = ComplexF64[0 -im; im 0]
    σ_z = ComplexF64[1 0; 0 -1]

    for j in 1:N
        # Build single-site Pauli operators on full Hilbert space
        X_j = kron(diagm(ones(ComplexF64, 2^(j - 1))), kron(σ_x, diagm(ones(ComplexF64, 2^(N - j)))))
        Y_j = kron(diagm(ones(ComplexF64, 2^(j - 1))), kron(σ_y, diagm(ones(ComplexF64, 2^(N - j)))))
        Z_j = kron(diagm(ones(ComplexF64, 2^(j - 1))), kron(σ_z, diagm(ones(ComplexF64, 2^(N - j)))))

        X_vals[j] = real(tr(ρ * X_j))
        Y_vals[j] = real(tr(ρ * Y_j))
        Z_vals[j] = real(tr(ρ * Z_j))
    end

    return (X=X_vals, Y=Y_vals, Z=Z_vals)
end

"""
    shadow_mean_correlators(snapshots::Vector{CliffordSnapshot}, N::Int)

Compute nearest-neighbor correlators from Clifford shadows.
Returns (means, stds) where means/stds are tuples (XX, YY, ZZ).
"""
function shadow_mean_correlators(snapshots::Vector{CliffordSnapshot}, N::Int)
    if N < 2
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    end

    # Reconstruct density matrix
    ρ = reconstruct_density_matrix_clifford(snapshots, N)

    σ_x = ComplexF64[0 1; 1 0]
    σ_y = ComplexF64[0 -im; im 0]
    σ_z = ComplexF64[1 0; 0 -1]
    I2 = diagm(ones(ComplexF64, 2))

    XX_vals = Float64[]
    YY_vals = Float64[]
    ZZ_vals = Float64[]

    for j in 1:(N-1)
        # Build XX_j,j+1, YY_j,j+1, ZZ_j,j+1
        pre = j > 1 ? diagm(ones(ComplexF64, 2^(j - 1))) : [1.0 + 0im;;]
        post = j + 1 < N ? diagm(ones(ComplexF64, 2^(N - j - 1))) : [1.0 + 0im;;]

        XX_jk = kron(pre, kron(kron(σ_x, σ_x), post))
        YY_jk = kron(pre, kron(kron(σ_y, σ_y), post))
        ZZ_jk = kron(pre, kron(kron(σ_z, σ_z), post))

        push!(XX_vals, real(tr(ρ * XX_jk)))
        push!(YY_vals, real(tr(ρ * YY_jk)))
        push!(ZZ_vals, real(tr(ρ * ZZ_jk)))
    end

    # Statistics already imported at module level
    means = (mean(XX_vals), mean(YY_vals), mean(ZZ_vals))
    stds = N > 2 ? (std(XX_vals), std(YY_vals), std(ZZ_vals)) : (0.0, 0.0, 0.0)

    return (means, stds)
end

"""
    estimate_local_observables(snapshots::Vector{GlobalCliffordSnapshot}, N::Int)

Estimate single-site observables ⟨X_j⟩, ⟨Y_j⟩, ⟨Z_j⟩ from global Clifford shadows.
Uses density matrix reconstruction to extract observables.
"""
function estimate_local_observables(snapshots::Vector{GlobalCliffordSnapshot}, N::Int)
    # Reconstruct mean density matrix
    ρ = reconstruct_density_matrix_global_clifford(snapshots, N)

    X_vals = zeros(Float64, N)
    Y_vals = zeros(Float64, N)
    Z_vals = zeros(Float64, N)

    # Extract local observables from ρ
    σ_x = ComplexF64[0 1; 1 0]
    σ_y = ComplexF64[0 -im; im 0]
    σ_z = ComplexF64[1 0; 0 -1]

    for j in 1:N
        X_j = kron(diagm(ones(ComplexF64, 2^(j - 1))), kron(σ_x, diagm(ones(ComplexF64, 2^(N - j)))))
        Y_j = kron(diagm(ones(ComplexF64, 2^(j - 1))), kron(σ_y, diagm(ones(ComplexF64, 2^(N - j)))))
        Z_j = kron(diagm(ones(ComplexF64, 2^(j - 1))), kron(σ_z, diagm(ones(ComplexF64, 2^(N - j)))))

        X_vals[j] = real(tr(ρ * X_j))
        Y_vals[j] = real(tr(ρ * Y_j))
        Z_vals[j] = real(tr(ρ * Z_j))
    end

    return (X=X_vals, Y=Y_vals, Z=Z_vals)
end


end # module CPUClassicalShadows

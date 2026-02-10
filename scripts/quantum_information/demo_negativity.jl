# =============================================================================
# DEMO: Pure-State Negativity Calculation - Schmidt vs Partial Transpose
# =============================================================================
#
# This didactical script demonstrates two equivalent methods for computing
# the entanglement negativity of a pure quantum state across a bipartition:
#
#   1. SCHMIDT METHOD (Fast): O(2^N) - Uses singular value decomposition
#   2. PARTIAL TRANSPOSE METHOD (Slow): O(4^N) - Full density matrix construction
#
# Both methods give identical results for pure states, but the Schmidt method
# is exponentially faster and should be used in practice.
#
# =============================================================================
# PARTIAL TRANSPOSE: DEFINITION AND INTUITION
# =============================================================================
#
# Consider a bipartite system AB with Hilbert space H_A ⊗ H_B.
# Let dim(H_A) = d_A and dim(H_B) = d_B, so total dimension is d_A × d_B.
#
# DENSITY MATRIX AS 4-INDEX TENSOR:
# ---------------------------------
# A density matrix ρ on H_A ⊗ H_B can be viewed as a 4-index tensor:
#
#     ρ = Σᵢⱼₖₗ ρ[i,k,j,l] |i⟩⟨j|_A ⊗ |k⟩⟨l|_B
#
# where:
#   - i, j ∈ {0, 1, ..., d_A-1} are indices for subsystem A
#   - k, l ∈ {0, 1, ..., d_B-1} are indices for subsystem B
#   - ρ[i,k,j,l] = ⟨i,k|ρ|j,l⟩ is the matrix element
#
# FLATTENING TO 2D MATRIX:
# ------------------------
# To work with ρ as a 2D matrix, we use a combined index:
#
#     (row) = i * d_B + k    ← combines (i,k) into single row index
#     (col) = j * d_B + l    ← combines (j,l) into single column index
#
# For 2 qubits (d_A = d_B = 2), the basis ordering is:
#
#     index:  0    1    2    3
#     state: |00⟩ |01⟩ |10⟩ |11⟩
#             ↑↑   ↑↑   ↑↑   ↑↑
#             AB   AB   AB   AB
#
# PARTIAL TRANSPOSE DEFINITION:
# -----------------------------
# The PARTIAL TRANSPOSE with respect to B swaps the B-indices (k ↔ l):
#
#     ρ^{T_B}[i,k,j,l] = ρ[i,l,j,k]
#                           ↑     ↑
#                        k and l swapped!
#
# In bra-ket notation:
#     ρ^{T_B} = Σᵢⱼₖₗ ρ[i,k,j,l] |i⟩⟨j|_A ⊗ |l⟩⟨k|_B
#                                          ↑   ↑
#                                        swapped!
#
# IMPLEMENTATION AS RESHAPE-TRANSPOSE-RESHAPE:
# --------------------------------------------
# In code, partial transpose can be done as:
#     1. Reshape ρ from (d_A*d_B, d_A*d_B) → (d_A, d_B, d_A, d_B)
#     2. Permute axes: (i, k, j, l) → (i, l, j, k)  [swap k ↔ l]
#     3. Reshape back to (d_A*d_B, d_A*d_B)
#
# PHYSICAL INTUITION:
# -------------------
# - Full TRANSPOSE flips the entire matrix: ρ^T [α,β] = ρ[β,α]
# - PARTIAL transpose only flips within one subsystem
# - For product states: (ρ_A ⊗ ρ_B)^{T_B} = ρ_A ⊗ (ρ_B)^T
# - Since transpose preserves eigenvalues, product states remain positive
# - Entangled states can acquire NEGATIVE EIGENVALUES!
#
# =============================================================================
# DETAILED EXAMPLE: Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
# =============================================================================
#
# STEP 1: Write the density matrix
# ---------------------------------
#     |Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩)
#
#     ρ = |Φ⁺⟩⟨Φ⁺| = (1/2)(|00⟩ + |11⟩)(⟨00| + ⟨11|)
#       = (1/2)(|00⟩⟨00| + |00⟩⟨11| + |11⟩⟨00| + |11⟩⟨11|)
#
# STEP 2: Write as 4×4 matrix (basis: |00⟩, |01⟩, |10⟩, |11⟩)
# ------------------------------------------------------------
#
#              |00⟩  |01⟩  |10⟩  |11⟩
#     ρ = (1/2) ┌                    ┐
#         ⟨00|  │  1    0    0    1  │   ← |00⟩⟨00| + |00⟩⟨11|
#         ⟨01|  │  0    0    0    0  │
#         ⟨10|  │  0    0    0    0  │
#         ⟨11|  │  1    0    0    1  │   ← |11⟩⟨00| + |11⟩⟨11|
#               └                    ┘
#
# STEP 3: Apply partial transpose (swap B-indices in each term)
# --------------------------------------------------------------
# For each matrix element ρ[i,k,j,l], compute ρ^{T_B}[i,k,j,l] = ρ[i,l,j,k]
#
# Key transformation rules for the 4 non-zero elements:
#     |00⟩⟨00| → |00⟩⟨00|  (indices: i=0,k=0,j=0,l=0 → no change)
#     |00⟩⟨11| → |01⟩⟨10|  (indices: i=0,k=0,j=1,l=1 → swap k↔l gives i=0,k=1,j=1,l=0)
#     |11⟩⟨00| → |10⟩⟨01|  (indices: i=1,k=1,j=0,l=0 → swap k↔l gives i=1,k=0,j=0,l=1)
#     |11⟩⟨11| → |11⟩⟨11|  (indices: i=1,k=1,j=1,l=1 → no change)
#
# STEP 4: Write the partially transposed matrix
# ----------------------------------------------
#
#                  |00⟩  |01⟩  |10⟩  |11⟩
#     ρ^{T_B} = (1/2) ┌                    ┐
#               ⟨00|  │  1    0    0    0  │
#               ⟨01|  │  0    0    1    0  │   ← |01⟩⟨10| appeared here!
#               ⟨10|  │  0    1    0    0  │   ← |10⟩⟨01| appeared here!
#               ⟨11|  │  0    0    0    1  │
#                     └                    ┘
#
# STEP 5: Compute eigenvalues
# ---------------------------
# The eigenvalues of ρ^{T_B} are: {1/2, 1/2, 1/2, -1/2}
#                                                  ↑
#                                           NEGATIVE EIGENVALUE!
#
# The presence of a negative eigenvalue proves |Φ⁺⟩ is ENTANGLED.
#
# Negativity: N = |λ_negative| = |-1/2| = 0.5
# Trace norm: ||ρ^{T_B}||₁ = 1/2 + 1/2 + 1/2 + 1/2 = 2
# Check: N = (||ρ^{T_B}||₁ - 1)/2 = (2 - 1)/2 = 0.5 ✓
#
# IMPORTANT NOTE: T_A ≡ T_B invariance
# ------------------------------------
# Performing partial transpose on subsystem A yields the SAME eigenvalue
# spectrum as transposing B. This is because ρ^{T_A} = (ρ^{T_B})^T for pure
# states, and transpose preserves eigenvalues. Thus:
#
#     N(ρ^{T_A}) = N(ρ^{T_B})
#
# In code, it doesn't matter which subsystem you transpose; the Negativity
# is invariant under this choice.
# =============================================================================
# ANALYTICAL EXAMPLE: N-qubit GHZ state
# =============================================================================
#
# The GHZ state for N qubits is:
#
#     |GHZ_N⟩ = (1/√2)(|00...0⟩ + |11...1⟩)
#
# SCHMIDT DECOMPOSITION for k:(N-k) bipartition:
# ----------------------------------------------
# Splitting into A (first k qubits) and B (last N-k qubits):
#
#     |GHZ_N⟩ = (1/√2)(|00...0⟩_A ⊗ |00...0⟩_B + |11...1⟩_A ⊗ |11...1⟩_B)
#
# This is ALREADY in Schmidt form with:
#     σ₁ = 1/√2,  |a₁⟩ = |00...0⟩_A,  |b₁⟩ = |00...0⟩_B
#     σ₂ = 1/√2,  |a₂⟩ = |11...1⟩_A,  |b₂⟩ = |11...1⟩_B
#
# Schmidt coefficients: {1/√2, 1/√2}  (for ANY k from 1 to N-1)
#
# NEGATIVITY CALCULATION:
# -----------------------
#     Σσᵢ = 1/√2 + 1/√2 = √2
#     (Σσᵢ)² = 2
#     N = ((Σσᵢ)² - 1)/2 = (2 - 1)/2 = 0.5
#
# KEY RESULT: N(GHZ_N, k) = 0.5 for ALL bipartitions k = 1, 2, ..., N-1
#
# This is because GHZ has Schmidt rank 2 regardless of where you cut!
# The entanglement is "fragile" - tracing out ANY qubit destroys it.
#
# =============================================================================
# ANALYTICAL EXAMPLE: N-qubit W state
# =============================================================================
#
# The W state for N qubits is:
#
#     |W_N⟩ = (1/√N)(|10...0⟩ + |01...0⟩ + ... + |00...1⟩)
#           = (1/√N) Σᵢ |0...010...0⟩  (single 1 at position i)
#
# SCHMIDT DECOMPOSITION for k:(N-k) bipartition:
# ----------------------------------------------
# Split into A (first k qubits) and B (last N-k qubits).
# The W state terms split into two groups:
#
#   - k terms have the "1" in subsystem A: |single 1⟩_A ⊗ |00...0⟩_B
#   - (N-k) terms have the "1" in subsystem B: |00...0⟩_A ⊗ |single 1⟩_B
#
# Define normalized states:
#     |W_A⟩ = (1/√k) Σᵢ₌₁ᵏ |0...010...0⟩_A  (uniform superposition in A)
#     |W_B⟩ = (1/√(N-k)) Σⱼ₌₁^{N-k} |0...010...0⟩_B  (uniform superposition in B)
#
# Schmidt decomposition:
#     |W_N⟩ = √(k/N) |W_A⟩ ⊗ |00...0⟩_B + √((N-k)/N) |00...0⟩_A ⊗ |W_B⟩
#
# Schmidt coefficients: {√(k/N), √((N-k)/N)}  (Schmidt rank = 2)
#
# NEGATIVITY CALCULATION:
# -----------------------
#     σ₁ = √(k/N),  σ₂ = √((N-k)/N)
#     Σσᵢ = √(k/N) + √((N-k)/N)
#     (Σσᵢ)² = k/N + (N-k)/N + 2√(k(N-k))/N = 1 + 2√(k(N-k))/N
#     N = ((Σσᵢ)² - 1)/2 = √(k(N-k))/N
#
# KEY RESULT: N(W_N, k) = √(k(N-k))/N
#
# EXAMPLES for N=10:
#     k=1:  N = √(1×9)/10 = 3/10 = 0.30
#     k=2:  N = √(2×8)/10 = 4/10 = 0.40
#     k=3:  N = √(3×7)/10 = √21/10 ≈ 0.458
#     k=4:  N = √(4×6)/10 = √24/10 ≈ 0.490
#     k=5:  N = √(5×5)/10 = 5/10 = 0.50  (maximum at k=N/2)
#
# The W state shows a SYMMETRIC profile peaking at k = N/2.
# Unlike GHZ, W has more "robust" entanglement under particle loss.
#
# =============================================================================
# ANALYTICAL EXAMPLE: Dicke state |D_N^{N/2}⟩
# =============================================================================
#
# The Dicke state with N/2 excitations is:
#
#     |D_N^{N/2}⟩ = (1/√C(N,N/2)) Σ_perms |half 0s, half 1s⟩
#
# where C(N,N/2) = N!/(N/2)!² is the binomial coefficient.
#
# SCHMIDT DECOMPOSITION for k:(N-k) bipartition:
# ----------------------------------------------
# This is more complex. The Schmidt rank is min(C(k,m), C(N-k, N/2-m)) for
# each possible number m of 1s in subsystem A (m = 0, 1, ..., min(k, N/2)).
#
# The Schmidt coefficients are:
#     σₘ = √(C(k,m) × C(N-k, N/2-m) / C(N, N/2))
#
# for m = max(0, N/2-(N-k)), ..., min(k, N/2).
#
# NEGATIVITY (numerical values for N=10, k=5):
#     Schmidt coefficients: multiple values (higher rank than GHZ/W)
#     Expected negativity: N ≈ 1.53  (much higher than W or GHZ!)
#
# The Dicke state is MAXIMALLY ENTANGLED among symmetric states.
#
# =============================================================================
# SUMMARY: ANALYTICAL BENCHMARKS
# =============================================================================
#
# For N=10 qubits, expected negativity values:
#
# | State    | k=1   | k=2   | k=3   | k=4   | k=5   |
# |----------|-------|-------|-------|-------|-------|
# | GHZ      | 0.50  | 0.50  | 0.50  | 0.50  | 0.50  |
# | W        | 0.30  | 0.40  | 0.458 | 0.490 | 0.50  |
# | Dicke    | 0.50  | 0.92  | 1.25  | 1.45  | 1.53  |
# | Random   | ~0.5  | ~1.5  | ~3.4  | ~7.0  | ~11   |
#
# Notes:
#   - GHZ: constant N=0.5 (Schmidt rank 2 everywhere)
#   - W: N = √(k(N-k))/N, symmetric around k=N/2
#   - Dicke: higher entanglement, symmetric around k=N/2
#   - Random: volume-law scaling, N ~ 2^{min(k,N-k)}/2
#
# =============================================================================
# PPT CRITERION (Peres-Horodecki)
# =============================================================================
#
# THEOREM (Peres, 1996): If ρ is separable, then ρ^{T_B} ≥ 0.
#
# CONTRAPOSITIVE: If ρ^{T_B} has negative eigenvalues → ρ is ENTANGLED.
#
# THEOREM (Horodecki³, 1996): For 2⊗2 and 2⊗3 systems, PPT is also SUFFICIENT.
#   - In these dimensions: ρ is separable ⟺ ρ^{T_B} ≥ 0
#
# For LARGER systems (3⊗3 and beyond):
#   - PPT is only NECESSARY, not sufficient
#   - There exist "BOUND ENTANGLED" states: entangled but satisfy PPT
#   - These states cannot be distilled into pure entanglement
#
# =============================================================================
# NEGATIVITY: QUANTIFYING PPT VIOLATION
# =============================================================================
#
# The NEGATIVITY quantifies "how much" the PPT criterion is violated:
#
#     N(ρ) = (||ρ^{T_B}||₁ - 1) / 2 = Σᵢ |λᵢ⁻|
#
# where:
#   - ||·||₁ is the trace norm: ||A||₁ = Tr(√(A†A)) = Σ|eigenvalues|
#   - λᵢ⁻ are the NEGATIVE eigenvalues of ρ^{T_B}
#
# Properties:
#   - N(ρ) ≥ 0 always
#   - N(ρ) = 0 for separable states (in 2⊗2 and 2⊗3: ⟺ separable)
#   - N(ρ) = (d-1)/2 for maximally entangled state of dimension d
#   - N(Bell state) = 0.5
#
# =============================================================================
# PURE STATE SIMPLIFICATION: SCHMIDT DECOMPOSITION
# =============================================================================
#
# For a PURE STATE |ψ⟩ with Schmidt decomposition:
#
#     |ψ⟩ = Σᵢ σᵢ |aᵢ⟩_A ⊗ |bᵢ⟩_B
#
# where σᵢ ≥ 0 are the SCHMIDT COEFFICIENTS (singular values), the negativity
# simplifies dramatically:
#
#     N(|ψ⟩) = ((Σᵢ σᵢ)² - 1) / 2
#
# This is because ||ρ^{T_B}||₁ = (Σᵢ σᵢ)² for pure states.
#
# The σᵢ are simply the singular values of the state vector |ψ⟩ reshaped
# as a d_A × d_B matrix, where d_A = 2^k and d_B = 2^{N-k} for k qubits in A.
#
# =============================================================================
# KEY INSIGHT: THE "ACTIVE SPACE" TRICK
# =============================================================================
#
# The exponential speedup comes from recognizing that a PURE STATE lives in
# a much smaller "active subspace" than what the full Hilbert space suggests.
#
# PHYSICAL INTUITION:
# -------------------
# Consider N=10 qubits with bipartition k=5 (5 qubits in A, 5 in B):
#   - Full Hilbert space: 2^10 = 1024 dimensions
#   - Density matrix ρ: 1024 × 1024 = ~1 million entries
#   - Partial transpose ρ^{T_B}: same size
#
# But for a PURE STATE |ψ⟩:
#   - Schmidt rank ≤ min(2^k, 2^{N-k}) = 32 (for k=5)
#   - Only 32 non-zero singular values!
#   - All information is in a 32-dimensional subspace
#
# MATHEMATICAL DERIVATION:
# ------------------------
# For pure state ρ = |ψ⟩⟨ψ|, write in Schmidt basis:
#
#     |ψ⟩ = Σᵢ σᵢ |aᵢ⟩ ⊗ |bᵢ⟩
#
#     ρ = Σᵢⱼ σᵢσⱼ |aᵢ⟩⟨aⱼ| ⊗ |bᵢ⟩⟨bⱼ|
#
# Partial transpose swaps b-indices:
#
#     ρ^{T_B} = Σᵢⱼ σᵢσⱼ |aᵢ⟩⟨aⱼ| ⊗ |bⱼ⟩⟨bᵢ|
#
# This is NOT a valid density matrix (can have negative eigenvalues).
# The trace norm ||ρ^{T_B}||₁ can be computed directly:
#
# THEOREM: For pure states, ||ρ^{T_B}||₁ = (Σᵢ σᵢ)²
#
# PROOF SKETCH:
# The eigenvalues of ρ^{T_B} for a pure state are {σᵢσⱼ} with signs.
# Direct calculation shows that the sum of absolute eigenvalues equals (Σσᵢ)².
#
# COMPLEXITY SAVINGS:
# -------------------
# Instead of:
#   1. Building 2^N × 2^N density matrix    → O(4^N) memory
#   2. Computing partial transpose          → O(4^N) operations
#   3. Finding all eigenvalues              → O(8^N) operations
#
# We compute:
#   1. Reshape |ψ⟩ to d_A × d_B matrix      → O(1) (just a view!)
#   2. SVD to get singular values           → O(d_A × d_B × min(d_A,d_B))
#   3. Sum singular values, square          → O(min(d_A, d_B))
#
# For N=10, k=5: Schmidt is ~1000× faster. For N=14: ~10000× faster!
#
# =============================================================================
# PARTIAL TRANSPOSE METHOD
# =============================================================================
#
#   1. Form full density matrix ρ = |ψ⟩⟨ψ| ∈ ℂ^{2^N × 2^N}     → O(4^N) memory
#   2. Reshape to ρ[a,b,a',b'] and swap b ↔ b'                   → O(4^N) time
#   3. Compute eigenvalues of ρ^{T_B}                            → O(8^N) time
#   4. Sum absolute eigenvalues                                   → O(2^N) time
#
# =============================================================================
# COMPLEXITY COMPARISON TABLE
# =============================================================================
#
# Concrete numbers for research-scale systems:
#
# | N (qubits) | Method           | Operations (approx) | Memory Needed |
# |------------|------------------|---------------------|---------------|
# |    10      | Schmidt (SVD)    |  ~10⁶               |  ~8 KB        |
# |    10      | Partial Transp.  |  ~10⁹               |  ~8 MB        |
# |    14      | Schmidt (SVD)    |  ~10⁸               |  ~130 KB      |
# |    14      | Partial Transp.  |  ~10¹²              |  ~2 GB        |
# |    20      | Schmidt (SVD)    |  ~10¹⁰              |  ~8 MB        |
# |    20      | Partial Transp.  |  ~10¹⁸              |  ~8 TB (!)    |
#
# The Schmidt method remains tractable for N ≈ 30 qubits (limited by SVD),
# while PT becomes infeasible beyond N ≈ 14 on typical workstations.
#
# LOGARITHMIC NEGATIVITY
# =============================================================================
#
# The LOGARITHMIC NEGATIVITY is often preferred as an entanglement measure:
#
#     E_N(ρ) = log₂(||ρ^{T_B}||₁) = log₂(2N(ρ) + 1)
#
# Advantages:
#   - ADDITIVE under tensor products: E_N(ρ₁ ⊗ ρ₂) = E_N(ρ₁) + E_N(ρ₂)
#   - Upper bound on distillable entanglement
#   - E_N = 0 for separable states, E_N > 0 indicates entanglement
#
# For a maximally entangled state of dimension d: E_N = log₂(d)
# For N-qubit GHZ state with k:N-k bipartition: E_N = 1 (one ebit)
#
# =============================================================================
# WHY SCHMIDT METHOD FAILS FOR MIXED STATES
# =============================================================================
#
# The Schmidt decomposition is UNIQUE TO PURE STATES:
#
#     |ψ⟩ = Σᵢ σᵢ |aᵢ⟩ ⊗ |bᵢ⟩  (pure state)
#
# For a MIXED STATE ρ, there is no such simple decomposition. Instead:
#
#     ρ = Σⱼ pⱼ |ψⱼ⟩⟨ψⱼ|  (ensemble of pure states)
#
# Each |ψⱼ⟩ has its own Schmidt decomposition, and they don't combine nicely.
# The partial transpose mixes terms from different pure states in complex ways.
#
# Therefore, for mixed states:
#   - Must use the FULL partial transpose method
#   - No shortcut exists in general
#   - Complexity remains O(4^N) to O(8^N)
#
# This is why mixed-state entanglement is computationally harder to quantify.
#
# =============================================================================
# PAGE CURVE: TYPICAL NEGATIVITY PROFILE
# =============================================================================
#
# For RANDOM (Haar-distributed) pure states, the negativity follows a
# characteristic "PAGE CURVE" as a function of bipartition size k:
#
#     N(k) ≈ (2^{min(k,N-k)} - 1) / 2  for typical random states
#
# Key features:
#   - SYMMETRIC: N(k) = N(N-k) due to swap symmetry A ↔ B
#   - MAXIMUM at k = N/2: half-half bipartition is most entangled
#   - ZERO at k = 0 or k = N: trivial bipartitions (no cut)
#   - GROWS EXPONENTIALLY with min(k, N-k): smaller subsystem limits entanglement
#
# This explains why:
#   - GHZ states have CONSTANT negativity (N = 0.5 for all k > 0)
#   - W states have a PEAKED profile (max at k = N/2)
#   - Dicke states show HIGHER negativity peaks than W states
#   - Random states show the HIGHEST negativity (volume-law entanglement)
#
# =============================================================================

using LinearAlgebra
using Printf
using CSV
using DataFrames

# Output directory
const OUTPUT_DIR = joinpath(@__DIR__, "demo_negativity")
mkpath(OUTPUT_DIR)

# Maximum N for Partial Transpose method (PT is O(4^N), too slow for larger N)
const N_MAX_PT = 11

# =============================================================================
# QUANTUM STATE GENERATORS
# =============================================================================

"""
    make_ghz(N::Int) -> Vector{ComplexF64}

Generate the N-qubit GHZ (Greenberger-Horne-Zeilinger) state:

    |GHZ_N⟩ = (|00...0⟩ + |11...1⟩) / √2

Properties:
- Maximally entangled across ANY bipartition
- Schmidt rank = 2 for any cut
- Negativity = 0.5 for any non-trivial cut
"""
function make_ghz(N::Int)
    ψ = zeros(ComplexF64, 2^N)
    ψ[1] = 1/√2          # |00...0⟩ → index 1
    ψ[end] = 1/√2        # |11...1⟩ → index 2^N
    return ψ
end

"""
    make_w(N::Int) -> Vector{ComplexF64}

Generate the N-qubit W state:

    |W_N⟩ = (|10...0⟩ + |01...0⟩ + ... + |00...1⟩) / √N

Properties:
- Equal superposition of all single-excitation states
- More robust to particle loss than GHZ
- Negativity depends on the bipartition size
"""
function make_w(N::Int)
    ψ = zeros(ComplexF64, 2^N)
    for i in 0:(N-1)
        # |00...010...0⟩ with 1 at position i (from right)
        idx = 1 << i + 1  # +1 for 1-based indexing
        ψ[idx] = 1/√N
    end
    return ψ
end

"""
    make_dicke(N::Int, k::Int) -> Vector{ComplexF64}

Generate the N-qubit Dicke state with k excitations:

    |D_N^k⟩ = (1/√C(N,k)) Σ_perm |perm of k ones and N-k zeros⟩

Properties:
- Symmetric under particle exchange
- W state is special case: D_N^1
- k = N/2 gives maximum entanglement
"""
function make_dicke(N::Int, k::Int)
    ψ = zeros(ComplexF64, 2^N)
    # Count combinations
    n_combs = binomial(N, k)
    amp = 1/√n_combs

    for idx in 0:(2^N - 1)
        # Count number of 1s in binary representation
        if count_ones(idx) == k
            ψ[idx + 1] = amp  # +1 for 1-based indexing
        end
    end
    return ψ
end

"""
    make_random(N::Int) -> Vector{ComplexF64}

Generate a random N-qubit pure state (Haar random).

Properties:
- Generic entanglement structure
- Negativity follows "Page curve" - peaks at k = N/2
"""
function make_random(N::Int)
    ψ = randn(ComplexF64, 2^N)
    ψ ./= norm(ψ)
    return ψ
end

# =============================================================================
# ANALYTICAL NEGATIVITY FORMULAS (for benchmarking)
# =============================================================================

"""
    negativity_ghz_analytical(N::Int, k::Int) -> Float64

Analytical negativity for GHZ state with k:(N-k) bipartition.

    N(GHZ, k) = 0.5  for any k ∈ {1, ..., N-1}

This is because GHZ always has Schmidt rank 2 with coefficients {1/√2, 1/√2}.
"""
negativity_ghz_analytical(N::Int, k::Int) = (1 <= k <= N-1) ? 0.5 : 0.0

"""
    negativity_w_analytical(N::Int, k::Int) -> Float64

Analytical negativity for W state with k:(N-k) bipartition.

    N(W, k) = √(k(N-k)) / N

Derivation:
- Schmidt coefficients: {√(k/N), √((N-k)/N)}
- Sum: Σσ = √(k/N) + √((N-k)/N)
- (Σσ)² = 1 + 2√(k(N-k))/N
- N = ((Σσ)² - 1)/2 = √(k(N-k))/N
"""
negativity_w_analytical(N::Int, k::Int) = sqrt(k * (N - k)) / N

"""
    negativity_dicke_analytical(N::Int, n_exc::Int, k::Int) -> Float64

Analytical negativity for Dicke state |D_N^{n_exc}⟩ with k:(N-k) bipartition.

The Schmidt coefficients are:
    σₘ = √(C(k,m) × C(N-k, n_exc-m) / C(N, n_exc))

for m = max(0, n_exc-(N-k)), ..., min(k, n_exc).
"""
function negativity_dicke_analytical(N::Int, n_exc::Int, k::Int)
    # Compute Schmidt coefficients
    m_min = max(0, n_exc - (N - k))
    m_max = min(k, n_exc)

    sigma_sum = 0.0
    for m in m_min:m_max
        if binomial(k, m) > 0 && binomial(N-k, n_exc-m) > 0
            σ = sqrt(binomial(k, m) * binomial(N-k, n_exc-m) / binomial(N, n_exc))
            sigma_sum += σ
        end
    end

    return (sigma_sum^2 - 1) / 2
end

# =============================================================================
# NEGATIVITY CALCULATION - SCHMIDT METHOD (FAST)
# =============================================================================

"""
    negativity_schmidt(ψ::Vector{ComplexF64}, N::Int, k::Int) -> Float64

Compute the negativity of pure state |ψ⟩ across the k|(N-k) bipartition
using the Schmidt decomposition (fast method).

# Arguments
- `ψ`: State vector of dimension 2^N
- `N`: Total number of qubits
- `k`: Number of qubits in subsystem B (to be "traced out" conceptually)

# Algorithm
1. Reshape ψ into matrix Ψ of dimensions (2^{N-k}) × (2^k)
2. Compute singular values σᵢ via SVD
3. Return Neg = ((Σσᵢ)² - 1) / 2

# Complexity
- Time: O(2^N)
- Space: O(1) additional (ψ is just reshaped as a view)

# Mathematical Justification
For pure state |ψ⟩ = Σᵢ σᵢ |aᵢ⟩⊗|bᵢ⟩, the partial transpose eigenvalues are:
  λ = {+σᵢσⱼ for i≤j, -σᵢσⱼ for i>j}
Thus ||ρ^{T_B}||₁ = Σᵢⱼ |σᵢσⱼ| = (Σσᵢ)²
And Neg = ((Σσᵢ)² - 1) / 2
"""
function negativity_schmidt(ψ::Vector{ComplexF64}, N::Int, k::Int)
    dim_kept = 1 << (N - k)    # 2^{N-k}
    dim_traced = 1 << k        # 2^k

    # Reshape as matrix: rows = kept subsystem, cols = traced subsystem
    Ψ_mat = reshape(ψ, dim_kept, dim_traced)

    # SVD gives Schmidt coefficients
    σ = svdvals(Ψ_mat)

    # Pure-state negativity formula
    return (sum(σ)^2 - 1) / 2
end

# =============================================================================
# NEGATIVITY CALCULATION - PARTIAL TRANSPOSE METHOD (SLOW, FOR VERIFICATION)
# =============================================================================

"""
    negativity_partial_transpose(ψ::Vector{ComplexF64}, N::Int, k::Int) -> Float64

Compute the negativity of pure state |ψ⟩ across the k|(N-k) bipartition
using the full partial transpose method (slow, for verification).

# Arguments
- `ψ`: State vector of dimension 2^N
- `N`: Total number of qubits
- `k`: Number of qubits in subsystem B (last k qubits)

# Algorithm
1. Form density matrix ρ = |ψ⟩⟨ψ|
2. Reshape ρ as ρ[a, b, a', b'] where a,a' ∈ {1..2^{N-k}} and b,b' ∈ {1..2^k}
3. Partial transpose: swap b ↔ b' to get ρ^{T_B}[a, b', a', b]
4. Compute eigenvalues of ρ^{T_B}
5. Return Neg = (Σ|λᵢ| - 1) / 2

# Complexity
- Time: O(4^N) for matrix formation, O(8^N) for eigenvalue decomposition
- Space: O(4^N) for density matrix

# Warning
Very slow for N > 10! Only use for verification on small systems.
"""
function negativity_partial_transpose(ψ::Vector{ComplexF64}, N::Int, k::Int)
    dim_kept = 1 << (N - k)
    dim_traced = 1 << k
    dim_total = 1 << N

    # Form full density matrix ρ = |ψ⟩⟨ψ|
    ρ = ψ * ψ'

    # Reshape as 4-index tensor: ρ[a, b, a', b']
    # where a indexes kept subsystem (first N-k qubits)
    # and b indexes traced subsystem (last k qubits)
    ρ_reshaped = reshape(ρ, dim_kept, dim_traced, dim_kept, dim_traced)

    # Partial transpose over B: swap b and b'
    # (a, b, a', b') → (a, b', a', b)
    ρ_pt = permutedims(ρ_reshaped, (1, 4, 3, 2))

    # Reshape back to matrix
    ρ_pt = reshape(ρ_pt, dim_total, dim_total)

    # Compute eigenvalues (Hermitian for numerical stability)
    eigs = eigvals(Hermitian(ρ_pt))

    # Negativity from trace norm
    return (sum(abs.(eigs)) - 1) / 2
end

# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

"""
    compare_methods(ψ::Vector{ComplexF64}, N::Int, state_name::String; max_k_pt::Int=6)

Compare Schmidt and Partial Transpose methods for all bipartitions k = 0..N.

Returns a DataFrame with columns:
- k: number of traced qubits
- neg_schmidt: negativity via Schmidt method
- neg_pt: negativity via Partial Transpose (or missing if k > max_k_pt)
- neg_analytical: analytical formula value (for GHZ, W, Dicke)
- diff: absolute difference between methods
- time_schmidt: computation time for Schmidt method
- time_pt: computation time for PT method
"""
function compare_methods(ψ::Vector{ComplexF64}, N::Int, state_name::String;
                         max_k_pt::Int=6, analytical_fn::Union{Function,Nothing}=nothing)
    results = DataFrame(
        state = String[],
        k = Int[],
        neg_schmidt = Float64[],
        neg_pt = Union{Float64, Missing}[],
        neg_analytical = Union{Float64, Missing}[],
        diff_num = Union{Float64, Missing}[],
        diff_ana = Union{Float64, Missing}[],
        time_schmidt_ms = Float64[],
        time_pt_s = Union{Float64, Missing}[]
    )

    for k in 1:(N-1)  # Skip trivial cuts k=0 (no cut) and k=N (full trace)
        # Schmidt method (always computed)
        t0 = time()
        neg_s = negativity_schmidt(ψ, N, k)
        t_schmidt = (time() - t0) * 1000  # ms

        # Partial transpose method (only for small k due to memory)
        if k <= max_k_pt && (N - k) >= 1
            t0 = time()
            neg_pt = negativity_partial_transpose(ψ, N, k)
            t_pt = time() - t0  # seconds
            diff_num = abs(neg_s - neg_pt)
        else
            neg_pt = missing
            t_pt = missing
            diff_num = missing
        end

        # Analytical value (if available)
        if analytical_fn !== nothing
            neg_ana = analytical_fn(k)
            diff_ana = abs(neg_s - neg_ana)
        else
            neg_ana = missing
            diff_ana = missing
        end

        push!(results, (state_name, k, neg_s, neg_pt, neg_ana, diff_num, diff_ana, t_schmidt, t_pt))
    end

    return results
end

# =============================================================================
# MAIN SCRIPT
# =============================================================================

function main()
    println("=" ^ 80)
    println("  PURE-STATE NEGATIVITY: SCHMIDT vs PARTIAL TRANSPOSE COMPARISON")
    println("=" ^ 80)
    println()

    # System sizes to test (larger sizes only use Schmidt method)
    N_vals = [6, 8, 10, 11, 12, 14]

    all_results = DataFrame()

    for N in N_vals
        println("\n" * "=" ^ 60)
        println("  N = $N qubits")
        println("=" ^ 60)

        # Only compute PT for small N (it's O(4^N))
        max_k_pt = N <= N_MAX_PT ? N : 0

        # Generate test states for this N with their analytical formulas
        states = [
            ("GHZ", make_ghz(N), k -> negativity_ghz_analytical(N, k)),
            ("W", make_w(N), k -> negativity_w_analytical(N, k)),
            ("Dicke k=N/2", make_dicke(N, N÷2), k -> negativity_dicke_analytical(N, N÷2, k)),
            ("Random", make_random(N), nothing)  # No analytical formula for random
        ]

        for (name, ψ, analytical_fn) in states
            println("─" ^ 80)
            println("  State: $name (N=$N)")
            println("─" ^ 80)

            results = compare_methods(ψ, N, "$name (N=$N)"; max_k_pt=max_k_pt, analytical_fn=analytical_fn)

        # Print table with analytical column
        println()
        if analytical_fn !== nothing
            println("  k │ Neg (Schmidt) │ Neg (Analyt.) │ Diff (Ana)  │    Neg (PT)   │ Diff (PT)   │ Time Schmidt │ Time PT  │ Speedup")
            println("────┼───────────────┼───────────────┼─────────────┼───────────────┼─────────────┼──────────────┼──────────┼─────────")
        else
            println("  k │ Neg (Schmidt) │    Neg (PT)   │  Diff (PT)  │ Time (Schmidt) │ Time (PT) │ Speedup")
            println("────┼───────────────┼───────────────┼─────────────┼────────────────┼───────────┼─────────")
        end

        for row in eachrow(results)
            if analytical_fn !== nothing
                # Table with analytical column
                if ismissing(row.neg_pt)
                    @printf("  %2d │   %10.5f   │   %10.5f   │  %9.2e  │   (skipped)   │      -      │    %6.2f ms  │    -     │    -\n",
                            row.k, row.neg_schmidt, row.neg_analytical, row.diff_ana, row.time_schmidt_ms)
                else
                    speedup = row.time_pt_s * 1000 / row.time_schmidt_ms
                    @printf("  %2d │   %10.5f   │   %10.5f   │  %9.2e  │   %10.5f   │  %9.2e  │    %6.2f ms  │  %5.2f s │  %5.0f×\n",
                            row.k, row.neg_schmidt, row.neg_analytical, row.diff_ana, row.neg_pt, row.diff_num, row.time_schmidt_ms, row.time_pt_s, speedup)
                end
            else
                # Table without analytical column (for Random state)
                if ismissing(row.neg_pt)
                    @printf("  %2d │   %10.5f   │   (skipped)   │      -      │     %6.2f ms   │     -     │    -\n",
                            row.k, row.neg_schmidt, row.time_schmidt_ms)
                else
                    speedup = row.time_pt_s * 1000 / row.time_schmidt_ms
                    @printf("  %2d │   %10.5f   │   %10.5f   │  %9.2e  │     %6.2f ms   │  %5.2f s  │  %5.0f×\n",
                            row.k, row.neg_schmidt, row.neg_pt, row.diff_num, row.time_schmidt_ms, row.time_pt_s, speedup)
                end
            end
        end
        println()

        append!(all_results, results)
        end  # states loop
    end  # N_vals loop

    # Save results
    csv_path = joinpath(OUTPUT_DIR, "negativity_comparison.csv")
    CSV.write(csv_path, all_results)
    println("  Results saved to: demo_negativity/negativity_comparison.csv")

    # Summary
    println()
    println("=" ^ 80)
    println("  SUMMARY")
    println("=" ^ 80)
    println()
    println("  - Schmidt and Partial Transpose methods give IDENTICAL results")
    println("    (differences are ~10⁻¹⁴, i.e., machine precision)")
    println()
    println("  - Schmidt method is O(2^N), PT method is O(4^N)")
    println("    Schmidt takes ~0.1ms, PT takes ~0.1-1s (1000-10000× slower)")
    println()
    println("  - For pure states, use the Schmidt formula:")
    println("      Neg = ((Σσᵢ)² - 1) / 2")
    println("    where σᵢ are the singular values of ψ reshaped as matrix.")
    println()
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

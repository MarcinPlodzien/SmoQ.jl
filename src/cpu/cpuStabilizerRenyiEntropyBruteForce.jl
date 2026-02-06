#=
╔══════════════════════════════════════════════════════════════════════════════╗
║            STABILIZER RÉNYI ENTROPY (SRE) MODULE                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Quantifying Nonstabilizerness ("Magic") in Quantum States                   ║
║  Using Fast Matrix-Free Bitwise Pauli String Evaluation                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

# REFERENCES
═══════════════════════════════════════════════════════════════════════════════

[1] Leone, Oliviero, Hamma (2022): "Stabilizer Rényi Entropy"
    Physical Review Letters 128, 050402
    https://doi.org/10.1103/PhysRevLett.128.050402
    
[2] Haug, Piroli (2023): "Stabilizer entropies and nonstabilizerness monotones"
    Quantum 7, 1092
    https://doi.org/10.22331/q-2023-08-28-1092

[3] Gottesman (1998): "The Heisenberg Representation of Quantum Computers"
    arXiv:quant-ph/9807006
    (Foundational paper on stabilizer formalism)


# BACKGROUND: THE PAULI GROUP
═══════════════════════════════════════════════════════════════════════════════

The **Pauli group** 𝒫_N on N qubits consists of all N-fold tensor products of 
Pauli matrices {I, X, Y, Z} with phases {±1, ±i}:

    𝒫_N = { ±1, ±i } × { I, X, Y, Z }^⊗N

Properties:
• |𝒫_N| = 4 × 4^N elements (4 phases × 4^N Pauli strings)
• Closed under multiplication (group structure)
• All elements are Hermitian (up to phase) with eigenvalues ±1
• For P ∈ 𝒫_N: P² = ±I (involutory up to phase)

Single-qubit Paulis:
    I = [1 0; 0 1]    X = [0 1; 1 0]    Y = [0 -i; i 0]    Z = [1 0; 0 -1]


# BACKGROUND: THE CLIFFORD GROUP
═══════════════════════════════════════════════════════════════════════════════

The **Clifford group** 𝒞_N is the normalizer of the Pauli group in U(2^N):

    𝒞_N = { U ∈ U(2^N) : U P U† ∈ 𝒫_N  ∀ P ∈ 𝒫_N }

Clifford gates map Pauli operators to Pauli operators under conjugation.

**Generators of 𝒞_N:**
• Hadamard:     H = (X + Z)/√2       maps X↔Z
• Phase gate:   S = diag(1, i)      maps X→Y, Y→-X
• CNOT:         CNOT_{ij}           maps X_i→X_i⊗X_j, Z_j→Z_i⊗Z_j

Key property: Clifford circuits can be efficiently simulated classically 
(Gottesman-Knill theorem) because Pauli operators remain Pauli under evolution.


# BACKGROUND: STABILIZER STATES
═══════════════════════════════════════════════════════════════════════════════

A **stabilizer state** |ψ⟩ is a +1 eigenstate of an abelian subgroup S ⊂ 𝒫_N:

    P |ψ⟩ = |ψ⟩  for all P ∈ S

The subgroup S is called the **stabilizer group** and has |S| = 2^N elements.

**Examples of stabilizer states:**
• Computational basis: |0...0⟩ stabilized by {Z₁, Z₂, ..., Z_N}
• Plus state: |+...+⟩ stabilized by {X₁, X₂, ..., X_N}
• GHZ state: (|0...0⟩+|1...1⟩)/√2 stabilized by {X₁X₂...X_N, Z₁Z₂, Z₂Z₃, ...}
• Bell state: (|00⟩+|11⟩)/√2 stabilized by {X₁X₂, Z₁Z₂}

**Key property:** Stabilizer states are exactly those reachable from |0...0⟩ 
by Clifford gates alone. They can be described efficiently with O(N²) bits.


# NONSTABILIZERNESS AND "MAGIC"
═══════════════════════════════════════════════════════════════════════════════

**Nonstabilizerness** (or "magic") quantifies how far a state is from being 
a stabilizer state. It is a quantum resource required for:

• Universal quantum computation (beyond Clifford)
• Quantum advantage over classical simulation
• Fault-tolerant quantum computing via magic state distillation

The **T gate** (π/8 rotation): T = diag(1, e^{iπ/4}) is the canonical source 
of magic. T|+⟩ is a "magic state" used in fault-tolerant protocols.


# STABILIZER RÉNYI ENTROPY: DEFINITION
═══════════════════════════════════════════════════════════════════════════════

The **n-th Stabilizer Rényi Entropy** (SRE) is defined as:

    Mₙ(ρ) = 1/(1-n) × log₂[ 1/d × Σ_P |Tr(ρP)|^{2n} ]

where:
• d = 2^N is the Hilbert space dimension
• P ranges over all 4^N Pauli strings (excluding global phases)
• n ≥ 2 is the Rényi index

For pure states |ψ⟩ with ρ = |ψ⟩⟨ψ|:

    Mₙ(ψ) = 1/(1-n) × log₂[ 1/d × Σ_P |⟨ψ|P|ψ⟩|^{2n} ]

The most commonly used variant is **M₂** (n=2).


# PROPERTIES OF STABILIZER RÉNYI ENTROPY
═══════════════════════════════════════════════════════════════════════════════

* **Faithfulness:** Mn(psi) = 0 if and only if |psi> is a stabilizer state.
  This follows from the characteristic function of stabilizer states.

* **Non-negativity:** Mn(psi) >= 0 for all states.

* **Invariance under Clifford gates:** 
  Mₙ(C|ψ⟩) = Mₙ(|ψ⟩) for any Clifford unitary C.
  Clifford gates permute Pauli operators, preserving the sum.

* **Additivity on tensor products:**
  Mₙ(|ψ⟩ ⊗ |φ⟩) = Mₙ(|ψ⟩) + Mₙ(|φ⟩)
  Magic of product states adds.

* **Bounded:** For N qubits, Mn <= N (approximately).
  Maximum magic is achieved by certain highly entangled states.

x **Not monotonic under general channels** (differs from proper resource monotones)
  However, M₂ has been shown to be related to certain resource monotones.


# COMPUTATIONAL COMPLEXITY
═══════════════════════════════════════════════════════════════════════════════

• 4^N Pauli strings to evaluate
• O(2^N) operations per Pauli string (bitwise evaluation)
• **Total: O(8^N)**, heavily parallelized over Pauli strings

Approximate timings (with multithreading on 16 cores):
• N=6:  4K Paulis  →  ~0.01s
• N=8:  65K Paulis →  ~0.1s
• N=10: 1M Paulis  →  ~2s
• N=12: 16M Paulis →  ~60s


# IMPLEMENTATION NOTES
═══════════════════════════════════════════════════════════════════════════════

This implementation uses:
• **Bitwise encoding:** Pauli strings as integers 0:(4^N-1)
• **O(2^N) Pauli evaluation:** XOR-based bit flipping for X/Y operators
• **Parallel summation:** Thread-local accumulators (no atomic operations)
• **Direct |⟨P⟩|² computation:** Avoids complex number overhead
=#

module CPUStabilizerRenyiEntropyBruteForce

# Primary verb-based API (recommended)
export get_stabilizer_renyi_entropy

# Internal/legacy exports  
export pauli_moment_sum
export is_stabilizer_state
export sre_summary

# Legacy alias (for backward compatibility)
export magic

# ============================================================================
# PAULI STRING ENCODING
# ============================================================================
# Encoding: pauli_idx = Σₖ pₖ × 4^(k-1), where pₖ ∈ {0,1,2,3} → {I,X,Y,Z}
# Qubit k (1-indexed) has operator (pauli_idx >> 2(k-1)) & 3

"""
    decode_pauli_masks(idx::Int, N::Int) -> (z_mask, flip_mask, y_mask, y_count)

Decode Pauli string index to bitwise masks for O(2^N) evaluation.
"""
@inline function decode_pauli_masks(idx::Int, N::Int)
    z_mask = 0
    flip_mask = 0
    y_mask = 0
    y_count = 0
    
    temp = idx
    @inbounds for k in 0:(N-1)
        op = temp & 3  # 0=I, 1=X, 2=Y, 3=Z
        if op == 1      # X: flip bit
            flip_mask |= (1 << k)
        elseif op == 2  # Y: flip bit + phase
            flip_mask |= (1 << k)
            y_mask |= (1 << k)
            y_count += 1
        elseif op == 3  # Z: parity
            z_mask |= (1 << k)
        end
        temp >>= 2
    end
    
    return z_mask, flip_mask, y_mask, y_count
end

# ============================================================================
# FAST BITWISE PAULI EXPECTATION: |⟨ψ|P|ψ⟩|²
# ============================================================================

"""
    expect_pauli_squared(ψ::Vector{ComplexF64}, pauli_idx::Int, N::Int) -> Float64

Compute |⟨ψ|P|ψ⟩|² using bitwise operations. Returns squared magnitude directly.

Complexity: O(2^N) - single pass over state vector.
"""
function expect_pauli_squared(ψ::Vector{ComplexF64}, pauli_idx::Int, N::Int)
    pauli_idx == 0 && return 1.0  # Identity
    
    z_mask, flip_mask, y_mask, y_count = decode_pauli_masks(pauli_idx, N)
    dim = 1 << N
    
    # Precompute i^y_count phase
    y_mod = y_count & 3
    # i^0=1, i^1=i, i^2=-1, i^3=-i
    base_re = (y_mod == 0) ? 1.0 : (y_mod == 2) ? -1.0 : 0.0
    base_im = (y_mod == 1) ? 1.0 : (y_mod == 3) ? -1.0 : 0.0
    
    result_re = 0.0
    result_im = 0.0
    
    @inbounds for bra in 0:(dim-1)
        ket = xor(bra, flip_mask)
        
        # Sign from Z operators: (-1)^popcount(bra & z_mask)
        z_sign = 1 - 2 * (count_ones(bra & z_mask) & 1)
        
        # Sign from Y operators: (-1)^popcount(bra & y_mask)  
        y_sign = 1 - 2 * (count_ones(bra & y_mask) & 1)
        total_sign = z_sign * y_sign
        
        # Compute conj(ψ[bra]) * ψ[ket]
        ψ_bra = ψ[bra + 1]
        ψ_ket = ψ[ket + 1]
        
        re_bra, im_bra = reim(ψ_bra)
        re_ket, im_ket = reim(ψ_ket)
        
        # conj(a+bi)(c+di) = (ac+bd) + (ad-bc)i
        prod_re = (re_bra * re_ket + im_bra * im_ket) * total_sign
        prod_im = (re_bra * im_ket - im_bra * re_ket) * total_sign
        
        # Multiply by base Y phase
        result_re += prod_re * base_re - prod_im * base_im
        result_im += prod_re * base_im + prod_im * base_re
    end
    
    return result_re * result_re + result_im * result_im
end

"""
    expect_pauli_squared_dm(ρ::Matrix{ComplexF64}, pauli_idx::Int, N::Int) -> Float64

Compute |Tr(ρP)|² for density matrix.
"""
function expect_pauli_squared_dm(ρ::Matrix{ComplexF64}, pauli_idx::Int, N::Int)
    pauli_idx == 0 && return 1.0  # Tr(ρI) = 1
    
    z_mask, flip_mask, y_mask, y_count = decode_pauli_masks(pauli_idx, N)
    dim = 1 << N
    
    y_mod = y_count & 3
    base_re = (y_mod == 0) ? 1.0 : (y_mod == 2) ? -1.0 : 0.0
    base_im = (y_mod == 1) ? 1.0 : (y_mod == 3) ? -1.0 : 0.0
    
    result_re = 0.0
    result_im = 0.0
    
    # Tr(ρP) = Σᵢ ρ[i, xor(i, flip)] × phase(i)
    @inbounds for i in 0:(dim-1)
        j = xor(i, flip_mask)
        
        z_sign = 1 - 2 * (count_ones(i & z_mask) & 1)
        y_sign = 1 - 2 * (count_ones(i & y_mask) & 1)
        total_sign = z_sign * y_sign
        
        ρ_ij = ρ[i + 1, j + 1]
        re_ρ, im_ρ = reim(ρ_ij)
        re_ρ *= total_sign
        im_ρ *= total_sign
        
        result_re += re_ρ * base_re - im_ρ * base_im
        result_im += re_ρ * base_im + im_ρ * base_re
    end
    
    return result_re * result_re + result_im * result_im
end

# ============================================================================
# PARALLEL PAULI MOMENT SUM
# ============================================================================
#
# THREADING BEST PRACTICES FOR JULIA 1.12+
# ═══════════════════════════════════════════════════════════════════════════
#
# This section documents critical lessons for parallel reduction in Julia,
# specifically patterns that avoid common pitfalls with `Threads.@threads`.
#
# PROBLEM 1: threadid() returns UNSTABLE values with default scheduler
# ─────────────────────────────────────────────────────────────────────
# The default `@threads` scheduler is task-based and can migrate tasks 
# between threads during execution. This means `Threads.threadid()` may
# return different values at different points in the same loop iteration!
#
# SOLUTION: Use `:static` scheduler
#   Threads.@threads :static for ...
#
# The `:static` scheduler assigns loop iterations to threads statically
# (no task migration), making threadid() stable within each iteration.
#
#
# PROBLEM 2: threadid() range changed in Julia 1.12
# ─────────────────────────────────────────────────────────────────────
# In Julia <1.12: threadid() returns 1:nthreads()
# In Julia 1.12+: threadid() returns 2:(nthreads()+1) for worker threads!
#
# If you size your accumulator array using nthreads(), accessing 
# partial_sums[threadid()] will cause BoundsError for the highest thread.
#
# SOLUTION: Use maxthreadid() for array sizing
#   max_tid = Threads.maxthreadid()  # Returns highest possible threadid
#   partial_sums = zeros(Float64, max_tid)
#
#
# PROBLEM 3: False sharing / cache contention
# ─────────────────────────────────────────────────────────────────────
# When multiple threads write to adjacent memory locations (e.g., 
# partial_sums[1], partial_sums[2], ...), they may thrash each other's
# CPU cache lines, causing severe slowdowns (up to 10-100x!).
#
# SOLUTION: Use thread-local accumulators with proper spacing
# For even better performance, pad with zeros or use a struct with 
# cache-line alignment (64 bytes typically).
#
#
# PATTERN: Thread-Local Reduction (Race-Condition Free)
# ─────────────────────────────────────────────────────────────────────
# ```julia
# max_tid = Threads.maxthreadid()
# partial_sums = zeros(Float64, max_tid)  # One accumulator per thread
#
# Threads.@threads :static for i in data
#     tid = Threads.threadid()
#     result = expensive_computation(i)
#     @inbounds partial_sums[tid] += result  # No race condition!
# end
#
# total = sum(partial_sums)  # Combine at the end
# ```
#
# This pattern is race-free because each thread only writes to its own slot.
#
# ═══════════════════════════════════════════════════════════════════════════

"""
    pauli_moment_sum(ψ::Vector{ComplexF64}, N::Int; power::Int=4) -> Float64

Compute Σ_P |⟨ψ|P|ψ⟩|^power over all 4^N Pauli strings.

For Mₙ, use power = 2n (e.g., power=4 for M₂).

**Threading implementation:**
- Uses `:static` scheduler to prevent task migration
- Uses `maxthreadid()` for Julia 1.12+ compatibility
- Thread-local accumulators avoid race conditions
- Final reduction via `sum(partial_sums)`

**Complexity:** O(4^N) parallel × O(2^N) per Pauli = O(8^N) total
"""
function pauli_moment_sum(ψ::Vector{ComplexF64}, N::Int; power::Int=4)
    num_paulis = 4^N
    half_power = power ÷ 2  # |⟨P⟩|^{2n} = (|⟨P⟩|²)^n
    
    # ──────────────────────────────────────────────────────────────────────
    # CRITICAL: Use maxthreadid(), NOT nthreads()!
    # In Julia 1.12+, threadid() returns values 2:(nthreads+1), not 1:nthreads
    # Using nthreads() would cause BoundsError for the highest thread ID.
    # ──────────────────────────────────────────────────────────────────────
    max_tid = Threads.maxthreadid()
    partial_sums = zeros(Float64, max_tid)
    
    # ──────────────────────────────────────────────────────────────────────
    # CRITICAL: Use :static scheduler!
    # Without :static, the default task-based scheduler can migrate tasks
    # between threads mid-iteration, making threadid() return unstable values.
    # This would cause partial_sums to be incorrectly accumulated.
    # ──────────────────────────────────────────────────────────────────────
    Threads.@threads :static for pauli_idx in 0:(num_paulis - 1)
        tid = Threads.threadid()
        abs_sq = expect_pauli_squared(ψ, pauli_idx, N)
        @inbounds partial_sums[tid] += abs_sq ^ half_power
    end
    
    return sum(partial_sums)
end

"""
    pauli_moment_sum(ρ::Matrix{ComplexF64}, N::Int; power::Int=4) -> Float64

Compute Σ_P |Tr(ρP)|^power for density matrix.
"""
function pauli_moment_sum(ρ::Matrix{ComplexF64}, N::Int; power::Int=4)
    num_paulis = 4^N
    half_power = power ÷ 2
    
    # Thread-local accumulators - size by maxthreadid() for Julia 1.12+ compatibility
    max_tid = Threads.maxthreadid()
    partial_sums = zeros(Float64, max_tid)
    
    # Use :static scheduler for stable threadid()
    Threads.@threads :static for pauli_idx in 0:(num_paulis - 1)
        tid = Threads.threadid()
        abs_sq = expect_pauli_squared_dm(ρ, pauli_idx, N)
        @inbounds partial_sums[tid] += abs_sq ^ half_power
    end
    
    return sum(partial_sums)
end

# ============================================================================
# STABILIZER RÉNYI ENTROPY
# ============================================================================

"""
    stabilizer_renyi_entropy(ψ::Vector{ComplexF64}, N::Int; n::Int=2) -> Float64

Compute the n-th Stabilizer Rényi Entropy Mₙ for pure state |ψ⟩.

# Definition
    Mₙ(ψ) = 1/(1-n) × log₂[ 1/d × Σ_P |⟨ψ|P|ψ⟩|^{2n} ]

# Arguments
- `ψ`: Normalized state vector of length 2^N
- `N`: Number of qubits  
- `n`: Rényi index (default 2, must be ≥ 2)

# Returns
- Float64: Mₙ value (0 for stabilizer states, >0 for magical states)

# Example
```julia
ψ_zero = zeros(ComplexF64, 4); ψ_zero[1] = 1.0  # |00⟩
M2 = stabilizer_renyi_entropy(ψ_zero, 2)        # ≈ 0 (stabilizer)

ψ_magic = [1.0, exp(im*π/4)] / sqrt(2)          # T|+⟩
M2 = stabilizer_renyi_entropy(ψ_magic, 1)       # > 0 (has magic)
```
"""
function stabilizer_renyi_entropy(ψ::Vector{ComplexF64}, N::Int; n::Int=2)
    n < 2 && error("Rényi index n must be ≥ 2 (n=1 limit requires log-sum)")
    
    d = 2^N
    moment_sum = pauli_moment_sum(ψ, N; power=2*n)
    
    # Mₙ = log₂(moment_sum/d) / (1-n)
    argument = moment_sum / d
    return argument > 0 ? log2(argument) / (1 - n) : Inf
end

"""
    stabilizer_renyi_entropy(ρ::Matrix{ComplexF64}, N::Int; n::Int=2) -> Float64

Compute Mₙ for density matrix ρ.
"""
function stabilizer_renyi_entropy(ρ::Matrix{ComplexF64}, N::Int; n::Int=2)
    n < 2 && error("Rényi index n must be ≥ 2")
    
    d = 2^N
    moment_sum = pauli_moment_sum(ρ, N; power=2*n)
    
    argument = moment_sum / d
    return argument > 0 ? log2(argument) / (1 - n) : Inf
end

# ============================================================================
# PRIMARY API: get_stabilizer_renyi_entropy
# ============================================================================

"""
    get_stabilizer_renyi_entropy(ψ::Vector{ComplexF64}; n::Int=2) -> Float64

Compute the n-th Stabilizer Rényi Entropy Mₙ for pure state |ψ⟩.

# Definition
    Mₙ(ψ) = 1/(1-n) × log₂[ 1/d × Σ_P |⟨ψ|P|ψ⟩|^{2n} ]

# Arguments
- `ψ`: Normalized state vector of length 2^N
- `n`: Rényi index (default 2, must be ≥ 2)

# Returns
- Float64: Mₙ value (0 for stabilizer states, >0 for magical states)

# Complexity
O(8^N) brute-force enumeration over all 4^N Pauli strings.

# Example
```julia
ψ_zero = zeros(ComplexF64, 4); ψ_zero[1] = 1.0  # |00⟩
M2 = get_stabilizer_renyi_entropy(ψ_zero)        # ≈ 0 (stabilizer)
```
"""
function get_stabilizer_renyi_entropy(ψ::Vector{ComplexF64}; n::Int=2)
    n < 2 && error("Rényi index n must be ≥ 2 (n=1 limit requires log-sum)")
    
    N = Int(log2(length(ψ)))
    d = 2^N
    moment_sum = pauli_moment_sum(ψ, N; power=2*n)
    
    # Mₙ = log₂(moment_sum/d) / (1-n)
    argument = moment_sum / d
    return argument > 0 ? log2(argument) / (1 - n) : Inf
end

"""
    get_stabilizer_renyi_entropy(ρ::Matrix{ComplexF64}; n::Int=2) -> Float64

Compute Mₙ for density matrix ρ.
"""
function get_stabilizer_renyi_entropy(ρ::Matrix{ComplexF64}; n::Int=2)
    n < 2 && error("Rényi index n must be ≥ 2")
    
    N = Int(log2(size(ρ, 1)))
    d = 2^N
    moment_sum = pauli_moment_sum(ρ, N; power=2*n)
    
    argument = moment_sum / d
    return argument > 0 ? log2(argument) / (1 - n) : Inf
end

# ============================================================================
# LEGACY API (kept for backward compatibility)
# ============================================================================

# Original functions with explicit N parameter
stabilizer_renyi_entropy(ψ::Vector{ComplexF64}, N::Int; n::Int=2) = get_stabilizer_renyi_entropy(ψ; n=n)
stabilizer_renyi_entropy(ρ::Matrix{ComplexF64}, N::Int; n::Int=2) = get_stabilizer_renyi_entropy(ρ; n=n)

"""
    magic(ψ_or_ρ, N::Int) -> Float64

Legacy shorthand for M₂. Use `get_stabilizer_renyi_entropy()` instead.
"""
magic(ψ::Vector{ComplexF64}, N::Int) = get_stabilizer_renyi_entropy(ψ; n=2)
magic(ρ::Matrix{ComplexF64}, N::Int) = get_stabilizer_renyi_entropy(ρ; n=2)

"""
    is_stabilizer_state(ψ_or_ρ, N::Int; tol::Float64=1e-10) -> Bool

True if M₂ ≈ 0 (state is a stabilizer state).
"""
is_stabilizer_state(ψ::Vector{ComplexF64}, N::Int; tol=1e-10) = abs(get_stabilizer_renyi_entropy(ψ)) < tol
is_stabilizer_state(ρ::Matrix{ComplexF64}, N::Int; tol=1e-10) = abs(get_stabilizer_renyi_entropy(ρ)) < tol

"""
    sre_summary(ψ::Vector{ComplexF64}, N::Int; max_n::Int=4)

Print M₂, M₃, ... Mₘₐₓ for diagnostic purposes.
"""
function sre_summary(ψ::Vector{ComplexF64}, N::Int; max_n::Int=4)
    println("═" ^ 50)
    println("  Stabilizer Rényi Entropy Summary")
    println("═" ^ 50)
    println("  N = $N qubits, 4^N = $(4^N) Pauli strings")
    println()
    for n in 2:max_n
        M_n = get_stabilizer_renyi_entropy(ψ; n=n)
        status = abs(M_n) < 1e-10 ? "[stabilizer]" : "[magic]"
        println("    M_$n = $(round(M_n; digits=6))  ($status)")
    end
    println("═" ^ 50)
end

end # module CPUStabilizerRenyiEntropyBruteForce

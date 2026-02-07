#!/usr/bin/env julia
"""
run_benchmark_smoq_Float64.jl - SmoQ.jl Baseline Benchmark (ComplexF64)

╔══════════════════════════════════════════════════════════════════════╗
║  BENCHMARK: SmoQ.jl Random Circuit Sampling — Float64 Baseline     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  This is the BASELINE SmoQ.jl implementation using ComplexF64       ║
║  (standard double precision complex numbers). It serves as the      ║
║  reference against which the optimized Float32-AVX variant is       ║
║  compared for internal speedup analysis.                            ║
║                                                                     ║
║  PRECISION: Float64 (ComplexF64)                                    ║
║  • 64-bit real + 64-bit imaginary = 128 bits per amplitude          ║
║  • Machine epsilon: ~2.2e-16                                        ║
║  • Full double precision — no accuracy concerns                     ║
║                                                                     ║
║  OPTIMIZATIONS IN THIS VERSION:                                     ║
║  1. Gate Fusion — consecutive single-qubit gates on the same qubit  ║
║     are multiplied into a single 2×2 unitary matrix, reducing the   ║
║     number of passes over the state vector.                         ║
║  2. @simd annotation — hints the Julia compiler to use SIMD         ║
║     vectorization in the inner gate-application loop.               ║
║  3. @inbounds — removes bounds checking in tight loops.             ║
║  4. Multi-threaded shots via Threads.@threads                       ║
║                                                                     ║
║  WHAT THIS VERSION DOES NOT DO:                                     ║
║  • No LoopVectorization.jl @turbo macro (→ Float32 AVX version)     ║
║  • No separated real/imaginary arrays (uses Complex struct)         ║
║  • No Float32 reduced precision                                     ║
║                                                                     ║
║  ARCHITECTURE:                                                      ║
║  The state vector is represented as Vector{ComplexF64} of length    ║
║  2^N.  Each gate is applied by iterating over (block, offset)       ║
║  pairs that identify the (|0⟩, |1⟩) amplitude pairs affected by    ║
║  the gate's target qubit.                                           ║
║                                                                     ║
║  TASK: Random circuit sampling                                      ║
║  • Generate N_GATES random gates (70% single-qubit, 30% two-qubit) ║
║  • Apply to |00...0⟩ state, then projective-measure all qubits     ║
║  • Repeat for N_SHOTS independent shots                             ║
║  • Time the full (circuit + measure) operation                      ║
║                                                                     ║
║  Usage: julia --threads=4 run_benchmark_smoq_Float64.jl             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

using LinearAlgebra
using Printf
using Statistics
using Dates

# ==============================================================================
# PATHS — Use local utils folder within the benchmark directory
# ==============================================================================

SCRIPT_DIR = @__DIR__
OUTPUT_DIR = SCRIPT_DIR
UTILS_CPU = joinpath(SCRIPT_DIR, "utils", "cpu")

# Include SmoQ's native gate implementations (CNOT, CZ use SmoQ's bit-masking)
include(joinpath(UTILS_CPU, "cpuQuantumChannelGates.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateMeasurements.jl"))
using .CPUQuantumStateMeasurements: projective_measurement_all!
using .CPUQuantumChannelGates: apply_cnot_psi!, apply_cz_psi!

# ==============================================================================
# CONFIGURATION — Benchmark parameters
# ==============================================================================

const N_SHOTS = 100           # Number of independent measurement shots
const N_RANGE = 2:2:20        # Qubit counts to benchmark (2, 4, 6, ..., 20)
const N_WARMUP = 1            # JIT warmup runs (first run compiles; discard timing)
const CIRCUIT_DIR = joinpath(SCRIPT_DIR, "circuits")

# ==============================================================================
# GATE MATRICES — Standard quantum gate unitaries (ComplexF64)
#
# All gates are stored as 2×2 ComplexF64 matrices. Fixed gates are const
# globals to avoid repeated allocation. Rotation gates are computed on-the-fly.
# ==============================================================================

const H_MAT = ComplexF64[1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)]  # Hadamard
const X_MAT = ComplexF64[0 1; 1 0]                                      # Pauli-X (NOT)
const Y_MAT = ComplexF64[0 -im; im 0]                                   # Pauli-Y
const Z_MAT = ComplexF64[1 0; 0 -1]                                     # Pauli-Z
const S_MAT = ComplexF64[1 0; 0 im]                                     # Phase (S)
const T_MAT = ComplexF64[1 0; 0 exp(im*π/4)]                            # π/8 (T)

# Rotation gates: parameterized by angle θ
@inline rx_mat(θ::Float64) = ComplexF64[cos(θ/2) -im*sin(θ/2); -im*sin(θ/2) cos(θ/2)]
@inline ry_mat(θ::Float64) = ComplexF64[cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
@inline rz_mat(θ::Float64) = ComplexF64[exp(-im*θ/2) 0; 0 exp(im*θ/2)]

# ==============================================================================
# GATE TYPES — Enum for fast dispatch; avoids string comparisons
# ==============================================================================

@enum GateType GATE_H=1 GATE_X=2 GATE_Y=3 GATE_Z=4 GATE_S=5 GATE_T=6 GATE_RX=7 GATE_RY=8 GATE_RZ=9 GATE_CNOT=10 GATE_CZ=11

const SINGLE_QUBIT_GATES = [GATE_H, GATE_X, GATE_Y, GATE_Z, GATE_S, GATE_T, GATE_RX, GATE_RY, GATE_RZ]
const TWO_QUBIT_GATES = [GATE_CNOT, GATE_CZ]

# GateOp: raw gate descriptor from the random circuit generator
# • q2 == 0  →  single-qubit gate on q1
# • q2 != 0  →  two-qubit gate with q1 = control, q2 = target
struct GateOp
    gate_type::GateType
    q1::Int
    q2::Int       # 0 for single-qubit gates
    theta::Float64
end

# ==============================================================================
# FUSED OPERATIONS
#
# Gate fusion is one of the key optimizations in this baseline:
# Instead of applying, e.g., H → Rz → T on qubit 3 as three separate
# sweeps over the 2^N state vector, we merge them into a single 2×2 unitary
# U_fused = T · Rz · H and apply it in ONE pass.
#
# The fusion is "greedy": we accumulate single-qubit gates on a per-qubit 
# basis, and flush accumulated gates whenever a two-qubit gate touches that 
# qubit (because two-qubit gates cannot be fused into a 2×2 matrix).
#
# For circuits with ~70% single-qubit gates, this typically reduces
# the number of state-vector sweeps by 3-5×.
# ==============================================================================

struct FusedOp
    is_single::Bool           # true = fused single-qubit 2×2 unitary
    qubit::Int                # target qubit (1-indexed)
    control::Int              # control qubit for CNOT/CZ (0 if single-qubit)
    U::Matrix{ComplexF64}     # 2×2 fused unitary matrix
    gate_type::GateType       # gate type (only relevant for two-qubit gates)
end

@inline function get_gate_matrix(gate_type::GateType, theta::Float64)
    gate_type == GATE_H  && return H_MAT
    gate_type == GATE_X  && return X_MAT
    gate_type == GATE_Y  && return Y_MAT
    gate_type == GATE_Z  && return Z_MAT
    gate_type == GATE_S  && return S_MAT
    gate_type == GATE_T  && return T_MAT
    gate_type == GATE_RX && return rx_mat(theta)
    gate_type == GATE_RY && return ry_mat(theta)
    gate_type == GATE_RZ && return rz_mat(theta)
    return ComplexF64[1 0; 0 1]
end

"""
    fuse_gates(ops::Vector{GateOp}) → Vector{FusedOp}

Fuse consecutive single-qubit gates on the same qubit into a single 2×2 unitary.

Algorithm:
  1. Maintain a `pending` dictionary: qubit → accumulated 2×2 unitary
  2. For each single-qubit gate: multiply into pending[qubit]
  3. For each two-qubit gate: flush pending entries for both qubits first,
     then emit the two-qubit gate as-is
  4. At the end, flush all remaining pending entries

This reduces N_GATES ≈ 1000 raw operations to ~200-400 fused operations
(depending on the circuit structure), cutting state-vector sweeps significantly.
"""
function fuse_gates(ops::Vector{GateOp})
    fused = FusedOp[]
    pending = Dict{Int, Matrix{ComplexF64}}()
    
    function flush_qubit!(q::Int)
        if haskey(pending, q)
            push!(fused, FusedOp(true, q, 0, pending[q], GATE_H))
            delete!(pending, q)
        end
    end
    
    for op in ops
        if op.q2 == 0  # Single-qubit gate
            U = get_gate_matrix(op.gate_type, op.theta)
            if haskey(pending, op.q1)
                # Matrix multiply: U_new * U_accumulated  (gate composition)
                pending[op.q1] = U * pending[op.q1]
            else
                pending[op.q1] = copy(U)
            end
        else  # Two-qubit gate — flush both qubits first
            flush_qubit!(op.q1)
            flush_qubit!(op.q2)
            push!(fused, FusedOp(false, op.q2, op.q1, ComplexF64[1 0; 0 1], op.gate_type))
        end
    end
    
    # Flush remaining accumulated gates
    for q in sort(collect(keys(pending)))
        flush_qubit!(q)
    end
    
    return fused
end

# ==============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │             BITWISE MATRIX-FREE GATE APPLICATION                        │
# │                                                                         │
# │  This is the core innovation of SmoQ's statevector simulator.           │
# │  Gates are applied WITHOUT constructing the full 2^N × 2^N unitary.    │
# │  Instead, we exploit the tensor-product structure of the Hilbert space  │
# │  to apply each gate as a sequence of 2×2 operations on amplitude       │
# │  pairs, selected via BITWISE ARITHMETIC on the state index.            │
# └──────────────────────────────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════
# 1. STATE VECTOR ENCODING (Little-Endian Bit Convention)
# ═══════════════════════════════════════════════════════════════════════════
#
#   The state |qN ... q2 q1⟩ is stored at index:
#     idx = q1·2⁰ + q2·2¹ + ... + qN·2^(N-1)
#
#   Example (N=3):  |q3 q2 q1⟩
#     |000⟩ → idx=0,  |001⟩ → idx=1,  |010⟩ → idx=2,  |011⟩ → idx=3
#     |100⟩ → idx=4,  |101⟩ → idx=5,  |110⟩ → idx=6,  |111⟩ → idx=7
#
#   Qubit k (1-indexed) controls bit position (k-1) in the index.
#   To read qubit k's value:  bit_k = (idx >> (k-1)) & 1
#
# ═══════════════════════════════════════════════════════════════════════════
# 2. SINGLE-QUBIT GATE APPLICATION (Block-Offset Iteration)
# ═══════════════════════════════════════════════════════════════════════════
#
#   A single-qubit gate U acts on qubit q. In the full 2^N space, this is
#   equivalent to  I ⊗ ... ⊗ U ⊗ ... ⊗ I  (U at position q).
#
#   The MATRIX-FREE trick: we don't build this 2^N × 2^N matrix.
#   Instead, we identify all PAIRS of amplitudes that differ ONLY in bit q:
#     (ψ[i], ψ[j])  where j = i + 2^(q-1)  and bit q of i is 0
#
#   Each pair forms a 2D subproblem:
#     |new_ψ[i]|     |u11 u12|   |ψ[i]|
#     |new_ψ[j]|  =  |u21 u22| × |ψ[j]|    (2×2 matrix-vector multiply)
#
#   There are 2^(N-1) such pairs, and they are independent — this is
#   what makes the operation O(2^N) instead of O(4^N) matrix multiply.
#
#   MEMORY ACCESS PATTERN (Block-Offset):
#   We iterate in blocks of size 2^q. Within each block:
#     - First 2^(q-1) entries have bit q = 0  (the |0⟩ subspace)
#     - Next  2^(q-1) entries have bit q = 1  (the |1⟩ subspace)
#
#   step      = 2^(q-1)        = distance between paired amplitudes
#   block_size = 2^q            = 2 × step
#   n_blocks   = 2^N / 2^q     = number of non-contiguous blocks
#
#   Visualization for N=4, gate on qubit 2 (step=2, block_size=4):
#   ┌────────────────────────────────────────────────────────────────────┐
#   │ Block 0        │ Block 1        │ Block 2        │ Block 3        │
#   │ idx: 0 1 2 3   │ idx: 4 5 6 7   │ idx: 8 9 10 11 │ idx: 12-15    │
#   │ bit2: 0 0 1 1  │ bit2: 0 0 1 1  │ bit2: 0 0 1  1 │ bit2: 0 0 1 1 │
#   │ pair: ↕ ↕       │ pair: ↕ ↕       │ pair: ↕ ↕       │ pair: ↕ ↕     │
#   │ (0,2)(1,3)      │ (4,6)(5,7)      │ (8,10)(9,11)   │ (12,14)(13,15)│
#   └────────────────────────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════
# 3. TWO-QUBIT GATES (Bit-Mask Iteration)
# ═══════════════════════════════════════════════════════════════════════════
#
#   CNOT(control, target): flip target qubit IF control qubit is |1⟩.
#   Implementation: for each index i where bit_control=1 AND bit_target=0,
#   swap ψ[i] ↔ ψ[j] where j = i XOR target_mask  (flips target bit).
#
#   CZ(q1, q2): apply -1 phase when BOTH qubits are |1⟩.
#   Implementation: for each index i where bit_q1=1 AND bit_q2=1,
#   multiply ψ[i] by -1. This is branchless: sign = 1 - 2·(b1 & b2).
#
#   Both are O(2^N) — we scan the state vector ONCE, testing bits.
#   No matrix construction, no Kronecker products.
#
# ═══════════════════════════════════════════════════════════════════════════
# 4. WHY THIS IS "MATRIX-FREE"
# ═══════════════════════════════════════════════════════════════════════════
#
#   Traditional approach: construct U_full = I⊗...⊗U⊗...⊗I (2^N × 2^N),
#   then multiply ψ_new = U_full × ψ.  Cost: O(4^N) storage, O(4^N) flops.
#
#   Matrix-free approach: apply U as 2^(N-1) independent 2×2 multiply ops.
#   Cost: O(2^N) storage (state vector only), O(2^N) flops per gate.
#
#   The speedup is exponential: we never store or compute with the full
#   unitary matrix. The gate's action is encoded entirely in the bit
#   arithmetic that selects amplitude pairs.
#
# ═══════════════════════════════════════════════════════════════════════════
# 5. SIMD VECTORIZATION (@simd)
# ═══════════════════════════════════════════════════════════════════════════
#
#   The inner offset loop iterates over CONTIGUOUS memory within each block.
#   @simd hints the compiler to vectorize the multiply-accumulate.
#   With ComplexF64, each amplitude is 16 bytes → AVX-256 fits 1 complex
#   number (2 Float64). This is why Float32 + separated arrays is faster:
#   it fits 8 values per AVX register instead of 1.
#
# ==============================================================================

"""
    apply_fused_gate!(ψ, U, q)

Apply a 2×2 unitary matrix U to qubit q of state vector ψ.

The state vector has 2^N entries. For qubit q:
- step = 2^(q-1): distance between paired |0⟩ and |1⟩ amplitudes
- block_size = 2^q: total pairs in one contiguous block
- n_blocks = 2^N / 2^q: number of such blocks

For each (i, j = i + step) pair:
  ψ[i] ← u11·ψ[i] + u12·ψ[j]
  ψ[j] ← u21·ψ[i] + u22·ψ[j]
"""
@inline function apply_fused_gate!(ψ::Vector{ComplexF64}, U::Matrix{ComplexF64}, q::Int)
    dim = length(ψ)
    step = 1 << (q - 1)
    block_size = step << 1
    n_blocks = dim >> q
    
    # Extract matrix elements to local variables (avoids repeated array access)
    u11, u12 = U[1,1], U[1,2]
    u21, u22 = U[2,1], U[2,2]
    
    @inbounds for block in 0:(n_blocks-1)
        base = block * block_size
        @simd for offset in 0:(step-1)
            i = base + offset + 1
            j = i + step
            a = ψ[i]   # |0⟩ amplitude
            b = ψ[j]   # |1⟩ amplitude
            ψ[i] = u11 * a + u12 * b
            ψ[j] = u21 * a + u22 * b
        end
    end
end

"""
    run_fused_circuit!(ψ, fused_ops, N)

Execute all fused gates on state vector ψ.

Dispatch logic:
- Single-qubit fused gates → apply_fused_gate! (block-offset 2×2 multiply)
- CNOT → apply_cnot_psi! (bit-mask swap: for each idx with control=1 & target=0,
  swap ψ[idx] ↔ ψ[idx ⊻ target_mask])
- CZ → apply_cz_psi! (branchless conditional negate: ψ[idx] *= 1 - 2·(b1 & b2))

All operations are O(2^N) per gate — no matrix construction.
The CNOT/CZ implementations come from cpuQuantumChannelGates.jl which uses
SmoQ's standard bitwise gate protocol.
"""
function run_fused_circuit!(ψ::Vector{ComplexF64}, fused_ops::Vector{FusedOp}, N::Int)
    @inbounds for op in fused_ops
        if op.is_single
            apply_fused_gate!(ψ, op.U, op.qubit)
        else
            if op.gate_type == GATE_CNOT
                apply_cnot_psi!(ψ, op.control, op.qubit, N)
            else
                apply_cz_psi!(ψ, op.control, op.qubit, N)
            end
        end
    end
    return ψ
end

# ==============================================================================
# CIRCUIT LOADING — Load shared circuits from circuits/ directory
#
# All frameworks load the EXACT SAME circuits (generated by generate_circuits.py
# with seed=42), ensuring a perfectly fair comparison.
# ==============================================================================

const GATE_NAME_MAP = Dict(
    "H" => GATE_H, "X" => GATE_X, "Y" => GATE_Y, "Z" => GATE_Z,
    "S" => GATE_S, "T" => GATE_T, "RX" => GATE_RX, "RY" => GATE_RY,
    "RZ" => GATE_RZ, "CNOT" => GATE_CNOT, "CZ" => GATE_CZ,
)

function load_circuit_from_file(N::Int)
    filepath = joinpath(CIRCUIT_DIR, @sprintf("circuit_N%02d.txt", N))
    ops = GateOp[]
    for line in eachline(filepath)
        line = strip(line)
        (isempty(line) || startswith(line, "#")) && continue
        parts = split(line)
        gate_type = GATE_NAME_MAP[parts[1]]
        q1 = parse(Int, parts[2]) + 1  # 0-indexed → 1-indexed
        if gate_type in (GATE_CNOT, GATE_CZ)
            q2 = parse(Int, parts[3]) + 1
            push!(ops, GateOp(gate_type, q1, q2, 0.0))
        elseif gate_type in (GATE_RX, GATE_RY, GATE_RZ)
            theta = parse(Float64, parts[3])
            push!(ops, GateOp(gate_type, q1, 0, theta))
        else
            push!(ops, GateOp(gate_type, q1, 0, 0.0))
        end
    end
    return ops
end

"""
    run_circuit_sampling(N, ops, n_shots)

Run the full benchmark task:
1. Fuse the circuit gates (done once)
2. Simulate the circuit ONCE to obtain the final state |ψ⟩
3. Born-sample n_shots bitstrings from P(i) = |ψ[i]|²
"""
function run_circuit_sampling(N::Int, ops::Vector{GateOp}, n_shots::Int)
    fused_ops = fuse_gates(ops)

    dim = 1 << N
    # --- Simulate the circuit ONCE ---
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1.0       # |00...0⟩ initial state
    run_fused_circuit!(ψ, fused_ops, N)

    # --- Born-rule sampling from |ψ|² ---
    probs = abs2.(ψ)
    cumprobs = cumsum(probs)
    results = Vector{Vector{Int}}(undef, n_shots)
    for shot in 1:n_shots
        r = rand()
        idx = searchsortedfirst(cumprobs, r) - 1  # 0-indexed bitstring
        bits = Vector{Int}(undef, N)
        for q in 1:N
            bits[q] = (idx >> (q - 1)) & 1
        end
        results[shot] = bits
    end
    return results
end

# ==============================================================================
# BENCHMARK EXECUTION
# ==============================================================================

println("=" ^ 70)
println("  SmoQ.jl — Random Circuit Sampling Benchmark")
println("=" ^ 70)
println("\n  Shots: $N_SHOTS")
println("  Qubit range: $(first(N_RANGE)):$(step(N_RANGE)):$(last(N_RANGE))")
println("  Precision: ComplexF64")
println("  Optimization: Gate Fusion + @simd")
println("  Circuits: shared (seed=42, from circuits/)")

timings = Dict{Int, Float64}()

println("\n     N  │    Time (s)")
println("  ──────┼──────────────")

for N in N_RANGE
    ops = load_circuit_from_file(N)

    # Warmup: first run triggers Julia's JIT compilation
    for _ in 1:N_WARMUP
        run_circuit_sampling(N, ops, N_SHOTS)
    end

    # Actual benchmark run
    t = @elapsed run_circuit_sampling(N, ops, N_SHOTS)
    timings[N] = t

    @printf("    %2d  │  %10.4f\n", N, t)
    flush(stdout)
end

# Save results
timing_file = joinpath(OUTPUT_DIR, "timings_smoq.txt")
open(timing_file, "w") do f
    println(f, "# SmoQ.jl benchmark - $(Dates.now())")
    println(f, "# Precision: ComplexF64, Optimization: Gate Fusion + @simd")
    println(f, "# $N_SHOTS shots, shared circuits (seed=42)")
    println(f, "# N\tTime_s")
    for N in N_RANGE
        @printf(f, "%d\t%.6f\n", N, timings[N])
    end
end
println("\nSaved: ", basename(timing_file))

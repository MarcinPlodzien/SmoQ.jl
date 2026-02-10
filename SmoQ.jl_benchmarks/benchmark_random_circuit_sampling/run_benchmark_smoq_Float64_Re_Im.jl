#!/usr/bin/env julia
"""
run_benchmark_smoq_Float64_Re_Im.jl - SmoQ.jl with Float64 separated Re/Im

╔══════════════════════════════════════════════════════════════════════╗
║  BENCHMARK: SmoQ.jl — Float64 with Separated Re/Im Arrays          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  This variant enables us to ISOLATE the effect of the memory layout ║
║  optimization (separated re/im arrays) from the Float32+@turbo     ║
║  optimization.                                                      ║
║                                                                     ║
║  Compared to the Float64 baseline:                                  ║
║  • SAME precision: Float64 (no accuracy loss)                       ║
║  • DIFFERENT layout: ψ_re::Vector{Float64}, ψ_im::Vector{Float64}  ║
║    instead of ψ::Vector{ComplexF64}                                 ║
║  • SAME vectorization hint: @simd (not @turbo, which needs Float32) ║
║                                                                     ║
║  WHY @turbo IS NOT USED HERE:                                       ║
║  LoopVectorization.jl's @turbo works with Float64 too, BUT the     ║
║  main win of @turbo is its ability to pack more values per SIMD     ║
║  register. With Float64, AVX-256 fits 4 values (vs 8 for Float32), ║
║  so the gain over Julia's built-in @simd is less dramatic.          ║
║  We keep @simd to show what separated re/im alone does.             ║
║                                                                     ║
║  EXPECTED RESULT:                                                   ║
║  Modest speedup over baseline from better cache-line utilization    ║
║  and more regular memory access patterns. The separated layout      ║
║  eliminates stride-2 access to real parts.                          ║
║                                                                     ║
║  Usage: julia --threads=4 run_benchmark_smoq_Float64_Re_Im.jl      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

using LinearAlgebra
using Printf
using Statistics
using Dates
using Random

# ==============================================================================
# CONFIGURATION
# ==============================================================================

const N_GATES = 1000
const N_SHOTS = 100
const N_RANGE = 2:2:20
const N_WARMUP = 1
const SINGLE_GATE_RATIO = 0.7

SCRIPT_DIR = @__DIR__
OUTPUT_DIR = SCRIPT_DIR

# ==============================================================================
# GATE MATRICES — Stored as SEPARATED Float64 Real/Imaginary arrays
# ==============================================================================

const H_RE = Float64[1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)]
const H_IM = zeros(Float64, 2, 2)
const X_RE = Float64[0 1; 1 0]
const X_IM = zeros(Float64, 2, 2)
const Y_RE = Float64[0 0; 0 0]
const Y_IM = Float64[0 -1; 1 0]
const Z_RE = Float64[1 0; 0 -1]
const Z_IM = zeros(Float64, 2, 2)
const S_RE = Float64[1 0; 0 0]
const S_IM = Float64[0 0; 0 1]
const T_RE = Float64[1 0; 0 cos(π/4)]
const T_IM = Float64[0 0; 0 sin(π/4)]

# ==============================================================================
# GATE TYPES
# ==============================================================================

@enum GateType GATE_H=1 GATE_X=2 GATE_Y=3 GATE_Z=4 GATE_S=5 GATE_T=6 GATE_RX=7 GATE_RY=8 GATE_RZ=9 GATE_CNOT=10 GATE_CZ=11

const SINGLE_QUBIT_GATES = [GATE_H, GATE_X, GATE_Y, GATE_Z, GATE_S, GATE_T, GATE_RX, GATE_RY, GATE_RZ]
const TWO_QUBIT_GATES = [GATE_CNOT, GATE_CZ]

struct GateOp
    gate_type::GateType
    q1::Int
    q2::Int
    theta::Float64
end

struct FusedOp
    is_single::Bool
    qubit::Int
    control::Int
    U_re::Matrix{Float64}
    U_im::Matrix{Float64}
    gate_type::GateType
end

# ==============================================================================
# GATE RETRIEVAL (separated re/im, Float64)
# ==============================================================================

function get_gate_matrices(gate_type::GateType, theta::Float64)
    if gate_type == GATE_H;  return H_RE, H_IM
    elseif gate_type == GATE_X; return X_RE, X_IM
    elseif gate_type == GATE_Y; return Y_RE, Y_IM
    elseif gate_type == GATE_Z; return Z_RE, Z_IM
    elseif gate_type == GATE_S; return S_RE, S_IM
    elseif gate_type == GATE_T; return T_RE, T_IM
    elseif gate_type == GATE_RX
        c, s = cos(theta/2), sin(theta/2)
        return Float64[c 0; 0 c], Float64[0 -s; -s 0]
    elseif gate_type == GATE_RY
        c, s = cos(theta/2), sin(theta/2)
        return Float64[c -s; s c], zeros(Float64, 2, 2)
    elseif gate_type == GATE_RZ
        c, s = cos(theta/2), sin(theta/2)
        return Float64[c 0; 0 c], Float64[-s 0; 0 s]
    end
    return Float64[1 0; 0 1], zeros(Float64, 2, 2)
end

# ==============================================================================
# GATE FUSION (same logic, Float64 re/im)
# ==============================================================================

function complex_matmul(A_re, A_im, B_re, B_im)
    C_re = A_re * B_re - A_im * B_im
    C_im = A_re * B_im + A_im * B_re
    return C_re, C_im
end

function fuse_gates(ops::Vector{GateOp})
    fused = FusedOp[]
    pending_re = Dict{Int, Matrix{Float64}}()
    pending_im = Dict{Int, Matrix{Float64}}()
    
    function flush_qubit!(q::Int)
        if haskey(pending_re, q)
            push!(fused, FusedOp(true, q, 0, pending_re[q], pending_im[q], GATE_H))
            delete!(pending_re, q)
            delete!(pending_im, q)
        end
    end
    
    for op in ops
        if op.q2 == 0
            U_re, U_im = get_gate_matrices(op.gate_type, op.theta)
            if haskey(pending_re, op.q1)
                pending_re[op.q1], pending_im[op.q1] = complex_matmul(
                    U_re, U_im, pending_re[op.q1], pending_im[op.q1])
            else
                pending_re[op.q1] = copy(U_re)
                pending_im[op.q1] = copy(U_im)
            end
        else
            flush_qubit!(op.q1)
            flush_qubit!(op.q2)
            push!(fused, FusedOp(false, op.q2, op.q1,
                Float64[1 0; 0 1], zeros(Float64, 2, 2), op.gate_type))
        end
    end
    
    for q in sort(collect(keys(pending_re)))
        flush_qubit!(q)
    end
    
    return fused
end

# ==============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │             BITWISE MATRIX-FREE GATE APPLICATION                        │
# │                                                                         │
# │  Gates are applied WITHOUT constructing the full 2^N × 2^N unitary.    │
# │  We exploit the tensor-product structure: each single-qubit gate       │
# │  becomes 2^(N-1) independent 2×2 operations on amplitude pairs,       │
# │  selected via BITWISE ARITHMETIC on the state index.                   │
# └──────────────────────────────────────────────────────────────────────────┘
#
# STATE VECTOR ENCODING (Little-Endian):
#   |qN...q2 q1⟩ stored at idx = q1·2⁰ + q2·2¹ + ... + qN·2^(N-1)
#   Qubit k (1-indexed) → bit position (k-1).  Read: bit_k = (idx >> (k-1)) & 1
#
# SINGLE-QUBIT GATE on qubit q:
#   Find all pairs (ψ[i], ψ[j]) where j = i + 2^(q-1) and bit q of i is 0.
#   Apply 2×2 unitary to each pair independently — O(2^N) total.
#
#   Iteration uses block-offset pattern:
#     step = 2^(q-1),  block_size = 2^q,  n_blocks = 2^N / 2^q
#     Inner loop: offset ∈ 0:(step-1) → contiguous memory → SIMD-friendly
#
# TWO-QUBIT GATES (CNOT, CZ):
#   CNOT: for each idx with bit_control=1 & bit_target=0,
#         swap ψ[idx] ↔ ψ[idx ⊻ target_mask]
#   CZ: for each idx with bit_q1=1 & bit_q2=1,
#       ψ[idx] *= -1  (branchless: sign = 1 - 2·(b1 & b2))
#
# WHY MATRIX-FREE:
#   Traditional: build I⊗...⊗U⊗...⊗I (2^N × 2^N), multiply → O(4^N)
#   Matrix-free: 2^(N-1) independent 2×2 ops → O(2^N) per gate
#
# SEPARATED RE/IM ADVANTAGE:
#   With ψ_re[] and ψ_im[] as separate Float64 arrays, the inner loop
#   loads contiguous real values without stride-2 interleaving. This
#   improves cache-line utilization and enables cleaner SIMD patterns.
#
# ==============================================================================
# GATE APPLICATION — Separated Float64 arrays with @simd
#
# This is the key comparison point: same @simd as baseline, but with
# separated re/im arrays. The compiler can now load contiguous Float64
# values without interleaved re/im pairs.
# ==============================================================================

@inline function apply_gate_simd!(ψ_re::Vector{Float64}, ψ_im::Vector{Float64},
                                   U_re::Matrix{Float64}, U_im::Matrix{Float64}, q::Int)
    dim = length(ψ_re)
    step = 1 << (q - 1)
    block_size = step << 1
    n_blocks = dim >> q
    
    u00_re, u01_re = U_re[1,1], U_re[1,2]
    u10_re, u11_re = U_re[2,1], U_re[2,2]
    u00_im, u01_im = U_im[1,1], U_im[1,2]
    u10_im, u11_im = U_im[2,1], U_im[2,2]
    
    @inbounds for block in 0:(n_blocks-1)
        base = block * block_size
        @simd for offset in 0:(step-1)
            i = base + offset + 1
            j = i + step
            
            a_re = ψ_re[i]; a_im = ψ_im[i]
            b_re = ψ_re[j]; b_im = ψ_im[j]
            
            ψ_re[i] = (u00_re * a_re - u00_im * a_im) + (u01_re * b_re - u01_im * b_im)
            ψ_im[i] = (u00_re * a_im + u00_im * a_re) + (u01_re * b_im + u01_im * b_re)
            ψ_re[j] = (u10_re * a_re - u10_im * a_im) + (u11_re * b_re - u11_im * b_im)
            ψ_im[j] = (u10_re * a_im + u10_im * a_re) + (u11_re * b_im + u11_im * b_re)
        end
    end
end

"""
    apply_cnot!(ψ_re, ψ_im, control, target)

CNOT gate via bit-mask iteration (separated re/im Float64 variant).

For each basis state index i where control bit is SET and target bit is CLEAR:
  j = i ⊻ target_mask   (XOR flips the target bit)
  swap ψ[i] ↔ ψ[j]     (both re and im parts)

The condition `target bit == 0` avoids double-swapping the same pair.
O(2^N) scan — no matrix construction.
"""
function apply_cnot!(ψ_re::Vector{Float64}, ψ_im::Vector{Float64}, control::Int, target::Int)
    dim = length(ψ_re)
    control_mask = 1 << (control - 1)
    target_mask = 1 << (target - 1)
    @inbounds for i in 0:(dim-1)
        if (i & control_mask) != 0 && (i & target_mask) == 0
            j = i ⊻ target_mask
            ψ_re[i+1], ψ_re[j+1] = ψ_re[j+1], ψ_re[i+1]
            ψ_im[i+1], ψ_im[j+1] = ψ_im[j+1], ψ_im[i+1]
        end
    end
end

"""
    apply_cz!(ψ_re, ψ_im, q1, q2)

CZ gate via branchless bit-mask iteration (separated re/im Float64 variant).

For each index where bit_q1=1 AND bit_q2=1, negate the amplitude.
Uses @simd with conditional branch — the compiler may auto-vectorize.
O(2^N) scan — no matrix construction.
"""
function apply_cz!(ψ_re::Vector{Float64}, ψ_im::Vector{Float64}, q1::Int, q2::Int)
    dim = length(ψ_re)
    mask1 = 1 << (q1 - 1)
    mask2 = 1 << (q2 - 1)
    @inbounds @simd for i in 0:(dim-1)
        if ((i & mask1) != 0) && ((i & mask2) != 0)
            ψ_re[i+1] = -ψ_re[i+1]
            ψ_im[i+1] = -ψ_im[i+1]
        end
    end
end

"""
    run_fused_circuit!(ψ_re, ψ_im, fused_ops)

Execute all fused gates on the separated Float64 state vector.

Dispatch logic:
- Single-qubit fused gates → apply_gate_simd! (block-offset 2×2 multiply)
- CNOT → apply_cnot! (bit-mask swap: idx where control=1 & target=0)
- CZ → apply_cz! (conditional negate: idx where both qubits are |1⟩)

All operations are O(2^N) per gate — no matrix construction.
"""
function run_fused_circuit!(ψ_re::Vector{Float64}, ψ_im::Vector{Float64},
                           fused_ops::Vector{FusedOp})
    @inbounds for op in fused_ops
        if op.is_single
            apply_gate_simd!(ψ_re, ψ_im, op.U_re, op.U_im, op.qubit)
        else
            if op.gate_type == GATE_CNOT
                apply_cnot!(ψ_re, ψ_im, op.control, op.qubit)
            else
                apply_cz!(ψ_re, ψ_im, op.control, op.qubit)
            end
        end
    end
end

# ==============================================================================
# CIRCUIT GENERATION & SAMPLING
# ==============================================================================

function generate_random_circuit(N::Int, n_gates::Int)
    ops = Vector{GateOp}(undef, n_gates)
    @inbounds for i in 1:n_gates
        if rand() < SINGLE_GATE_RATIO || N == 1
            gate = rand(SINGLE_QUBIT_GATES)
            q = rand(1:N)
            theta = (gate in [GATE_RX, GATE_RY, GATE_RZ]) ? rand() * 2π : 0.0
            ops[i] = GateOp(gate, q, 0, theta)
        else
            gate = rand(TWO_QUBIT_GATES)
            q1 = rand(1:N)
            q2 = rand(1:N-1)
            q2 = q2 >= q1 ? q2 + 1 : q2
            ops[i] = GateOp(gate, q1, q2, 0.0)
        end
    end
    return ops
end

function sample_measurement(ψ_re::Vector{Float64}, ψ_im::Vector{Float64})
    r = rand()
    cumsum = 0.0
    @inbounds for i in 1:length(ψ_re)
        prob = ψ_re[i]^2 + ψ_im[i]^2
        cumsum += prob
        if r < cumsum
            return i - 1
        end
    end
    return length(ψ_re) - 1
end

function run_circuit_sampling(N::Int, ops::Vector{GateOp}, n_shots::Int)
    fused_ops = fuse_gates(ops)
    dim = 1 << N

    # --- Simulate the circuit ONCE ---
    ψ_re = zeros(Float64, dim)
    ψ_im = zeros(Float64, dim)
    ψ_re[1] = 1.0
    run_fused_circuit!(ψ_re, ψ_im, fused_ops)

    # --- Born-rule sampling from |ψ|² ---
    probs = ψ_re.^2 .+ ψ_im.^2
    cumprobs = cumsum(probs)
    results = Vector{Int}(undef, n_shots)
    for shot in 1:n_shots
        r = rand()
        results[shot] = searchsortedfirst(cumprobs, r) - 1
    end
    return results
end

# ==============================================================================
# BENCHMARK
# ==============================================================================

println("=" ^ 70)
println("  SmoQ.jl Float64-Re_Im Benchmark (separated layout)")
println("=" ^ 70)
println("\n  Threads: $(Threads.nthreads())")
println("  Gates: $N_GATES, Shots: $N_SHOTS")
println("  Qubit range: $(first(N_RANGE)):$(step(N_RANGE)):$(last(N_RANGE))")
println("  Precision: Float64 (separated re/im)")
println("  Vectorization: @simd")

timings = Dict{Int, Float64}()

println("\n     N  │    Time (s)")
println("  ──────┼──────────────")

for N in N_RANGE
    for _ in 1:N_WARMUP
        ops = generate_random_circuit(N, N_GATES)
        run_circuit_sampling(N, ops, N_SHOTS)
    end
    
    ops = generate_random_circuit(N, N_GATES)
    t = @elapsed run_circuit_sampling(N, ops, N_SHOTS)
    timings[N] = t
    
    @printf("    %2d  │  %10.4f\n", N, t)
    flush(stdout)
end

timing_file = joinpath(OUTPUT_DIR, "timings_smoq_Float64_Re_Im.txt")
open(timing_file, "w") do f
    println(f, "# SmoQ.jl Float64-Re_Im benchmark - $(Dates.now())")
    println(f, "# Float64, separated re/im, @simd")
    println(f, "# $(Threads.nthreads()) threads, $N_GATES gates, $N_SHOTS shots")
    println(f, "# N\tTime_s")
    for N in N_RANGE
        @printf(f, "%d\t%.6f\n", N, timings[N])
    end
end
println("\nSaved: ", basename(timing_file))

#!/usr/bin/env julia
"""
bench_smoq.jl - SmoQ.jl (ComplexF64) timing sweep over N
"""

using LinearAlgebra
using Printf

SCRIPT_DIR = @__DIR__
WORKSPACE = dirname(dirname(dirname(SCRIPT_DIR)))
UTILS_CPU = joinpath(WORKSPACE, "utils", "cpu")

include(joinpath(UTILS_CPU, "cpuQuantumChannelGates.jl"))
using .CPUQuantumChannelGates: apply_hadamard_psi!, apply_s_psi!,
    apply_ry_psi!, apply_rx_psi!, apply_cnot_psi!, apply_cz_psi!

const L = 4
const ANGLES_RY = [π/4, π/3, π/6, π/5, π/7, π/8, π/9, π/10, π/11, π/12,
                   π/13, π/14, π/15, π/16, π/17, π/18, π/19, π/20, π/21, π/22, π/23, π/24, π/25, π/26]
const ANGLES_RX = [π/5, π/7, π/4, π/9, π/3, π/11, π/6, π/13, π/8, π/10,
                   π/12, π/14, π/15, π/17, π/16, π/19, π/18, π/20, π/21, π/22, π/23, π/24, π/25, π/26]
const N_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

function run_circuit(N::Int)
    dim = 1 << N
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1.0
    for layer in 1:L
        for q in 2:2:N;       apply_hadamard_psi!(ψ, q, N); end
        for q in 1:N;         apply_ry_psi!(ψ, q, ANGLES_RY[q], N); end
        for q in 1:2:(N-1);   apply_cnot_psi!(ψ, q, q+1, N); end
        for q in 1:2:N;       apply_s_psi!(ψ, q, N); end
        for q in 1:N;         apply_rx_psi!(ψ, q, ANGLES_RX[q], N); end
        for q in 2:2:(N-1);   apply_cz_psi!(ψ, q, q+1, N); end
    end
    return ψ
end

function _expect_Z(ψ, k, N)
    bp = k-1; r = 0.0
    @inbounds @simd for i in 0:(length(ψ)-1)
        r += abs2(ψ[i+1]) * (1 - 2*((i>>bp)&1))
    end; return r
end
function _expect_X(ψ, k, N)
    bp = k-1; step = 1<<bp; r = 0.0
    @inbounds for i in 0:(length(ψ)-1)
        if ((i>>bp)&1)==0; r += 2*real(conj(ψ[i+1])*ψ[i+step+1]); end
    end; return r
end
function _expect_Y(ψ, k, N)
    bp = k-1; step = 1<<bp; r = 0.0
    @inbounds for i in 0:(length(ψ)-1)
        if ((i>>bp)&1)==0; r += 2*imag(conj(ψ[i+1])*ψ[i+step+1]); end
    end; return r
end
function _expect_ZZ(ψ, i, j, N)
    bi=i-1; bj=j-1; r=0.0
    @inbounds @simd for s in 0:(length(ψ)-1)
        r += abs2(ψ[s+1]) * (1-2*xor((s>>bi)&1,(s>>bj)&1))
    end; return r
end
function _expect_XX(ψ, i, j, N)
    bi=i-1;bj=j-1;si=1<<bi;sj=1<<bj;r=0.0
    @inbounds for s in 0:(length(ψ)-1)
        if ((s>>bi)&1)==0 && ((s>>bj)&1)==0
            r += 2*real(conj(ψ[s+1])*ψ[s+si+sj+1]) + 2*real(conj(ψ[s+sj+1])*ψ[s+si+1])
        end
    end; return r
end
function _expect_YY(ψ, i, j, N)
    bi=i-1;bj=j-1;si=1<<bi;sj=1<<bj;r=0.0
    @inbounds for s in 0:(length(ψ)-1)
        if ((s>>bi)&1)==0 && ((s>>bj)&1)==0
            r += -2*real(conj(ψ[s+1])*ψ[s+si+sj+1]) + 2*real(conj(ψ[s+sj+1])*ψ[s+si+1])
        end
    end; return r
end

function compute_observables(ψ, N)
    for q in 1:N; _expect_X(ψ,q,N); _expect_Y(ψ,q,N); _expect_Z(ψ,q,N); end
    for b in 1:(N-1); _expect_XX(ψ,b,b+1,N); _expect_YY(ψ,b,b+1,N); _expect_ZZ(ψ,b,b+1,N); end
end

println("=" ^ 60)
println("  SmoQ.jl (ComplexF64) — Timing Sweep")
println("=" ^ 60)

ψ_w = run_circuit(2); compute_observables(ψ_w, 2)  # JIT warmup

output_file = joinpath(SCRIPT_DIR, "timing_smoq.txt")
open(output_file, "w") do f
    write(f, "# SmoQ.jl (ComplexF64) timing\n# N  time_s\n")
    for N_val in N_VALUES
        t = time_ns()
        ψ = run_circuit(N_val)
        compute_observables(ψ, N_val)
        dt = (time_ns() - t) / 1e9
        @printf(f, "%d  %.6f\n", N_val, dt)
        @printf("  N = %2d : %.6f s\n", N_val, dt)
    end
end
println("  Saved: ", basename(output_file))

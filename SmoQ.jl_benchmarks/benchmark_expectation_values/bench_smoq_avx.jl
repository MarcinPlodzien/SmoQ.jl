#!/usr/bin/env julia
"""
bench_smoq_avx.jl - SmoQ.jl AVX (Float32 Re/Im) timing sweep over N
"""

using LinearAlgebra
using Printf
using LoopVectorization

SCRIPT_DIR = @__DIR__

const L = 4
const ANGLES_RY = Float32[π/4, π/3, π/6, π/5, π/7, π/8, π/9, π/10, π/11, π/12,
                           π/13, π/14, π/15, π/16, π/17, π/18, π/19, π/20, π/21, π/22, π/23, π/24, π/25, π/26]
const ANGLES_RX = Float32[π/5, π/7, π/4, π/9, π/3, π/11, π/6, π/13, π/8, π/10,
                           π/12, π/14, π/15, π/17, π/16, π/19, π/18, π/20, π/21, π/22, π/23, π/24, π/25, π/26]
const N_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

function apply_gate_turbo!(ψ_re::Vector{Float32}, ψ_im::Vector{Float32},
                           u00_re::Float32, u00_im::Float32, u01_re::Float32, u01_im::Float32,
                           u10_re::Float32, u10_im::Float32, u11_re::Float32, u11_im::Float32, q::Int)
    dim = length(ψ_re); step = 1<<(q-1); bs = step<<1; nb = dim>>q
    @inbounds for block in 0:(nb-1)
        base = block * bs
        @turbo for offset in 0:(step-1)
            i = base + offset + 1; j = i + step
            a_re = ψ_re[i]; a_im = ψ_im[i]; b_re = ψ_re[j]; b_im = ψ_im[j]
            ψ_re[i] = (u00_re*a_re - u00_im*a_im) + (u01_re*b_re - u01_im*b_im)
            ψ_im[i] = (u00_re*a_im + u00_im*a_re) + (u01_re*b_im + u01_im*b_re)
            ψ_re[j] = (u10_re*a_re - u10_im*a_im) + (u11_re*b_re - u11_im*b_im)
            ψ_im[j] = (u10_re*a_im + u10_im*a_re) + (u11_re*b_im + u11_im*b_re)
        end
    end
end

apply_h_avx!(re, im, q, N) = begin s = Float32(1/sqrt(2f0)); apply_gate_turbo!(re, im, s, 0f0, s, 0f0, s, 0f0, -s, 0f0, q) end
apply_s_avx!(re, im, q, N) = apply_gate_turbo!(re, im, 1f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 1f0, q)
apply_ry_avx!(re, im, q, θ, N) = begin c,s = Float32(cos(θ/2)),Float32(sin(θ/2)); apply_gate_turbo!(re, im, c, 0f0, -s, 0f0, s, 0f0, c, 0f0, q) end
apply_rx_avx!(re, im, q, θ, N) = begin c,s = Float32(cos(θ/2)),Float32(sin(θ/2)); apply_gate_turbo!(re, im, c, 0f0, 0f0, -s, 0f0, -s, c, 0f0, q) end

function apply_cnot_avx!(re, im, ctrl, tgt)
    dim = length(re); cm = 1<<(ctrl-1); tm = 1<<(tgt-1)
    @inbounds for i in 0:(dim-1)
        if (i & cm) != 0 && (i & tm) == 0
            j = i ⊻ tm
            re[i+1], re[j+1] = re[j+1], re[i+1]
            im[i+1], im[j+1] = im[j+1], im[i+1]
        end
    end
end

function apply_cz_avx!(re, im, q1, q2)
    dim = length(re); m1 = 1<<(q1-1); m2 = 1<<(q2-1)
    @turbo for i in 0:(dim-1)
        flag = ((i & m1) != 0) & ((i & m2) != 0)
        re[i+1] = ifelse(flag, -re[i+1], re[i+1])
        im[i+1] = ifelse(flag, -im[i+1], im[i+1])
    end
end

function run_circuit(N::Int)
    dim = 1 << N
    re = zeros(Float32, dim); im = zeros(Float32, dim); re[1] = 1f0
    for layer in 1:L
        for q in 2:2:N;     apply_h_avx!(re, im, q, N); end
        for q in 1:N;        apply_ry_avx!(re, im, q, ANGLES_RY[q], N); end
        for q in 1:2:(N-1);  apply_cnot_avx!(re, im, q, q+1); end
        for q in 1:2:N;      apply_s_avx!(re, im, q, N); end
        for q in 1:N;        apply_rx_avx!(re, im, q, ANGLES_RX[q], N); end
        for q in 2:2:(N-1);  apply_cz_avx!(re, im, q, q+1); end
    end
    return re, im
end

function _expect_Z(re, im, k, N)
    bp=k-1; r=0.0
    @inbounds @simd for i in 0:(length(re)-1)
        r += (Float64(re[i+1])^2+Float64(im[i+1])^2) * (1-2*((i>>bp)&1))
    end; return r
end
function _expect_X(re, im, k, N)
    bp=k-1; step=1<<bp; r=0.0
    @inbounds for i in 0:(length(re)-1)
        if ((i>>bp)&1)==0; j=i+step
            r += 2*(Float64(re[i+1])*Float64(re[j+1])+Float64(im[i+1])*Float64(im[j+1]))
        end
    end; return r
end
function _expect_Y(re, im, k, N)
    bp=k-1; step=1<<bp; r=0.0
    @inbounds for i in 0:(length(re)-1)
        if ((i>>bp)&1)==0; j=i+step
            r += 2*(Float64(re[i+1])*Float64(im[j+1])-Float64(im[i+1])*Float64(re[j+1]))
        end
    end; return r
end
function _expect_ZZ(re, im, i, j, N)
    bi=i-1;bj=j-1;r=0.0
    @inbounds @simd for s in 0:(length(re)-1)
        r += (Float64(re[s+1])^2+Float64(im[s+1])^2) * (1-2*xor((s>>bi)&1,(s>>bj)&1))
    end; return r
end
function _expect_XX(re, im, i, j, N)
    bi=i-1;bj=j-1;si=1<<bi;sj=1<<bj;r=0.0
    @inbounds for s in 0:(length(re)-1)
        if ((s>>bi)&1)==0 && ((s>>bj)&1)==0
            r += 2*(Float64(re[s+1])*Float64(re[s+si+sj+1])+Float64(im[s+1])*Float64(im[s+si+sj+1]))
            r += 2*(Float64(re[s+sj+1])*Float64(re[s+si+1])+Float64(im[s+sj+1])*Float64(im[s+si+1]))
        end
    end; return r
end
function _expect_YY(re, im, i, j, N)
    bi=i-1;bj=j-1;si=1<<bi;sj=1<<bj;r=0.0
    @inbounds for s in 0:(length(re)-1)
        if ((s>>bi)&1)==0 && ((s>>bj)&1)==0
            r += -2*(Float64(re[s+1])*Float64(re[s+si+sj+1])+Float64(im[s+1])*Float64(im[s+si+sj+1]))
            r +=  2*(Float64(re[s+sj+1])*Float64(re[s+si+1])+Float64(im[s+sj+1])*Float64(im[s+si+1]))
        end
    end; return r
end

function compute_observables(re, im, N)
    for q in 1:N; _expect_X(re,im,q,N); _expect_Y(re,im,q,N); _expect_Z(re,im,q,N); end
    for b in 1:(N-1); _expect_XX(re,im,b,b+1,N); _expect_YY(re,im,b,b+1,N); _expect_ZZ(re,im,b,b+1,N); end
end

println("=" ^ 60)
println("  SmoQ.jl AVX (Float32 Re/Im) — Timing Sweep")
println("=" ^ 60)

rw, iw = run_circuit(2); compute_observables(rw, iw, 2)

output_file = joinpath(SCRIPT_DIR, "timing_smoq_avx.txt")
open(output_file, "w") do f
    write(f, "# SmoQ.jl AVX (Float32 Re/Im) timing\n# N  time_s\n")
    for N_val in N_VALUES
        t = time_ns()
        re, im = run_circuit(N_val)
        compute_observables(re, im, N_val)
        dt = (time_ns() - t) / 1e9
        @printf(f, "%d  %.6f\n", N_val, dt)
        @printf("  N = %2d : %.6f s\n", N_val, dt)
    end
end
println("  Saved: ", basename(output_file))

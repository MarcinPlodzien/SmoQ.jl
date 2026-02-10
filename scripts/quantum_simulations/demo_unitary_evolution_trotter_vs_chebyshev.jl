# Date: 2026
#
#!/usr/bin/env julia
#=
================================================================================
    Benchmark: Trotter vs Chebyshev - Timing Comparison Only
================================================================================
Fast benchmark script - no observable measurements, just timing.
Generates 2x2 timing comparison plots.
================================================================================
=#

using LinearAlgebra
using Plots
using Printf




# Load modules
using SmoQ.CPUHamiltonianBuilder
using SmoQ.CPUQuantumChannelUnitaryEvolutionTrotter
using SmoQ.CPUQuantumChannelUnitaryEvolutionChebyshev

# ==============================================================================
# CONFIGURATION
# ==============================================================================
const OUTPUT_DIR = joinpath(@__DIR__, "demo_time_integrators_trotter_chebyshev")
const FIG_DIR = joinpath(OUTPUT_DIR, "figures")
mkpath(FIG_DIR)

const T_max = 5.0
const dt = 0.01
const N_time_shots = 100
const TEST_SIZES = collect(2:1:26)

const Jxx, Jyy, Jzz, hx = 1.0, 1.0, 0.5, 1.0

function format_time(seconds::Float64)
    hours = floor(Int, seconds / 3600)
    mins = floor(Int, (seconds % 3600) / 60)
    secs = floor(Int, seconds % 60)
    ms = floor(Int, (seconds * 1000) % 1000)
    return @sprintf("%02d:%02d:%02d:%03d", hours, mins, secs, ms)
end

# ==============================================================================
# BENCHMARK
# ==============================================================================
println("="^70)
println("  TROTTER vs CHEBYSHEV - TIMING BENCHMARK")
Δt_cheb = T_max / (N_time_shots - 1)
println("  $(Threads.nthreads()) threads | T_max=$T_max | dt=$dt | Δt_cheb=$(round(Δt_cheb, digits=3))")
println("="^70)

timing_data = []

for N in TEST_SIZES
    println("\n--- N = $N ---")

    dim = 1 << N
    n_trotter_steps = floor(Int, T_max / dt)

    if (dim * 16 / 1e9) > 64  # Skip if > 64GB
        println("  SKIPPED (too large)")
        continue
    end

    # Build Hamiltonian parameters and gates
    params = build_hamiltonian_parameters(N, 1;
        J_x_direction=(Jxx, Jyy, Jzz),
        J_y_direction=(0.0, 0.0, 0.0),
        h_field=(hx, 0.0, 0.0))

    gates = precompute_trotter_gates_bitwise_cpu(params, dt)

    ψ0 = zeros(ComplexF64, dim); ψ0[1] = 1.0

    # Pre-compute spectral bounds
    E_min, E_max = estimate_spectral_range_lanczos(params)
    spectral_bounds = (E_min, E_max)

    # =========== TROTTER ===========
    ψ_tr = copy(ψ0)
    print("  Trotter ($n_trotter_steps steps): ")
    t_trotter = @elapsed begin
        for step in 1:n_trotter_steps
            apply_fast_trotter_step_cpu!(ψ_tr, gates, N)
        end
    end
    println("$(format_time(t_trotter)) ($(round(t_trotter, digits=2))s)")

    # =========== CHEBYSHEV ===========
    ψ_ch = copy(ψ0); M_max = 0
    print("  Chebyshev ($(N_time_shots-1) steps): ")
    t_cheb = @elapsed begin
        for i in 2:N_time_shots
            M, _, _ = chebyshev_evolve_psi!(ψ_ch, params, Δt_cheb; spectral_bounds=spectral_bounds)
            M_max = max(M_max, M)
        end
    end
    println("$(format_time(t_cheb)) ($(round(t_cheb, digits=2))s) (M_max=$M_max)")

    # Fidelity
    F_final = abs(dot(ψ_tr, ψ_ch))^2
    println("  Fidelity: $(round(F_final, digits=10))")

    push!(timing_data, (N=N, t_tr=t_trotter, t_ch=t_cheb, M=M_max, F=F_final))
end

# ==============================================================================
# 2x2 TIMING PLOT
# ==============================================================================
println("\n" * "="^70)
println("  TIMING SUMMARY")
println("="^70)
println("  N  | Trotter(s) | Cheb(s) | M  | Fidelity")
println("  " * "-"^50)
for d in timing_data
    @printf("  %2d | %10.2f | %7.2f | %2d | %.10f\n",
            d.N, d.t_tr, d.t_ch, d.M, d.F)
end

if length(timing_data) > 0
    Ns = [d.N for d in timing_data]
    t_tr_s = [d.t_tr for d in timing_data]
    t_ch_s = [d.t_ch for d in timing_data]

    # Row 1: Linear y-scale
    p11 = plot(Ns, t_tr_s, label="Trotter", lw=2, marker=:circle, ms=5, color=:blue,
               xlabel="N (qubits)", ylabel="Time (s)", title="Linear N, Linear Time", legend=:topleft)
    plot!(p11, Ns, t_ch_s, label="Chebyshev", lw=2, marker=:square, ms=5, color=:red)

    p12 = plot(Ns, t_tr_s, label="Trotter", lw=2, marker=:circle, ms=5, color=:blue,
               xlabel="log₂(dim) = N", ylabel="Time (s)", title="Log₂(dim), Linear Time", legend=:topleft)
    plot!(p12, Ns, t_ch_s, label="Chebyshev", lw=2, marker=:square, ms=5, color=:red)

    # Row 2: Log10 y-scale
    p21 = plot(Ns, t_tr_s, label="Trotter", lw=2, marker=:circle, ms=5, color=:blue,
               xlabel="N (qubits)", ylabel="Time (s)", title="Linear N, Log₁₀ Time",
               legend=:topleft, yscale=:log10)
    plot!(p21, Ns, t_ch_s, label="Chebyshev", lw=2, marker=:square, ms=5, color=:red)

    p22 = plot(Ns, t_tr_s, label="Trotter", lw=2, marker=:circle, ms=5, color=:blue,
               xlabel="log₂(dim) = N", ylabel="Time (s)", title="Log₂(dim), Log₁₀ Time",
               legend=:topleft, yscale=:log10)
    plot!(p22, Ns, t_ch_s, label="Chebyshev", lw=2, marker=:square, ms=5, color=:red)

    fig_timing = plot(p11, p12, p21, p22, layout=(2,2), size=(1000, 800),
                      plot_title="Trotter vs Chebyshev Timing ($(Threads.nthreads()) threads)")
    savefig(fig_timing, joinpath(FIG_DIR, "timing_benchmark.png"))
    println("\nSaved: $(joinpath(FIG_DIR, "timing_benchmark.png"))")
end

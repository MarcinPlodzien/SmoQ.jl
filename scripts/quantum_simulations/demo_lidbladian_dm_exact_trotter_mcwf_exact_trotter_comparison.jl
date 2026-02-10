#!/usr/bin/env julia
#=
================================================================================
    4-WAY COMPARISON: DM×{Exact,Trotter} vs MCWF×{Exact,Trotter}

    LINDBLADIAN: dρ/dt = -i[H,ρ] + γ Σₖ D[Lₖ]ρ

    JUMP OPERATOR: Lₖ = σ_zₖ (pure dephasing)
                   Applied to ALL qubits with rate γ

    PHYSICS: ⟨X⟩,⟨Y⟩ → 0 (coherences decay)
             ⟨Z⟩ unchanged (populations preserved)

    Methods:
    1. DM + Exact propagator (gold standard)
    2. DM + Trotter gates
    3. MCWF + Exact propagator
    4. MCWF + Trotter gates

    All 4 should overlap if implementations are correct!
================================================================================
=#

using LinearAlgebra
using Statistics
using Plots

SCRIPT_DIR = @__DIR__
WORKSPACE = dirname(SCRIPT_DIR)
UTILS_CPU = joinpath(WORKSPACE, "utils", "cpu")

include(joinpath(UTILS_CPU, "cpuHamiltonianBuilder.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumChannelUnitaryEvolutionExact.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumChannelUnitaryEvolutionTrotter.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateObservables.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumChannelLindbladianEvolutionDensityMatrix.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumChannelLindbladianEvolutionMonteCarloWaveFunction.jl"))

using .CPUHamiltonianBuilder: build_hamiltonian_parameters, construct_sparse_hamiltonian
using .CPUQuantumChannelUnitaryEvolutionExact: precompute_exact_propagator_cpu
using .CPUQuantumChannelUnitaryEvolutionTrotter: precompute_trotter_gates_bitwise_cpu, apply_fast_trotter_step_cpu!, evolve_trotter_rho_cpu!
using .CPUQuantumStateObservables: expect_local, expect_corr
using .CPUQuantumChannelLindbladianEvolutionDensityMatrix: lindblad_dm_step!, lindblad_dm_step_trotter!
using .CPUQuantumChannelLindbladianEvolutionMonteCarloWaveFunction: lindblad_mcwf_step!, lindblad_mcwf_step_trotter!, lindblad_mcwf_batched_step!

# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================
const OUTPUT_DIR = joinpath(SCRIPT_DIR, "demo_lidbladian_dm_vs_mcwf_exact_vs_trotter")
const DATA_DIR = joinpath(OUTPUT_DIR, "data")
const FIGURES_DIR = joinpath(OUTPUT_DIR, "figures")
mkpath(DATA_DIR)
mkpath(FIGURES_DIR)

# ==============================================================================
# PARAMETERS
# ==============================================================================
const N_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22] # Capped at 14 (memory/threading issues above)
const STATE_NAME = "allzero"           # Initial state name for filenames
const DM_CUTOFF = 10                  # Skip DM for N >= this value
const MCWF_EXACT_CUTOFF = 10          # Skip MCWF+Exact for N >= this value (matches SPARSE_CUTOFF)
const γ = 0.1
const dt = 0.05
const T_max = 10
const n_traj = 1000
const n_steps = Int(T_max / dt)

println("=" ^ 80)
println("  4-WAY LINDBLADIAN COMPARISON (N-SCALING)")
println("  DM×{Exact,Trotter} vs MCWF×{Exact,Trotter}")
println("=" ^ 80)
println("\nParameters:")
println("  N values: $N_values")
println("  DM cutoff: N < $DM_CUTOFF")
println("  γ=$γ, dt=$dt, n_steps=$n_steps, T_max=$T_max")
println("  n_traj=$n_traj (for MCWF)")

# Global timing storage for N-scaling plot
timing_all = Dict{Int, Dict{String, Float64}}()

# ==============================================================================
# MAIN LOOP OVER N
# ==============================================================================
for N in N_values

println("\n" * "=" ^ 80)
println("  N = $N (dim = $(1 << N))")
println("=" ^ 80)

# ==============================================================================
# BUILD HAMILTONIAN & PROPAGATORS
# ==============================================================================
println("\nBuilding Hamiltonian...")
H_params = build_hamiltonian_parameters(N, 1;
    J_x_direction=(1.0, 1.0, 0.5), J_y_direction=(0., 0., 0.), h_field=(1.0, 0., 0.))

# Only build sparse H and exact propagator for small N (memory-intensive)
SPARSE_CUTOFF = 11  # Skip sparse H and exact propagator for N >= this
H_sparse = nothing
U_exact = nothing
if N < SPARSE_CUTOFF
    H_sparse = construct_sparse_hamiltonian(H_params)
    U_exact, _ = precompute_exact_propagator_cpu(H_sparse, dt)
    println("  -- Exact propagator U = exp(-iH*dt)")
else
    println("  [SKIPPED] Sparse H & exact propagator (N >= $SPARSE_CUTOFF)")
end

# Always build Trotter gates (matrix-free)
gates = precompute_trotter_gates_bitwise_cpu(H_params, dt)
println("  -- Trotter gates ($(length(gates)) gates)")

# Initial state
dim = 1 << N

# ---- All spins |0...0⟩ ----
ψ0 = zeros(ComplexF64, dim)
ψ0[1] = 1.0  # |000...0⟩ is index 1 (1-based)

# ---- GHZ state (commented out) ----
# ψ0 = zeros(ComplexF64, dim)
# ψ0[1] = 1.0 / sqrt(2); ψ0[end] = 1.0 / sqrt(2)

# ---- Néel state: |0101...⟩ (commented out) ----
# neel_index = sum(1 << (k-1) for k in 1:2:N)
# ψ0 = zeros(ComplexF64, dim)
# ψ0[neel_index + 1] = 1.0

# Only build density matrix for small N (DM methods)
ρ0 = nothing
if N < DM_CUTOFF
    ρ0 = ψ0 * ψ0'  # dim × dim matrix - huge for large N!
end

# Helpers: mean per qubit / per pair
n_pairs = N - 1
expect_x_mean(state, N) = sum(expect_local(state, k, N, :x) for k in 1:N) / N
expect_y_mean(state, N) = sum(expect_local(state, k, N, :y) for k in 1:N) / N
expect_z_mean(state, N) = sum(expect_local(state, k, N, :z) for k in 1:N) / N
expect_xx_mean(state, N) = sum(expect_corr(state, i, i+1, N, :xx) for i in 1:N-1) / n_pairs
expect_yy_mean(state, N) = sum(expect_corr(state, i, i+1, N, :yy) for i in 1:N-1) / n_pairs
expect_zz_mean(state, N) = sum(expect_corr(state, i, i+1, N, :zz) for i in 1:N-1) / n_pairs

# ==============================================================================
# STORAGE FOR RESULTS
# ==============================================================================
times = collect((0:n_steps) .* dt)

# 4 methods × 6 observables
method_names = ["DM+Exact", "DM+Trotter", "MCWF+Exact", "MCWF+Trotter"]
obs_names = ["⟨X⟩", "⟨Y⟩", "⟨Z⟩", "⟨XX⟩", "⟨YY⟩", "⟨ZZ⟩"]

results = Dict{String, Dict{String, Vector{Float64}}}()
for m in method_names
    results[m] = Dict{String, Vector{Float64}}()
    for o in obs_names
        results[m][o] = zeros(n_steps + 1)
    end
end

# ==============================================================================
# TIMING STORAGE
# ==============================================================================
timing = Dict{String, Float64}()

# ==============================================================================
# 1. DM + EXACT PROPAGATOR (Gold Standard) - SKIP FOR LARGE N
# ==============================================================================
if N < DM_CUTOFF
    println("\n" * "-"^50)
    println("1. DM + Exact propagator (gold standard)...")

    # Warmup
    ρ_warmup = copy(ρ0)
    lindblad_dm_step!(ρ_warmup, U_exact, N, γ, dt)

    # Evolution with fair timing
    ρ = copy(ρ0)
    t_start = time()
    for step in 0:n_steps
        results["DM+Exact"]["⟨X⟩"][step+1] = expect_x_mean(ρ, N)
        results["DM+Exact"]["⟨Y⟩"][step+1] = expect_y_mean(ρ, N)
        results["DM+Exact"]["⟨Z⟩"][step+1] = expect_z_mean(ρ, N)
        results["DM+Exact"]["⟨XX⟩"][step+1] = expect_xx_mean(ρ, N)
        results["DM+Exact"]["⟨YY⟩"][step+1] = expect_yy_mean(ρ, N)
        results["DM+Exact"]["⟨ZZ⟩"][step+1] = expect_zz_mean(ρ, N)
        if step < n_steps
            lindblad_dm_step!(ρ, U_exact, N, γ, dt)
        end
    end
    timing["DM+Exact"] = time() - t_start
    println("   Done in $(round(timing["DM+Exact"], digits=2))s")

    # ==============================================================================
    # 2. DM + TROTTER
    # ==============================================================================
    println("2. DM + Trotter gates...")
    ρ = copy(ρ0)
    t_start = time()
    for step in 0:n_steps
        results["DM+Trotter"]["⟨X⟩"][step+1] = expect_x_mean(ρ, N)
        results["DM+Trotter"]["⟨Y⟩"][step+1] = expect_y_mean(ρ, N)
        results["DM+Trotter"]["⟨Z⟩"][step+1] = expect_z_mean(ρ, N)
        results["DM+Trotter"]["⟨XX⟩"][step+1] = expect_xx_mean(ρ, N)
        results["DM+Trotter"]["⟨YY⟩"][step+1] = expect_yy_mean(ρ, N)
        results["DM+Trotter"]["⟨ZZ⟩"][step+1] = expect_zz_mean(ρ, N)
        if step < n_steps
            lindblad_dm_step_trotter!(ρ, gates, N, γ, dt)
        end
    end
    timing["DM+Trotter"] = time() - t_start
    println("   Done in $(round(timing["DM+Trotter"], digits=2))s")
else
    println("\n[SKIPPED] DM methods (N >= $DM_CUTOFF too large)")
    timing["DM+Exact"] = NaN
    timing["DM+Trotter"] = NaN
end

# ==============================================================================
# 3. MCWF + EXACT PROPAGATOR - SKIP FOR LARGE N
# ==============================================================================
mcwf_exact_std = Dict{String, Vector{Float64}}()
if N < MCWF_EXACT_CUTOFF
    println("3. MCWF + Exact propagator ($n_traj trajectories)...")
    t_start = time()
    traj_data_exact = zeros(n_steps + 1, 6, n_traj)

    Threads.@threads for traj in 1:n_traj
        ψ = copy(ψ0)
        for step in 0:n_steps
            traj_data_exact[step+1, 1, traj] = expect_x_mean(ψ, N)
            traj_data_exact[step+1, 2, traj] = expect_y_mean(ψ, N)
            traj_data_exact[step+1, 3, traj] = expect_z_mean(ψ, N)
            traj_data_exact[step+1, 4, traj] = expect_xx_mean(ψ, N)
            traj_data_exact[step+1, 5, traj] = expect_yy_mean(ψ, N)
            traj_data_exact[step+1, 6, traj] = expect_zz_mean(ψ, N)
            if step < n_steps
                lindblad_mcwf_step!(ψ, U_exact, N, γ, dt)
            end
        end
        traj % 1000 == 0 && print("   $traj/$n_traj\r")
    end

    # Store mean and std for MCWF+Exact
    for (i, o) in enumerate(obs_names)
        results["MCWF+Exact"][o] .= vec(mean(traj_data_exact[:, i, :], dims=2))
        mcwf_exact_std[o] = vec(std(traj_data_exact[:, i, :], dims=2)) ./ sqrt(n_traj)
    end
    timing["MCWF+Exact"] = time() - t_start
    println("   Done in $(round(timing["MCWF+Exact"], digits=2))s")
else
    println("[SKIPPED] MCWF+Exact (N >= $MCWF_EXACT_CUTOFF, exact propagator too slow)")
    timing["MCWF+Exact"] = NaN
    for o in obs_names
        mcwf_exact_std[o] = zeros(n_steps + 1)
    end
end

# ==============================================================================
# 4. MCWF + TROTTER
# ==============================================================================
println("4. MCWF + Trotter gates ($n_traj trajectories)...")
t_start = time()
traj_data_trotter = zeros(n_steps + 1, 6, n_traj)

Threads.@threads for traj in 1:n_traj
    ψ = copy(ψ0)
    for step in 0:n_steps
        traj_data_trotter[step+1, 1, traj] = expect_x_mean(ψ, N)
        traj_data_trotter[step+1, 2, traj] = expect_y_mean(ψ, N)
        traj_data_trotter[step+1, 3, traj] = expect_z_mean(ψ, N)
        traj_data_trotter[step+1, 4, traj] = expect_xx_mean(ψ, N)
        traj_data_trotter[step+1, 5, traj] = expect_yy_mean(ψ, N)
        traj_data_trotter[step+1, 6, traj] = expect_zz_mean(ψ, N)
        if step < n_steps
            lindblad_mcwf_step_trotter!(ψ, gates, apply_fast_trotter_step_cpu!, N, γ, dt)
        end
    end
    traj % 1000 == 0 && print("   $traj/$n_traj\r")
end

# Store mean and std for MCWF+Trotter
mcwf_trotter_std = Dict{String, Vector{Float64}}()
for (i, o) in enumerate(obs_names)
    results["MCWF+Trotter"][o] .= vec(mean(traj_data_trotter[:, i, :], dims=2))
    mcwf_trotter_std[o] = vec(std(traj_data_trotter[:, i, :], dims=2)) ./ sqrt(n_traj)
end
timing["MCWF+Trotter"] = time() - t_start
println("   Done in $(round(timing["MCWF+Trotter"], digits=2))s")

# ==============================================================================
# GENERATE 2×3 GRID PLOT
# ==============================================================================
println("\n" * "-"^50)
println("Generating 2×3 grid plot with error bands...")

plots = []
for obs in obs_names
    p = plot(title=obs, xlabel="t", legend=:topright, ylims=(-1, 1))

    # DM methods only if N < DM_CUTOFF
    if N < DM_CUTOFF
        t_dm_exact = round(timing["DM+Exact"], sigdigits=2)
        t_dm_trotter = round(timing["DM+Trotter"], sigdigits=2)
        # DM+Exact (thick blue solid)
        plot!(p, times, results["DM+Exact"][obs], lw=3, color=:blue, label="DM+Exact ($(t_dm_exact)s)")

        # DM+Trotter (thick cyan dashed)
        plot!(p, times, results["DM+Trotter"][obs], lw=3, ls=:dash, color=:cyan, label="DM+Trotter ($(t_dm_trotter)s)")
    end

    # MCWF+Exact only if N < MCWF_EXACT_CUTOFF
    if N < MCWF_EXACT_CUTOFF
        t_mcwf_exact = round(timing["MCWF+Exact"], sigdigits=2)
        plot!(p, times, results["MCWF+Exact"][obs], ribbon=2mcwf_exact_std[obs],
              fillalpha=0.3, lw=2, color=:red, label="MCWF+Exact ($(n_traj)traj, $(t_mcwf_exact)s)")
    end

    # MCWF+Trotter with shaded error (orange) - always plotted
    t_mcwf_trotter = round(timing["MCWF+Trotter"], sigdigits=2)
    plot!(p, times, results["MCWF+Trotter"][obs], ribbon=2mcwf_trotter_std[obs],
          fillalpha=0.3, lw=2, color=:orange, label="MCWF+Trotter ($(n_traj)traj, $(t_mcwf_trotter)s)")

    push!(plots, p)
end

fig = plot(plots..., layout=(2,3), size=(1500, 800),
    plot_title="|ψ₀⟩=|0...0⟩  |  dρ/dt=-i[H,ρ]+γΣᵢ(σᶻᵢρσᶻᵢ-ρ)  |  H=Jˣˣσˣσˣ+Jʸʸσʸσʸ+Jᶻᶻσᶻσᶻ+hˣσˣ  |  N=$N, γ=$γ")

figpath = joinpath(FIGURES_DIR, "fig_lindbladian_$(STATE_NAME)_N$(lpad(N, 2, '0'))_Ntraj$(n_traj).png")
savefig(fig, figpath)
println("-- Saved figure: $(figpath)")

# ==============================================================================
# SAVE DATA TO FILES
# ==============================================================================
using DelimitedFiles

println("\nSaving data files...")

# Save time array
writedlm(joinpath(DATA_DIR, "times_N$(N)_ntraj$(n_traj).dat"), times)

# Save DM results
for (method_short, method) in [("dm_exact", "DM+Exact"), ("dm_trotter", "DM+Trotter")]
    for obs in obs_names
        obs_clean = replace(obs, "Σ" => "sum_")
        filename = "$(method_short)_$(obs_clean)_N$(N)_ntraj$(n_traj).dat"
        writedlm(joinpath(DATA_DIR, filename), results[method][obs])
    end
end

# Save MCWF results with std
for (method_short, method, std_dict) in [
    ("mcwf_exact", "MCWF+Exact", mcwf_exact_std),
    ("mcwf_trotter", "MCWF+Trotter", mcwf_trotter_std)
]
    for obs in obs_names
        obs_clean = replace(obs, "Σ" => "sum_")
        # Mean
        filename = "$(method_short)_$(obs_clean)_N$(N)_ntraj$(n_traj).dat"
        writedlm(joinpath(DATA_DIR, filename), results[method][obs])
        # Std
        filename_std = "$(method_short)_$(obs_clean)_std_N$(N)_ntraj$(n_traj).dat"
        writedlm(joinpath(DATA_DIR, filename_std), std_dict[obs])
    end
end

println("-- Saved data to: $(DATA_DIR)")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
println("\n" * "=" ^ 80)
println("FINAL VALUES (t = $T_max)")
println("=" ^ 80)
println()
println("Observable | DM+Exact  | DM+Trotter| MCWF+Exact|MCWF+Trotter")
println("-" ^ 80)
for obs in obs_names
    vals = [round(results[m][obs][end], digits=4) for m in method_names]
    println("  $(rpad(obs, 8)) | $(join([lpad(v, 9) for v in vals], " | "))")
end
println("-" ^ 80)

println("\nDEVIATIONS FROM DM+Exact (gold standard):")
println("-" ^ 60)
println("Observable | DM+Trotter | MCWF+Exact |MCWF+Trotter")
println("-" ^ 60)
for obs in obs_names
    gold = results["DM+Exact"][obs][end]
    deltas = [round(results[m][obs][end] - gold, digits=5) for m in method_names[2:end]]
    println("  $(rpad(obs, 8)) | $(join([lpad(d, 10) for d in deltas], " | "))")
end
println("=" ^ 80)

# ==============================================================================
# TIMING SUMMARY
# ==============================================================================
println("\n" * "=" ^ 80)
println("TIMING SUMMARY (includes evolution + observables)")
println("=" ^ 80)
println()
println("Method       | Time (s) | Time/step (ms)")
println("-" ^ 48)
for m in method_names
    t = timing[m]
    if isnan(t)
        println("  $(rpad(m, 12)) |      -- |           --")
    else
        t_per_step = t / n_steps * 1000
        println("  $(rpad(m, 12)) | $(lpad(round(t, digits=2), 8)) | $(lpad(round(t_per_step, digits=3), 14))")
    end
end
println("=" ^ 80)

# Store timing for this N
timing_all[N] = copy(timing)

end  # END OF N LOOP

# ==============================================================================
# N-SCALING SUMMARY
# ==============================================================================
println("\n" * "=" ^ 80)
println("  N-SCALING TIMING SUMMARY")
println("=" ^ 80)
println()
println("  N   | DM+Exact | DM+Trotter | MCWF+Exact | MCWF+Trotter")
println("-" ^ 64)
for N in N_values
    vals = [get(timing_all[N], m, NaN) for m in method_names]
    vals_str = [isnan(v) ? "   --   " : lpad(round(v, digits=2), 8) for v in vals]
    println("  $(lpad(N, 3)) | $(join(vals_str, " | "))")
end
println("=" ^ 80)

#!/usr/bin/env julia
#=
╔══════════════════════════════════════════════════════════════════════════════╗
║       MANY-BODY BELL CORRELATOR DEMO                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
--------
Demonstration of many-body Bell correlator optimization for entanglement
certification. Shows optimization over local measurement bases and noise
resilience analysis.

PHYSICAL QUANTITY:
------------------
The many-body Bell correlator:

    ℰ = max_θ |Tr[ρ · 𝓑(θ)]|²

where the Bell operator:

    𝓑(θ) = ⊗ⱼ Uⱼ(θⱼ,φⱼ) σ⁺ⱼ Uⱼ†(θⱼ,φⱼ)

with Uⱼ = Rz(φⱼ)Ry(θⱼ) local rotations and σ⁺ = |1⟩⟨0| the raising operator.

The Q-correlators are defined as:

    Q_bell = log₂(ℰ) + N
    Q_ent  = log₄(ℰ) + N = (1/2)log₂(ℰ) + N

PHYSICAL INTERPRETATION:
------------------------
- For entangled states (GHZ, graph states): Q_bell ≈ N, Q_ent > 0
- For product states: Q_bell ≤ 0 (no genuine multipartite entanglement)
- Maximum Q_bell = N-1 for GHZ state
- Q_bell > N-2 certifies genuine multipartite entanglement (non-2-separable)

REFERENCES:
-----------
[1] Płodzień et al., PRL 129, 250402 (2022) - One-axis twisting Bell correlations
[2] Płodzień et al., PRR 6, 023050 (2024) - Scalable Bell correlations
[3] Płodzień et al., PRA 110, 032428 (2024) - Non-k-separability certification

OUTPUT:
-------
  demo_many_body_bell_correlator/
  ├── data/   → Q_bell vs p_noise data files
  └── figures/ → Grid plots: rows=N, columns=states
=#

using Printf
using Dates
using LinearAlgebra
using Random
using Optim  # L-BFGS optimizer
using Plots; gr()
using ProgressMeter  # tqdm-like progress bars

# Setup paths
SCRIPT_DIR = @__DIR__
ROOT_DIR = dirname(SCRIPT_DIR)
UTILS_DIR = joinpath(ROOT_DIR, "utils", "cpu")

# Output
OUTPUT_DIR = joinpath(SCRIPT_DIR, "demo_many_body_bell_correlator")
DATA_DIR = joinpath(OUTPUT_DIR, "data")
FIG_DIR = joinpath(OUTPUT_DIR, "figures")
mkpath(DATA_DIR)
mkpath(FIG_DIR)

# Include modules
include(joinpath(UTILS_DIR, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_DIR, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_DIR, "cpuQuantumChannelGates.jl"))
include(joinpath(UTILS_DIR, "cpuQuantumChannelKrausOperators.jl"))
include(joinpath(UTILS_DIR, "cpuQuantumStateManyBodyBellCorrelator.jl"))

using .CPUQuantumStatePartialTrace
using .CPUQuantumStatePreparation
using .CPUQuantumChannelGates
using .CPUQuantumChannelKraus
using .CPUQuantumStateManyBodyBellCorrelator

# =============================================================================
# CONFIGURATION
# =============================================================================

# Optimizers to compare
optimizers = [
    (name="spsa_adam", label="SPSA+Adam", method=:spsa_adam, max_iter=300),
    (name="lbfgs", label="L-BFGS", method=:lbfgs, max_iter=300),
    (name="autograd_adam", label="Autograd+Adam", method=:autograd_adam, max_iter=300),
]

# Default optimizer for quick tests
opt_default = optimizers[1]
precision_digits = 2

# Test ranges
N_range_quick = 3:4  # Quick tests for tables
N_range_sweep = [6]  # Full Q vs p sweep

# Noise parameters
noise_models = [:dephasing, :depolarizing, :amplitude_damping]
noise_strengths_quick = collect(0.0:0.02:0.5)  # Same as p_noise_range
p_noise_range = collect(0.0:0.02:0.5)  # 21 points from 0 to 1

# Modes to compare
modes = [
            :dm, 
            :mcwf
        ]

# MCWF parameters
N_trajectories_quick = 1000
N_trajectories_sweep = 1000

# Format helper
fmt_val(x) = @sprintf("%+*.*f", precision_digits+4, precision_digits, x)

println("═"^70)
println("  MANY-BODY BELL CORRELATOR DEMO")
println("═"^70)
println("  Optimizers: $([o.label for o in optimizers])")
println("  Quick tests: N = $(collect(N_range_quick))")
println("  Sweep tests: N = $N_range_sweep")
println("  Modes: $modes")
println("  Output: demo_many_body_bell_correlator/")
println()

# =============================================================================
# STATE GENERATORS
# =============================================================================

"""Product state |+⟩⊗N"""
make_plus_product(N::Int) = make_ket("|+>", N)

"""GHZ state (|0...0⟩ + |1...1⟩)/√2 - returns (ψ, bases_string)"""
function make_ghz_z(N::Int)
    d = 2^N
    ψ = zeros(ComplexF64, d)
    ψ[1] = 1/sqrt(2)
    ψ[d] = 1/sqrt(2)
    bases = repeat("Z", N)
    return ψ, bases
end

"""GHZ in random local bases - returns (ψ, bases_string)"""
function make_ghz_random(N::Int)
    ψ, _ = make_ghz_z(N)
    bases = Char[]
    
    for k in 1:N
        b = rand(['X', 'Y', 'Z'])
        push!(bases, b)
        
        if b == 'X'
            apply_hadamard_psi!(ψ, k, N)
        elseif b == 'Y'
            apply_s_psi!(ψ, k, N)
            apply_hadamard_psi!(ψ, k, N)
        end
    end
    
    normalize_state!(ψ)
    return ψ, String(bases)
end

"""Star graph state"""
function make_star_graph(N::Int)
    @assert N >= 2
    ψ = make_ket("|+>", N)
    for k in 2:N
        apply_cz_psi!(ψ, 1, k, N)
    end
    return ψ
end

"""Star graph + T gates"""
function make_star_graph_T(N::Int)
    ψ = make_star_graph(N)
    for k in 1:N
        apply_t_psi!(ψ, k, N)
    end
    normalize_state!(ψ)
    return ψ
end

# State configurations
test_states = [
    (name="plus_product", label="|+⟩⊗N", gen=make_plus_product, entangled=false),
    (name="ghz_z", label="GHZ (Z)", gen=make_ghz_z, entangled=true),
    (name="ghz_random", label="GHZ (rand)", gen=make_ghz_random, entangled=true),
    (name="star_graph", label="Star Graph", gen=make_star_graph, entangled=true),
    (name="star_T", label="Star+T", gen=make_star_graph_T, entangled=true),
]

# =============================================================================
# NOISE FUNCTIONS
# =============================================================================

function apply_noise_dm(ψ, noise::Symbol, p::Float64, N::Int)
    ρ = ψ * ψ'
    qubits = collect(1:N)
    if noise == :dephasing
        apply_channel_dephasing!(ρ, p, qubits, N)
    elseif noise == :depolarizing
        apply_channel_depolarizing!(ρ, p, qubits, N)
    elseif noise == :amplitude_damping
        apply_channel_amplitude_damping!(ρ, p, qubits, N)
    end
    return ρ
end

function generate_trajectories(ψ, noise::Symbol, p::Float64, N::Int, M::Int)
    trajectories = Vector{Vector{ComplexF64}}(undef, M)
    qubits = collect(1:N)
    
    # Multithreaded trajectory generation
    Threads.@threads for i in 1:M
        ψ_t = copy(ψ)
        if noise == :dephasing
            apply_channel_dephasing!(ψ_t, p, qubits, N)
        elseif noise == :depolarizing
            apply_channel_depolarizing!(ψ_t, p, qubits, N)
        elseif noise == :amplitude_damping
            apply_channel_amplitude_damping!(ψ_t, p, qubits, N)
        end
        trajectories[i] = ψ_t
    end
    return trajectories
end

# =============================================================================
# PART 1: PURE STATE TESTS (with local basis optimization)
# =============================================================================

println("═"^70)
println("  PART 1: PURE STATES - Local Basis Optimization")
println("═"^70)
println()

all_results = []

for state in test_states
    # Check if this is a GHZ state (returns tuple with bases)
    test_result = state.gen(N_range_quick[1])
    is_ghz_state = test_result isa Tuple
    
    # Print table header with |ψ⟩ = prefix
    if state.name == "plus_product"
        println("  ── |ψ⟩ = |+⟩⊗N ──")
    elseif state.name == "ghz_z"
        println("  ── |ψ⟩ = |GHZ⟩ (Z basis) ──")
    elseif state.name == "ghz_random"
        println("  ── |ψ⟩ = |GHZ⟩ (random bases) ──")
    elseif state.name == "star_graph"
        println("  ── |ψ⟩ = |Star Graph⟩ ──")
    elseif state.name == "star_T"
        println("  ── |ψ⟩ = |Star+T⟩ ──")
    else
        println("  ── |ψ⟩ = $(state.label) ──")
    end
    
    # Table format depends on whether we have bases
    if is_ghz_state
        println("  ┌─────┬─────────┬──────────┬──────────┬──────────┐")
        println("  │  N  │  bases  │  Q_ent   │  Q_bell  │ time (s) │")
        println("  ├─────┼─────────┼──────────┼──────────┼──────────┤")
    else
        println("  ┌─────┬──────────┬──────────┬──────────┐")
        println("  │  N  │  Q_ent   │  Q_bell  │ time (s) │")
        println("  ├─────┼──────────┼──────────┼──────────┤")
    end
    
    for N in N_range_quick
        result = state.gen(N)
        
        if result isa Tuple
            ψ, bases = result
        else
            ψ = result
            bases = ""
        end
        
        t = @elapsed Q_bell, Q_ent, θ_opt = get_bell_correlator(ψ; 
            method=opt_default.method, max_iter=opt_default.max_iter)
        
        # Validation: product states should have Q_ent <= 0
        if !state.entangled && Q_ent > 0.1
            @warn "Product state has Q_ent > 0 ($(Q_ent)) - check optimizer!"
        end
        
        push!(all_results, (
            state=state.name, N=N, rep=:pure, mixed=false, 
            noise="none", p=0.0, Q_bell=Q_bell, Q_ent=Q_ent, time=t,
            bases=bases
        ))
        
        if is_ghz_state
            @printf("  │ %2d  │ %s │  %+6.3f  │  %+6.3f  │  %6.3f  │\n", N, bases, Q_ent, Q_bell, t)
        else
            @printf("  │ %2d  │  %+6.3f  │  %+6.3f  │  %6.3f  │\n", N, Q_ent, Q_bell, t)
        end
    end
    
    if is_ghz_state
        println("  └─────┴─────────┴──────────┴──────────┴──────────┘")
    else
        println("  └─────┴──────────┴──────────┴──────────┴──────────┘")
    end
    println()
end

# =============================================================================
# PART 2: DENSITY MATRIX WITH NOISE
# =============================================================================

println("═"^70)
println("  PART 2: DENSITY MATRIX WITH NOISE (N=3)")
println("═"^70)
println()

N_dm = 3

# Collect DM results
for state in test_states[1:3]
    for noise in noise_models[1:2]  # dephasing and depolarizing
        for p in noise_strengths_quick
            result = state.gen(N_dm)
            if result isa Tuple
                ψ, bases = result
            else
                ψ = result
                bases = ""
            end
            ρ = p > 0 ? apply_noise_dm(ψ, noise, p, N_dm) : ψ * ψ'
            
            t = @elapsed Q_bell, Q_ent, θ_opt = get_bell_correlator(ρ; 
                method=opt_default.method, max_iter=opt_default.max_iter)
            
            push!(all_results, (
                state=state.name, N=N_dm, rep=:dm, mixed=(p>0), 
                noise=String(noise), p=p, Q_bell=Q_bell, Q_ent=Q_ent, time=t,
                bases=bases
            ))
        end
    end
end

# Print tables for each state
for state in test_states[1:3]
    is_ghz_random = (state.name == "ghz_random")
    
    if state.name == "ghz_z"
        println("  ── |ψ⟩ = |GHZ⟩ (Z basis) ──")
    elseif state.name == "ghz_random"
        println("  ── |ψ⟩ = |GHZ⟩ (random bases) ──")
    else
        println("  ── |ψ⟩ = $(state.label) ──")
    end
    
    if is_ghz_random
        println("  ┌───────┬───────┬──────────────────────┬──────────────────────┐")
        println("  │   p   │ bases │      dephasing       │     depolarizing     │")
        println("  │       │       │  Q_bell  │   Q_ent   │  Q_bell  │   Q_ent   │")
        println("  ├───────┼───────┼──────────┼───────────┼──────────┼───────────┤")
    else
        println("  ┌───────┬──────────────────────┬──────────────────────┐")
        println("  │   p   │      dephasing       │     depolarizing     │")
        println("  │       │  Q_bell  │   Q_ent   │  Q_bell  │   Q_ent   │")
        println("  ├───────┼──────────┼───────────┼──────────┼───────────┤")
    end
    
    for p in noise_strengths_quick
        deph = filter(r -> r.state == state.name && r.noise == "dephasing" && 
                      r.rep == :dm && abs(r.p - p) < 0.01, all_results)
        depol = filter(r -> r.state == state.name && r.noise == "depolarizing" && 
                       r.rep == :dm && abs(r.p - p) < 0.01, all_results)
        
        deph_bell = !isempty(deph) ? fmt_val(deph[1].Q_bell) : "  ---  "
        deph_ent = !isempty(deph) ? fmt_val(deph[1].Q_ent) : "  ---  "
        depol_bell = !isempty(depol) ? fmt_val(depol[1].Q_bell) : "  ---  "
        depol_ent = !isempty(depol) ? fmt_val(depol[1].Q_ent) : "  ---  "
        
        if is_ghz_random
            bases = !isempty(deph) ? deph[1].bases : "---"
            @printf("  │ %.2f  │ %s │ %s │  %s  │ %s │  %s  │\n", 
                    p, bases, deph_bell, deph_ent, depol_bell, depol_ent)
        else
            @printf("  │ %.2f  │ %s │  %s  │ %s │  %s  │\n", 
                    p, deph_bell, deph_ent, depol_bell, depol_ent)
        end
    end
    
    if is_ghz_random
        println("  └───────┴───────┴──────────┴───────────┴──────────┴───────────┘")
    else
        println("  └───────┴──────────┴───────────┴──────────┴───────────┘")
    end
    println()
end

# =============================================================================
# PART 3: MCWF COMPARISON
# =============================================================================

println("═"^70)
println("  PART 3: MCWF (N=3, M=$N_trajectories_quick)")
println("═"^70)
println()

# Collect MCWF results
for state in test_states[1:2]
    for noise in noise_models[1:1]  # dephasing only for quick test
        for p in [0.1, 0.2]
            result = state.gen(N_dm)
            if result isa Tuple
                ψ, bases = result
            else
                ψ = result
                bases = ""
            end
            trajectories = generate_trajectories(ψ, noise, p, N_dm, N_trajectories_quick)
            
            t = @elapsed Q_bell, Q_ent, θ_opt = get_bell_correlator(trajectories; 
                method=opt_default.method, max_iter=opt_default.max_iter)
            
            push!(all_results, (
                state=state.name, N=N_dm, rep=:mcwf, mixed=true, 
                noise=String(noise), p=p, Q_bell=Q_bell, Q_ent=Q_ent, time=t,
                bases=bases
            ))
        end
    end
end

# Print MCWF tables
for state in test_states[1:2]
    println("  ── $(state.label) ──")
    println("  ┌───────┬──────────────────────┐")
    println("  │   p   │      dephasing       │")
    println("  │       │  Q_bell  │   Q_ent   │")
    println("  ├───────┼──────────┼───────────┤")
    
    for p in [0.1, 0.2]
        deph = filter(r -> r.state == state.name && r.noise == "dephasing" && 
                      r.rep == :mcwf && abs(r.p - p) < 0.01, all_results)
        
        deph_bell = !isempty(deph) ? @sprintf("%+6.2f", deph[1].Q_bell) : "  ---  "
        deph_ent = !isempty(deph) ? @sprintf("%+6.2f", deph[1].Q_ent) : "  ---  "
        
        @printf("  │ %.2f  │ %s │  %s  │\n", p, deph_bell, deph_ent)
    end
    println("  └───────┴──────────┴───────────┘")
    println()
end

# =============================================================================
# PART 4: Q_bell vs p_noise SWEEP (Grid Plot)
# =============================================================================

println("═"^70)
println("  PART 4: Q_bell vs p_noise SWEEP")
println("═"^70)
println()

# Reduced state set for sweep (skip ghz_random which regenerates each call)
sweep_states = [
    #(name="plus_product", label="|+⟩⊗N", gen=make_plus_product, entangled=false),
    (name="ghz_z", label="GHZ (Z)", gen=N -> make_ghz_z(N)[1], entangled=true),
   # (name="star_graph", label="Star Graph", gen=make_star_graph, entangled=true),
    (name="star_T", label="Star+T", gen=make_star_graph_T, entangled=true),
]

# Storage: Dict[(state, N, mode, opt_name)] -> Dict[noise -> (p_values, Q_values, Q_errors, times)]
# Q_errors is 0 for DM, bootstrap std for MCWF
sweep_data = Dict{Tuple{String,Int,Symbol,String}, Dict{Symbol,Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}}}()

# Generate filename helper
function make_data_filename(; state::String, N::Int, mode::Symbol, 
                             optimizer::String, M_traj::Int)
    if mode == :dm
        return @sprintf("data_Q_vs_p_%s_N%02d_dm_%s_M0.txt", state, N, optimizer)
    else
        return @sprintf("data_Q_vs_p_%s_N%02d_mcwf_%s_M%d.txt", state, N, optimizer, M_traj)
    end
end

# Grid search using Iterators.product - include optimizers!
configs = collect(Iterators.product(sweep_states, N_range_sweep, modes, optimizers))
total_configs = length(configs)
n_noise = length(noise_models)
n_p = length(p_noise_range)

println("  Total configurations: $total_configs × $n_noise noise models × $n_p p values")
println()

for (idx, (state, N, mode, opt)) in enumerate(configs)
    M_traj = mode == :mcwf ? N_trajectories_sweep : 0
    
    # Print current config with counter
    mode_str = mode == :dm ? "DM" : "MCWF(M=$M_traj)"
    println("  [$idx/$total_configs] $(state.label) | N=$N | $mode_str | $(opt.label)")
    
    ψ_init = state.gen(N)
    config_data = Dict{Symbol,Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}}()
    
    for (noise_idx, noise) in enumerate(noise_models)
        p_vals = Float64[]
        Q_vals = Float64[]
        Q_errs = Float64[]  # Bootstrap errors for MCWF, 0 for DM
        t_vals = Float64[]
        
        # Progress bar for p values
        progress_desc = "      $noise: "
        @showprogress dt=0.5 desc=progress_desc for p in p_noise_range
            if mode == :dm
                ρ = apply_noise_dm(ψ_init, noise, p, N)
                t = @elapsed Q_bell, Q_ent, θ_opt = get_bell_correlator(ρ; 
                    method=opt.method, max_iter=opt.max_iter)
                Q_err = 0.0  # No error for DM
            else
                trajectories = generate_trajectories(ψ_init, noise, p, N, M_traj)
                t = @elapsed Q_bell, Q_ent, θ_opt = get_bell_correlator(trajectories; 
                    method=opt.method, max_iter=opt.max_iter)
                # Bootstrap error estimation for MCWF (with error handling)
                Q_err = try
                    _, _, Q_bell_std, _ = bootstrap_bell_error(trajectories, θ_opt; n_bootstrap=30)
                    isfinite(Q_bell_std) ? Q_bell_std : 0.0
                catch
                    0.0  # Fallback if bootstrap fails
                end
            end
            
            push!(p_vals, p)
            push!(Q_vals, Q_bell)
            push!(Q_errs, Q_err)
            push!(t_vals, t)
        end
        
        config_data[noise] = (p_vals, Q_vals, Q_errs, t_vals)
        avg_time = round(sum(t_vals)/length(t_vals), digits=3)
        println("        → Q_bell ∈ [$(round(minimum(Q_vals), digits=2)), $(round(maximum(Q_vals), digits=2))], avg_t=$(avg_time)s")
    end
    
    sweep_data[(state.name, N, mode, opt.name)] = config_data
    
    # Save data file
    filename = make_data_filename(state=state.name, N=N, mode=mode, 
                                   optimizer=opt.name, M_traj=M_traj)
    open(joinpath(DATA_DIR, filename), "w") do io
        println(io, "# Q_bell vs p_noise for $(state.label), N=$N, mode=$mode")
        println(io, "# Optimizer: $(opt.label), max_iter=$(opt.max_iter)")
        mode == :mcwf && println(io, "# Trajectories: M=$M_traj")
        println(io, "# Generated: $(Dates.now())")
        println(io, "#")
        println(io, "# p_noise  dephasing  depolarizing  amplitude_damping")
        for (i, p) in enumerate(p_noise_range)
            @printf(io, "%.4f  %+.6f  %+.6f  %+.6f\n", 
                    p, config_data[:dephasing][2][i], 
                    config_data[:depolarizing][2][i], 
                    config_data[:amplitude_damping][2][i])
        end
    end
    println()
end

# =============================================================================
# GENERATE GRID PLOTS
# =============================================================================

println("═"^70)
println("  GENERATING GRID PLOTS")
println("═"^70)
println()

# Noise and state configuration
noise_colors = Dict(:dephasing => :blue, :depolarizing => :red, :amplitude_damping => :green)
noise_labels = Dict(:dephasing => "Dephasing", :depolarizing => "Depolarizing", 
                    :amplitude_damping => "Amp. Damping")
opt_colors = Dict("spsa_adam" => :blue, "lbfgs" => :green, "autograd_adam" => :red)
opt_markers = Dict("spsa_adam" => :circle, "lbfgs" => :square, "autograd_adam" => :diamond)

# Filter to entangled states only for main plots
entangled_states = filter(s -> s.entangled, sweep_states)

# -------------------------------------------------------------------------
# PLOT 1: Q_bell vs p_noise - one grid per optimizer/mode
# Rows = noise models, Cols = states
# -------------------------------------------------------------------------
for opt in optimizers
    for mode in modes
        n_rows = length(noise_models)
        n_cols = length(entangled_states)
        plots_array = []
        
        for (row_idx, noise) in enumerate(noise_models)
            for (col_idx, state) in enumerate(entangled_states)
                
                # Use first N in sweep range
                N = N_range_sweep[1]
                data = get(sweep_data, (state.name, N, mode, opt.name), nothing)
                
                p = plot(
                    xlabel = row_idx == n_rows ? L"p_\mathrm{noise}" : "",
                    ylabel = col_idx == 1 ? L"Q_\mathrm{Bell}" : "",
                    title = row_idx == 1 ? state.label : "",
                    legend = false,
                    xlims = (0, maximum(p_noise_range)),
                    ylims = (0, 1.1),
                    grid = true,
                    framestyle = :box,
                    tickfontsize = 10,
                    guidefontsize = 12,
                    titlefontsize = 12
                )
                
                # Add noise label on right side for first column
                if col_idx == n_cols
                    annotate!(p, maximum(p_noise_range)*0.95, 0.95, 
                             text(noise_labels[noise], 9, :right))
                end
                
                if data !== nothing && haskey(data, noise)
                    p_vals, Q_vals, Q_errs, _ = data[noise]
                    # Clamp to non-negative for display
                    Q_display = max.(Q_vals, 0.0)
                    
                    if mode == :mcwf && any(Q_errs .> 0)
                        plot!(p, p_vals, Q_display, 
                              ribbon=Q_errs, fillalpha=0.3,
                              color=noise_colors[noise], linewidth=2.5, 
                              marker=:circle, markersize=3,
                              label="")
                    else
                        plot!(p, p_vals, Q_display, 
                              color=noise_colors[noise], linewidth=2.5, 
                              marker=:circle, markersize=3,
                              label="")
                    end
                end
                
                push!(plots_array, p)
            end
        end
        
        mode_label = mode == :dm ? "DM" : "MCWF (M=$N_trajectories_sweep)"
        fig = plot(plots_array..., 
                   layout=(n_rows, n_cols), 
                   size=(400*n_cols, 300*n_rows),
                   margin=8Plots.mm,
                   plot_title=L"Q_\mathrm{Bell}\ \mathrm{vs}\ p_\mathrm{noise}" * " - $(opt.label), $mode_label, N=$(N_range_sweep[1])")
        
        fig_filename = @sprintf("fig_Q_bell_grid_%s_%s.png", mode, opt.name)
        savefig(fig, joinpath(FIG_DIR, fig_filename))
        println("  Saved: figures/$fig_filename")
    end
end

# -------------------------------------------------------------------------
# PLOT 2: DM vs MCWF comparison
# Rows = noise models, Cols = states
# -------------------------------------------------------------------------
println()
println("  Generating DM vs MCWF comparison plots...")

for opt in optimizers
    n_rows = length(noise_models)
    n_cols = length(entangled_states)
    plots_array = []
    
    for (row_idx, noise) in enumerate(noise_models)
        for (col_idx, state) in enumerate(entangled_states)
            
            N = N_range_sweep[1]
            dm_data = get(sweep_data, (state.name, N, :dm, opt.name), nothing)
            mcwf_data = get(sweep_data, (state.name, N, :mcwf, opt.name), nothing)
            
            p = plot(
                xlabel = row_idx == n_rows ? L"p_\mathrm{noise}" : "",
                ylabel = col_idx == 1 ? L"Q_\mathrm{Bell}" : "",
                title = row_idx == 1 ? state.label : "",
                legend = (row_idx == 1 && col_idx == n_cols) ? :topright : false,
                xlims = (0, maximum(p_noise_range)),
                ylims = (0, 1.1),
                grid = true,
                framestyle = :box,
                tickfontsize = 10,
                guidefontsize = 12,
                titlefontsize = 12
            )
            
            # Add noise label
            if col_idx == n_cols && row_idx > 1
                annotate!(p, maximum(p_noise_range)*0.95, 0.95, 
                         text(noise_labels[noise], 9, :right))
            end
            
            if dm_data !== nothing && haskey(dm_data, noise)
                p_vals, Q_vals, _, _ = dm_data[noise]
                Q_display = max.(Q_vals, 0.0)
                lbl = (row_idx == 1 && col_idx == n_cols) ? "DM (exact)" : ""
                plot!(p, p_vals, Q_display, 
                      color=noise_colors[noise], linewidth=2.5, linestyle=:solid,
                      label=lbl)
            end
            
            if mcwf_data !== nothing && haskey(mcwf_data, noise)
                p_vals, Q_vals, Q_errs, _ = mcwf_data[noise]
                Q_display = max.(Q_vals, 0.0)
                lbl = (row_idx == 1 && col_idx == n_cols) ? "MCWF (M=$N_trajectories_sweep)" : ""
                if any(Q_errs .> 0)
                    plot!(p, p_vals, Q_display, ribbon=Q_errs, fillalpha=0.3,
                          color=noise_colors[noise], linewidth=2, linestyle=:dash,
                          label=lbl)
                else
                    plot!(p, p_vals, Q_display, 
                          color=noise_colors[noise], linewidth=2, linestyle=:dash,
                          label=lbl)
                end
            end
            
            push!(plots_array, p)
        end
    end
    
    fig_combined = plot(plots_array..., 
                        layout=(n_rows, n_cols), 
                        size=(400*n_cols, 300*n_rows),
                        margin=8Plots.mm,
                        plot_title=L"Q_\mathrm{Bell}" * ": DM vs MCWF - $(opt.label), N=$(N_range_sweep[1])")
    
    fig_filename = @sprintf("fig_Q_bell_dm_vs_mcwf_%s.png", opt.name)
    savefig(fig_combined, joinpath(FIG_DIR, fig_filename))
    println("  Saved: figures/$fig_filename")
end

# -------------------------------------------------------------------------
# PLOT 3: Optimizer comparison - one combined grid
# Rows = noise models, Cols = states, all optimizers overlaid
# -------------------------------------------------------------------------
println()
println("  Generating optimizer comparison plots...")

for mode in modes
    n_rows = length(noise_models)
    n_cols = length(entangled_states)
    plots_array = []
    
    for (row_idx, noise) in enumerate(noise_models)
        for (col_idx, state) in enumerate(entangled_states)
            
            N = N_range_sweep[1]
            
            p = plot(
                xlabel = row_idx == n_rows ? L"p_\mathrm{noise}" : "",
                ylabel = col_idx == 1 ? L"Q_\mathrm{Bell}" : "",
                title = row_idx == 1 ? state.label : "",
                legend = (row_idx == 1 && col_idx == n_cols) ? :topright : false,
                xlims = (0, maximum(p_noise_range)),
                ylims = (0, 1.1),
                grid = true,
                framestyle = :box,
                tickfontsize = 10,
                guidefontsize = 12,
                titlefontsize = 12,
                legendfontsize = 8
            )
            
            # Add noise label
            if col_idx == n_cols && row_idx > 1
                annotate!(p, maximum(p_noise_range)*0.95, 0.95, 
                         text(noise_labels[noise], 9, :right))
            end
            
            for opt in optimizers
                data = get(sweep_data, (state.name, N, mode, opt.name), nothing)
                if data !== nothing && haskey(data, noise)
                    p_vals, Q_vals, Q_errs, t_vals = data[noise]
                    Q_display = max.(Q_vals, 0.0)
                    avg_t = round(sum(t_vals)/length(t_vals)*1000, digits=1)  # ms
                    opt_label = (row_idx == 1 && col_idx == n_cols) ? "$(opt.label) ($(avg_t)ms)" : ""
                    
                    if mode == :mcwf && any(Q_errs .> 0)
                        plot!(p, p_vals, Q_display,
                              ribbon=Q_errs, fillalpha=0.15,
                              color=opt_colors[opt.name], linewidth=2.5,
                              marker=opt_markers[opt.name], markersize=3,
                              label=opt_label)
                    else
                        plot!(p, p_vals, Q_display,
                              color=opt_colors[opt.name], linewidth=2.5,
                              marker=opt_markers[opt.name], markersize=3,
                              label=opt_label)
                    end
                end
            end
            
            push!(plots_array, p)
        end
    end
    
    mode_label = mode == :dm ? "DM" : "MCWF (M=$N_trajectories_sweep)"
    fig = plot(plots_array...,
               layout=(n_rows, n_cols),
               size=(400*n_cols, 300*n_rows),
               margin=8Plots.mm,
               plot_title="Optimizer Comparison: $mode_label, N=$(N_range_sweep[1])")
    
    fig_filename = @sprintf("fig_optimizer_comparison_%s.png", mode)
    savefig(fig, joinpath(FIG_DIR, fig_filename))
    println("  Saved: figures/$fig_filename")
end

# =============================================================================
# SAVE SUMMARY CSV
# =============================================================================

csv_file = joinpath(DATA_DIR, "bell_correlator_results.csv")
open(csv_file, "w") do io
    println(io, "state,N,representation,mixed,noise,p_noise,Q_bell,Q_ent,time_s,bases")
    for r in all_results
        @printf(io, "%s,%d,%s,%s,%s,%.2f,%.6f,%.6f,%.4f,%s\n",
                r.state, r.N, r.rep, r.mixed, r.noise, r.p, r.Q_bell, r.Q_ent, r.time, r.bases)
    end
end
println()
println("  Saved: data/bell_correlator_results.csv")

println()
println("═"^70)
println("  COMPLETED!")
println("  Data files: $(DATA_DIR)")
println("  Figures: $(FIG_DIR)")
println("═"^70)

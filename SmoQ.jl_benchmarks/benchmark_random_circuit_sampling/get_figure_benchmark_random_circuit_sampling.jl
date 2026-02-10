#!/usr/bin/env julia
"""
get_benchmark_figure.jl - Generate comparison plot of all quantum framework benchmarks

Usage: julia get_benchmark_figure.jl
"""

using Plots
using Printf

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR = @__DIR__
OUTPUT_FILE = joinpath(SCRIPT_DIR, "fig_random_circuit_sampling_evaluation_vs_time.png")

# Define frameworks to plot with colors and markers
FRAMEWORKS = [
    (file="timings_smoq.txt", name="SmoQ.jl", color=:blue, marker=:circle, lw=3),
    (file="timings_qiskit_aer.txt", name="Qiskit Aer", color=:red, marker=:square, lw=2),
    (file="timings_cirq.txt", name="Cirq", color=:green, marker=:diamond, lw=2),
    (file="timings_pennylane.txt", name="PennyLane Lightning", color=:orange, marker=:utriangle, lw=2),
    (file="timings_qilisdk_qilisim.txt", name="QiliSim", color=:purple, marker=:pentagon, lw=2),
    (file="timings_qilisdk_qutip.txt", name="QuTiP", color=:brown, marker=:hexagon, lw=2),
]

# ==============================================================================
# DATA LOADING
# ==============================================================================

function load_timings(filepath::String)
    N_vals = Int[]
    times = Float64[]
    
    if !isfile(filepath)
        @warn "File not found: $filepath"
        return N_vals, times
    end
    
    open(filepath, "r") do f
        for line in eachline(f)
            # Skip comments and empty lines
            strip(line) == "" && continue
            startswith(line, "#") && continue
            
            parts = split(line)
            if length(parts) >= 2
                try
                    push!(N_vals, parse(Int, parts[1]))
                    push!(times, parse(Float64, parts[2]))
                catch
                    continue
                end
            end
        end
    end
    
    return N_vals, times
end

# ==============================================================================
# PLOTTING
# ==============================================================================

println("=" ^ 70)
println("  Generating Benchmark Comparison Figure")
println("=" ^ 70)

# Create plot
plt = plot(
    xlabel="Number of Qubits (N)",
    ylabel="Time (seconds)",
    title="Quantum Framework Benchmark: Random Circuit Sampling\n(1000 gates, 100 shots, shared circuit seed=42)",
    legend=:topleft,
    yscale=:log10,
    size=(1000, 600),
    dpi=150,
    grid=true,
    gridlinewidth=0.5,
    gridstyle=:dash,
    titlefontsize=12,
    legendfontsize=9,
    guidefontsize=11,
    tickfontsize=10,
    margin=5Plots.mm,
    xticks=2:2:20,
    yticks=([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], ["10⁻⁴", "10⁻³", "10⁻²", "10⁻¹", "10⁰", "10¹", "10²"]),
    ylims=(1e-4, 1e3)
)

# Plot each framework
for fw in FRAMEWORKS
    filepath = joinpath(SCRIPT_DIR, fw.file)
    N_vals, times = load_timings(filepath)
    
    if !isempty(N_vals)
        println("  Loaded: $(fw.name) ($(length(N_vals)) data points)")
        plot!(plt, N_vals, times,
            label=fw.name,
            color=fw.color,
            marker=fw.marker,
            markersize=6,
            linewidth=fw.lw
        )
    else
        println("  SKIPPED: $(fw.name) (no data)")
    end
end

# Add horizontal reference lines
hline!(plt, [1.0], linestyle=:dash, color=:gray, alpha=0.5, label="1 second")
hline!(plt, [60.0], linestyle=:dash, color=:lightgray, alpha=0.5, label="1 minute")

# Save plot
savefig(plt, OUTPUT_FILE)
println("\n  Saved: ", basename(OUTPUT_FILE))
println("=" ^ 70)

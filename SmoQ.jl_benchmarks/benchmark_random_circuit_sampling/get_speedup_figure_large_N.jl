#!/usr/bin/env julia
"""
get_speedup_figure_large_N.jl - Speedup ratio bar chart for N ≥ 10 only

Top row:    SmoQ-Float64 (baseline) vs external frameworks
Middle row: SmoQ-Float64-Re_Im (separated layout) vs external frameworks
Bottom row: SmoQ-Float32-AVX-Re_Im (optimized) vs external frameworks

Usage: julia get_speedup_figure_large_N.jl
"""

using Plots
using Printf

SCRIPT_DIR = @__DIR__
OUTPUT_FILE = joinpath(SCRIPT_DIR, "benchmark_speedup_large_N.png")

# Reference files
SMOQ_F64_FILE = "timings_smoq_Float64.txt"
SMOQ_F64_REIM_FILE = "timings_smoq_Float64_Re_Im.txt"
SMOQ_AVX_FILE = "timings_smoq_Float32_AVX_Re_Im.txt"

N_SHOW = [10, 12, 14, 16, 18, 20]

N_COLORS = [:cornflowerblue, :dodgerblue, :steelblue, :royalblue4, :navy, :midnightblue]

# Competitors (external frameworks only)
COMPETITORS = [
    (file="timings_qiskit_aer.txt", name="Qiskit Aer"),
    (file="timings_cirq.txt", name="Cirq"),
    (file="timings_pennylane.txt", name="PennyLane"),
    (file="timings_qilisdk_qilisim.txt", name="QiliSim"),
]

function load_timings(filepath::String)
    data = Dict{Int, Float64}()
    isfile(filepath) || return data
    open(filepath, "r") do f
        for line in eachline(f)
            strip(line) == "" && continue
            startswith(line, "#") && continue
            parts = split(line)
            length(parts) >= 2 || continue
            try; data[parse(Int, parts[1])] = parse(Float64, parts[2]); catch; end
        end
    end
    return data
end

# Load all SmoQ references
smoq_f64 = load_timings(joinpath(SCRIPT_DIR, SMOQ_F64_FILE))
smoq_f64_reim = load_timings(joinpath(SCRIPT_DIR, SMOQ_F64_REIM_FILE))
smoq_avx = load_timings(joinpath(SCRIPT_DIR, SMOQ_AVX_FILE))

# Load competitors
comp_data = []
for comp in COMPETITORS
    data = load_timings(joinpath(SCRIPT_DIR, comp.file))
    push!(comp_data, (name=comp.name, data=data))
end

# ==============================================================================
# Helper: create one row of grouped bars
# ==============================================================================

function make_subplot(ref_data, comp_list, title_str; show_legend=false)
    n_comps = length(comp_list)
    n_sub = length(N_SHOW)
    gap = 1.5
    bw = 0.7

    plt = plot(
        ylabel="Speedup Ratio\n(other / SmoQ)",
        title=title_str,
        titlefontsize=18,
        legendfontsize=10,
        guidefontsize=14,
        tickfontsize=14,
        margin=6Plots.mm,
        bottom_margin=10Plots.mm,
        grid=true,
        gridstyle=:dash,
        gridlinewidth=0.3,
    )

    first_of_N = trues(n_sub)

    for (g, comp) in enumerate(comp_list)
        base_x = (g - 1) * (n_sub + gap)

        for (s, N) in enumerate(N_SHOW)
            x = base_x + s

            if haskey(comp.data, N) && haskey(ref_data, N) && ref_data[N] > 0
                r = comp.data[N] / ref_data[N]
            else
                r = 0.0
            end

            h = log10(max(r, 0.001))
            lbl = (g == 1 && first_of_N[s]) ? "N = $N" : ""
            first_of_N[s] = false

            bar!(plt, [x], [h],
                bar_width=bw,
                color=N_COLORS[s],
                label=lbl,
                alpha=0.9,
                linecolor=:gray40,
                linewidth=0.5,
            )

            val_str = r >= 10 ? @sprintf("%.0f×", r) : @sprintf("%.1f×", r)
            if r < 1.0
                y_ann = h - 0.25
                annotate!(plt, x, y_ann, text(val_str, 10, :left, rotation=45, color=:red))
            else
                y_ann = h + 0.15
                annotate!(plt, x, y_ann, text(val_str, 10, :left, rotation=45, color=:black))
            end
        end
    end

    hline!(plt, [0.0], linestyle=:solid, color=:black, linewidth=2.5, label=false)

    yt_vals = [-2, -1, 0, 1, 2]
    yt_labels = ["0.01×", "0.1×", "1×", "10×", "100×"]
    group_centers = [(g-1) * (n_sub + gap) + (n_sub + 1) / 2 for g in 1:n_comps]
    
    plot!(plt,
        xticks=(group_centers, [c.name for c in comp_list]),
        yticks=(yt_vals, yt_labels),
        xlims=(0, n_comps * (n_sub + gap)),
        ylims=(-2.5, 2.5),
        legend=show_legend ? :bottomright : false,
    )
    return plt
end

# ==============================================================================
# Build three-row figure
# ==============================================================================

# Row 1: SmoQ-Float64 vs external frameworks
top = make_subplot(smoq_f64, comp_data,
    "SmoQ-Float64 (baseline) vs External Frameworks";
    show_legend=true)

# Row 2: SmoQ-Float64-Re_Im vs external frameworks
mid = make_subplot(smoq_f64_reim, comp_data,
    "SmoQ-Float64-Re_Im (separated layout) vs External Frameworks";
    show_legend=false)

# Row 3: SmoQ-Float32-AVX-Re_Im vs external frameworks
bot = make_subplot(smoq_avx, comp_data,
    "SmoQ-Float32-AVX-Re_Im (optimized) vs External Frameworks";
    show_legend=false)

combined = plot(top, mid, bot,
    layout=(3, 1),
    size=(1600, 1400),
    dpi=150,
    plot_title="SmoQ.jl: Random Circuit Sampling — N ≥ 10 (1000 gates, 100 samples)",
    plot_titlefontsize=22,
)

savefig(combined, OUTPUT_FILE)
println("Saved: ", basename(OUTPUT_FILE))

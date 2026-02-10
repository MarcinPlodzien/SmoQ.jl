#!/usr/bin/env julia
"""
get_speedup_figure.jl - Speedup ratio bar chart: SmoQ.jl vs external frameworks

Usage: julia get_speedup_figure.jl
"""

using Plots
using Printf

SCRIPT_DIR = @__DIR__
OUTPUT_FILE = joinpath(SCRIPT_DIR, "fig_random_circuit_sampling_speedup.png")

SMOQ_FILE = "timings_smoq.txt"

N_SHOW = [4, 6, 8, 10, 12, 14, 16, 18, 20]

N_COLORS = [:lightskyblue, :deepskyblue, :steelblue1, :cornflowerblue, :dodgerblue, :steelblue, :royalblue4, :navy, :midnightblue]

COMPETITORS = [
    (file="timings_cirq.txt", name="Cirq"),
    (file="timings_pennylane.txt", name="PennyLane"),
    (file="timings_qiskit_aer.txt", name="Qiskit Aer"),
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

# Load data
smoq = load_timings(joinpath(SCRIPT_DIR, SMOQ_FILE))
comp_data = []
for comp in COMPETITORS
    data = load_timings(joinpath(SCRIPT_DIR, comp.file))
    push!(comp_data, (name=comp.name, data=data))
end

# Build figure
n_comps = length(comp_data)
n_sub = length(N_SHOW)
gap = 1.5
bw = 0.7

plt = plot(
    ylabel="Execution Time Ratio\n(other / SmoQ.jl)",
    title="SmoQ.jl: Random Circuit Sampling Speedup\n(1000 gates, 100 shots, shared circuit seed=42)",
    titlefontsize=16,
    legendfontsize=10,
    guidefontsize=14,
    tickfontsize=14,
    margin=6Plots.mm,
    left_margin=12Plots.mm,
    bottom_margin=10Plots.mm,
    grid=true,
    gridstyle=:dash,
    gridlinewidth=0.3,
    size=(1400, 700),
    dpi=150,
)

first_of_N = trues(n_sub)

for (g, comp) in enumerate(comp_data)
    base_x = (g - 1) * (n_sub + gap)

    for (s, N) in enumerate(N_SHOW)
        x = base_x + s

        if haskey(comp.data, N) && haskey(smoq, N) && smoq[N] > 0
            r = comp.data[N] / smoq[N]
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

yt_vals = [0, 1, 2]
yt_labels = ["1×", "10×", "100×"]
group_centers = [(g-1) * (n_sub + gap) + (n_sub + 1) / 2 for g in 1:n_comps]

plot!(plt,
    xticks=(group_centers, [c.name for c in comp_data]),
    yticks=(yt_vals, yt_labels),
    xlims=(0, n_comps * (n_sub + gap)),
    ylims=(-0.5, 2.8),
    legend=:bottomleft,
    legend_columns=3,
)

savefig(plt, OUTPUT_FILE)
println("Saved: ", basename(OUTPUT_FILE))

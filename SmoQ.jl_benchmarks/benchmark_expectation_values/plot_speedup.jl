#!/usr/bin/env julia
"""
plot_speedup.jl - Speedup ratio bar chart: SmoQ.jl vs external frameworks
                  for expectation value benchmarks

Usage: julia plot_speedup.jl
"""

using Printf

SCRIPT_DIR = @__DIR__
OUTPUT_FILE = joinpath(SCRIPT_DIR, "fig_observables_calculation_speedup.png")

# Baseline framework
SMOQ_KEY = "smoq"

# Competitors to compare against (must match column names in timing_all.csv)
COMPETITORS = [
    (key="cirq",      name="Cirq"),
    (key="pennylane",  name="PennyLane"),
    (key="qiskit",     name="Qiskit"),
]

# ==============================================================================
# DATA LOADING (from timing_all.csv)
# ==============================================================================

function load_csv(path::String)
    data = Dict{String, Dict{Int, Float64}}()
    isfile(path) || error("CSV not found: $path")

    lines = readlines(path)
    header = split(strip(lines[1]), ",")
    keys = header[2:end]  # skip "N"
    for k in keys
        data[k] = Dict{Int, Float64}()
    end

    for line in lines[2:end]
        strip(line) == "" && continue
        parts = split(strip(line), ",")
        N = parse(Int, parts[1])
        for (i, k) in enumerate(keys)
            val = strip(parts[i+1])
            val == "" && continue
            data[k][N] = parse(Float64, val)
        end
    end
    return data
end

csv_path = joinpath(SCRIPT_DIR, "timing_all.csv")
all_data = load_csv(csv_path)

# Get all N values from baseline
smoq_data = all_data[SMOQ_KEY]
N_SHOW = sort(collect(keys(smoq_data)))

N_COLORS = [
    :lightskyblue, :deepskyblue, :steelblue1, :cornflowerblue, :dodgerblue,
    :steelblue, :royalblue4, :navy, :midnightblue, :indigo,
    :purple4, :mediumpurple4, :darkviolet
]
# Trim to match number of N values
N_COLORS = N_COLORS[1:min(length(N_COLORS), length(N_SHOW))]

println("=" ^ 70)
println("  Generating Expectation Values Speedup Figure")
println("=" ^ 70)
println("  Baseline: SmoQ.jl (ComplexF64)")
println("  N values: ", N_SHOW)

# ==============================================================================
# PLOTTING
# ==============================================================================

using Plots

n_comps = length(COMPETITORS)
n_sub = length(N_SHOW)
gap = 1.5
bw = 0.7

plt = plot(
    ylabel="Execution Time Ratio\n(other / SmoQ.jl)",
    title="SmoQ.jl Speedup: Expectation Values (State Prep + Local & NN Correlators)",
    titlefontsize=14,
    legendfontsize=10,
    guidefontsize=14,
    tickfontsize=14,
    margin=6Plots.mm,
    left_margin=12Plots.mm,
    bottom_margin=10Plots.mm,
    top_margin=20Plots.mm,
    grid=true,
    gridstyle=:dash,
    gridlinewidth=0.3,
    size=(1400, 750),
    dpi=150,
)

first_of_N = trues(n_sub)

for (g, comp) in enumerate(COMPETITORS)
    base_x = (g - 1) * (n_sub + gap)

    for (s, N) in enumerate(N_SHOW)
        x = base_x + s

        comp_data = get(all_data, comp.key, nothing)
        if !isnothing(comp_data) && haskey(comp_data, N) && haskey(smoq_data, N) && smoq_data[N] > 0
            r = comp_data[N] / smoq_data[N]
        else
            r = 0.0
        end

        h = log10(max(r, 0.001))
        h_capped = min(h, 3.0)  # cap bars at 1000×
        lbl = (g == 1 && first_of_N[s]) ? "N = $N" : ""
        first_of_N[s] = false

        bar!(plt, [x], [h_capped],
            bar_width=bw,
            color=N_COLORS[s],
            label=lbl,
            alpha=0.9,
            linecolor=:gray40,
            linewidth=0.5,
        )

        val_str = r >= 10 ? @sprintf("%.0f×", r) : @sprintf("%.1f×", r)
        if r < 1.0
            y_ann = h_capped - 0.25
            annotate!(plt, x, y_ann, text(val_str, 9, :left, rotation=45, color=:red))
        else
            y_ann = h_capped + 0.1
            annotate!(plt, x, y_ann, text(val_str, 9, :left, rotation=45, color=:black))
        end
    end
end

hline!(plt, [0.0], linestyle=:solid, color=:black, linewidth=2.5, label=false)

yt_vals = [0, 1, 2, 3]
yt_labels = ["1×", "10×", "100×", "1000×"]
group_centers = [(g-1) * (n_sub + gap) + (n_sub + 1) / 2 for g in 1:n_comps]

plot!(plt,
    xticks=(group_centers, [c.name for c in COMPETITORS]),
    yticks=(yt_vals, yt_labels),
    xlims=(0, n_comps * (n_sub + gap)),
    ylims=(-0.5, 3.8),
    legend=:bottomleft,
    legend_columns=min(n_sub, 5),
)

savefig(plt, OUTPUT_FILE)
println("\n  Saved: ", basename(OUTPUT_FILE))
println("=" ^ 70)

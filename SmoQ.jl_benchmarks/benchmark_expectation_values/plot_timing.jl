#!/usr/bin/env julia
"""
plot_timing.jl - Collect timing data and produce t vs N figure
"""

using Printf

SCRIPT_DIR = @__DIR__

const FRAMEWORKS = [
    ("smoq",        "SmoQ.jl (ComplexF64)",    :blue,    :circle),
    ("smoq_avx",    "SmoQ.jl (Float32 AVX)",   :red,     :diamond),
    ("qiskit",      "Qiskit",                  :green,   :utriangle),
    ("cirq",        "Cirq",                    :purple,  :square),
    ("pennylane",   "PennyLane Lightning",     :orange,  :dtriangle),
]

function read_timing(path)
    ns = Int[]; ts = Float64[]
    isfile(path) || return ns, ts
    for line in readlines(path)
        startswith(line, "#") && continue; strip(line) == "" && continue
        parts = split(strip(line))
        length(parts) >= 2 || continue
        push!(ns, parse(Int, parts[1]))
        push!(ts, parse(Float64, parts[2]))
    end
    return ns, ts
end

println("=" ^ 60)
println("  Timing Data Collection")
println("=" ^ 60)

all_data = Dict{String, Tuple{Vector{Int}, Vector{Float64}}}()
for (key, label, _, _) in FRAMEWORKS
    ns, ts = read_timing(joinpath(SCRIPT_DIR, "timing_$(key).txt"))
    all_data[key] = (ns, ts)
    if !isempty(ns)
        println("  ✓ $label: $(length(ns)) data points")
    else
        println("  ✗ $label: no data")
    end
end

# --- Save CSV ---
csv_path = joinpath(SCRIPT_DIR, "timing_all.csv")
open(csv_path, "w") do f
    write(f, "N," * join([k for (k,_,_,_) in FRAMEWORKS], ",") * "\n")
    all_ns = sort(unique(vcat([ns for (ns,_) in values(all_data)]...)))
    for n in all_ns
        row = ["$n"]
        for (key,_,_,_) in FRAMEWORKS
            ns, ts = all_data[key]
            idx = findfirst(==(n), ns)
            push!(row, isnothing(idx) ? "" : @sprintf("%.6f", ts[idx]))
        end
        write(f, join(row, ",") * "\n")
    end
end
println("\n  Saved CSV: ", basename(csv_path))

let
    all_ns = sort(unique(vcat([ns for (ns,_) in values(all_data)]...)))
    keys_list = [k for (k,_,_,_) in FRAMEWORKS]
    println("\n" * "=" ^ 80)
    println("  Timing Summary: t(N) [seconds]")
    println("=" ^ 80)
    header = @sprintf("  %-5s", "N")
    for k in keys_list; header *= " │ " * @sprintf("%14s", k); end
    println(header)
    println("  " * "─"^5 * ("┼" * "─"^16)^length(keys_list))
    for n in all_ns
        line = @sprintf("  %2d  ", n)
        for k in keys_list
            ns, ts = all_data[k]; idx = findfirst(==(n), ns)
            line *= " │ " * (isnothing(idx) ? @sprintf("%14s", "—") : @sprintf("%14.6f", ts[idx]))
        end
        println(line)
    end
    println("=" ^ 80)
end

# --- Plot ---
using CairoMakie

fig = Figure(size = (900, 600), fontsize = 14)
ax = Axis(fig[1, 1],
    xlabel = "Number of qubits N",
    ylabel = "Execution time [s]",
    title  = "State Preparation + Local & NN Correlator Expectation Values",
    yscale = log10,
    xticks = 2:2:24,
    yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
    ytickformat = values -> [let e = Int(round(log10(v))); e >= 0 ? "10$(Makie.UnicodeFun.to_superscript(e))" : "10⁻$(Makie.UnicodeFun.to_superscript(-e))" end for v in values],
)

colors = Dict("smoq" => :royalblue, "smoq_avx" => :crimson,
              "qiskit" => :forestgreen, "cirq" => :purple,
              "pennylane" => :darkorange, "qilisdk" => :gray50)
markers = Dict("smoq" => :circle, "smoq_avx" => :diamond,
               "qiskit" => :utriangle, "cirq" => :rect,
               "pennylane" => :dtriangle, "qilisdk" => :star5)

for (key, label, _, _) in FRAMEWORKS
    ns, ts = all_data[key]
    isempty(ns) && continue
    scatterlines!(ax, ns, ts,
        label = label,
        color = get(colors, key, :black),
        marker = get(markers, key, :circle),
        markersize = 10,
        linewidth = 2,
    )
end

axislegend(ax, position = :lt, framevisible = true, labelsize = 11)
fig_path = joinpath(SCRIPT_DIR, "fig_observables_calculation_evaluation_vs_time.png")
save(fig_path, fig, px_per_unit = 2)
println("  Saved figure: ", basename(fig_path))

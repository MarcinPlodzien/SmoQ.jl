# Date: 2026
#
#=
================================================================================
    DataManagementHelper.jl - Filename Building & Data Structure Management
================================================================================

Provides reusable functions for:
- Consistent filename building from config_params
- Directory structure creation
- Data loading and saving utilities

USAGE:
    include("utils/helpers/DataManagementHelper.jl")
    using .DataManagementHelper

================================================================================
=#

module DataManagementHelper

using Printf
using DelimitedFiles
using Statistics

export params_to_string, build_filename, create_output_dirs
export build_config_params, load_input_sequence, normalize_sequence
export save_capacity, save_evaluation, save_pearson_correlation, save_summary

# ==============================================================================
# DEFAULT KEY ORDER FOR FILENAMES
# ==============================================================================

# Standard key order for consistent filenames across all scripts
# Uses full descriptive names with Ny (n_rails) before Nx (chain length)
const DEFAULT_KEY_ORDER = [
    "Ny", "Nx", "T", "hamiltonian", "representation", "integrator",
    "encoding", "protocol", "hardware", "task", "traj"
]

# ==============================================================================
# FILENAME BUILDERS
# ==============================================================================

"""
    params_to_string(params; key_order=DEFAULT_KEY_ORDER, separator="_")

Convert config_params dict to string for filenames.
Keys are processed in order; special formatting for 'T' (time) and 'prot' (protocol).

# Arguments
- `params`: Dict{String,Any} of configuration parameters
- `key_order`: Order of keys in output string
- `separator`: String separator between key=value pairs

# Returns
- Formatted string like "task=prediction_Ham=heisenberg_L=4_..."
"""
function params_to_string(params; key_order=DEFAULT_KEY_ORDER, separator="_")
    parts = String[]
    for k in key_order
        if haskey(params, k)
            v = params[k]
            # Clean up protocol names
            if k == "protocol"
                v = replace(string(v), " " => "", "(" => "", ")" => "")
            end
            # Format time with fixed width
            if k == "T" && v isa Number
                v = @sprintf("%05.2f", v)
            end
            # Zero-pad integer values for Ny and Nx
            if (k == "Ny" || k == "Nx") && v isa Integer
                v = @sprintf("%02d", v)
            end
            # Use hyphen between key and value: key-value
            push!(parts, "$(k)-$(v)")
        end
    end
    return join(parts, separator)
end

"""
    build_filename(prefix::String, params, ext::String; key_order=DEFAULT_KEY_ORDER)

Build complete filename from prefix, parameters, and extension.

# Arguments
- `prefix`: Filename prefix (e.g., "capacity", "evaluation_grid")
- `params`: Config parameters dict
- `ext`: File extension (e.g., ".txt", ".png")
- `key_order`: Order of keys in filename

# Returns
- Complete filename like "capacity_task=prediction_Ham=heisenberg_L=4.txt"

# Example
```julia
params = Dict("task"=>:prediction, "Ham"=>:heisenberg, "L"=>4, "hardware"=>"gpu")
build_filename("capacity", params, ".txt")
# => "capacity_task=prediction_Ham=heisenberg_L=4_hardware=gpu.txt"
```
"""
function build_filename(prefix::String, params, ext::String; key_order=DEFAULT_KEY_ORDER)
    return prefix * "_" * params_to_string(params; key_order=key_order) * ext
end

# ==============================================================================
# DIRECTORY STRUCTURE
# ==============================================================================

"""
    create_output_dirs(results_root, config_name, task)

Create standard output directory structure for QRC experiments.

Creates:
    {results_root}/{config_name}/{task}/
    ├── data/
    │   ├── capacity/
    │   ├── evaluation/
    │   ├── observables/
    │   └── pearson_correlation/
    └── figures/
        ├── capacity/
        ├── evaluation/
        └── pearson_correlation/

# Arguments
- `results_root`: Root results directory (e.g., "/path/to/RESULTS")
- `config_name`: Name of this config/run (e.g., "master_run_001")
- `task`: Task type symbol (e.g., :prediction, :memory)

# Returns
- Base directory path: {results_root}/{config_name}/{task}
"""
function create_output_dirs(results_root, config_name, task)
    base = joinpath(results_root, config_name, string(task))

    # Data directories
    mkpath(joinpath(base, "data", "capacity"))
    mkpath(joinpath(base, "data", "evaluation"))
    mkpath(joinpath(base, "data", "observables"))
    mkpath(joinpath(base, "data", "pearson_correlation"))

    # Figure directories
    mkpath(joinpath(base, "figures", "capacity"))
    mkpath(joinpath(base, "figures", "evaluation"))
    mkpath(joinpath(base, "figures", "pearson_correlation"))

    return base
end

# ==============================================================================
# CONFIG PARAMS BUILDER
# ==============================================================================

"""
    build_config_params(; Ny, Nx, T, hamiltonian, representation, integrator,
                         encoding, protocol, task=nothing, hardware=nothing,
                         traj=nothing)

Build standardized config_params dict for filenames and titles.
Uses full descriptive key names matching the DEFAULT_KEY_ORDER.

# Arguments (all keyword)
- `Ny`: Number of rails (vertical grid dimension)
- `Nx`: Chain length (horizontal grid dimension)
- `T`: Evolution time
- `hamiltonian`: Hamiltonian type (:heisenberg, :TFIM_ZZ_X)
- `representation`: Simulation representation (:dm, :mcwf)
- `integrator`: Integrator type (:exact, :trotter)
- `encoding`: Encoding mode (:batch_causal, :batch_lookahead, :broadcast)
- `protocol`: Protocol name string (e.g., "Protocol1")
- `task`: Task type (:prediction, :memory) - optional
- `hardware`: Hardware string ("cpu" or "gpu") - optional
- `traj`: Number of trajectories (optional, for MCWF)

# Returns
- Dict{String,Any} ready for build_filename and plotting functions
"""
function build_config_params(; Ny, Nx, T, hamiltonian, representation, integrator,
                               encoding, protocol, task=nothing, hardware=nothing,
                               traj=nothing)
    N = Nx * Ny
    params = Dict{String, Any}(
        "Ny" => Ny,
        "Nx" => Nx,
        "N" => N,
        "T" => T,
        "hamiltonian" => hamiltonian,
        "representation" => representation,
        "integrator" => integrator,
        "encoding" => encoding,
        "protocol" => protocol
    )
    if !isnothing(task)
        params["task"] = task
    end
    if !isnothing(traj)
        params["traj"] = traj
    end
    if !isnothing(hardware)
        params["hardware"] = string(hardware)
    end
    return params
end

# ==============================================================================
# DATA LOADING UTILITIES
# ==============================================================================

"""
    load_input_sequence(path; n_samples=nothing)

Load input sequence from file and optionally truncate.

# Arguments
- `path`: Path to input file (e.g., santafe.txt)
- `n_samples`: Number of samples to use (nothing = all)

# Returns
- Vector{Float64} of input values
"""
function load_input_sequence(path; n_samples=nothing)
    data = vec(readdlm(path))
    if !isnothing(n_samples)
        data = data[1:min(n_samples, length(data))]
    end
    return data
end

"""
    normalize_sequence(seq; method=:minmax)

Normalize sequence to [0,1] range.

# Arguments
- `seq`: Input sequence vector
- `method`: Normalization method (:minmax or :zscore)

# Returns
- Normalized sequence
- Normalization parameters (min, max) or (mean, std)
"""
function normalize_sequence(seq; method=:minmax)
    if method == :minmax
        min_v, max_v = extrema(seq)
        range_v = max_v - min_v
        if range_v < 1e-12
            range_v = 1.0
        end
        normalized = (seq .- min_v) ./ range_v
        return normalized, (min_v, max_v)
    else  # :zscore
        μ = mean(seq)
        σ = std(seq)
        if σ < 1e-12
            σ = 1.0
        end
        normalized = (seq .- μ) ./ σ
        return normalized, (μ, σ)
    end
end

# ==============================================================================
# DATA SAVING UTILITIES
# ==============================================================================

"""
    save_capacity(config_params, capacities, data_dir)

Save C(τ) capacity trace to file.

# Arguments
- `config_params`: Config parameters dict for filename
- `capacities`: Vector of C(τ) values
- `data_dir`: Directory to save to (e.g., base_dir/data/capacity)
"""
function save_capacity(config_params, capacities, data_dir)
    fname = build_filename("capacity", config_params, ".txt")
    fpath = joinpath(data_dir, fname)
    open(fpath, "w") do io
        println(io, "# Memory capacity C(τ) per delay τ")
        println(io, "# tau\tC")
        for (τ, C) in enumerate(capacities)
            println(io, "$(τ)\t$(round(C, digits=8))")
        end
    end
end

"""
    save_evaluation(config_params, preds, data_dir)

Save evaluation predictions for each tau value.

# Arguments
- `config_params`: Config parameters dict for filename
- `preds`: Dict{τ => (k_obs, k_target, y_true, y_pred)}
- `data_dir`: Directory to save to (e.g., base_dir/data/evaluation)
"""
function save_evaluation(config_params, preds, data_dir)
    key_order = [DEFAULT_KEY_ORDER..., "tau"]
    for (tau, (k_obs, k_target, y_true, y_pred)) in preds
        params_with_tau = copy(config_params)
        params_with_tau["tau"] = tau
        fname = build_filename("evaluation", params_with_tau, ".txt"; key_order=key_order)
        open(joinpath(data_dir, fname), "w") do io
            println(io, "# k_obs\tk_target\ty_true\ty_pred")
            for i in 1:length(y_true)
                println(io, "$(k_obs[i])\t$(k_target[i])\t$(y_true[i])\t$(y_pred[i])")
            end
        end
    end
end

"""
    save_pearson_correlation(config_params, preds, data_dir)

Save Pearson correlation r(τ) values to file.

# Arguments
- `config_params`: Config parameters dict for filename
- `preds`: Dict{τ => (k_obs, k_target, y_true, y_pred)}
- `data_dir`: Directory to save to (e.g., base_dir/data/pearson_correlation)
"""
function save_pearson_correlation(config_params, preds, data_dir)
    tau_r_pairs = Tuple{Int, Float64}[]
    for (tau, (k_obs, k_target, y_true, y_pred)) in preds
        r = cov(y_true, y_pred) / (std(y_true) * std(y_pred))
        push!(tau_r_pairs, (tau, r))
    end
    sort!(tau_r_pairs, by=x->x[1])

    fname = build_filename("pearson_correlation", config_params, ".txt")
    open(joinpath(data_dir, fname), "w") do io
        println(io, "# Pearson correlation r = cov(y_true, y_pred) / (σ_y_true * σ_y_pred)")
        println(io, "# tau\tr")
        for (tau, r) in tau_r_pairs
            println(io, "$tau\t$(round(r, digits=6))")
        end
    end
end

"""
    save_summary(results_root, config_name, results::Dict; config=nothing)

Save summary file with all results.

# Arguments
- `results_root`: Root results directory
- `config_name`: Config name
- `results`: Dict of results keyed by config string
- `config`: Optional CONFIG dict to include metadata
"""
function save_summary(results_root, config_name, results::Dict; config=nothing)
    summary_path = joinpath(results_root, config_name, "summary.txt")
    open(summary_path, "w") do f
        println(f, "QRC Results Summary")
        println(f, "Config: $config_name")
        if !isnothing(config)
            haskey(config, "L_range") && println(f, "L: $(config["L_range"])")
            haskey(config, "T_evol_range") && println(f, "T_evol: $(config["T_evol_range"])")
            haskey(config, "hardware") && println(f, "Hardware: $(config["hardware"])")
        end
        println(f, "=" ^ 60)
        for (cfg, r) in sort(collect(results), by=x->x[1])
            @printf(f, "\n[%s]\n", cfg)
            for (k, v) in pairs(r)
                if v isa Float64
                    @printf(f, "  %s = %.4f\n", k, v)
                else
                    println(f, "  $k = $v")
                end
            end
        end
    end
    return summary_path
end

end # module DataManagementHelper

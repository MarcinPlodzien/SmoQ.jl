# Date: 2026
#
#=
================================================================================
    PlottingHelper.jl - Reusable QRC Plotting Functions
================================================================================

Provides consistent plotting for QRC experiments:
- plot_capacity: C(τ) curves with stretched exponential fit
- plot_evaluation_grid: 2×4 grid of predictions vs truth
- plot_pearson_correlation: r(τ) correlation curves
- save_* functions for data export

USAGE:
    include("utils/PlottingHelper.jl")
    using .PlottingHelper

================================================================================
=#

module PlottingHelper

using Plots
using LsqFit
using Statistics
using Printf
using DelimitedFiles

# Import filename utilities from DataManagementHelper (single source of truth)
include("DataManagementHelper.jl")
using .DataManagementHelper: params_to_string, build_filename, DEFAULT_KEY_ORDER

export plot_capacity, plot_evaluation_grid, plot_pearson_correlation
export save_capacity, save_evaluation, save_pearson_correlation
export fit_stretched_exp, compute_usable_horizon, fit_all_capacity_models
export params_to_string, build_filename

# ==============================================================================
# CAPACITY FITTING
# ==============================================================================

"""Fit stretched exponential: C(τ) = C_∞ + α·exp(-(τ/τ_d)^β)"""
function fit_stretched_exp(tau_range, capacities)
    try
        model(τ, p) = p[1] .+ p[2] .* exp.(-(τ ./ p[3]).^p[4])

        C_inf0 = minimum(capacities)
        alpha0 = maximum(capacities) - C_inf0
        tau_d0 = 10.0
        beta0 = 1.0
        p0 = [C_inf0, alpha0, tau_d0, beta0]

        lower = [0.0, 0.0, 0.1, 0.1]
        upper = [1.0, 10.0, 500.0, 3.0]

        fit = curve_fit(model, Float64.(tau_range), Float64.(capacities), p0, lower=lower, upper=upper)
        params = coef(fit)

        y_pred = model(tau_range, params)
        SS_res = sum((capacities .- y_pred).^2)
        SS_tot = sum((capacities .- mean(capacities)).^2)
        R_sq = SS_tot > 0 ? max(0.0, min(1.0, 1.0 - SS_res / SS_tot)) : 0.0

        return (C_inf=params[1], alpha=params[2], tau_d=params[3], beta=params[4]), R_sq, true
    catch
        return (C_inf=0.0, alpha=1.0, tau_d=10.0, beta=1.0), 0.0, false
    end
end

"""Compute usable horizon τ_θ from stretched exponential fit."""
function compute_usable_horizon(str_params, theta)
    C_inf, alpha, tau_d, beta = str_params.C_inf, str_params.alpha, str_params.tau_d, str_params.beta

    if theta <= C_inf; return Inf, false; end
    if theta >= alpha + C_inf; return 0.0, false; end

    try
        inner = (theta - C_inf) / alpha
        if inner <= 0 || inner >= 1; return 0.0, false; end
        tau_theta = tau_d * (-log(inner))^(1/beta)
        if isnan(tau_theta) || isinf(tau_theta) || tau_theta < 0; return 0.0, false; end
        return tau_theta, true
    catch
        return 0.0, false
    end
end

"""Fit all capacity models and return best."""
function fit_all_capacity_models(tau_range, capacities)
    str_params, str_R2, str_ok = fit_stretched_exp(tau_range, capacities)
    return Dict(
        "stretched" => (params=str_params, R2=str_R2, success=str_ok),
        "best_model" => "stretched",
        "best_R2" => str_R2
    )
end

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

"""
    plot_capacity(config_params; capacities, c_total, figures_dir, theta_thresholds)

Generate C(τ) vs τ plot with stretched exponential fit overlaid.
Takes config_params dict with keys: task, Ham, R, L, N, T, enc, prot, mode, int, [traj], [hardware]
"""
function plot_capacity(config_params; capacities, c_total, figures_dir,
                       theta_thresholds=[0.81, 0.64, 0.49, 0.36, 0.25])
    tau_range = collect(1:length(capacities))
    tau_fit = range(1, length(capacities), length=100)

    fits = fit_all_capacity_models(tau_range, capacities)
    str_fit = fits["stretched"]

    # Extract values from config_params for title (using new full key names)
    traj_str = haskey(config_params, "traj") ? ", N_traj=$(config_params["traj"])" : ""
    hw_str = haskey(config_params, "hardware") ? " | $(uppercase(string(config_params["hardware"])))" : ""
    rep_str = haskey(config_params, "representation") ? uppercase(string(config_params["representation"])) : ""
    int_str = haskey(config_params, "integrator") ? uppercase(string(config_params["integrator"])) : ""
    task_label = haskey(config_params, "task") && config_params["task"] == :prediction ? "PREDICTION" : "MEMORY"
    Nx = config_params["Nx"]
    Ny = config_params["Ny"]
    title_str = "$(task_label) [$rep_str$traj_str, $int_str] | C_total=$(round(c_total, digits=2)) | R²=$(round(fits["best_R2"], digits=3))\n$(config_params["hamiltonian"]) | Nx=$Nx, Ny=$Ny, N=$(Nx*Ny) | T=$(config_params["T"]) | $(config_params["encoding"]) | $(config_params["protocol"])$hw_str"



    p = plot(tau_range, capacities,
             title=title_str,
             xlabel="τ", ylabel="C(τ)",
             label="Data", lw=2.5, marker=:circle, markersize=4, color=:black,
             grid=true, size=(1000, 700), titlefontsize=10,
             legend=:topright, legendfontsize=7, ylims=(0, 1))

    if str_fit.success
        sp = str_fit.params
        C_str = sp.C_inf .+ sp.alpha .* exp.(-(tau_fit ./ sp.tau_d).^sp.beta)
        label_str = "C(τ) = C_∞ + α·exp(-(τ/τ_d)^β)\nC_∞=$(round(sp.C_inf,digits=3)), α=$(round(sp.alpha,digits=3)), τ_d=$(round(sp.tau_d,digits=2)), β=$(round(sp.beta,digits=3))"
        plot!(p, tau_fit, C_str, label=label_str, lw=2.5, linestyle=:solid, color=:blue)
    end

    theta_colors = [:orange, :magenta, :cyan, :yellow, :green]
    tau_thetas = Dict{Float64, Float64}()

    for (i, theta) in enumerate(theta_thresholds)
        tau_theta, tau_theta_valid = compute_usable_horizon(str_fit.params, theta)
        color = theta_colors[mod1(i, length(theta_colors))]
        rho = sqrt(theta)

        hline!([theta], label="C≥$(theta) (|ρ|≥$(round(rho, digits=2)))", lw=1.5, linestyle=:dash, color=color)

        if tau_theta_valid && tau_theta > 0 && tau_theta < maximum(tau_range)
            vline!([tau_theta], label="τ_$(theta)=$(round(tau_theta, digits=1))", lw=1.5, linestyle=:dash, color=color)
            tau_thetas[theta] = tau_theta
        end
    end

    fname = build_filename("capacity", config_params, ".png")
    savefig(p, joinpath(figures_dir, fname))

    fits["tau_thetas"] = tau_thetas
    return fits
end

"""
    plot_evaluation_grid(config_params; capacities, preds, figures_dir)

Plot 2×4 evaluation grid showing predictions vs truth for selected τ values.
"""
function plot_evaluation_grid(config_params; capacities, preds, figures_dir)
    tau_vals = [1, 2, 5, 10, 15, 20, 25, 50]

    # Build title from config_params (using new full key names)
    traj_str = haskey(config_params, "traj") ? " | N_traj=$(config_params["traj"])" : ""
    hw_str = haskey(config_params, "hardware") ? " | $(uppercase(config_params["hardware"]))" : ""
    task_label = haskey(config_params, "task") && config_params["task"] == :prediction ? "PREDICTION" : "MEMORY"
    int_str = haskey(config_params, "integrator") ? " | $(uppercase(string(config_params["integrator"])))" : ""
    title_str = "$(task_label) | $(config_params["hamiltonian"]) | Ny=$(config_params["Ny"]), Nx=$(config_params["Nx"]) | T=$(config_params["T"]) | $(config_params["encoding"]) | $(config_params["protocol"])$int_str$traj_str$hw_str"

    p = plot(layout=(2, 4), size=(1600, 800), plot_title=title_str, titlefontsize=11)

    for (idx, tau) in enumerate(tau_vals)
        if haskey(preds, tau) && tau <= length(capacities)
            k_obs, k_target, y_test, y_pred = preds[tau]
            sample_idx = 1:length(y_test)
            cap = capacities[tau]

            plot!(p[idx], sample_idx, y_test, label="True", lw=2, color=:black, marker=:circle, markersize=2)
            plot!(p[idx], sample_idx, y_pred, label="Pred", lw=2, color=:red, linestyle=:dash, marker=:utriangle, markersize=2)
            r = cov(y_test, y_pred) / (std(y_test) * std(y_pred))
            title!(p[idx], "τ=$tau | C=$(round(cap, digits=3)) | r=$(round(r, digits=3))")
        else
            plot!(p[idx], [], [], title="τ=$tau N/A")
        end
    end

    fname = build_filename("evaluation_grid", config_params, ".png")
    savefig(p, joinpath(figures_dir, fname))
end

"""
    plot_pearson_correlation(config_params; preds, figures_dir)

Plot Pearson correlation r vs τ.
"""
function plot_pearson_correlation(config_params; preds, figures_dir)
    tau_r_pairs = Tuple{Int, Float64}[]
    for (tau, (k_obs, k_target, y_true, y_pred)) in preds
        r = cov(y_true, y_pred) / (std(y_true) * std(y_pred))
        push!(tau_r_pairs, (tau, r))
    end
    sort!(tau_r_pairs, by=x->x[1])

    tau_vals = [p[1] for p in tau_r_pairs]
    r_vals = [p[2] for p in tau_r_pairs]

    traj_str = haskey(config_params, "traj") ? " | N_traj=$(config_params["traj"])" : ""
    hw_str = haskey(config_params, "hardware") ? " | $(uppercase(config_params["hardware"]))" : ""
    task_label = haskey(config_params, "task") && config_params["task"] == :prediction ? "PREDICTION" : "MEMORY"
    Nx = config_params["Nx"]
    Ny = config_params["Ny"]
    title_str = "$(task_label) | Pearson Correlation r(τ)\n$(config_params["hamiltonian"]) | Nx=$Nx, Ny=$Ny, N=$(Nx*Ny) | T=$(config_params["T"]) | $(config_params["encoding"]) | $(config_params["protocol"])$traj_str$hw_str"

    p = plot(tau_vals, r_vals,
             title=title_str, xlabel="τ", ylabel="r (Pearson correlation)",
             label="r = cov(y,ŷ)/(σy·σŷ)", lw=2.5, marker=:circle, markersize=4, color=:blue,
             grid=true, size=(1000, 700), titlefontsize=10,
             legend=:topright, legendfontsize=8, ylims=(0, 1))

    hline!([0.9], label="r=0.9", lw=1.5, linestyle=:dash, color=:orange)
    hline!([0.8], label="r=0.8", lw=1.5, linestyle=:dash, color=:magenta)
    hline!([0.7], label="r=0.7", lw=1.5, linestyle=:dash, color=:cyan)

    fname = build_filename("pearson_correlation", config_params, ".png")
    savefig(p, joinpath(figures_dir, fname))
end

# ==============================================================================
# SAVE FUNCTIONS
# ==============================================================================

"""Save C(τ) capacity trace to file."""
function save_capacity(config_params, capacities, data_dir)
    fname = build_filename("capacity", config_params, ".txt")
    writedlm(joinpath(data_dir, fname), capacities)
end

"""Save evaluation predictions for each tau."""
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

"""Save Pearson correlation r(τ) to file."""
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

end # module PlottingHelper

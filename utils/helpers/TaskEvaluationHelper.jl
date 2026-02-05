# Date: 2026
#
#=
================================================================================
    TaskEvaluationHelper.jl - Ridge Regression & QRC Task Evaluation
================================================================================

Provides reusable functions for:
- Ridge regression for readout layer
- Memory capacity C(τ) computation (past recall: predict u(t-τ) from x(t))
- Future prediction capacity computation (predict u(t+τ) from x(t))
- Unified evaluate_task() for both memory and prediction tasks
- Train/test splitting with standardization

USAGE:
    include("utils/helpers/TaskEvaluationHelper.jl")
    using .TaskEvaluationHelper

================================================================================
=#

module TaskEvaluationHelper

using LinearAlgebra
using Statistics

export ridge_regression, compute_capacity, compute_memory_capacity, compute_prediction_capacity
export evaluate_task, prediction_task

# ==============================================================================
# RIDGE REGRESSION
# ==============================================================================

"""
    ridge_regression(X, y, α=1e-3)

Solve ridge regression: W = (X'X + αI)⁻¹ X'y

# Arguments
- `X`: Feature matrix (n_samples × n_features)
- `y`: Target vector (n_samples)
- `α`: Regularization parameter

# Returns
- `W`: Weight vector (n_features)
"""
function ridge_regression(X, y, α=1e-3)
    return (X'*X + α*I) \ (X'*y)
end

# ==============================================================================
# MEMORY CAPACITY
# ==============================================================================

"""
    compute_capacity(X, input_seq, tau_range; train_ratio=0.8, alpha=1e-3)

Compute memory capacity C(τ) for each delay τ.

C(τ) = cor(y_pred, y_true)² measures how well the reservoir can reconstruct
the input from τ timesteps ago (memory) or predict τ timesteps ahead (prediction).

# Arguments
- `X`: Feature matrix from reservoir (n_steps × n_features)
- `input_seq`: Original input sequence
- `tau_range`: Range of delays to evaluate (e.g., 1:50)
- `train_ratio`: Fraction of data for training (default 0.8)
- `alpha`: Ridge regularization parameter (default 1e-3)

# Returns
- `C`: Vector of capacity values C(τ) for each τ
- `C_total`: Sum of all capacities Σ C(τ)
- `preds`: Dict{τ => (k_obs, k_target, y_test, y_pred)} for each τ
"""
function compute_capacity(X, input_seq, tau_range; train_ratio=0.8, alpha=1e-3)
    n_samples = size(X, 1)
    
    # Standardize features
    X_mean, X_std = mean(X, dims=1), std(X, dims=1)
    X_std[X_std .< 1e-12] .= 1.0  # Avoid division by zero
    X_scaled = (X .- X_mean) ./ X_std
    
    C = Float64[]
    preds = Dict{Int, Tuple}()
    
    for τ in tau_range
        # Check if τ is valid
        τ >= n_samples && (push!(C, 0.0); continue)
        
        # Build regression problem: predict input at time (k+τ) from features at time k
        X_effective = X_scaled[1:(n_samples-τ), :]
        y_raw = input_seq[(τ+1):n_samples]
        
        # Standardize targets
        y_mean, y_std = mean(y_raw), std(y_raw)
        y_std < 1e-12 && (y_std = 1.0)
        y_scaled = (y_raw .- y_mean) ./ y_std
        
        n_effective = length(y_scaled)
        n_effective < 10 && (push!(C, 0.0); continue)
        
        # Train/test split
        n_train = floor(Int, n_effective * train_ratio)
        
        # Ridge regression
        weights = ridge_regression(X_effective[1:n_train, :], y_scaled[1:n_train], alpha)
        
        # Predict on test set
        y_predicted = X_effective[n_train+1:end, :] * weights
        y_test_scaled = y_scaled[n_train+1:end]
        
        length(y_test_scaled) < 2 && (push!(C, 0.0); continue)
        
        # Store predictions (denormalized)
        preds[τ] = (
            collect(1:length(y_test_scaled)),           # k_obs
            collect((n_train+1):n_effective),           # k_target
            y_test_scaled .* y_std .+ y_mean,           # y_test (denormalized)
            y_predicted .* y_std .+ y_mean              # y_pred (denormalized)
        )
        
        # Capacity = squared correlation
        capacity = std(y_predicted) > 1e-9 && std(y_test_scaled) > 1e-9 ? cor(y_predicted, y_test_scaled)^2 : 0.0
        push!(C, capacity)
    end
    
    return C, sum(C), preds
end

# Alias for clarity - compute_capacity computes memory (past recall)
const compute_memory_capacity = compute_capacity

"""
    compute_prediction_capacity(X, input_seq, tau_range; train_ratio=0.8, alpha=1e-3)

Compute future prediction capacity C(τ) for each delay τ.
Predicts u(t+τ) from features x(t) - how well can we predict future inputs?

# Arguments
- `X`: Feature matrix (n_samples × n_features) from reservoir
- `input_seq`: Original input sequence u(t)
- `tau_range`: Range of future delays (e.g., 1:50)
- `train_ratio`: Fraction of data for training (default 0.8)
- `alpha`: Ridge regularization parameter (default 1e-3)

# Returns
- `C`: Vector of prediction capacity values C(τ) for each τ
- `C_total`: Sum of all capacities Σ C(τ)
- `preds`: Dict{τ => (k_obs, k_target, y_test, y_pred)} for each τ

# Note
This is opposite to memory capacity which predicts u(t-τ) from x(t).
Prediction capacity measures forecasting ability.
"""
function compute_prediction_capacity(X, input_seq, tau_range; train_ratio=0.8, alpha=1e-3)
    n_samples = size(X, 1)
    
    # Standardize features
    X_mean, X_std = mean(X, dims=1), std(X, dims=1)
    X_std[X_std .< 1e-12] .= 1.0
    X_scaled = (X .- X_mean) ./ X_std
    
    C = Float64[]
    preds = Dict{Int, Tuple}()
    
    for τ in tau_range
        # Check if τ is valid
        τ >= n_samples && (push!(C, 0.0); continue)
        
        # Build regression problem: predict FUTURE input u(t+τ) from features x(t)
        # Use features from time 1 to (n-τ), predict input from (τ+1) to n
        X_effective = X_scaled[1:(n_samples-τ), :]
        y_raw = input_seq[(τ+1):n_samples]  # Future inputs
        
        # Standardize targets
        y_mean, y_std = mean(y_raw), std(y_raw)
        y_std < 1e-12 && (y_std = 1.0)
        y_scaled = (y_raw .- y_mean) ./ y_std
        
        n_effective = length(y_scaled)
        n_effective < 10 && (push!(C, 0.0); continue)
        
        # Train/test split
        n_train = floor(Int, n_effective * train_ratio)
        
        # Ridge regression
        weights = ridge_regression(X_effective[1:n_train, :], y_scaled[1:n_train], alpha)
        
        # Predict on test set
        y_predicted = X_effective[n_train+1:end, :] * weights
        y_test_scaled = y_scaled[n_train+1:end]
        
        length(y_test_scaled) < 2 && (push!(C, 0.0); continue)
        
        # Store predictions (denormalized)
        preds[τ] = (
            collect(1:length(y_test_scaled)),
            collect((n_train+1):n_effective),
            y_test_scaled .* y_std .+ y_mean,
            y_predicted .* y_std .+ y_mean
        )
        
        # Capacity = squared correlation
        capacity = std(y_predicted) > 1e-9 && std(y_test_scaled) > 1e-9 ? cor(y_predicted, y_test_scaled)^2 : 0.0
        push!(C, capacity)
    end
    
    return C, sum(C), preds
end

# ==============================================================================
# PREDICTION TASK
# ==============================================================================

"""
    prediction_task(X, input_seq; train_ratio=0.8, horizon=1, alpha=1e-3)

Evaluate prediction task: predict u(t+horizon) from reservoir state at time t.

# Arguments
- `X`: Feature matrix from reservoir (n_steps × n_features)
- `input_seq`: Original input sequence
- `train_ratio`: Fraction of data for training
- `horizon`: Prediction horizon (default 1)
- `alpha`: Ridge regularization parameter

# Returns
- Named tuple with: y_pred, y_test, rmse, r2
"""
function prediction_task(X, input_seq; train_ratio=0.8, horizon=1, alpha=1e-3)
    n = size(X, 1)
    
    # Standardize features
    X_m, X_s = mean(X, dims=1), std(X, dims=1)
    X_s[X_s .< 1e-12] .= 1.0
    X_sc = (X .- X_m) ./ X_s
    
    X_e = X_sc[1:(n-horizon), :]
    y_r = input_seq[(horizon+1):n]
    
    y_m, y_s = mean(y_r), std(y_r)
    y_s < 1e-12 && (y_s = 1.0)
    y_n = (y_r .- y_m) ./ y_s
    
    n_tr = floor(Int, length(y_n) * train_ratio)
    
    W = ridge_regression(X_e[1:n_tr, :], y_n[1:n_tr], alpha)
    
    yp = X_e[n_tr+1:end, :] * W
    yt = y_n[n_tr+1:end]
    
    # Denormalize
    yp_d = yp .* y_s .+ y_m
    yt_d = yt .* y_s .+ y_m
    
    # Metrics
    rmse = sqrt(mean((yp_d .- yt_d).^2))
    r2 = cor(yp, yt)^2
    
    return (y_pred=yp_d, y_test=yt_d, rmse=rmse, r2=r2)
end

# ==============================================================================
# FULL EVALUATION PIPELINE
# ==============================================================================

"""
    evaluate_task(X, input_seq, tau_range; task_type=:prediction, train_ratio=0.8, alpha=1e-3)

Full task evaluation: compute capacity for all τ and return predictions.

# Arguments
- `X`: Feature matrix from reservoir
- `input_seq`: Original input sequence
- `tau_range`: Range of delays to evaluate
- `task_type`: :prediction or :memory
- `train_ratio`: Fraction of data for training
- `alpha`: Ridge regularization parameter

# Returns
- `c_trace`: Vector of C(τ) values
- `c_total`: Total capacity Σ C(τ)
- `preds`: Dict of predictions for each τ
- `trained_models`: Dict of trained weight matrices
"""
function evaluate_task(X, input_seq, tau_range; 
                       task_type=:prediction, train_ratio=0.8, alpha=1e-3)
    
    n = size(X, 1)
    
    # Standardize features
    X_m, X_s = mean(X, dims=1), std(X, dims=1)
    X_s[X_s .< 1e-12] .= 1.0
    X_sc = (X .- X_m) ./ X_s
    
    c_trace = Float64[]
    preds = Dict{Int, Tuple}()
    trained_models = Dict{Int, Any}()
    
    for τ in tau_range
        τ >= n && (push!(c_trace, 0.0); continue)
        
        # Build regression problem based on task type
        if task_type == :prediction
            # Predict future: u(t+τ) from x(t)
            X_e = X_sc[1:(n-τ), :]
            y_r = input_seq[(τ+1):n]
        else  # :memory
            # Recall past: u(t-τ) from x(t)
            X_e = X_sc[(τ+1):n, :]
            y_r = input_seq[1:(n-τ)]
        end
        
        y_m, y_s = mean(y_r), std(y_r)
        y_s < 1e-12 && (y_s = 1.0)
        y_e = (y_r .- y_m) ./ y_s
        
        n_e = length(y_e)
        n_e < 10 && (push!(c_trace, 0.0); continue)
        
        n_tr = floor(Int, n_e * train_ratio)
        
        # Train
        W = ridge_regression(X_e[1:n_tr, :], y_e[1:n_tr], alpha)
        trained_models[τ] = W
        
        # Test
        yp = X_e[n_tr+1:end, :] * W
        yt = y_e[n_tr+1:end]
        
        length(yt) < 2 && (push!(c_trace, 0.0); continue)
        
        # Store predictions
        preds[τ] = (
            collect(1:length(yt)),
            collect((n_tr+1):n_e),
            yt .* y_s .+ y_m,
            yp .* y_s .+ y_m
        )
        
        # Capacity
        capacity = std(yp) > 1e-9 && std(yt) > 1e-9 ? cor(yp, yt)^2 : 0.0
        push!(c_trace, capacity)
    end
    
    return c_trace, sum(c_trace), preds, trained_models
end

# ==============================================================================
# TRAINED MODEL (for cross-encoding experiments)
# ==============================================================================

"""
    TrainedModel

Stores all information needed to apply trained weights to new features.
Enables cross-encoding experiments (e.g., train on lookahead, test on causal).

Fields:
- weights::Vector{Float64}   - Ridge regression weights
- X_mean::Vector{Float64}    - Feature means used for standardization
- X_std::Vector{Float64}     - Feature stds used for standardization  
- y_mean::Float64            - Target mean
- y_std::Float64             - Target std
- n_train::Int               - Number of training samples
- alpha::Float64             - Regularization strength used
"""
struct TrainedModel
    weights::Vector{Float64}
    X_mean::Vector{Float64}
    X_std::Vector{Float64}
    y_mean::Float64
    y_std::Float64
    n_train::Int
    alpha::Float64
end

"""
    save_trained_models(filepath, trained_models)

Saves trained models to a text file with full reproducibility.
Format is human-readable and can be loaded with load_trained_models().
"""
function save_trained_models(filepath::String, trained_models::Dict{Int, TrainedModel})
    open(filepath, "w") do io
        println(io, "# TrainedModels export - TaskEvaluationHelper.jl")
        println(io, "# tau\tn_features\tweights...\tX_mean...\tX_std...\ty_mean\ty_std\tn_train\talpha")
        
        for tau in sort(collect(keys(trained_models)))
            m = trained_models[tau]
            n_feat = length(m.weights)
            
            parts = String[]
            push!(parts, string(tau))
            push!(parts, string(n_feat))
            append!(parts, string.(m.weights))
            append!(parts, string.(m.X_mean))
            append!(parts, string.(m.X_std))
            push!(parts, string(m.y_mean))
            push!(parts, string(m.y_std))
            push!(parts, string(m.n_train))
            push!(parts, string(m.alpha))
            
            println(io, join(parts, "\t"))
        end
    end
end

"""
    load_trained_models(filepath) -> Dict{Int, TrainedModel}

Loads trained models from file saved by save_trained_models().
"""
function load_trained_models(filepath::String)::Dict{Int, TrainedModel}
    models = Dict{Int, TrainedModel}()
    
    open(filepath, "r") do io
        for line in eachline(io)
            startswith(line, "#") && continue
            isempty(strip(line)) && continue
            
            parts = split(line, "\t")
            tau = parse(Int, parts[1])
            n_feat = parse(Int, parts[2])
            
            weights = [parse(Float64, parts[2+i]) for i in 1:n_feat]
            X_mean = [parse(Float64, parts[2+n_feat+i]) for i in 1:n_feat]
            X_std = [parse(Float64, parts[2+2*n_feat+i]) for i in 1:n_feat]
            base_idx = 2 + 3*n_feat
            y_mean = parse(Float64, parts[base_idx + 1])
            y_std = parse(Float64, parts[base_idx + 2])
            n_train = parse(Int, parts[base_idx + 3])
            alpha = parse(Float64, parts[base_idx + 4])
            
            models[tau] = TrainedModel(weights, X_mean, X_std, y_mean, y_std, n_train, alpha)
        end
    end
    
    return models
end

"""
    apply_trained_model(X_new, target, model::TrainedModel, tau; task_type=:prediction)

Apply a trained model to new features (e.g., from a different encoding).
Returns predicted values, test values, and capacity score.
"""
function apply_trained_model(X_new::Matrix{Float64}, target::Vector{Float64}, 
                            model::TrainedModel, tau::Int; 
                            task_type::Symbol=:prediction, train_ratio::Float64=0.8)
    n_samples = size(X_new, 1)
    
    if task_type == :prediction
        range_X = 1:(n_samples-tau)
        range_y = (tau+1):n_samples
    else
        range_X = (tau+1):n_samples
        range_y = 1:(n_samples-tau)
    end
    
    # Standardize using TRAINING params
    X_scaled = (X_new .- model.X_mean') ./ model.X_std'
    X_eff = X_scaled[range_X, :]
    y_raw = target[range_y]
    y_eff = (y_raw .- model.y_mean) ./ model.y_std
    
    n_eff = length(y_eff)
    n_train = model.n_train
    
    X_test = X_eff[n_train+1:end, :]
    y_test = y_eff[n_train+1:end]
    y_pred = X_test * model.weights
    
    if length(y_test) < 2 || std(y_pred) < 1e-9 || std(y_test) < 1e-9
        c = 0.0
    else
        c = cor(y_pred, y_test)^2
    end
    
    return y_pred, y_test, c
end

export TrainedModel, save_trained_models, load_trained_models, apply_trained_model

end # module TaskEvaluationHelper

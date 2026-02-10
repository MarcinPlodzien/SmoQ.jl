# Date: 2026
#
#=
================================================================================
    cpuVariationalQuantumCircuitOptimizers.jl - VQC Optimization Algorithms
================================================================================

PURPOSE:
--------
Optimize variational quantum circuit parameters using gradient-based and
gradient-free methods. Integrates with Optim.jl for advanced algorithms.

OPTIMIZERS:
-----------
Built-in:
  SPSAOptimizer       - Simultaneous Perturbation Stochastic Approximation
  AdamOptimizer       - Adaptive Moment Estimation
  GradientDescent     - Simple gradient descent with learning rate

Optim.jl Integration:
  optimize_optim()    - Use any Optim.jl algorithm (LBFGS, BFGS, NelderMead, etc.)

SPSA DETAILS:
-------------
SPSA is hardware-friendly: only 2 circuit evaluations per iteration,
regardless of parameter count. Uses decaying step size and perturbation:
  aₖ = a / (k + A)^α     (step size)
  cₖ = c / k^γ           (perturbation)

Default parameters follow Spall (1998): α=0.602, γ=0.101, A=10

USAGE:
------
```julia
include("cpuVariationalQuantumCircuitOptimizers.jl")
using .CPUVariationalQuantumCircuitOptimizers

# SPSA optimization
opt = SPSAOptimizer()
θ_final, history = optimize!(cost_fn, θ_init, opt; max_iter=200)

# Adam with gradients
opt = AdamOptimizer(lr=0.01)
θ_final, history = optimize!(cost_fn, θ_init, opt; grad_fn=calculate_gradients)

# Optim.jl integration
result = optimize_optim(cost_fn, grad_fn, θ_init; method=:LBFGS)
```

================================================================================
=#

module CPUVariationalQuantumCircuitOptimizers

using LinearAlgebra

# Optional Optim.jl integration
const OPTIM_AVAILABLE = try
    @eval using Optim
    true
catch
    false
end

export SPSAOptimizer, AdamOptimizer, GradientDescentOptimizer
export optimize!, step!
export optimize_optim

# ==============================================================================
# SPSA OPTIMIZER (Hardware-Friendly)
# ==============================================================================

"""
    SPSAOptimizer

Simultaneous Perturbation Stochastic Approximation optimizer.

Only requires 2 function evaluations per iteration, regardless of parameter count!
This makes it ideal for large parameter counts or hardware deployment.

# Fields
- `a`: Initial step size (default: 0.1)
- `c`: Initial perturbation (default: 0.1)
- `A`: Stability constant (default: 10)
- `α`: Step decay exponent (default: 0.602)
- `γ`: Perturbation decay exponent (default: 0.101)

# Reference
Spall, J. C. (1998). "An Overview of the Simultaneous Perturbation Method
for Efficient Optimization"
"""
mutable struct SPSAOptimizer
    a::Float64      # Initial step size
    c::Float64      # Initial perturbation
    A::Float64      # Stability constant
    α::Float64      # Step decay exponent
    γ::Float64      # Perturbation decay exponent

    function SPSAOptimizer(; a::Float64=0.1, c::Float64=0.1, A::Float64=10.0,
                            α::Float64=0.602, γ::Float64=0.101)
        new(a, c, A, α, γ)
    end
end

"""
    step!(opt::SPSAOptimizer, θ, cost_fn, k) -> θ_new

Perform one SPSA optimization step.

# Arguments
- `opt`: SPSA optimizer
- `θ`: Current parameters
- `cost_fn`: Cost function f(θ) → Float64
- `k`: Iteration number (1-indexed)

# Returns
- Updated parameters
"""
function step!(opt::SPSAOptimizer, θ::Vector{Float64}, cost_fn, k::Int)
    n_params = length(θ)

    # Decaying gains (Spall's recommended schedule)
    aₖ = opt.a / (k + opt.A)^opt.α
    cₖ = opt.c / k^opt.γ

    # Sample Bernoulli ±1 perturbation
    Δ = 2.0 .* (rand(n_params) .> 0.5) .- 1.0

    # Only 2 function evaluations!
    θ_plus = θ .+ cₖ .* Δ
    θ_minus = θ .- cₖ .* Δ

    c_plus = cost_fn(θ_plus)
    c_minus = cost_fn(θ_minus)

    # SPSA gradient estimate
    gₖ = (c_plus - c_minus) ./ (2.0 * cₖ .* Δ)

    # Update parameters
    return θ .- aₖ .* gₖ
end

# ==============================================================================
# ADAM OPTIMIZER
# ==============================================================================

"""
    AdamOptimizer

Adaptive Moment Estimation optimizer (Kingma & Ba, 2015).

Requires gradient computation, but often converges faster than SPSA
when exact gradients are available.

# Fields
- `lr`: Learning rate (default: 0.01)
- `β1`: First moment decay (default: 0.9)
- `β2`: Second moment decay (default: 0.999)
- `ε`: Numerical stability (default: 1e-8)
"""
mutable struct AdamOptimizer
    lr::Float64
    β1::Float64
    β2::Float64
    ε::Float64
    m::Union{Nothing, Vector{Float64}}  # First moment estimate
    v::Union{Nothing, Vector{Float64}}  # Second moment estimate
    t::Int                               # Time step

    function AdamOptimizer(; lr::Float64=0.01, β1::Float64=0.9,
                            β2::Float64=0.999, ε::Float64=1e-8)
        new(lr, β1, β2, ε, nothing, nothing, 0)
    end
end

"""
    step!(opt::AdamOptimizer, θ, gradient) -> θ_new

Perform one Adam optimization step.

# Arguments
- `opt`: Adam optimizer
- `θ`: Current parameters
- `gradient`: Gradient at current parameters

# Returns
- Updated parameters
"""
function step!(opt::AdamOptimizer, θ::Vector{Float64}, gradient::Vector{Float64})
    n_params = length(θ)

    # Initialize moments if needed
    if isnothing(opt.m)
        opt.m = zeros(n_params)
        opt.v = zeros(n_params)
    end

    opt.t += 1

    # Update biased first moment estimate
    opt.m .= opt.β1 .* opt.m .+ (1 - opt.β1) .* gradient

    # Update biased second moment estimate
    opt.v .= opt.β2 .* opt.v .+ (1 - opt.β2) .* (gradient .^ 2)

    # Bias correction
    m_hat = opt.m ./ (1 - opt.β1^opt.t)
    v_hat = opt.v ./ (1 - opt.β2^opt.t)

    # Update parameters
    return θ .- opt.lr .* m_hat ./ (sqrt.(v_hat) .+ opt.ε)
end

"""
    reset!(opt::AdamOptimizer)

Reset Adam optimizer state (moments and time step).
"""
function reset!(opt::AdamOptimizer)
    opt.m = nothing
    opt.v = nothing
    opt.t = 0
end

# ==============================================================================
# GRADIENT DESCENT OPTIMIZER
# ==============================================================================

"""
    GradientDescentOptimizer

Simple gradient descent with fixed or decaying learning rate.
"""
mutable struct GradientDescentOptimizer
    lr::Float64
    decay::Float64  # lr_k = lr / (1 + decay * k)

    function GradientDescentOptimizer(; lr::Float64=0.01, decay::Float64=0.0)
        new(lr, decay)
    end
end

"""
    step!(opt::GradientDescentOptimizer, θ, gradient, k) -> θ_new
"""
function step!(opt::GradientDescentOptimizer, θ::Vector{Float64},
               gradient::Vector{Float64}, k::Int=1)
    lr_k = opt.lr / (1 + opt.decay * k)
    return θ .- lr_k .* gradient
end

# ==============================================================================
# MAIN OPTIMIZATION LOOP
# ==============================================================================

"""
    optimize!(cost_fn, θ_init, opt; kwargs...) -> (θ_final, history)

Run optimization loop.

# Arguments
- `cost_fn`: Cost function f(θ) → Float64
- `θ_init`: Initial parameters
- `opt`: Optimizer (SPSAOptimizer, AdamOptimizer, or GradientDescentOptimizer)

# Keyword Arguments
- `grad_fn`: Gradient function (required for Adam/GD, ignored for SPSA)
- `max_iter`: Maximum iterations (default: 100)
- `tol`: Convergence tolerance on cost change (default: 1e-6)
- `verbose`: Print progress (default: false)
- `callback`: Optional callback(k, θ, cost) called each iteration

# Returns
- `θ_final`: Optimized parameters
- `history`: Vector of (iteration, cost) tuples
"""
function optimize!(cost_fn, θ_init::Vector{Float64}, opt::SPSAOptimizer;
                   max_iter::Int=100, tol::Float64=1e-6,
                   verbose::Bool=false, callback=nothing)
    θ = copy(θ_init)
    history = Tuple{Int, Float64}[]

    cost_prev = Inf

    for k in 1:max_iter
        θ = step!(opt, θ, cost_fn, k)
        cost = cost_fn(θ)
        push!(history, (k, cost))

        if !isnothing(callback)
            callback(k, θ, cost)
        end

        if verbose && k % 10 == 0
            println("SPSA Iter $k: cost = $cost")
        end

        # Check convergence
        if abs(cost - cost_prev) < tol && k > 10
            verbose && println("Converged at iteration $k")
            break
        end
        cost_prev = cost
    end

    return θ, history
end

function optimize!(cost_fn, θ_init::Vector{Float64}, opt::AdamOptimizer;
                   grad_fn, max_iter::Int=100, tol::Float64=1e-6,
                   verbose::Bool=false, callback=nothing)
    θ = copy(θ_init)
    history = Tuple{Int, Float64}[]

    reset!(opt)
    cost_prev = Inf

    for k in 1:max_iter
        gradient = grad_fn(cost_fn, θ)
        θ = step!(opt, θ, gradient)
        cost = cost_fn(θ)
        push!(history, (k, cost))

        if !isnothing(callback)
            callback(k, θ, cost)
        end

        if verbose && k % 10 == 0
            println("Adam Iter $k: cost = $cost")
        end

        if abs(cost - cost_prev) < tol && k > 10
            verbose && println("Converged at iteration $k")
            break
        end
        cost_prev = cost
    end

    return θ, history
end

function optimize!(cost_fn, θ_init::Vector{Float64}, opt::GradientDescentOptimizer;
                   grad_fn, max_iter::Int=100, tol::Float64=1e-6,
                   verbose::Bool=false, callback=nothing)
    θ = copy(θ_init)
    history = Tuple{Int, Float64}[]

    cost_prev = Inf

    for k in 1:max_iter
        gradient = grad_fn(cost_fn, θ)
        θ = step!(opt, θ, gradient, k)
        cost = cost_fn(θ)
        push!(history, (k, cost))

        if !isnothing(callback)
            callback(k, θ, cost)
        end

        if verbose && k % 10 == 0
            println("GD Iter $k: cost = $cost")
        end

        if abs(cost - cost_prev) < tol && k > 10
            verbose && println("Converged at iteration $k")
            break
        end
        cost_prev = cost
    end

    return θ, history
end

# ==============================================================================
# OPTIM.JL INTEGRATION
# ==============================================================================

"""
    optimize_optim(cost_fn, grad_fn, θ_init; method=:LBFGS, kwargs...)

Use Optim.jl for optimization. Requires Optim.jl to be installed.

# Arguments
- `cost_fn`: Cost function f(θ) → Float64
- `grad_fn`: Gradient function (θ → Vector{Float64})
- `θ_init`: Initial parameters
- `method`: :LBFGS, :BFGS, :GradientDescent, :NelderMead, :ConjugateGradient

# Returns
- Optim.OptimizationResults
"""
function optimize_optim(cost_fn, grad_fn, θ_init::Vector{Float64};
                        method::Symbol=:LBFGS, kwargs...)
    if !OPTIM_AVAILABLE
        error("Optim.jl not available. Install with: ] add Optim")
    end

    # Wrapper for Optim's expected gradient signature g!(G, x)
    function g!(G, x)
        grad = grad_fn(cost_fn, x)
        G .= grad
    end

    # Select method
    if method == :LBFGS
        opt = Optim.LBFGS()
    elseif method == :BFGS
        opt = Optim.BFGS()
    elseif method == :GradientDescent
        opt = Optim.GradientDescent()
    elseif method == :ConjugateGradient
        opt = Optim.ConjugateGradient()
    elseif method == :NelderMead
        # Gradient-free
        return Optim.optimize(cost_fn, θ_init, Optim.NelderMead(); kwargs...)
    else
        error("Unknown method: $method. Use :LBFGS, :BFGS, :GradientDescent, :ConjugateGradient, :NelderMead")
    end

    return Optim.optimize(cost_fn, g!, θ_init, opt; kwargs...)
end

end # module CPUVariationalQuantumCircuitOptimizers

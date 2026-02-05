# Date: 2026
#
#=
================================================================================
    cpuVQCEnzymeWrapper.jl - Enzyme-Compatible VQC Execution
================================================================================

PURPOSE:
--------
Provide Enzyme-differentiable VQC execution by avoiding closures over complex
circuit structs. Uses a "flattened" circuit representation.

APPROACH:
---------
Instead of closing over ParameterizedCircuit (which Enzyme can't handle),
we extract circuit info into simple arrays that Enzyme can differentiate through.

USAGE:
------
```julia
include("cpuVQCEnzymeWrapper.jl")
using .CPUVQCEnzymeWrapper

# Create wrapper from circuit
wrapper = EnzymeCircuitWrapper(circuit, N, energy_fn)

# Cost function (Enzyme-compatible)
cost = cost_mcwf(wrapper, θ)    # Pure state mode
cost = cost_dm(wrapper, θ)      # Density matrix mode

# Gradients with Enzyme
grad = gradient_enzyme(wrapper, θ, :mcwf)
grad = gradient_enzyme(wrapper, θ, :dm)
```

================================================================================
=#

module CPUVQCEnzymeWrapper

using LinearAlgebra
using Random

# Import gate operations
include("cpuQuantumChannelGates.jl")
using .CPUQuantumChannelGates

# Try to import Enzyme
const ENZYME_AVAILABLE = try
    @eval using Enzyme
    true
catch
    false
end

export EnzymeCircuitWrapper, build_enzyme_wrapper
export make_cost_pure, make_cost_dm, gradient_enzyme
export apply_circuit_enzyme!

# ==============================================================================
# FLATTENED CIRCUIT REPRESENTATION
# ==============================================================================

"""
    EnzymeCircuitWrapper

Enzyme-compatible circuit representation using simple arrays.
Avoids complex struct closures that break Enzyme.

# Fields
- `N::Int`: Number of qubits
- `n_params::Int`: Number of variational parameters
- `op_types::Vector{Symbol}`: Operation types
- `op_qubits::Vector{Tuple{Int,Int}}`: Qubit indices (q1, q2) - q2=0 for single qubit
- `op_param_idx::Vector{Int}`: Parameter index (0 = not parameterized)
- `op_noise_p::Vector{Float64}`: Noise probability for each op
"""
struct EnzymeCircuitWrapper
    N::Int
    n_params::Int
    op_types::Vector{Symbol}
    op_qubits::Vector{Tuple{Int,Int}}
    op_param_idx::Vector{Int}
    op_noise_p::Vector{Float64}
end

"""
    build_enzyme_wrapper(circuit::ParameterizedCircuit) -> EnzymeCircuitWrapper

Convert a ParameterizedCircuit to Enzyme-compatible wrapper.
"""
function build_enzyme_wrapper(circuit)
    op_types = Symbol[]
    op_qubits = Tuple{Int,Int}[]
    op_param_idx = Int[]
    op_noise_p = Float64[]
    
    for op in circuit.operations
        push!(op_types, op.type)
        
        q1 = length(op.qubits) >= 1 ? op.qubits[1] : 0
        q2 = length(op.qubits) >= 2 ? op.qubits[2] : 0
        push!(op_qubits, (q1, q2))
        
        push!(op_param_idx, op.param_idx)
        push!(op_noise_p, op.p)
    end
    
    return EnzymeCircuitWrapper(
        circuit.N, circuit.n_params,
        op_types, op_qubits, op_param_idx, op_noise_p
    )
end

# ==============================================================================
# CIRCUIT EXECUTION (PURE STATE - MCWF)
# ==============================================================================

"""
    apply_circuit_enzyme!(ψ, wrapper, θ, N)

Apply circuit to pure state. Enzyme-compatible.
"""
function apply_circuit_enzyme!(ψ::Vector{ComplexF64}, wrapper::EnzymeCircuitWrapper, 
                                θ::Vector{Float64})
    N = wrapper.N
    
    for i in eachindex(wrapper.op_types)
        op_type = wrapper.op_types[i]
        q1, q2 = wrapper.op_qubits[i]
        param_idx = wrapper.op_param_idx[i]
        noise_p = wrapper.op_noise_p[i]
        
        # Get angle if parameterized
        angle = param_idx > 0 ? θ[param_idx] : 0.0
        
        # Apply gate
        if op_type == :ry
            apply_ry_psi!(ψ, q1, angle, N)
        elseif op_type == :rx
            apply_rx_psi!(ψ, q1, angle, N)
        elseif op_type == :rz
            apply_rz_psi!(ψ, q1, angle, N)
        elseif op_type == :h
            apply_hadamard_psi!(ψ, q1, N)
        elseif op_type == :x
            apply_pauli_x_psi!(ψ, q1, N)
        elseif op_type == :z
            apply_pauli_z_psi!(ψ, q1, N)
        elseif op_type == :cz
            apply_cz_psi!(ψ, q1, q2, N)
        elseif op_type == :cnot || op_type == :cx
            apply_cnot_psi!(ψ, q1, q2, N)
        elseif op_type == :rxx
            apply_rxx_psi!(ψ, q1, q2, angle, N)
        elseif op_type == :ryy
            apply_ryy_psi!(ψ, q1, q2, angle, N)
        elseif op_type == :rzz
            apply_rzz_psi!(ψ, q1, q2, angle, N)
        elseif op_type == :depolarizing && noise_p > 0
            # Stochastic noise - skip for gradient computation (or use mean)
            # For MCWF: would apply random Pauli with prob p
            # For gradient: treat as identity (zero noise gradient)
            nothing
        elseif op_type == :dephasing && noise_p > 0
            # Similar to depolarizing
            nothing
        elseif op_type == :amplitude_damping && noise_p > 0
            nothing
        end
    end
    
    return ψ
end

# ==============================================================================
# CIRCUIT EXECUTION (DENSITY MATRIX)
# ==============================================================================

"""
    apply_circuit_enzyme!(ρ, wrapper, θ)

Apply circuit to density matrix. Enzyme-compatible.
"""
function apply_circuit_enzyme!(ρ::Matrix{ComplexF64}, wrapper::EnzymeCircuitWrapper,
                                θ::Vector{Float64})
    N = wrapper.N
    
    for i in eachindex(wrapper.op_types)
        op_type = wrapper.op_types[i]
        q1, q2 = wrapper.op_qubits[i]
        param_idx = wrapper.op_param_idx[i]
        noise_p = wrapper.op_noise_p[i]
        
        angle = param_idx > 0 ? θ[param_idx] : 0.0
        
        # Apply gate (DM version)
        if op_type == :ry
            apply_ry_rho!(ρ, q1, angle, N)
        elseif op_type == :rx
            apply_rx_rho!(ρ, q1, angle, N)
        elseif op_type == :rz
            apply_rz_rho!(ρ, q1, angle, N)
        elseif op_type == :h
            apply_hadamard_rho!(ρ, q1, N)
        elseif op_type == :x
            apply_pauli_x_rho!(ρ, q1, N)
        elseif op_type == :z
            apply_pauli_z_rho!(ρ, q1, N)
        elseif op_type == :cz
            apply_cz_rho!(ρ, q1, q2, N)
        elseif op_type == :cnot || op_type == :cx
            apply_cnot_rho!(ρ, q1, q2, N)
        elseif op_type == :rxx
            apply_rxx_rho!(ρ, q1, q2, angle, N)
        elseif op_type == :ryy
            apply_ryy_rho!(ρ, q1, q2, angle, N)
        elseif op_type == :rzz
            apply_rzz_rho!(ρ, q1, q2, angle, N)
        elseif op_type == :depolarizing && noise_p > 0
            _apply_depolarizing_kraus!(ρ, q1, noise_p, N)
        elseif op_type == :dephasing && noise_p > 0
            _apply_dephasing_kraus!(ρ, q1, noise_p, N)
        end
    end
    
    return ρ
end

# Kraus operators for DM mode
function _apply_depolarizing_kraus!(ρ::Matrix{ComplexF64}, k::Int, p::Float64, N::Int)
    ρ_new = (1 - p) * ρ
    
    ρ_x = copy(ρ)
    apply_pauli_x_rho!(ρ_x, k, N)
    ρ_new .+= (p/3) .* ρ_x
    
    ρ_z = copy(ρ)
    apply_pauli_z_rho!(ρ_z, k, N)
    ρ_new .+= (p/3) .* ρ_z
    
    # Y = -iXZ
    ρ_y = copy(ρ)
    apply_pauli_z_rho!(ρ_y, k, N)
    apply_pauli_x_rho!(ρ_y, k, N)
    ρ_new .+= (p/3) .* ρ_y
    
    ρ .= ρ_new
end

function _apply_dephasing_kraus!(ρ::Matrix{ComplexF64}, k::Int, p::Float64, N::Int)
    ρ_z = copy(ρ)
    apply_pauli_z_rho!(ρ_z, k, N)
    ρ .= (1 - p/2) .* ρ .+ (p/2) .* ρ_z
end

# ==============================================================================
# COST FUNCTIONS
# ==============================================================================

"""
Create pure-state cost function (for VQE energy).
"""
function make_cost_pure(wrapper::EnzymeCircuitWrapper, energy_fn::Function)
    function cost(θ::Vector{Float64})::Float64
        dim = 1 << wrapper.N
        ψ = zeros(ComplexF64, dim)
        ψ[1] = 1.0
        apply_circuit_enzyme!(ψ, wrapper, θ)
        return energy_fn(ψ)
    end
    return cost
end

"""
Create density matrix cost function (for noisy VQE).
"""
function make_cost_dm(wrapper::EnzymeCircuitWrapper, energy_fn::Function)
    function cost(θ::Vector{Float64})::Float64
        dim = 1 << wrapper.N
        ρ = zeros(ComplexF64, dim, dim)
        ρ[1, 1] = 1.0
        apply_circuit_enzyme!(ρ, wrapper, θ)
        return energy_fn(ρ)
    end
    return cost
end

# ==============================================================================
# ENZYME GRADIENT
# ==============================================================================

"""
    gradient_enzyme(cost_fn, θ) -> Vector{Float64}

Compute gradient using Enzyme reverse-mode autodiff.

NOTE: The cost function is wrapped in Const() because it may capture
immutable data (like the EnzymeCircuitWrapper).
"""
function gradient_enzyme(cost_fn, θ::Vector{Float64})
    if !ENZYME_AVAILABLE
        error("Enzyme not available")
    end
    
    dθ = zeros(Float64, length(θ))
    θ_copy = copy(θ)
    # Use Const(cost_fn) because the function closes over immutable data
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(cost_fn), Enzyme.Active, Enzyme.Duplicated(θ_copy, dθ))
    return dθ
end

# ==============================================================================
# SPSA WITH PROPER HYPERPARAMETERS
# ==============================================================================

export SPSAState, spsa_step!, spsa_gradient

"""
SPSA optimizer state with proper scheduling.

Standard hyperparameter schedule (Spall 1998):
- a_k = a / (A + k)^α
- c_k = c / k^γ

Recommended: α=0.602, γ=0.101, A=10% of iterations
"""
mutable struct SPSAState
    a::Float64      # Learning rate scale
    c::Float64      # Perturbation scale
    A::Float64      # Stability constant
    α::Float64      # LR decay exponent
    γ::Float64      # Perturbation decay exponent
    k::Int          # Current iteration
end

function SPSAState(; a=0.5, c=0.1, A=10.0, α=0.602, γ=0.101)
    SPSAState(a, c, A, α, γ, 0)
end

"""
Compute SPSA gradient estimate with proper scheduling.
"""
function spsa_gradient(cost_fn, θ::Vector{Float64}, state::SPSAState)
    state.k += 1
    k = state.k
    
    # Scheduled parameters
    a_k = state.a / (state.A + k)^state.α
    c_k = state.c / k^state.γ
    
    # Random perturbation (Bernoulli ±1)
    n = length(θ)
    Δ = 2.0 .* (rand(n) .> 0.5) .- 1.0
    
    # Evaluate cost at ±perturbation
    θ_plus = θ .+ c_k .* Δ
    θ_minus = θ .- c_k .* Δ
    
    f_plus = cost_fn(θ_plus)
    f_minus = cost_fn(θ_minus)
    
    # Gradient estimate
    g = (f_plus - f_minus) ./ (2 * c_k .* Δ)
    
    return g, a_k  # Return gradient and current LR
end

"""
Single SPSA optimization step.
"""
function spsa_step!(θ::Vector{Float64}, cost_fn, state::SPSAState)
    g, a_k = spsa_gradient(cost_fn, θ, state)
    θ .-= a_k .* g
    return θ
end

end # module CPUVQCEnzymeWrapper

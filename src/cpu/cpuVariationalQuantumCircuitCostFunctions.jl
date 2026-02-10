# Date: 2026
#
#=
================================================================================
    cpuVariationalQuantumCircuitCostFunctions.jl - Cost Functions for VQC ML
================================================================================

PURPOSE:
--------
Define cost functions for variational quantum circuit machine learning tasks.

COST FUNCTIONS:
---------------
Expectation Values:
  expectation_cost        - ⟨ψ(θ)|H|ψ(θ)⟩ for VQE-like tasks
  local_observable_cost   - Single qubit expectation ⟨σₖ⟩

Classification:
  classification_cost     - Binary/multi-class classification
  cross_entropy_cost      - Cross-entropy loss

Regression:
  regression_cost         - MSE between predictions and targets

Fidelity:
  fidelity_cost          - |⟨target|ψ(θ)⟩|² for state preparation

DATA ENCODING:
--------------
  encode_amplitude!       - Amplitude encoding of classical data
  encode_angle!           - Angle encoding via rotation gates

USAGE:
------
```julia
include("cpuVariationalQuantumCircuitCostFunctions.jl")
using .CPUVariationalQuantumCircuitCostFunctions

# Simple cost: minimize ⟨Z₁⟩ to prepare |1⟩ state
function my_cost(θ)
    ψ = make_ket("|0>", N)
    apply_circuit!(ψ, circuit, θ)
    return local_observable_cost(ψ, 1, N, :z)  # Returns ⟨Z₁⟩
end

# Optimize to find |1⟩ state
θ_opt, _ = optimize!(my_cost, θ_init, SPSAOptimizer())
```

================================================================================
=#

module CPUVariationalQuantumCircuitCostFunctions

using LinearAlgebra

export expectation_cost, local_observable_cost
export classification_cost, cross_entropy_cost
export regression_cost
export fidelity, fidelity_cost
export encode_amplitude!, encode_angle!

# ==============================================================================
# LOCAL OBSERVABLE COST
# ==============================================================================

"""
    local_observable_cost(ψ, k, N, pauli) -> Float64

Compute ⟨ψ|σₖ|ψ⟩ for a local Pauli observable.

# Arguments
- `ψ`: State vector
- `k`: Qubit index (1-based)
- `N`: Total qubits
- `pauli`: :x, :y, or :z

# Returns
- Expectation value (real Float64)

# Example
```julia
cost = local_observable_cost(ψ, 1, 4, :z)  # ⟨Z₁⟩
```
"""
function local_observable_cost(ψ::Vector{ComplexF64}, k::Int, N::Int, pauli::Symbol)
    if pauli == :z
        return _expect_Z(ψ, k, N)
    elseif pauli == :x
        return _expect_X(ψ, k, N)
    elseif pauli == :y
        return _expect_Y(ψ, k, N)
    else
        error("Unknown Pauli: $pauli. Use :x, :y, or :z")
    end
end

# Bitwise Z expectation (diagonal)
function _expect_Z(ψ::Vector{ComplexF64}, k::Int, N::Int)
    bit_pos = k - 1
    result = 0.0
    @inbounds for i in 0:(length(ψ)-1)
        bit = (i >> bit_pos) & 1
        result += abs2(ψ[i+1]) * (1 - 2*bit)
    end
    return result
end

# Bitwise X expectation
function _expect_X(ψ::Vector{ComplexF64}, k::Int, N::Int)
    bit_pos = k - 1
    step = 1 << bit_pos
    result = 0.0
    @inbounds for i in 0:(length(ψ)-1)
        if ((i >> bit_pos) & 1) == 0
            result += 2 * real(conj(ψ[i+1]) * ψ[i+step+1])
        end
    end
    return result
end

# Bitwise Y expectation
function _expect_Y(ψ::Vector{ComplexF64}, k::Int, N::Int)
    bit_pos = k - 1
    step = 1 << bit_pos
    result = 0.0
    @inbounds for i in 0:(length(ψ)-1)
        if ((i >> bit_pos) & 1) == 0
            result += 2 * imag(conj(ψ[i+1]) * ψ[i+step+1])
        end
    end
    return result
end

# ==============================================================================
# EXPECTATION VALUE COST (Hamiltonian)
# ==============================================================================

"""
    expectation_cost(ψ, coeffs, paulis, qubits, N) -> Float64

Compute ⟨ψ|H|ψ⟩ for a Hamiltonian H = Σᵢ cᵢ Pᵢ (sum of Pauli terms).

# Arguments
- `ψ`: State vector
- `coeffs`: Coefficients [c₁, c₂, ...]
- `paulis`: Pauli types [[:z], [:x, :x], [:z, :z], ...]
- `qubits`: Qubit indices [[1], [1,2], [2,3], ...]
- `N`: Total qubits

# Example
```julia
# H = -Z₁ - Z₂ + 0.5 X₁X₂
coeffs = [-1.0, -1.0, 0.5]
paulis = [[:z], [:z], [:x, :x]]
qubits = [[1], [2], [1, 2]]
E = expectation_cost(ψ, coeffs, paulis, qubits, 4)
```
"""
function expectation_cost(ψ::Vector{ComplexF64},
                          coeffs::Vector{Float64},
                          paulis::Vector{Vector{Symbol}},
                          qubits::Vector{Vector{Int}},
                          N::Int)
    @assert length(coeffs) == length(paulis) == length(qubits)

    energy = 0.0
    for i in eachindex(coeffs)
        if length(paulis[i]) == 1
            # Single-qubit term
            energy += coeffs[i] * local_observable_cost(ψ, qubits[i][1], N, paulis[i][1])
        elseif length(paulis[i]) == 2
            # Two-qubit term
            energy += coeffs[i] * _two_body_correlator(ψ, qubits[i][1], qubits[i][2],
                                                        paulis[i][1], paulis[i][2], N)
        else
            error("Only 1 and 2-body terms supported")
        end
    end

    return energy
end

# Two-body correlator ⟨σᵢσⱼ⟩
function _two_body_correlator(ψ::Vector{ComplexF64}, i::Int, j::Int,
                               p1::Symbol, p2::Symbol, N::Int)
    if p1 == :z && p2 == :z
        return _expect_ZZ(ψ, i, j, N)
    elseif p1 == :x && p2 == :x
        return _expect_XX(ψ, i, j, N)
    elseif p1 == :y && p2 == :y
        return _expect_YY(ψ, i, j, N)
    else
        error("Only ZZ, XX, YY correlators implemented")
    end
end

function _expect_ZZ(ψ::Vector{ComplexF64}, i::Int, j::Int, N::Int)
    bit_i, bit_j = i - 1, j - 1
    result = 0.0
    @inbounds @simd for s in 0:(length(ψ)-1)
        parity = xor((s >> bit_i) & 1, (s >> bit_j) & 1)
        result += abs2(ψ[s+1]) * (1 - 2*parity)
    end
    return result
end

function _expect_XX(ψ::Vector{ComplexF64}, i::Int, j::Int, N::Int)
    step_i, step_j = 1 << (i-1), 1 << (j-1)
    result = 0.0
    @inbounds for s in 0:(length(ψ)-1)
        bi, bj = (s >> (i-1)) & 1, (s >> (j-1)) & 1
        if bi == 0 && bj == 0
            s11 = xor(xor(s, step_i), step_j)
            result += 2 * real(conj(ψ[s+1]) * ψ[s11+1])
        elseif bi == 0 && bj == 1
            s10 = xor(xor(s, step_i), step_j)
            result += 2 * real(conj(ψ[s+1]) * ψ[s10+1])
        end
    end
    return result
end

function _expect_YY(ψ::Vector{ComplexF64}, i::Int, j::Int, N::Int)
    step_i, step_j = 1 << (i-1), 1 << (j-1)
    result = 0.0
    @inbounds for s in 0:(length(ψ)-1)
        bi, bj = (s >> (i-1)) & 1, (s >> (j-1)) & 1
        if bi == 0 && bj == 0
            s11 = xor(xor(s, step_i), step_j)
            result += -2 * real(conj(ψ[s+1]) * ψ[s11+1])
        elseif bi == 0 && bj == 1
            s10 = xor(xor(s, step_i), step_j)
            result += 2 * real(conj(ψ[s+1]) * ψ[s10+1])
        end
    end
    return result
end

# ==============================================================================
# FIDELITY COST (State Preparation)
# ==============================================================================

"""
    fidelity_cost(ψ, target) -> Float64

Compute 1 - |⟨target|ψ⟩|² (to minimize for state preparation).

# Returns
- 0 when ψ = target (up to global phase)
- 1 when ψ ⊥ target
"""
function fidelity_cost(ψ::Vector{ComplexF64}, target::Vector{ComplexF64})
    overlap = dot(target, ψ)
    return 1.0 - abs2(overlap)
end

"""
    fidelity(ψ, target) -> Float64

Compute |⟨target|ψ⟩|² (fidelity, not cost).
"""
function fidelity(ψ::Vector{ComplexF64}, target::Vector{ComplexF64})
    overlap = dot(target, ψ)
    return abs2(overlap)
end

# ==============================================================================
# CLASSIFICATION COST
# ==============================================================================

"""
    classification_cost(predictions, labels) -> Float64

Compute mean squared error for classification.

# Arguments
- `predictions`: Vector of predicted values (from circuit readout)
- `labels`: Vector of target labels (0 or 1 for binary)
"""
function classification_cost(predictions::Vector{Float64}, labels::Vector{Float64})
    @assert length(predictions) == length(labels)
    return sum((predictions .- labels) .^ 2) / length(labels)
end

"""
    cross_entropy_cost(p_pred, labels) -> Float64

Binary cross-entropy loss.

# Arguments
- `p_pred`: Predicted probabilities (0 to 1)
- `labels`: Binary labels (0 or 1)
"""
function cross_entropy_cost(p_pred::Vector{Float64}, labels::Vector{Float64})
    ε = 1e-10  # Numerical stability
    p = clamp.(p_pred, ε, 1 - ε)
    return -mean(labels .* log.(p) .+ (1 .- labels) .* log.(1 .- p))
end

# ==============================================================================
# REGRESSION COST
# ==============================================================================

"""
    regression_cost(predictions, targets) -> Float64

Mean squared error for regression tasks.
"""
function regression_cost(predictions::Vector{Float64}, targets::Vector{Float64})
    @assert length(predictions) == length(targets)
    return sum((predictions .- targets) .^ 2) / length(targets)
end

# ==============================================================================
# DATA ENCODING
# ==============================================================================

"""
    encode_amplitude!(ψ, x, N)

Amplitude encoding: normalize data vector and use as state amplitudes.

# Warning
Requires dim(ψ) = 2^N ≥ length(x). Pads with zeros if needed.
"""
function encode_amplitude!(ψ::Vector{ComplexF64}, x::Vector{Float64}, N::Int)
    dim = 1 << N
    @assert dim >= length(x) "Data length $(length(x)) > state dimension $dim"

    # Normalize data
    norm_x = norm(x)
    if norm_x < 1e-10
        error("Data vector is nearly zero")
    end

    # Set amplitudes
    fill!(ψ, 0.0)
    for i in eachindex(x)
        ψ[i] = ComplexF64(x[i] / norm_x)
    end

    return ψ
end

"""
    encode_angle(x, feature_map=:ry) -> Vector{Float64}

Angle encoding: convert data features to rotation angles.

# Arguments
- `x`: Feature vector
- `feature_map`: :ry (default), :rx, :rz

# Returns
- Vector of angles for parameterized gates
"""
function encode_angle(x::Vector{Float64}; feature_map::Symbol=:ry)
    # Simple linear scaling: angle = π * x (assuming x ∈ [0, 1])
    return π .* x
end

end # module CPUVariationalQuantumCircuitCostFunctions

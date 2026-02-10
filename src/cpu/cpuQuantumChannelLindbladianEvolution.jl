# Date: 2026
#
#=
================================================================================
    cpuQuantumChannelLindbladianEvolution.jl - Lindbladian Master Equation
    
    HIGH-LEVEL INTERFACE with Jump Operators:
    
    dρ/dt = -i[H, ρ] + Σⱼ γⱼ (Lⱼ ρ Lⱼ† - ½{Lⱼ†Lⱼ, ρ})
    
    Supports both:
    - Density Matrix (DM) evolution
    - Monte Carlo Wave Function (MCWF) / Quantum Trajectories
    
    DESIGN: Delegates coherent evolution to existing UnitaryEvolution modules.
================================================================================
=#

module CPUQuantumChannelLindbladianEvolution

using LinearAlgebra
using Random
using SparseArrays

# Import from sibling modules
using ..CPUHamiltonianBuilder: HamiltonianParams, construct_sparse_hamiltonian
using ..CPUQuantumChannelUnitaryEvolutionTrotter: FastTrotterGate, apply_fast_trotter_step_cpu!, 
    precompute_trotter_gates_bitwise_cpu
using ..CPUQuantumChannelUnitaryEvolutionChebyshev: chebyshev_evolve_psi!, estimate_spectral_range_lanczos
using ..CPUQuantumChannelUnitaryEvolutionExact: precompute_exact_propagator_cpu, evolve_exact_psi_cpu!, evolve_exact_rho_cpu!

export JumpOperator, create_jump_operator
export LindbladianEvolver, create_lindbladian_evolver
export lindbladian_evolve_dm!, lindbladian_evolve_mcwf!

# ==============================================================================
# JUMP OPERATOR
# ==============================================================================

"""
    JumpOperator

Represents a Lindblad jump operator Lⱼ with rate γⱼ.

# Standard operators (bitwise efficient)
- :sigma_minus (σ⁻) - Lowering operator |0⟩⟨1| (spontaneous emission)
- :sigma_plus (σ⁺)  - Raising operator |1⟩⟨0|
- :sigma_z    (σᶻ)  - Dephasing operator
- :sigma_x    (σˣ)  - Bit flip operator

# Custom operators
- Provide your own matrix L

# Usage
```julia
# Standard operator on qubit 3 with rate 0.1
L1 = create_jump_operator(:sigma_minus, qubit=3, γ=0.1)

# Custom operator
L_custom = [0 1; 0 0]  # |0⟩⟨1|
L2 = create_jump_operator(:custom, matrix=L_custom, qubit=1, γ=0.05)
```
"""
struct JumpOperator
    type::Symbol            # :sigma_minus, :sigma_plus, :sigma_z, :sigma_x, :custom
    qubit::Int              # Target qubit (1-indexed)
    γ::Float64              # Jump rate
    L::Union{Matrix{ComplexF64}, Nothing}  # Explicit matrix (for :custom or precomputed)
    L_dag_L::Union{Matrix{ComplexF64}, Nothing}  # L†L (precomputed for DM)
end

"""Create a jump operator."""
function create_jump_operator(type::Symbol; 
                               qubit::Int=1, 
                               γ::Float64=0.1,
                               matrix::Union{Matrix, Nothing}=nothing)
    
    # Standard 2x2 operators
    σ_minus = ComplexF64[0 1; 0 0]  # |0⟩⟨1|
    σ_plus  = ComplexF64[0 0; 1 0]  # |1⟩⟨0|
    σ_z     = ComplexF64[1 0; 0 -1]
    σ_x     = ComplexF64[0 1; 1 0]
    
    if type == :sigma_minus
        L = σ_minus
    elseif type == :sigma_plus
        L = σ_plus
    elseif type == :sigma_z
        L = σ_z
    elseif type == :sigma_x
        L = σ_x
    elseif type == :custom
        @assert matrix !== nothing "Must provide matrix for :custom type"
        L = ComplexF64.(matrix)
    else
        error("Unknown jump operator type: $type")
    end
    
    L_dag_L = L' * L
    
    return JumpOperator(type, qubit, γ, L, L_dag_L)
end

# ==============================================================================
# LINDBLADIAN EVOLVER
# ==============================================================================

"""
    LindbladianEvolver

Pre-computed resources for Lindbladian evolution.
"""
struct LindbladianEvolver
    # System parameters
    N::Int                  # Number of qubits
    dt::Float64             # Time step
    
    # Jump operators
    jump_ops::Vector{JumpOperator}
    
    # Coherent evolution (from existing modules)
    integrator::Symbol      # :exact, :trotter, :chebyshev
    trotter_gates::Union{Vector{FastTrotterGate}, Nothing}
    exact_U::Union{Matrix{ComplexF64}, Nothing}
    H_params::Union{HamiltonianParams, Nothing}
    spectral_bounds::Union{Tuple{Float64, Float64}, Nothing}
    
    # Precomputed for DM evolution
    full_L::Vector{Union{SparseMatrixCSC{ComplexF64}, Nothing}}  # Full-space L operators
    full_L_dag_L::Vector{Union{SparseMatrixCSC{ComplexF64}, Nothing}}
end

"""
    create_lindbladian_evolver(H_params, dt, jump_ops; integrator=:trotter)

Create Lindbladian evolver with jump operators.

# Arguments
- `H_params` - Hamiltonian parameters
- `dt` - Time step
- `jump_ops` - Vector of JumpOperator
- `integrator` - :exact, :trotter, or :chebyshev
"""
function create_lindbladian_evolver(H_params::HamiltonianParams,
                                     dt::Float64,
                                     jump_ops::Vector{JumpOperator};
                                     integrator::Symbol=:trotter)
    N = H_params.N_x * H_params.N_y
    dim = 1 << N
    
    # Precompute coherent evolution (delegate to existing modules)
    trotter_gates = nothing
    exact_U = nothing
    spectral_bounds = nothing
    
    if integrator == :trotter
        trotter_gates = precompute_trotter_gates_bitwise_cpu(H_params, dt)
    elseif integrator == :exact
        exact_U = precompute_exact_propagator_cpu(H_params, dt)
    elseif integrator == :chebyshev
        spectral_bounds = estimate_spectral_range_lanczos(H_params)
    end
    
    # Precompute full-space jump operators for DM
    full_L = Vector{Union{SparseMatrixCSC{ComplexF64}, Nothing}}(undef, length(jump_ops))
    full_L_dag_L = Vector{Union{SparseMatrixCSC{ComplexF64}, Nothing}}(undef, length(jump_ops))
    
    for (i, op) in enumerate(jump_ops)
        if op.type in (:sigma_minus, :sigma_plus, :sigma_z, :sigma_x, :custom)
            # Build full-space operator via Kronecker products
            full_L[i] = _embed_operator(op.L, op.qubit, N)
            full_L_dag_L[i] = _embed_operator(op.L_dag_L, op.qubit, N)
        else
            full_L[i] = nothing
            full_L_dag_L[i] = nothing
        end
    end
    
    return LindbladianEvolver(N, dt, jump_ops, integrator,
                              trotter_gates, exact_U, H_params, spectral_bounds,
                              full_L, full_L_dag_L)
end

"""Embed single-qubit operator into full Hilbert space."""
function _embed_operator(op2::Matrix{ComplexF64}, qubit::Int, N::Int)
    dim = 1 << N
    I2 = sparse(ComplexF64[1 0; 0 1])
    op_sparse = sparse(op2)
    
    # Build kronecker product: I ⊗ ... ⊗ L ⊗ ... ⊗ I
    result = qubit == 1 ? op_sparse : I2
    for q in 2:N
        if q == qubit
            result = kron(result, op_sparse)
        else
            result = kron(result, I2)
        end
    end
    return result
end

# ==============================================================================
# MCWF EVOLUTION (Quantum Trajectories)
# ==============================================================================

"""
    lindbladian_evolve_mcwf!(ψ, evolver)

One Lindblad step using MCWF (quantum trajectories):
1. Non-Hermitian evolution with effective Hamiltonian
2. Stochastic quantum jump
3. Renormalize
"""
function lindbladian_evolve_mcwf!(ψ::Vector{ComplexF64}, evolver::LindbladianEvolver)
    N = evolver.N
    dt = evolver.dt
    dim = length(ψ)
    
    # 1. Coherent evolution (unitary part)
    if evolver.integrator == :exact
        evolve_exact_psi_cpu!(ψ, evolver.exact_U)
    elseif evolver.integrator == :trotter
        apply_fast_trotter_step_cpu!(ψ, evolver.trotter_gates, N)
    elseif evolver.integrator == :chebyshev
        chebyshev_evolve_psi!(ψ, evolver.H_params, evolver.dt;
                              spectral_bounds=evolver.spectral_bounds)
    end
    
    # 2. Stochastic quantum jumps
    for (i, op) in enumerate(evolver.jump_ops)
        γ = op.γ
        qubit = op.qubit
        
        # Compute jump probability: p = γ dt ⟨ψ|L†L|ψ⟩
        p_jump = γ * dt * _expect_LdagL(ψ, op, qubit)
        
        if rand() < p_jump
            # Apply jump: |ψ⟩ → L|ψ⟩ / ||L|ψ⟩||
            _apply_jump!(ψ, op, qubit)
        else
            # No jump: apply non-Hermitian correction (optional for 1st order)
            # For simplicity, we skip this in first-order approximation
        end
    end
    
    # 3. Renormalize
    ψ ./= norm(ψ)
    
    return ψ
end

"""Compute ⟨ψ|L†L|ψ⟩ for single-qubit operator using bitwise ops."""
function _expect_LdagL(ψ::Vector{ComplexF64}, op::JumpOperator, qubit::Int)
    dim = length(ψ)
    mask = 1 << (qubit - 1)
    
    if op.type == :sigma_minus
        # L†L = |1⟩⟨1|, so ⟨ψ|L†L|ψ⟩ = Σᵢ |ψᵢ|² where bit(i, qubit) = 1
        result = 0.0
        @inbounds for i in 0:(dim-1)
            if (i & mask) != 0
                result += abs2(ψ[i+1])
            end
        end
        return result
        
    elseif op.type == :sigma_plus
        # L†L = |0⟩⟨0|
        result = 0.0
        @inbounds for i in 0:(dim-1)
            if (i & mask) == 0
                result += abs2(ψ[i+1])
            end
        end
        return result
        
    elseif op.type == :sigma_z
        # L†L = I
        return 1.0
        
    elseif op.type == :sigma_x
        # L†L = I
        return 1.0
        
    else
        # Fall back to explicit calculation
        L = evolver.full_L[findfirst(x -> x === op, evolver.jump_ops)]
        Lψ = L * ψ
        return real(dot(Lψ, Lψ))
    end
end

"""Apply jump operator L to state vector using bitwise ops."""
function _apply_jump!(ψ::Vector{ComplexF64}, op::JumpOperator, qubit::Int)
    dim = length(ψ)
    mask = 1 << (qubit - 1)
    
    if op.type == :sigma_minus
        # σ⁻|1⟩ = |0⟩, σ⁻|0⟩ = 0
        @inbounds for i in 0:(dim-1)
            if (i & mask) != 0
                # |1⟩ → |0⟩
                i0 = xor(i, mask)
                ψ[i0+1] = ψ[i+1]
                ψ[i+1] = 0.0
            end
        end
        
    elseif op.type == :sigma_plus
        # σ⁺|0⟩ = |1⟩, σ⁺|1⟩ = 0
        @inbounds for i in 0:(dim-1)
            if (i & mask) == 0
                # |0⟩ → |1⟩
                i1 = xor(i, mask)
                ψ[i1+1] = ψ[i+1]
                ψ[i+1] = 0.0
            end
        end
        
    elseif op.type == :sigma_z
        # σᶻ|0⟩ = |0⟩, σᶻ|1⟩ = -|1⟩
        @inbounds for i in 0:(dim-1)
            if (i & mask) != 0
                ψ[i+1] = -ψ[i+1]
            end
        end
        
    elseif op.type == :sigma_x
        # σˣ swaps |0⟩ ↔ |1⟩
        @inbounds for i in 0:(dim-1)
            if (i & mask) == 0
                i1 = xor(i, mask)
                ψ[i+1], ψ[i1+1] = ψ[i1+1], ψ[i+1]
            end
        end
    end
    
    return ψ
end

# ==============================================================================
# DENSITY MATRIX EVOLUTION
# ==============================================================================

"""
    lindbladian_evolve_dm!(ρ, evolver)

One Lindblad step on density matrix:
ρ(t+dt) = U ρ U† + dt Σⱼ γⱼ (Lⱼ ρ Lⱼ† - ½{Lⱼ†Lⱼ, ρ})
"""
function lindbladian_evolve_dm!(ρ::Matrix{ComplexF64}, evolver::LindbladianEvolver)
    N = evolver.N
    dt = evolver.dt
    
    # 1. Coherent evolution (delegate to existing)
    if evolver.integrator == :exact
        evolve_exact_rho_cpu!(ρ, evolver.exact_U)
    elseif evolver.integrator == :trotter
        error("Trotter DM evolution requires evolve_trotter_rho_cpu! - use :exact instead")
    elseif evolver.integrator == :chebyshev
        error("Chebyshev DM evolution requires Liouvillian superoperator - use :exact instead")
    end
    
    # 2. Dissipative part (BITWISE matrix-free for standard operators)
    for (i, op) in enumerate(evolver.jump_ops)
        _apply_dissipator_dm!(ρ, op, dt)
    end
    
    return ρ
end

"""
Apply dissipator D[L]ρ = γ dt (L ρ L† - ½{L†L, ρ}) using bitwise operations.
MATRIX-FREE for standard operators.
"""
function _apply_dissipator_dm!(ρ::Matrix{ComplexF64}, op::JumpOperator, dt::Float64)
    γ = op.γ
    qubit = op.qubit
    dim = size(ρ, 1)
    mask = 1 << (qubit - 1)
    factor = γ * dt
    
    if op.type == :sigma_minus
        # L = σ⁻ = |0⟩⟨1|
        # L†L = |1⟩⟨1| (projects to excited state)
        # LρL† transfers population from |1⟩ to |0⟩
        @inbounds for j in 0:(dim-1)
            for i in 0:(dim-1)
                bit_i = (i >> (qubit - 1)) & 1
                bit_j = (j >> (qubit - 1)) & 1
                
                if bit_i == 1 && bit_j == 1
                    # Transfer: ρ[i0,j0] += γ dt ρ[i,j]
                    i0 = xor(i, mask)
                    j0 = xor(j, mask)
                    ρ[i0+1, j0+1] += factor * ρ[i+1, j+1]
                    # Decay excited population
                    ρ[i+1, j+1] *= (1 - factor)
                elseif bit_i == 1 || bit_j == 1
                    # Coherence decay (off-diagonal involving |1⟩)
                    ρ[i+1, j+1] *= (1 - 0.5 * factor)
                end
            end
        end
        
    elseif op.type == :sigma_plus
        # L = σ⁺ = |1⟩⟨0|
        # L†L = |0⟩⟨0|
        @inbounds for j in 0:(dim-1)
            for i in 0:(dim-1)
                bit_i = (i >> (qubit - 1)) & 1
                bit_j = (j >> (qubit - 1)) & 1
                
                if bit_i == 0 && bit_j == 0
                    # Transfer: ρ[i1,j1] += γ dt ρ[i,j]
                    i1 = xor(i, mask)
                    j1 = xor(j, mask)
                    ρ[i1+1, j1+1] += factor * ρ[i+1, j+1]
                    ρ[i+1, j+1] *= (1 - factor)
                elseif bit_i == 0 || bit_j == 0
                    ρ[i+1, j+1] *= (1 - 0.5 * factor)
                end
            end
        end
        
    elseif op.type == :sigma_z
        # L = σᶻ, L†L = I
        # D[σᶻ]ρ = σᶻρσᶻ - ρ (pure dephasing)
        # Only off-diagonals with different qubit states are affected
        @inbounds for j in 0:(dim-1)
            for i in 0:(dim-1)
                bit_i = (i >> (qubit - 1)) & 1
                bit_j = (j >> (qubit - 1)) & 1
                
                if bit_i != bit_j
                    # Off-diagonal: decays as exp(-2γt) ≈ 1 - 2γdt
                    ρ[i+1, j+1] *= (1 - 2 * factor)
                end
                # Diagonal unchanged (σᶻρσᶻ = ρ for diagonal)
            end
        end
        
    elseif op.type == :sigma_x
        # L = σˣ, L†L = I
        # D[σˣ]ρ = σˣρσˣ - ρ
        @inbounds for j in 0:(dim-1)
            for i in 0:(dim-1)
                bit_i = (i >> (qubit - 1)) & 1
                bit_j = (j >> (qubit - 1)) & 1
                
                # Swap indices
                i_flip = xor(i, mask)
                j_flip = xor(j, mask)
                
                # D[σˣ]ρ[i,j] = γ dt (ρ[i_flip, j_flip] - ρ[i,j])
                # Only update once per pair (i < i_flip)
                if i < i_flip || (i == i_flip && j < j_flip)
                    val_orig = ρ[i+1, j+1]
                    val_flip = ρ[i_flip+1, j_flip+1]
                    ρ[i+1, j+1] += factor * (val_flip - val_orig)
                    ρ[i_flip+1, j_flip+1] += factor * (val_orig - val_flip)
                end
            end
        end
        
    else
        # Fall back to explicit matrix for custom operators (not matrix-free)
        L = evolver.full_L[findfirst(x -> x === op, evolver.jump_ops)]
        LdL = evolver.full_L_dag_L[findfirst(x -> x === op, evolver.jump_ops)]
        Lρ = L * ρ
        LρLd = Lρ * L'
        anticomm = LdL * ρ + ρ * LdL
        ρ .+= factor * (LρLd - 0.5 * anticomm)
    end
    
    return ρ
end

end # module

# Date: 2026
#
#=
================================================================================
    cpuQuantumChannelUnitaryEvolutionExact.jl - Exact Time Evolution (CPU)
================================================================================

OVERVIEW
--------
Standalone module for exact time evolution using exp(-iHt).
In-place operations for memory efficiency.

NAMING
------
- _rho suffix: Density matrix operations
- _psi suffix: Pure state operations
- _cpu suffix: CPU implementation

EXPORTS
-------
- precompute_exact_propagator_cpu(H, T_evol)
- evolve_exact_rho_cpu!(ρ, U, U_adj)
- evolve_exact_psi_cpu!(ψ, U)

================================================================================
=#

module CPUQuantumChannelUnitaryEvolutionExact

using LinearAlgebra
using SparseArrays

export precompute_exact_propagator_cpu
export evolve_exact_rho_cpu!, evolve_exact_psi_cpu!

"""
    precompute_exact_propagator_cpu(H, T_evol) -> (U, U_adj)

Compute exact propagator U = exp(-i*H*T_evol) and its adjoint U†.
Works with both dense and sparse Hamiltonians.
"""
function precompute_exact_propagator_cpu(H, T_evol::Real)
    H_dense = Matrix(H)
    U = exp(-im * H_dense * T_evol)
    U_adj = U'
    return U, U_adj
end

"""
    evolve_exact_rho_cpu!(ρ, U, U_adj) -> Matrix{ComplexF64}

In-place exact evolution for density matrix: ρ → U ρ U†
"""
function evolve_exact_rho_cpu!(ρ::Matrix{ComplexF64}, U::Matrix{ComplexF64}, U_adj::Matrix{ComplexF64})
    tmp = U * ρ
    mul!(ρ, tmp, U_adj)
    return ρ
end

"""
    evolve_exact_psi_cpu!(ψ, U) -> Vector{ComplexF64}

In-place exact evolution for pure state: |ψ⟩ → U|ψ⟩
"""
function evolve_exact_psi_cpu!(ψ::Vector{ComplexF64}, U::Matrix{ComplexF64})
    ψ .= U * ψ
    return ψ
end

end # module

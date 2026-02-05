# Date: 2026
#
#=
================================================================================
    cpuQuantumStateTomography.jl - MCWF State Reconstruction
================================================================================

OVERVIEW
--------
Functions for reconstructing density matrices from MCWF trajectories.

KEY INSIGHT: Entanglement measures (S_vN, Negativity, Concurrence) are NONLINEAR
functions of ρ. You CANNOT average them over trajectories! Instead:
  1. Collect M trajectories: {|ψ_1⟩, |ψ_2⟩, ..., |ψ_M⟩}
  2. Reconstruct ρ = (1/M) Σ_m |ψ_m⟩⟨ψ_m|
  3. Compute entanglement measures on the full ρ (use CPUQuantumStateCharacteristic)

FUNCTIONS
---------
Trajectory → Density Matrix:
- reconstruct_density_matrix(trajectories) -> Matrix{ComplexF64}
- reconstruct_reduced_density_matrix(trajectories, trace_out, N) -> Matrix{ComplexF64}

Analysis wrapper:
- analyze_mcwf_trajectories(trajectories) -> NamedTuple

For Classical Shadows tomography, see cpuClassicalShadows.jl

================================================================================
=#

module CPUQuantumStateTomography

using LinearAlgebra

export reconstruct_density_matrix, reconstruct_reduced_density_matrix
export analyze_mcwf_trajectories

# ==============================================================================
# DENSITY MATRIX RECONSTRUCTION FROM MCWF TRAJECTORIES
# ==============================================================================

"""
    reconstruct_density_matrix(trajectories::Vector{Vector{ComplexF64}}) -> Matrix{ComplexF64}

Reconstruct mixed state density matrix from MCWF pure state trajectories.

    ρ = (1/M) Σ_m |ψ_m⟩⟨ψ_m|

This is the CORRECT way to compute entanglement from MCWF:
- First reconstruct ρ (this function)
- Then compute measures on ρ using CPUQuantumStateCharacteristic

# Example
```julia
trajectories = [ψ1, ψ2, ψ3, ...]  # Vector of state vectors
ρ = reconstruct_density_matrix(trajectories)
# Then use functions from CPUQuantumStateCharacteristic:
S = von_neumann_entropy(ρ)
```
"""
function reconstruct_density_matrix(trajectories::Vector{Vector{ComplexF64}})
    M = length(trajectories)
    @assert M > 0 "Need at least one trajectory"
    
    dim = length(trajectories[1])
    ρ = zeros(ComplexF64, dim, dim)
    
    @inbounds for ψ in trajectories
        for j in 1:dim
            for i in 1:dim
                ρ[i, j] += ψ[i] * conj(ψ[j])
            end
        end
    end
    
    ρ ./= M
    return ρ
end

"""
    reconstruct_reduced_density_matrix(trajectories, trace_out, N) -> Matrix{ComplexF64}

Reconstruct reduced density matrix by tracing out specified qubits.
More memory efficient than reconstructing full ρ and then tracing.

# Arguments
- `trajectories`: Vector of pure state vectors
- `trace_out`: Vector of qubit indices to trace out (1-indexed)
- `N`: Total number of qubits
"""
function reconstruct_reduced_density_matrix(
    trajectories::Vector{Vector{ComplexF64}},
    trace_out::Vector{Int},
    N::Int
)
    M = length(trajectories)
    @assert M > 0 "Need at least one trajectory"
    
    keep = setdiff(1:N, trace_out)
    n_keep = length(keep)
    d_keep = 1 << n_keep
    
    ρ_red = zeros(ComplexF64, d_keep, d_keep)
    
    for ψ in trajectories
        for i_red in 0:(d_keep-1)
            for j_red in 0:(d_keep-1)
                for k_trace in 0:(1 << length(trace_out) - 1)
                    i_full = build_full_index(i_red, k_trace, keep, trace_out, N)
                    j_full = build_full_index(j_red, k_trace, keep, trace_out, N)
                    ρ_red[i_red+1, j_red+1] += ψ[i_full+1] * conj(ψ[j_full+1])
                end
            end
        end
    end
    
    ρ_red ./= M
    return ρ_red
end

# Helper function to build full index from reduced and traced parts
function build_full_index(idx_keep::Int, idx_trace::Int, 
                          keep::Vector{Int}, trace_out::Vector{Int}, N::Int)
    idx_full = 0
    keep_bit = 0
    trace_bit = 0
    
    for q in 1:N
        if q in keep
            bit = (idx_keep >> keep_bit) & 1
            idx_full |= (bit << (q-1))
            keep_bit += 1
        else
            bit = (idx_trace >> trace_bit) & 1
            idx_full |= (bit << (q-1))
            trace_bit += 1
        end
    end
    
    return idx_full
end

# ==============================================================================
# ANALYSIS WRAPPER
# ==============================================================================

"""
    analyze_mcwf_trajectories(trajectories) -> NamedTuple

Basic analysis of MCWF trajectories. Reconstructs ρ and computes purity.
For entanglement measures, use the reconstructed ρ with CPUQuantumStateCharacteristic.

# Returns NamedTuple with:
- `rho`: Reconstructed density matrix
- `purity`: Tr(ρ²)
- `trace`: Tr(ρ) - should be 1.0

# Example
```julia
using .CPUQuantumStateCharacteristic  # For entanglement measures

trajectories = [ψ1, ψ2, ...]
result = analyze_mcwf_trajectories(trajectories)
S = von_neumann_entropy(result.rho)
```
"""
function analyze_mcwf_trajectories(trajectories::Vector{Vector{ComplexF64}})
    ρ = reconstruct_density_matrix(trajectories)
    
    return (
        rho = ρ,
        purity = real(tr(ρ * ρ)),
        trace = real(tr(ρ)),
    )
end

end # module

# Date: 2026
#
#=
================================================================================
    cpuQuantumStateCharacterization.jl - Quantum State Diagnostics (CPU)
================================================================================

OVERVIEW
--------
Functions for characterizing quantum states: entanglement measures, 
localization measures, and purity.

No bitwise operations dependency - uses bitwise operations for efficiency.

FUNCTIONS
---------
Localization & Purity:
- inverse_participation_ratio(ρ/ψ) -> Float64  (IPR)
- purity(ρ) -> Float64

Entanglement Entropy (von Neumann):
- von_neumann_entropy(ρ) -> Float64
- entanglement_entropy_schmidt(ψ, d_A, d_B) -> Float64
- entanglement_entropy_horizontal(ρ, L, n_rails) -> Float64
- entanglement_entropy_vertical(ρ, L, n_rails) -> Float64

Negativity (True Entanglement):
- negativity_horizontal(ρ, L, n_rails) -> Float64
- log_negativity_horizontal(ρ, L, n_rails) -> Float64

================================================================================
=#

module CPUQuantumStateCharacteristic

using LinearAlgebra

export inverse_participation_ratio, purity
export von_neumann_entropy, entanglement_entropy_schmidt
export entanglement_entropy_horizontal, entanglement_entropy_vertical
export entanglement_entropy_schmidt_horizontal
export entanglement_entropy_1d_chain, negativity_1d_chain
export negativity_horizontal, log_negativity_horizontal
export negativity_vertical, log_negativity_vertical
export partial_transpose_rail1, partial_transpose_left

# ==============================================================================
# INVERSE PARTICIPATION RATIO (IPR) & PURITY
# ==============================================================================

"""
    inverse_participation_ratio(ρ::Matrix{ComplexF64}) -> Float64

Compute IPR = Σᵢ ρᵢᵢ² for density matrix (diagonal localization).

# Properties
- IPR = 1 for state localized on single basis state
- IPR = 1/d for maximally delocalized state
- Measures basis localization 
# Complexity
O(2^N) for density matrix (single diagonal pass).
"""
function inverse_participation_ratio(ρ::Matrix{ComplexF64})
    result = 0.0
    @inbounds for i in 1:size(ρ, 1)
        result += abs2(ρ[i, i])  # Only diagonal: ρᵢᵢ²
    end
    return result
end

"""
    inverse_participation_ratio(ψ::Vector{ComplexF64}) -> Float64

Compute IPR = Σᵢ |ψᵢ|⁴ for pure state.

# Complexity  
O(2^N) — very cheap, single pass over amplitudes.
"""
function inverse_participation_ratio(ψ::Vector{ComplexF64})
    result = 0.0
    @inbounds @simd for i in 1:length(ψ)
        result += abs2(ψ[i])^2  # |ψᵢ|⁴
    end
    return result
end

"""
    purity(ρ::Matrix{ComplexF64}) -> Float64

Compute purity γ = Tr(ρ²) = Σᵢⱼ |ρᵢⱼ|².

# Properties
- γ = 1 for pure states
- γ = 1/d for maximally mixed state
- Measures state mixedness (NOT basis localization)
"""
function purity(ρ::Matrix{ComplexF64})
    result = 0.0
    @inbounds for j in 1:size(ρ, 2)
        for i in 1:size(ρ, 1)
            result += abs2(ρ[i, j])
        end
    end
    return result
end

# ==============================================================================
# VON NEUMANN ENTROPY
# ==============================================================================

"""
    von_neumann_entropy(ρ::Matrix{ComplexF64}) -> Float64

Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ) via eigenvalue decomposition.

# Properties
- S(ρ) = 0 for pure states
- S(ρ) = log(d) for maximally mixed state
"""
function von_neumann_entropy(ρ::Matrix{ComplexF64})
    λs = real.(eigvals(Hermitian(ρ)))
    entropy = 0.0
    @inbounds for λ in λs
        if λ > 1e-15
            entropy -= λ * log(λ)
        end
    end
    return entropy
end

"""
    entanglement_entropy_schmidt(ψ::Vector{ComplexF64}, d_A::Int, d_B::Int) -> Float64

Compute entanglement entropy for pure state using Schmidt decomposition (SVD).
"""
function entanglement_entropy_schmidt(ψ::Vector{ComplexF64}, d_A::Int, d_B::Int)
    @assert length(ψ) == d_A * d_B "State vector length must equal d_A × d_B"
    M = reshape(ψ, d_A, d_B)
    σs = svdvals(M)
    entropy = 0.0
    @inbounds for σ in σs
        p = σ^2
        if p > 1e-15
            entropy -= p * log(p)
        end
    end
    return entropy
end

"""
    entanglement_entropy_1d_chain(ψ::Vector{ComplexF64}, N::Int) -> Float64

Matrix-free entanglement entropy for 1D chain with half-chain bipartition.
Uses Schmidt decomposition (SVD) - O(2^N) complexity.

# Example
```julia
ψ = [1/√2, 0, 0, 1/√2]  # Bell state
entanglement_entropy_1d_chain(ψ, 2)  # Returns log(2) ≈ 0.693
```
"""
function entanglement_entropy_1d_chain(ψ::Vector{ComplexF64}, N::Int)
    N_left = N ÷ 2
    d_left = 1 << N_left
    d_right = 1 << (N - N_left)
    M = reshape(ψ, d_left, d_right)
    σs = svdvals(M)
    entropy = 0.0
    @inbounds for σ in σs
        p = σ^2
        if p > 1e-15
            entropy -= p * log(p)
        end
    end
    return entropy
end

"""
    negativity_1d_chain(ψ::Vector{ComplexF64}, N::Int) -> Float64

Matrix-free negativity for 1D chain with half-chain bipartition.
Uses Schmidt decomposition: ||ρ^{T_A}||_1 = (Σσ)² for pure states.
N = ((Σσ)² - 1) / 2.  O(2^N) complexity via SVD.

# Example
```julia
ψ = [1/√2, 0, 0, 1/√2]  # Bell state
negativity_1d_chain(ψ, 2)  # Returns 0.5 (maximally entangled)
```
"""
function negativity_1d_chain(ψ::Vector{ComplexF64}, N::Int)
    N_left = N ÷ 2
    d_left = 1 << N_left
    d_right = 1 << (N - N_left)
    M = reshape(ψ, d_left, d_right)
    σs = svdvals(M)
    sum_σ = sum(σs)
    return (sum_σ^2 - 1.0) / 2.0
end

"""
    entanglement_entropy_schmidt_horizontal(ψ::Vector{ComplexF64}, L::Int, n_rails::Int) -> Float64

Compute entanglement entropy for horizontal (Rail 1 | Rail 2) bipartition using Schmidt decomposition.
"""
function entanglement_entropy_schmidt_horizontal(ψ::Vector{ComplexF64}, L::Int, n_rails::Int)
    @assert n_rails == 2 "Horizontal cut requires exactly 2 rails"
    d_rail = 2^L
    
    M = zeros(ComplexF64, d_rail, d_rail)
    @inbounds for rail2 in 0:(d_rail-1)
        for rail1 in 0:(d_rail-1)
            idx = (rail2 << L) | rail1
            M[rail1+1, rail2+1] = ψ[idx+1]
        end
    end
    
    σs = svdvals(M)
    entropy = 0.0
    @inbounds for σ in σs
        p = σ^2
        if p > 1e-15
            entropy -= p * log(p)
        end
    end
    return entropy
end

"""
    entanglement_entropy_horizontal(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int) -> Float64

Compute von Neumann entropy for horizontal (Rail 1 | Rail 2) bipartition.
"""
function entanglement_entropy_horizontal(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int)
    @assert n_rails == 2 "Horizontal cut requires exactly 2 rails"
    
    d_rail = 2^L
    ρ_rail2 = zeros(ComplexF64, d_rail, d_rail)
    
    @inbounds for i2 in 0:(d_rail-1)
        for j2 in 0:(d_rail-1)
            for k1 in 0:(d_rail-1)
                i_full = (i2 << L) | k1
                j_full = (j2 << L) | k1
                ρ_rail2[i2+1, j2+1] += ρ_full[i_full+1, j_full+1]
            end
        end
    end
    
    return von_neumann_entropy(ρ_rail2)
end

"""
    entanglement_entropy_vertical(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int) -> Float64

Compute von Neumann entropy for vertical (Left | Right) bipartition.
"""
function entanglement_entropy_vertical(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int)
    @assert n_rails == 2 "Vertical cut requires exactly 2 rails"
    @assert L % 2 == 0 "Vertical cut requires even L"
    
    L_half = L ÷ 2
    d_left = 2^(2 * L_half)
    d_right = 2^(2 * L_half)
    
    ρ_left = zeros(ComplexF64, d_left, d_left)
    mask_half = (1 << L_half) - 1
    
    @inbounds for i_left in 0:(d_left-1)
        for j_left in 0:(d_left-1)
            i_r1_left = i_left & mask_half
            i_r2_left = (i_left >> L_half) & mask_half
            j_r1_left = j_left & mask_half
            j_r2_left = (j_left >> L_half) & mask_half
            
            for k_right in 0:(d_right-1)
                k_r1_right = k_right & mask_half
                k_r2_right = (k_right >> L_half) & mask_half
                
                i_r1 = i_r1_left | (k_r1_right << L_half)
                i_r2 = i_r2_left | (k_r2_right << L_half)
                i_full = (i_r2 << L) | i_r1
                
                j_r1 = j_r1_left | (k_r1_right << L_half)
                j_r2 = j_r2_left | (k_r2_right << L_half)
                j_full = (j_r2 << L) | j_r1
                
                ρ_left[i_left+1, j_left+1] += ρ_full[i_full+1, j_full+1]
            end
        end
    end
    
    return von_neumann_entropy(ρ_left)
end

# ==============================================================================
# NEGATIVITY (True Quantum Entanglement)
# ==============================================================================

"""
    partial_transpose_rail1(ρ_full::Matrix{ComplexF64}, L::Int) -> Matrix{ComplexF64}

Compute partial transpose with respect to Rail 1.
"""
function partial_transpose_rail1(ρ_full::Matrix{ComplexF64}, L::Int)
    d_rail = 2^L
    d_full = 2^(2*L)
    ρ_pt = zeros(ComplexF64, d_full, d_full)
    
    @inbounds for i1 in 0:(d_rail-1)
        for i2 in 0:(d_rail-1)
            for j1 in 0:(d_rail-1)
                for j2 in 0:(d_rail-1)
                    i_orig = (i2 << L) | j1
                    j_orig = (j2 << L) | i1
                    i_pt = (i2 << L) | i1
                    j_pt = (j2 << L) | j1
                    ρ_pt[i_pt+1, j_pt+1] = ρ_full[i_orig+1, j_orig+1]
                end
            end
        end
    end
    
    return ρ_pt
end

"""
    negativity_horizontal(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int) -> Float64

Compute negativity N(ρ) = (||ρ^{T₁}||₁ - 1) / 2 for horizontal bipartition.

# Properties
- N = 0 for separable states
- N > 0 iff ρ is entangled
"""
function negativity_horizontal(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int)
    @assert n_rails == 2 "Horizontal cut requires exactly 2 rails"
    ρ_pt = partial_transpose_rail1(ρ_full, L)
    λs = real.(eigvals(Hermitian(ρ_pt)))
    trace_norm = sum(abs, λs)
    return (trace_norm - 1.0) / 2.0
end

"""
    log_negativity_horizontal(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int) -> Float64

Compute log-negativity E_N = log₂(2N + 1).
"""
function log_negativity_horizontal(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int)
    neg = negativity_horizontal(ρ_full, L, n_rails)
    return log2(2*neg + 1)
end

"""
    partial_transpose_left(ρ_full::Matrix{ComplexF64}, L::Int) -> Matrix{ComplexF64}

Compute partial transpose with respect to left half (for vertical bipartition).
"""
function partial_transpose_left(ρ_full::Matrix{ComplexF64}, L::Int)
    @assert L % 2 == 0 "Vertical cut requires even L"
    L_half = L ÷ 2
    d_full = 2^(2*L)
    d_half = 2^L_half
    
    ρ_pt = zeros(ComplexF64, d_full, d_full)
    
    # Left half = sites 1..L_half on both rails
    # This is more complex - swap left indices while keeping right fixed
    @inbounds for i in 0:(d_full-1)
        for j in 0:(d_full-1)
            # Extract left parts (bits 0..L_half-1 and L..L+L_half-1)
            # and right parts (bits L_half..L-1 and L+L_half..2L-1)
            mask_half = (1 << L_half) - 1
            
            i_r1_left = i & mask_half
            i_r1_right = (i >> L_half) & mask_half
            i_r2_left = (i >> L) & mask_half
            i_r2_right = (i >> (L + L_half)) & mask_half
            
            j_r1_left = j & mask_half
            j_r1_right = (j >> L_half) & mask_half
            j_r2_left = (j >> L) & mask_half
            j_r2_right = (j >> (L + L_half)) & mask_half
            
            # Swap left indices: i_left <-> j_left
            new_i = j_r1_left | (i_r1_right << L_half) | (j_r2_left << L) | (i_r2_right << (L + L_half))
            new_j = i_r1_left | (j_r1_right << L_half) | (i_r2_left << L) | (j_r2_right << (L + L_half))
            
            ρ_pt[new_i+1, new_j+1] = ρ_full[i+1, j+1]
        end
    end
    
    return ρ_pt
end

"""
    negativity_vertical(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int) -> Float64

Compute negativity for vertical (Left | Right) bipartition.
Warning: O(2^{4L}) eigenvalue decomposition — expensive!
"""
function negativity_vertical(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int)
    @assert n_rails == 2 "Vertical cut requires exactly 2 rails"
    @assert L % 2 == 0 "Vertical cut requires even L"
    
    ρ_pt = partial_transpose_left(ρ_full, L)
    λs = real.(eigvals(Hermitian(ρ_pt)))
    trace_norm = sum(abs, λs)
    return (trace_norm - 1.0) / 2.0
end

"""
    log_negativity_vertical(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int) -> Float64

Compute log-negativity for vertical bipartition.
"""
function log_negativity_vertical(ρ_full::Matrix{ComplexF64}, L::Int, n_rails::Int)
    neg = negativity_vertical(ρ_full, L, n_rails)
    return log2(2*neg + 1)
end

end # module

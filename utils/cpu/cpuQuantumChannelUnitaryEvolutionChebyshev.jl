# Date: 2026
#
#=
================================================================================
    cpuQuantumChannelUnitaryEvolutionChebyshev.jl - Matrix-Free Chebyshev (CPU)
================================================================================

OVERVIEW
--------
Implements time evolution using Chebyshev polynomial expansion:
    e^{-iHt} |ψ⟩ ≈ Σₖ aₖ(t) Tₖ(H̃) |ψ⟩

CRITICAL: This is a **completely matrix-free** implementation.
All H|ψ⟩ operations use bitwise indexing O(2^N), not sparse matrices.

ALGORITHM
---------
1. Lanczos iteration to estimate spectral bounds (E_min, E_max)
2. Rescale Hamiltonian: H̃ = (H - b)/a where a=(E_max-E_min)/2, b=(E_max+E_min)/2
3. Chebyshev recurrence: Tₖ₊₁ = 2H̃Tₖ - Tₖ₋₁
4. Sum with Bessel coefficients: aₖ = (-i)^k × (2-δₖ₀) × Jₖ(a×t)

================================================================================
=#

module CPUQuantumChannelUnitaryEvolutionChebyshev

using LinearAlgebra
using SpecialFunctions: besselj
using Base.Threads

# Import HamiltonianParams from sibling module
using ..CPUHamiltonianBuilder: HamiltonianParams, CouplingTerm, FieldTerm

export apply_hamiltonian_bitwise!
export estimate_spectral_range_lanczos
export get_rescaling_factors
export chebyshev_evolve_psi!
export chebyshev_evolve_psi

# ==============================================================================
# MATRIX-FREE HAMILTONIAN APPLICATION (BITWISE, OPTIMIZED)
# ==============================================================================

"""
    apply_hamiltonian_bitwise!(ψ_out, ψ_in, params::HamiltonianParams)

Optimized "Pull" implementation of H|ψ⟩:
1. Parallelize over state index 'k' (rows of the vector)
2. For each 'k', iterate over all Hamiltonian terms to accumulate the result
3. Writes to ψ_out[k+1] exactly once (Thread-Safe & Cache-Friendly)

For H = Σ(Jxx·XX + Jyy·YY + Jzz·ZZ)_{bonds} + Σ(hx·X + hy·Y + hz·Z)_{sites}
"""
function apply_hamiltonian_bitwise!(ψ_out::Vector{ComplexF64}, 
                                     ψ_in::Vector{ComplexF64}, 
                                     params::HamiltonianParams)
    N = params.N
    dim = 1 << N
    
    # Pre-extract terms to avoid struct access inside hot loop
    x_bonds = params.x_bonds
    y_bonds = params.y_bonds
    fields = params.fields
    
    # PARALLEL LOOP OVER STATE INDICES (Pull pattern)
    @threads for k in 0:(dim-1)
        acc = zero(ComplexF64)
        
        # --- Apply X-direction bonds (XX, YY, ZZ) ---
        @inbounds for bond in x_bonds
            bit_i, bit_j = bond.i - 1, bond.j - 1
            mask_flip = (1 << bit_i) | (1 << bit_j)
            
            val_i = (k >> bit_i) & 1
            val_j = (k >> bit_j) & 1
            
            # ZZ (Diagonal): read from self
            if bond.Jzz != 0.0
                sign_zz = (val_i == val_j) ? 1.0 : -1.0
                acc += bond.Jzz * sign_zz * ψ_in[k+1]
            end
            
            # XX (Off-diagonal): read from neighbor with flipped bits
            k_flip = xor(k, mask_flip)
            if bond.Jxx != 0.0
                acc += bond.Jxx * ψ_in[k_flip+1]
            end
            
            # YY (Off-diagonal): phase is -1 if bits equal, +1 if different
            if bond.Jyy != 0.0
                sign_yy = (val_i == val_j) ? -1.0 : 1.0
                acc += bond.Jyy * sign_yy * ψ_in[k_flip+1]
            end
        end
        
        # --- Apply Y-direction bonds (same structure) ---
        @inbounds for bond in y_bonds
            bit_i, bit_j = bond.i - 1, bond.j - 1
            mask_flip = (1 << bit_i) | (1 << bit_j)
            
            val_i = (k >> bit_i) & 1
            val_j = (k >> bit_j) & 1
            
            if bond.Jzz != 0.0
                sign_zz = (val_i == val_j) ? 1.0 : -1.0
                acc += bond.Jzz * sign_zz * ψ_in[k+1]
            end
            
            k_flip = xor(k, mask_flip)
            if bond.Jxx != 0.0
                acc += bond.Jxx * ψ_in[k_flip+1]
            end
            
            if bond.Jyy != 0.0
                sign_yy = (val_i == val_j) ? -1.0 : 1.0
                acc += bond.Jyy * sign_yy * ψ_in[k_flip+1]
            end
        end
        
        # --- Apply Fields (X, Y, Z) ---
        @inbounds for field in fields
            bit_idx = field.idx - 1
            mask = 1 << bit_idx
            val = (k >> bit_idx) & 1
            
            # Z (Diagonal)
            if field.hz != 0.0
                sign_z = (val == 0) ? 1.0 : -1.0
                acc += field.hz * sign_z * ψ_in[k+1]
            end
            
            k_flip = xor(k, mask)
            
            # X (Off-diagonal)
            if field.hx != 0.0
                acc += field.hx * ψ_in[k_flip+1]
            end
            
            # Y (Off-diagonal): Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
            if field.hy != 0.0
                phase_y = (val == 0) ? 1im : -1im
                acc += field.hy * phase_y * ψ_in[k_flip+1]
            end
        end
        
        # Single write to memory (thread-safe)
        ψ_out[k+1] = acc
    end
    
    return ψ_out
end

# ==============================================================================
# LANCZOS SPECTRAL BOUNDS ESTIMATION (MATRIX-FREE)
# ==============================================================================

"""
    estimate_spectral_range_lanczos(params::HamiltonianParams; k_iter=30)

Estimate extremal eigenvalues (E_min, E_max) using Lanczos algorithm.

Uses matrix-free H|ψ⟩ application. Returns Ritz values that converge
very fast to the true extremal eigenvalues.
"""
function estimate_spectral_range_lanczos(params::HamiltonianParams; k_iter::Int=30)
    N = params.N
    dim = 1 << N
    
    # Initialize random normalized vector
    v = randn(ComplexF64, dim)
    v ./= norm(v)
    v_prev = zeros(ComplexF64, dim)
    w = similar(v)
    
    # Storage for tridiagonal matrix elements
    alphas = Float64[]
    betas = Float64[]
    
    # Lanczos iterations
    for j in 1:k_iter
        # w = H × v (Matrix-Free Application)
        apply_hamiltonian_bitwise!(w, v, params)
        
        # Orthogonalize against v_prev (if j > 1)
        if j > 1
            w .-= betas[end] .* v_prev
        end
        
        # α_j = ⟨v|H|v⟩ (Rayleigh quotient)
        alpha = real(dot(v, w))
        push!(alphas, alpha)
        
        # Orthogonalize against v
        w .-= alpha .* v
        
        # β_j = ||w||
        beta = norm(w)
        
        # Convergence check
        if beta < 1e-12
            break
        end
        push!(betas, beta)
        
        # Update vectors for next iteration
        v_prev .= v
        v .= w ./ beta
    end
    
    # Diagonalize small tridiagonal matrix T_k
    # SymTridiagonal is efficient for this
    T_k = SymTridiagonal(alphas, betas)
    evals = eigvals(T_k)
    
    E_min = minimum(evals)
    E_max = maximum(evals)
    
    return E_min, E_max
end

"""
    get_rescaling_factors(E_min, E_max; safety_margin=0.01)

Compute rescaling factors (a, b) such that H̃ = (H - b)/a has spectrum in [-1, 1].

a = (E_max - E_min)/2 (half-width)
b = (E_max + E_min)/2 (center)

Safety margin ensures no eigenvalue lands exactly at ±1.
"""
function get_rescaling_factors(E_min::Float64, E_max::Float64; safety_margin::Float64=0.01)
    width = E_max - E_min
    
    # Add safety buffer to avoid boundary issues
    E_min_safe = E_min - safety_margin * abs(width)
    E_max_safe = E_max + safety_margin * abs(width)
    
    a = (E_max_safe - E_min_safe) / 2.0  # half-width
    b = (E_max_safe + E_min_safe) / 2.0  # center
    
    return a, b
end

# ==============================================================================
# CHEBYSHEV EVOLUTION (COMPLETELY MATRIX-FREE)
# ==============================================================================

"""
    chebyshev_evolve_psi!(ψ, params::HamiltonianParams, t; tol=1e-12)

Evolve state |ψ⟩ → e^{-iHt}|ψ⟩ using Chebyshev expansion.

COMPLETELY MATRIX-FREE: uses bitwise H|ψ⟩ application.

Algorithm:
1. Lanczos to get spectral bounds (E_min, E_max)
2. Rescale: H̃ = (H - b)/a 
3. Chebyshev recursion: Tₖ₊₁ = 2H̃Tₖ - Tₖ₋₁
4. Sum: |ψ_out⟩ = exp(-i·b·t) × Σₖ aₖ Tₖ(H̃)|ψ₀⟩
"""
function chebyshev_evolve_psi!(ψ::Vector{ComplexF64}, 
                                params::HamiltonianParams, 
                                t::Float64;
                                tol::Float64=1e-12,
                                max_order::Int=1000,
                                spectral_bounds::Union{Nothing,Tuple{Float64,Float64}}=nothing)
    N = params.N
    dim = 1 << N
    
    # Get spectral bounds via Lanczos (or use provided)
    if isnothing(spectral_bounds)
        E_min, E_max = estimate_spectral_range_lanczos(params)
    else
        E_min, E_max = spectral_bounds
    end
    
    # Get rescaling factors with safety margin
    a, b = get_rescaling_factors(E_min, E_max)
    
    # Scaled time for Bessel functions: τ = a × t
    τ = a * t
    
    # Temporary vectors for Chebyshev recursion
    ψ_in = copy(ψ)              # Original input state
    T_prev = copy(ψ_in)         # T₀(H̃)|ψ⟩ = |ψ⟩
    T_curr = similar(ψ_in)      # T₁(H̃)|ψ⟩ = H̃|ψ⟩
    T_next = similar(ψ_in)      # Tₖ₊₁
    H_temp = similar(ψ_in)      # Temporary for H|ψ⟩
    
    # Compute T₁ = H̃|ψ⟩ = (H - b·I)|ψ⟩ / a = (H|ψ⟩ - b|ψ⟩) / a
    apply_hamiltonian_bitwise!(H_temp, ψ_in, params)
    @inbounds @simd for i in 1:dim
        T_curr[i] = (H_temp[i] - b * ψ_in[i]) / a
    end
    
    # Initialize result with a₀·T₀ (a₀ = J₀(τ))
    a0 = besselj(0, τ)
    @inbounds @simd for i in 1:dim
        ψ[i] = a0 * T_prev[i]
    end
    
    # Add a₁·T₁ (a₁ = -2i·J₁(τ))
    a1 = -2im * besselj(1, τ)
    @inbounds @simd for i in 1:dim
        ψ[i] += a1 * T_curr[i]
    end
    
    # Chebyshev recursion: Tₖ₊₁ = 2H̃Tₖ - Tₖ₋₁
    M_used = 2
    
    for k in 2:max_order
        # Compute H̃·T_curr = (H·T_curr - b·T_curr) / a
        apply_hamiltonian_bitwise!(H_temp, T_curr, params)
        
        # T_next = 2·H̃·T_curr - T_prev
        @inbounds @simd for i in 1:dim
            H_tilde_T = (H_temp[i] - b * T_curr[i]) / a
            T_next[i] = 2 * H_tilde_T - T_prev[i]
        end
        
        # Coefficient: aₖ = 2·(-i)^k·Jₖ(τ) for k > 0
        ak = 2 * ((-1im)^k) * besselj(k, τ)
        
        # Accumulate: ψ += aₖ·Tₖ
        @inbounds @simd for i in 1:dim
            ψ[i] += ak * T_next[i]
        end
        
        M_used = k + 1
        
        # Check convergence: stop when Bessel function is negligible
        if abs(besselj(k, τ)) < tol
            break
        end
        
        # Shift: T_prev ← T_curr, T_curr ← T_next
        T_prev, T_curr, T_next = T_curr, T_next, T_prev
    end
    
    # Apply global phase from energy shift: exp(-i·b·t)
    global_phase = exp(-1im * b * t)
    ψ .*= global_phase
    
    return M_used, a, b
end

"""
    chebyshev_evolve_psi(ψ0, params, t; kwargs...) -> (ψ_out, M_used, a, b)

Non-mutating version of chebyshev_evolve_psi!
"""
function chebyshev_evolve_psi(ψ0::Vector{ComplexF64}, 
                               params::HamiltonianParams, 
                               t::Float64; kwargs...)
    ψ_out = copy(ψ0)
    M_used, a, b = chebyshev_evolve_psi!(ψ_out, params, t; kwargs...)
    return ψ_out, M_used, a, b
end

end # module CPUQuantumChannelUnitaryEvolutionChebyshev

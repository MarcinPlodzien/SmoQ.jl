# Date: 2026
#
#=
================================================================================
    cpuQuantumStatePartialTrace.jl - Partial Trace Operations (CPU)
================================================================================

OVERVIEW
--------
Bitwise partial trace operations for pure states and density matrices.
All operations use matrix-free bitwise implementations for optimal performance.
Uses Julia multiple dispatch - single function name works on both psi and rho.

FUNCTIONS
---------

Standard Partial Trace:
  - partial_trace(psi::Vector, trace_qubits, N)   Trace from pure state -> rho
  - partial_trace(rho::Matrix, trace_qubits, N)   Trace from density matrix

Region-Splitting Partial Trace:
  - partial_trace_regions(psi, trace_sites, N)    Returns tuple of DMs for 
                                                   disconnected kept regions
  - partial_trace_regions(rho, trace_sites, N)    Same for density matrices

Helper Functions:
  - find_connected_regions(sites)                  Find consecutive groups

ALGORITHM
---------
Uses precomputed index mappings between kept/traced bits for O(2^N) efficiency.
For partial_trace: O(2^N_keep * 2^N) for pure states.
For partial_trace_regions: computes separate traces for each connected region.

BIT CONVENTION
--------------
LITTLE-ENDIAN: qubit 1 = bit 0 (LSB), qubit N = bit N-1 (MSB).
Example: For N=4, trace_qubits=[2,4] keeps qubits [1,3].

================================================================================
=#

module CPUQuantumStatePartialTrace

using LinearAlgebra

export partial_trace
export partial_trace_regions

# ==============================================================================
# PARTIAL TRACE - UNIFIED INTERFACE (Julia dispatch)
# ==============================================================================

"""
    partial_trace(ψ::Vector{ComplexF64}, trace_qubits, N) -> Matrix{ComplexF64}

Compute reduced density matrix by tracing out specified qubits from pure state ψ.
Uses bitwise operations for O(2^N_keep × 2^N) performance.

Arguments:
- ψ: Pure state vector (length 2^N)
- trace_qubits: Vector/Range of qubit indices to TRACE OUT (1-indexed)
- N: Total number of qubits

Returns:
- ρ_reduced: Reduced density matrix of dimension 2^N_keep × 2^N_keep
"""
function partial_trace(ψ::Vector{ComplexF64}, trace_qubits, N::Int)
    trace_set = Set(trace_qubits)
    keep_indices = [q for q in 1:N if !(q in trace_set)]
    
    N_keep = length(keep_indices)
    N_trace = N - N_keep
    dim_keep = 1 << N_keep
    dim_trace = 1 << N_trace
    
    # 0-indexed bit positions
    keep_bits = [k - 1 for k in keep_indices]
    trace_bits = [k - 1 for k in sort(collect(trace_set))]
    
    # Precompute trace -> full index mapping
    trace_to_full = Vector{Int}(undef, dim_trace)
    @inbounds for t in 0:(dim_trace-1)
        idx = 0
        for (bit_idx, full_bit) in enumerate(trace_bits)
            if (t >> (bit_idx - 1)) & 1 == 1
                idx |= (1 << full_bit)
            end
        end
        trace_to_full[t+1] = idx
    end
    
    # Precompute keep -> full index mapping
    keep_to_full = Vector{Int}(undef, dim_keep)
    @inbounds for k in 0:(dim_keep-1)
        idx = 0
        for (bit_idx, full_bit) in enumerate(keep_bits)
            if (k >> (bit_idx - 1)) & 1 == 1
                idx |= (1 << full_bit)
            end
        end
        keep_to_full[k+1] = idx
    end
    
    # Compute reduced density matrix: ρ[i,j] = Σ_t ψ[i⊕t] ψ*[j⊕t]
    ρ = zeros(ComplexF64, dim_keep, dim_keep)
    
    @inbounds for i_keep in 0:(dim_keep-1)
        i_base = keep_to_full[i_keep+1]
        for j_keep in 0:(dim_keep-1)
            j_base = keep_to_full[j_keep+1]
            val = zero(ComplexF64)
            
            for t in 0:(dim_trace-1)
                t_contrib = trace_to_full[t+1]
                i_full = i_base | t_contrib
                j_full = j_base | t_contrib
                val += ψ[i_full + 1] * conj(ψ[j_full + 1])
            end
            
            ρ[i_keep + 1, j_keep + 1] = val
        end
    end
    
    return ρ
end

"""
    partial_trace(ρ::Matrix{ComplexF64}, trace_qubits, N) -> Matrix{ComplexF64}

Compute partial trace of density matrix over specified qubits.

Arguments:
- ρ: Density matrix (2^N × 2^N)
- trace_qubits: Vector/Range of qubit indices to TRACE OUT (1-indexed)
- N: Total number of qubits

Returns:
- ρ_reduced: Reduced density matrix of dimension 2^N_keep × 2^N_keep
"""
function partial_trace(ρ::Matrix{ComplexF64}, trace_qubits, N::Int)
    trace_set = Set(trace_qubits)
    keep_qubits = [q for q in 1:N if !(q in trace_set)]
    
    n_trace = length(trace_qubits)
    n_keep = N - n_trace
    dim_trace = 1 << n_trace
    dim_keep = 1 << n_keep
    
    # 0-indexed bit positions
    keep_bits = [k - 1 for k in keep_qubits]
    trace_bits = [k - 1 for k in sort(collect(trace_set))]
    
    ρ_reduced = zeros(ComplexF64, dim_keep, dim_keep)
    
    @inbounds for i_keep in 0:(dim_keep-1)
        # Build base index for kept qubits
        i_base = 0
        for (bit_idx, full_bit) in enumerate(keep_bits)
            if (i_keep >> (bit_idx - 1)) & 1 == 1
                i_base |= (1 << full_bit)
            end
        end
        
        for j_keep in 0:(dim_keep-1)
            j_base = 0
            for (bit_idx, full_bit) in enumerate(keep_bits)
                if (j_keep >> (bit_idx - 1)) & 1 == 1
                    j_base |= (1 << full_bit)
                end
            end
            
            val = zero(ComplexF64)
            
            # Sum over traced indices
            for t in 0:(dim_trace-1)
                i_full = i_base
                j_full = j_base
                for (bit_idx, full_bit) in enumerate(trace_bits)
                    if (t >> (bit_idx - 1)) & 1 == 1
                        i_full |= (1 << full_bit)
                        j_full |= (1 << full_bit)
                    end
                end
                val += ρ[i_full + 1, j_full + 1]
            end
            
            ρ_reduced[i_keep + 1, j_keep + 1] = val
        end
    end
    
    return ρ_reduced
end

# ==============================================================================
# PARTIAL TRACE WITH REGION SPLITTING
# ==============================================================================
#
# OVERVIEW
# --------
# When tracing out qubits from a quantum system, the remaining (kept) qubits may
# form disconnected regions. For example, tracing out sites [4,7,9] from a 10-qubit
# system leaves kept sites [1,2,3,5,6,8,10], which form 4 disconnected regions:
#   Region 1: [1,2,3]  (3 qubits)
#   Region 2: [5,6]    (2 qubits)
#   Region 3: [8]      (1 qubit)
#   Region 4: [10]     (1 qubit)
#
# The `partial_trace_regions` function computes the reduced density matrix for
# EACH disconnected region separately, returning a tuple of density matrices.
#
# ALGORITHM
# ---------
# 1. Identify kept sites (complement of trace_sites)
# 2. Find connected regions among kept sites (consecutive groups)
# 3. For each region, compute its reduced density matrix by tracing out:
#    - All originally traced sites
#    - All sites belonging to OTHER kept regions
#
# COMPLEXITY
# ----------
# For M disconnected regions with sizes n₁, n₂, ..., nₘ:
#   - Region detection: O(N log N) for sorting
#   - Per-region partial trace: O(2^N × 2^(2nᵢ)) for region i
#   - Total: O(M × 2^N × 2^(2·max(nᵢ)))
#
# USE CASES
# ---------
# - Analyzing entanglement structure after local measurements
# - Computing mutual information between non-adjacent subsystems
# - Studying propagation of correlations in spin chains
# - Extracting local observables from global state
#
# ==============================================================================

"""
    find_connected_regions(sites::Vector{Int}) -> Vector{Vector{Int}}

Find connected (consecutive) regions in a sorted list of site indices.

This helper function identifies groups of consecutive integers in the input,
which correspond to spatially connected qubit regions after tracing operations.

# Algorithm
1. Sort the input sites
2. Iterate through sorted sites, grouping consecutive integers
3. Start new group when gap > 1 is encountered

# Arguments
- `sites`: Vector of site indices (1-indexed, need not be sorted)

# Returns  
- Vector of vectors, each inner vector containing consecutive site indices

# Complexity
- Time: O(N log N) for sorting + O(N) for grouping = O(N log N)
- Space: O(N) for output storage

# Examples
```julia
find_connected_regions([1,2,3,5,6,8,10])  # [[1,2,3], [5,6], [8], [10]]
find_connected_regions([5,3,4,1,2])       # [[1,2], [3,4,5]]
find_connected_regions([1,3,5,7])         # [[1], [3], [5], [7]]
find_connected_regions([1,2,3])           # [[1,2,3]]
```
"""
function find_connected_regions(sites::Vector{Int})
    # Handle empty input
    isempty(sites) && return Vector{Int}[]
    
    # Sort sites to identify consecutive groups
    sorted_sites = sort(sites)
    
    # Storage for identified regions
    regions = Vector{Int}[]
    
    # Initialize first region with first site
    current_region = [sorted_sites[1]]
    
    # Iterate through remaining sites
    for i in 2:length(sorted_sites)
        if sorted_sites[i] == sorted_sites[i-1] + 1
            # Site is consecutive with previous → extend current region
            push!(current_region, sorted_sites[i])
        else
            # Gap detected → finalize current region, start new one
            push!(regions, current_region)
            current_region = [sorted_sites[i]]
        end
    end
    
    # Don't forget to add the last region
    push!(regions, current_region)
    
    return regions
end

"""
    partial_trace_regions(ψ::Vector{ComplexF64}, trace_sites, N) -> Tuple{Matrix{ComplexF64}...}

Trace out specified sites from pure state and return reduced density matrices 
for each spatially disconnected region of the remaining system.

When tracing out non-contiguous sites, the kept sites may form multiple
disconnected regions. This function computes the reduced density matrix for
EACH region separately, which is useful for analyzing local properties and
entanglement structure.

# Mathematical Description
Given an N-qubit pure state |ψ⟩ and sites S to trace out, let K = {1,...,N} \\ S
be the kept sites. If K decomposes into M connected regions R₁, R₂, ..., Rₘ,
then this function returns:

    (ρ_{R₁}, ρ_{R₂}, ..., ρ_{Rₘ})

where ρ_{Rᵢ} = Tr_{all except Rᵢ}(|ψ⟩⟨ψ|)

# Arguments
- `ψ::Vector{ComplexF64}`: Pure state vector of length 2^N
- `trace_sites`: Vector/Range of site indices to trace out (1-indexed)
- `N::Int`: Total number of qubits in the system

# Returns
- Tuple of density matrices, one per connected region
- Regions are ordered by their minimum site index
- Each ρᵢ has dimension 2^|Rᵢ| × 2^|Rᵢ|

# Implementation Details
- Uses bitwise partial trace for each region (matrix-free)
- For region Rᵢ, traces out ALL sites not in Rᵢ (including other kept regions)
- Little-endian bit ordering: qubit 1 = bit 0 (LSB)

# Examples
```julia
# 10-qubit system, trace out sites 4, 7, 9
# Kept sites: [1,2,3,5,6,8,10] → 4 regions
ψ = randn(ComplexF64, 1024)  # Random 10-qubit state
normalize_state!(ψ)

ρ_123, ρ_56, ρ_8, ρ_10 = partial_trace_regions(ψ, [4,7,9], 10)
# ρ_123: 8×8 (3 qubits)
# ρ_56:  4×4 (2 qubits)  
# ρ_8:   2×2 (1 qubit)
# ρ_10:  2×2 (1 qubit)

# Verify traces
@assert tr(ρ_123) ≈ 1.0
@assert tr(ρ_56) ≈ 1.0
```

See also: [`partial_trace`](@ref), [`find_connected_regions`](@ref)
"""
function partial_trace_regions(ψ::Vector{ComplexF64}, trace_sites, N::Int)
    # Convert trace_sites to Set for O(1) lookup
    trace_set = Set(trace_sites)
    
    # Determine which sites are kept (complement of traced sites)
    keep_sites = [q for q in 1:N if !(q in trace_set)]
    
    # Validate: must keep at least one site
    isempty(keep_sites) && error("Cannot trace out all sites - no subsystem remains")
    
    # Identify connected regions among kept sites
    regions = find_connected_regions(keep_sites)
    
    # Optimization: if only one connected region, use standard partial_trace
    if length(regions) == 1
        return (partial_trace(ψ, trace_sites, N),)
    end
    
    # For multiple regions: compute reduced DM for each region separately
    # Key insight: for region Rᵢ, we trace out EVERYTHING except Rᵢ
    # This includes both the originally traced sites AND other kept regions
    results = Matrix{ComplexF64}[]
    
    for region in regions
        # Build set of sites in this region
        region_set = Set(region)
        
        # Sites to trace = all sites NOT in this specific region
        sites_to_trace = [q for q in 1:N if !(q in region_set)]
        
        # Compute reduced density matrix for this region
        ρ_region = partial_trace(ψ, sites_to_trace, N)
        
        push!(results, ρ_region)
    end
    
    return Tuple(results)
end

"""
    partial_trace_regions(ρ::Matrix{ComplexF64}, trace_sites, N) -> Tuple{Matrix{ComplexF64}...}

Trace out specified sites from density matrix and return reduced density matrices
for each spatially disconnected region of the remaining system.

This is the density matrix version of `partial_trace_regions`. It handles mixed
states correctly, unlike the pure state version which starts from |ψ⟩.

# Mathematical Description
Given an N-qubit density matrix ρ and sites S to trace out, let K = {1,...,N} \\ S
be the kept sites. If K decomposes into M connected regions R₁, R₂, ..., Rₘ,
then this function returns:

    (ρ_{R₁}, ρ_{R₂}, ..., ρ_{Rₘ})

where ρ_{Rᵢ} = Tr_{all except Rᵢ}(ρ)

# Arguments
- `ρ::Matrix{ComplexF64}`: Density matrix of dimension 2^N × 2^N  
- `trace_sites`: Vector/Range of site indices to trace out (1-indexed)
- `N::Int`: Total number of qubits in the system

# Returns
- Tuple of density matrices, one per connected region
- Regions are ordered by their minimum site index
- Each ρᵢ has dimension 2^|Rᵢ| × 2^|Rᵢ|

# Implementation Details
- Uses bitwise partial trace for each region (matrix-free)
- Handles mixed states correctly (no pure state assumption)
- Trace of each output ρᵢ equals 1 (properly normalized)

# Examples
```julia
# 6-qubit thermal state, trace out sites 3 and 5
# Kept sites: [1,2,4,6] → 3 regions: [1,2], [4], [6]
ρ = make_initial_rho(6)  # |000000⟩⟨000000|

ρ_12, ρ_4, ρ_6 = partial_trace_regions(ρ, [3,5], 6)
# ρ_12: 4×4 (2 qubits)
# ρ_4:  2×2 (1 qubit)
# ρ_6:  2×2 (1 qubit)
```

See also: [`partial_trace`](@ref), [`find_connected_regions`](@ref)
"""
function partial_trace_regions(ρ::Matrix{ComplexF64}, trace_sites, N::Int)
    # Convert trace_sites to Set for O(1) lookup
    trace_set = Set(trace_sites)
    
    # Determine which sites are kept (complement of traced sites)
    keep_sites = [q for q in 1:N if !(q in trace_set)]
    
    # Validate: must keep at least one site
    isempty(keep_sites) && error("Cannot trace out all sites - no subsystem remains")
    
    # Identify connected regions among kept sites
    regions = find_connected_regions(keep_sites)
    
    # Optimization: if only one connected region, use standard partial_trace
    if length(regions) == 1
        return (partial_trace(ρ, trace_sites, N),)
    end
    
    # For multiple regions: compute reduced DM for each region separately
    results = Matrix{ComplexF64}[]
    
    for region in regions
        # Build set of sites in this region
        region_set = Set(region)
        
        # Sites to trace = all sites NOT in this specific region
        sites_to_trace = [q for q in 1:N if !(q in region_set)]
        
        # Compute reduced density matrix for this region using bitwise partial trace
        ρ_region = partial_trace(ρ, sites_to_trace, N)
        
        push!(results, ρ_region)
    end
    
    return Tuple(results)
end

end # module

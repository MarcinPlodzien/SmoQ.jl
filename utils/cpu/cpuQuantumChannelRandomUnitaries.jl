# Date: 2026
#
#=
================================================================================
    cpuQuantumChannelRandomUnitaries.jl - Random Unitaries & Brick-Wall (CPU)
================================================================================

OVERVIEW
--------
Construction and application of quantum channels, focusing on:
- Haar random unitaries (single-qubit, 2-qubit, N-qubit)
- Brick-wall quantum circuits with random or specified gates

All gate applications use matrix-free bitwise indexing for O(2^N) state updates.

FUNCTIONS
---------

Haar Random Unitaries:
  random_unitary(1)               Single-qubit Haar random (2x2 matrix)
  random_unitary(2)               Two-qubit Haar random (4x4 matrix)
  random_unitary(N)               N-qubit Haar random (2^N x 2^N matrix)
  
  ALGORITHM: QR decomposition of complex Gaussian matrix
  - Draw: A[i,j] ~ N(0,1) + i*N(0,1) for complex Gaussian
  - Compute: Q, R = qr(A)
  - Fix phase: Q = Q * diag(sign(diag(R))) to ensure Haar measure
  
  COMPLEXITY:
    Storage: O(4^N) - unavoidable for random unitary
    Generation: O(4^N) for QR decomposition
    
  EXAMPLES:
    U1 = random_unitary(1)    # 2x2 Haar random
    U2 = random_unitary(2)    # 4x4 Haar random  
    U5 = random_unitary(5)    # 32x32 Haar random (use cautiously N>10)

Brick-Wall Circuit:
  apply_brickwall_layer!(psi, gates_even, gates_odd, N)
      Apply one layer of brick-wall circuit (even + odd pairs)
      MATRIX-FREE: Uses bitwise 2-qubit gate application
  
  apply_brickwall!(psi, depth, gates, N)
      Apply full brick-wall circuit of given depth
      
  random_brickwall!(psi, depth, N)
      Apply brick-wall with fresh Haar random 2-qubit gates per layer
      
  BRICK-WALL STRUCTURE (N=6 qubits, one layer):
      Layer even: (1,2)   (3,4)   (5,6)
      Layer odd:     (2,3)   (4,5)     (6,1) [optional periodic]
      
  COMPLEXITY:
    Storage: O(16 * N/2) per layer = O(N) for 4x4 gate matrices
    Application: O(N * 2^N) per layer using bitwise indexing
    
  EXAMPLES:
    # Random brick-wall circuit
    random_brickwall!(psi, 10, N)   # Apply 10 layers of random gates
    
    # Custom brick-wall with specified gates
    gates = [random_unitary(2) for _ in 1:N÷2]
    apply_brickwall_layer!(psi, gates, [], N)

Single/Two-Qubit Gate Application:
  apply_gate!(psi, U, qubit, N)
      Apply single-qubit gate U to qubit k
      BITWISE: O(2^N) with @inbounds @simd
      
  apply_2qubit_gate!(psi, U, q1, q2, N)
      Apply 2-qubit gate U to qubits (q1, q2)
      BITWISE: O(2^N) with precomputed index mapping

BIT CONVENTION
--------------
LITTLE-ENDIAN: Qubit 1 = LSB (bit position 0).
Same convention as cpuQuantumStatePreparation.jl
Gate U acts on computational basis as: |i⟩ → U|i⟩

================================================================================
=#

module CPUQuantumChannelRandomUnitaries

using LinearAlgebra
using Random

export random_unitary
export apply_gate!, apply_2qubit_gate!
export apply_brickwall_layer!, apply_brickwall!, random_brickwall!

# ==============================================================================
# HAAR RANDOM UNITARIES
# ==============================================================================
#
# ALGORITHM
# ---------
# To generate a Haar-distributed random unitary:
# 1. Generate complex Gaussian matrix: A[i,j] = randn() + im*randn()
# 2. QR decompose: Q, R = qr(A)
# 3. Adjust phases: Q = Q * Diagonal(sign.(diag(R)))
#
# This gives an exactly Haar-distributed unitary matrix.
#
# Reference: Mezzadri, "How to Generate Random Matrices from the Classical
#            Compact Groups", Notices AMS, 2007
#
# COMPLEXITY
# ----------
# Time: O(d^3) for QR decomposition where d = 2^N
# Space: O(d^2) to store the matrix
#
# For N > 10 qubits: d = 2^10 = 1024, matrix is 1M complex elements (16MB).
# For N > 14: d = 16384, matrix is 268M elements (4GB). Use with caution!
#
# ==============================================================================

"""
    random_unitary(n_qubits::Int) -> Matrix{ComplexF64}

Generate a Haar-distributed random unitary matrix on n_qubits.

# Arguments
- `n_qubits`: Number of qubits (matrix dimension will be 2^n_qubits)

# Returns
- A 2^n_qubits × 2^n_qubits unitary matrix drawn from Haar measure

# Algorithm
Uses QR decomposition of complex Gaussian matrix with phase correction
to ensure exact Haar distribution.

# Complexity
- Time: O(8^n_qubits) for QR decomposition
- Space: O(4^n_qubits) for matrix storage

# Examples
```julia
U1 = random_unitary(1)   # 2×2 random unitary
U2 = random_unitary(2)   # 4×4 random unitary
U3 = random_unitary(3)   # 8×8 random unitary

# Verify unitarity
@assert norm(U3' * U3 - I) < 1e-12
```

# Performance Notes
- n_qubits ≤ 10: Very fast, <1 second
- n_qubits = 12: ~10 seconds, 64MB matrix
- n_qubits ≥ 14: Memory-intensive (>4GB), use cautiously
"""
function random_unitary(n_qubits::Int)
    @assert n_qubits >= 1 "Need at least 1 qubit"
    
    dim = 1 << n_qubits  # 2^n_qubits
    
    # Generate complex Gaussian matrix
    A = randn(ComplexF64, dim, dim)
    
    # QR decomposition
    Q, R = qr(A)
    
    # Extract Q as dense matrix
    Q_dense = Matrix(Q)
    
    # Phase correction to ensure Haar measure
    # Multiply each column by sign of corresponding diagonal of R
    @inbounds for j in 1:dim
        phase = sign(R[j, j])
        if phase == 0
            phase = one(ComplexF64)  # Handle edge case
        end
        for i in 1:dim
            Q_dense[i, j] *= phase
        end
    end
    
    return Q_dense
end

# ==============================================================================
# SINGLE-QUBIT GATE APPLICATION (BITWISE)
# ==============================================================================
#
# ALGORITHM
# ---------
# For a single-qubit gate U on qubit k:
#   - For each pair of indices (i, i⊕2^(k-1)) where bit k-1 is 0 in i:
#     - [psi[i], psi[i⊕mask]] = U * [psi[i], psi[i⊕mask]]
#
# This is O(2^N) with 2^(N-1) pairs, each requiring a 2x2 matrix multiply.
#
# COMPLEXITY
# ----------
# Time: O(2^N)
# Space: O(1) auxiliary (in-place)
#
# ==============================================================================

"""
    apply_gate!(psi::Vector{ComplexF64}, U::Matrix{ComplexF64}, qubit::Int, N::Int)

Apply single-qubit gate U to qubit k in-place.

# Arguments
- `psi`: State vector (length 2^N), modified in-place
- `U`: 2×2 unitary matrix
- `qubit`: Target qubit (1-indexed, 1 = LSB)
- `N`: Total number of qubits

# Algorithm
Uses bitwise indexing: for each computational basis state |i⟩ where bit (qubit-1) is 0,
apply U to the pair (|i⟩, |i ⊕ 2^(qubit-1)⟩).

# Complexity
- Time: O(2^N) - single pass with SIMD
- Space: O(1) - in-place

# Examples
```julia
psi = zeros(ComplexF64, 8)
psi[1] = 1.0  # |000⟩

# Apply Hadamard to qubit 1
H = [1 1; 1 -1] / sqrt(2)
apply_gate!(psi, H, 1, 3)
# psi now in (|000⟩ + |001⟩)/√2
```
"""
function apply_gate!(psi::Vector{ComplexF64}, U::Matrix{ComplexF64}, qubit::Int, N::Int)
    @assert 1 <= qubit <= N "Qubit index must be in 1..N"
    @assert size(U) == (2, 2) "U must be 2×2"
    
    mask = 1 << (qubit - 1)  # Bit mask for qubit position
    dim = length(psi)
    
    # Extract gate elements for speed
    u00, u01 = U[1, 1], U[1, 2]
    u10, u11 = U[2, 1], U[2, 2]
    
    @inbounds for i in 0:(dim-1)
        # Only process indices where target bit is 0
        if (i & mask) == 0
            j = i | mask  # Index with target bit = 1
            
            # Apply 2x2 gate
            a = psi[i + 1]
            b = psi[j + 1]
            
            psi[i + 1] = u00 * a + u01 * b
            psi[j + 1] = u10 * a + u11 * b
        end
    end
    
    return psi
end

# ==============================================================================
# TWO-QUBIT GATE APPLICATION (BITWISE)
# ==============================================================================
#
# ALGORITHM
# ---------
# For a 2-qubit gate U on qubits (q1, q2) with q1 < q2:
#   - Bit positions: k1 = q1-1, k2 = q2-1
#   - For each index i where bits k1 and k2 are both 0:
#     - Form 4-tuple of indices: i, i|2^k1, i|2^k2, i|2^k1|2^k2
#     - Apply 4x4 gate to these amplitudes
#
# The ordering of the 4-tuple is determined by little-endian convention:
#   Index 0: both bits = 0
#   Index 1: bit k1 = 1, bit k2 = 0
#   Index 2: bit k1 = 0, bit k2 = 1
#   Index 3: both bits = 1
#
# COMPLEXITY
# ----------
# Time: O(2^N) with 2^(N-2) groups of 4
# Space: O(1) auxiliary (in-place)
#
# ==============================================================================

"""
    apply_2qubit_gate!(psi::Vector{ComplexF64}, U::Matrix{ComplexF64}, 
                       q1::Int, q2::Int, N::Int)

Apply two-qubit gate U to qubits (q1, q2) in-place.

# Arguments
- `psi`: State vector (length 2^N), modified in-place
- `U`: 4×4 unitary matrix
- `q1`, `q2`: Target qubits (1-indexed), q1 ≠ q2
- `N`: Total number of qubits

# Algorithm
Uses bitwise indexing: for each computational basis state where both target bits are 0,
apply the 4×4 gate to the quartet of related states.

# Matrix Convention
The 4×4 gate U acts on the 2-qubit computational basis in order:
|00⟩, |10⟩, |01⟩, |11⟩ (where first bit is q1, second is q2)

# Complexity
- Time: O(2^N) - single pass
- Space: O(1) - in-place

# Examples
```julia
psi = zeros(ComplexF64, 8)
psi[1] = 1.0  # |000⟩

# Apply CNOT (control=q1, target=q2)
CNOT = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
apply_2qubit_gate!(psi, CNOT, 1, 2, 3)
```
"""
function apply_2qubit_gate!(psi::Vector{ComplexF64}, U::Matrix{ComplexF64}, 
                             q1::Int, q2::Int, N::Int)
    @assert 1 <= q1 <= N && 1 <= q2 <= N "Qubit indices must be in 1..N"
    @assert q1 != q2 "Qubits must be different"
    @assert size(U) == (4, 4) "U must be 4×4"
    
    # Ensure q1 < q2 for consistent indexing
    if q1 > q2
        q1, q2 = q2, q1
        # Swap qubits in gate: apply SWAP * U * SWAP
        SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
        U = SWAP * U * SWAP
    end
    
    mask1 = 1 << (q1 - 1)
    mask2 = 1 << (q2 - 1)
    mask_both = mask1 | mask2
    
    dim = length(psi)
    
    @inbounds for i in 0:(dim-1)
        # Only process indices where both target bits are 0
        if (i & mask_both) == 0
            # Four indices in computational basis order
            i00 = i
            i10 = i | mask1
            i01 = i | mask2
            i11 = i | mask_both
            
            # Extract amplitudes
            a00 = psi[i00 + 1]
            a10 = psi[i10 + 1]
            a01 = psi[i01 + 1]
            a11 = psi[i11 + 1]
            
            # Apply 4x4 gate
            psi[i00 + 1] = U[1,1]*a00 + U[1,2]*a10 + U[1,3]*a01 + U[1,4]*a11
            psi[i10 + 1] = U[2,1]*a00 + U[2,2]*a10 + U[2,3]*a01 + U[2,4]*a11
            psi[i01 + 1] = U[3,1]*a00 + U[3,2]*a10 + U[3,3]*a01 + U[3,4]*a11
            psi[i11 + 1] = U[4,1]*a00 + U[4,2]*a10 + U[4,3]*a01 + U[4,4]*a11
        end
    end
    
    return psi
end

# ==============================================================================
# BRICK-WALL CIRCUIT
# ==============================================================================
#
# STRUCTURE
# ---------
# A brick-wall circuit alternates between even and odd layers:
#
# Even layer (for N=6):  Gates on pairs (1,2), (3,4), (5,6)
# Odd layer:             Gates on pairs (2,3), (4,5), [optional (6,1)]
#
# Each layer applies N/2 independent 2-qubit gates, allowing parallelism.
#
# WHY BRICK-WALL?
# ---------------
# - Efficient: O(N) gates per layer, O(K*N) total for depth K
# - Universal: Depth O(N) suffices for any unitary (with random gates)
# - Natural: Models realistic quantum hardware connectivity
# - Thermalization: Random brick-wall circuits thermalize in O(N) depth
#
# COMPLEXITY
# ----------
# Per layer: O(N * 2^N) for applying N/2 gates
# Depth K:   O(K * N * 2^N) total
#
# ==============================================================================

"""
    apply_brickwall_layer!(psi::Vector{ComplexF64}, gates_even::Vector, 
                           gates_odd::Vector, N::Int; periodic::Bool=false)

Apply one brick-wall layer (even pairs, then odd pairs).

# Arguments
- `psi`: State vector, modified in-place
- `gates_even`: Vector of 4×4 gates for even pairs (1,2), (3,4), ...
- `gates_odd`: Vector of 4×4 gates for odd pairs (2,3), (4,5), ...
- `N`: Total number of qubits
- `periodic`: If true, include (N,1) pair in odd layer (default false)

# Brick-Wall Pattern (N=6)
```
Even:  ─[U1]─   ─[U2]─   ─[U3]─
       q1-q2   q3-q4   q5-q6
       
Odd:      ─[V1]─   ─[V2]─   [V3]  (periodic only)
         q2-q3   q4-q5   q6-q1
```

# Examples
```julia
N = 6
gates_even = [random_unitary(2) for _ in 1:N÷2]
gates_odd = [random_unitary(2) for _ in 1:(N÷2 - 1)]
apply_brickwall_layer!(psi, gates_even, gates_odd, N)
```
"""
function apply_brickwall_layer!(psi::Vector{ComplexF64}, 
                                 gates_even::Vector, 
                                 gates_odd::Vector, 
                                 N::Int; 
                                 periodic::Bool=false)
    # Even layer: (1,2), (3,4), (5,6), ...
    n_even = N ÷ 2
    @assert length(gates_even) >= n_even "Need $n_even gates for even layer"
    
    for k in 1:n_even
        q1 = 2k - 1
        q2 = 2k
        apply_2qubit_gate!(psi, gates_even[k], q1, q2, N)
    end
    
    # Odd layer: (2,3), (4,5), ...
    n_odd = periodic ? (N ÷ 2) : (N ÷ 2 - 1)
    if n_odd > 0 && length(gates_odd) >= n_odd
        for k in 1:min(n_odd, length(gates_odd))
            if k < N ÷ 2 || !periodic
                q1 = 2k
                q2 = 2k + 1
            else
                # Periodic boundary: (N, 1)
                q1 = N
                q2 = 1
            end
            apply_2qubit_gate!(psi, gates_odd[k], q1, q2, N)
        end
    end
    
    return psi
end

"""
    apply_brickwall!(psi::Vector{ComplexF64}, depth::Int, 
                     even_gates::Vector{Vector}, odd_gates::Vector{Vector}, N::Int;
                     periodic::Bool=false)

Apply full brick-wall circuit of specified depth.

# Arguments
- `psi`: State vector, modified in-place
- `depth`: Number of layers (each layer = even + odd)
- `even_gates`: Vector of gate vectors, one per layer
- `odd_gates`: Vector of gate vectors, one per layer
- `N`: Total number of qubits
- `periodic`: Include (N,1) boundary in odd layers

# Examples
```julia
# Pre-generate all gates
depth = 10
even_gates = [[random_unitary(2) for _ in 1:N÷2] for _ in 1:depth]
odd_gates = [[random_unitary(2) for _ in 1:(N÷2-1)] for _ in 1:depth]
apply_brickwall!(psi, depth, even_gates, odd_gates, N)
```
"""
function apply_brickwall!(psi::Vector{ComplexF64}, depth::Int,
                          even_gates::Vector, odd_gates::Vector, N::Int;
                          periodic::Bool=false)
    for layer in 1:depth
        apply_brickwall_layer!(psi, even_gates[layer], odd_gates[layer], N; 
                               periodic=periodic)
    end
    return psi
end

"""
    random_brickwall!(psi::Vector{ComplexF64}, depth::Int, N::Int; 
                      periodic::Bool=false)

Apply brick-wall circuit with fresh Haar random 2-qubit gates at each layer.

# Arguments
- `psi`: State vector, modified in-place
- `depth`: Number of brick-wall layers
- `N`: Total number of qubits (should be even)
- `periodic`: Include boundary gates (N,1)

# Algorithm
For each layer:
1. Generate N/2 fresh Haar random 4×4 unitaries for even pairs
2. Generate (N/2 - 1) fresh random unitaries for odd pairs (or N/2 if periodic)
3. Apply brick-wall layer

# Complexity
- Gate generation: O(depth * N * 64) for 4×4 QR decompositions
- Application: O(depth * N * 2^N)

# Examples
```julia
psi = zeros(ComplexF64, 2^10)
psi[1] = 1.0
random_brickwall!(psi, 20, 10)  # 20 layers on 10 qubits
# psi is now approximately a random state (Page scrambled)
```

# Physical Interpretation
- Depth ~ N: State becomes Page-scrambled (maximally entangled)
- Depth ~ log(N): Long-range correlations begin to develop
- Random brick-wall models chaotic quantum dynamics
"""
function random_brickwall!(psi::Vector{ComplexF64}, depth::Int, N::Int;
                            periodic::Bool=false)
    @assert iseven(N) "N should be even for brick-wall"
    
    n_even = N ÷ 2
    n_odd = periodic ? n_even : (n_even - 1)
    
    for _ in 1:depth
        # Generate fresh random gates
        gates_even = [random_unitary(2) for _ in 1:n_even]
        gates_odd = n_odd > 0 ? [random_unitary(2) for _ in 1:n_odd] : Matrix{ComplexF64}[]
        
        # Apply layer
        apply_brickwall_layer!(psi, gates_even, gates_odd, N; periodic=periodic)
    end
    
    return psi
end

end # module

# Date: 2026
#
#=
================================================================================
    cpuQuantumStatePreparation.jl - Core Quantum State Operations (CPU)
================================================================================

OVERVIEW
--------
Core operations for quantum state construction and manipulation. All operations
use matrix-free bitwise implementations for optimal performance.

FUNCTIONS
---------

State Construction:
  make_ket(state::Symbol)              Single-qubit: make_ket(:zero) -> |0>
  make_ket(state_str, N)               Uniform: make_ket("|0>", 4) -> |0000>
  make_ket("|0+0-1+>")                 Per-qubit: 6-qubit state
  make_rho(...)                        Density matrix versions of above
  make_product_ket([:zero,:one])       Product: |0> (x) |1> = |01>
  make_initial_psi(N, [0,1,1,0])       Computational basis: |0110>
  make_product_state(N, angles)        Rotated product state
  make_maximally_mixed(N)              I/2^N (Diagonal for efficiency)

Normalization and Metrics:
  normalize_state!(psi)       In-place: psi = psi/||psi||
  normalize_state!(rho)       In-place: rho = rho/Tr(rho)
  get_norm(psi)         SIMD-optimized ||psi||_2
  get_trace(rho)        SIMD-optimized Tr(rho)

Tensor Products:
  tensor(psi_A, psi_B)                       |psi_A> (x) |psi_B>
  tensor(rho_A, rho_B)                       rho_A (x) rho_B
  tensor_product([psi_A, psi_B, psi_C])      Multi-state tensor
  tensor_product([rho_A, psi_B, rho_C])      Mixed: psi auto-converted to |.><.|

Qubit Reset:
  Traces out specified qubit(s), then tensors fresh state back in.
  Result: Tr_{qubits}(state) (x) |fresh><fresh|
  
  UNIFIED API:
  reset(psi, qubits, states, N)     Works for Vector (pure) or Matrix (rho)
  reset(rho, qubits, states, N)     qubits: Int or Vector, states: Symbol or Vector
  
  EXAMPLES:
  reset(psi, 2, :zero, 4)           Reset qubit 2 of 4-qubit psi to |0>
  reset(rho, 3, :plus, 4)           Reset qubit 3 of 4-qubit rho to |+><+|
  reset(psi, [1,2], [:zero,:one], 4)  Reset qubits 1,2 to |0>,|1>
  reset(psi, 1:L, input_states, N)    QRC: reset input rail each timestep
  
  LEGACY (still available):
  reset_qubit_psi, reset_qubit_rho, reset_qubits_psi, reset_qubits_rho

Entangled States (with local basis support):
  All entangled states support local basis specification:
  - Single char ("z", "x", "y"): uniform basis for all qubits
  - Per-qubit string ("xyzzx"): different basis per qubit
  
  Basis meanings:
    'z' - computational: |0_z>=|0>, |1_z>=|1>
    'x' - X eigenbasis:  |0_x>=|+>, |1_x>=|->
    'y' - Y eigenbasis:  |0_y>=|+i>, |1_y>=|-i>  where |+i>=(|0>+i|1>)/sqrt(2)
  
  make_ghz(N, basis="z")
      GHZ = (|0_B...0_B> + |1_B...1_B>) / sqrt(2)
      Examples:
        make_ghz(4)          # (|0000> + |1111>) / sqrt(2)
        make_ghz(4, "x")     # (|++++> + |---->) / sqrt(2)
        make_ghz(5, "xyzzx") # Per-qubit basis GHZ
        
  make_w(N, basis="z")
      W = sum_k |0_B...1_B...0_B> / sqrt(N)  (one excitation)
      Examples:
        make_w(3)        # (|100> + |010> + |001>) / sqrt(3)
        make_w(3, "x")   # (|--+> + |-+-> + |+-->) / sqrt(3)
        
  make_bell(type, basis="z")
      type: :phi_plus, :phi_minus, :psi_plus, :psi_minus
      Examples:
        make_bell(:phi_plus)        # (|00> + |11>) / sqrt(2)
        make_bell(:phi_plus, "x")   # (|++> + |-->) / sqrt(2)
        make_bell(:psi_minus, "xy") # (|0_x 1_y> - |1_x 0_y>) / sqrt(2)

BIT CONVENTION
--------------
LITTLE-ENDIAN: Qubit 1 = LSB (bit position 0).
State |q_N, ..., q_2, q_1> has index = q_1 + 2*q_2 + 4*q_3 + ...

Example: |01> means qubit 1 = 0, qubit 2 = 1, stored at index 2 (0-indexed).

================================================================================
=#

module CPUQuantumStatePreparation

using LinearAlgebra

# Import partial trace for reset operations
using ..CPUQuantumStatePartialTrace: partial_trace

export make_ket, make_rho
export normalize_state!, get_norm, get_trace
export tensor, tensor_product, tensor_product_ket, tensor_product_ket!, tensor_product_rho
export make_product_ket, make_product_rho
export make_initial_psi, make_initial_rho, make_product_state
export make_maximally_mixed
export make_ghz, make_w, make_bell
export reset
export reset_qubit_psi, reset_qubit_rho
export reset_qubits_psi, reset_qubits_rho

# ==============================================================================
# SINGLE-QUBIT BASIS STATES
# ==============================================================================

"""
    make_ket(state::Symbol) -> Vector{ComplexF64}

Create single-qubit pure state.

# Arguments
- `state`: One of :zero (|0⟩), :one (|1⟩), :plus (|+⟩), :minus (|-⟩)

# Examples
```julia
make_ket(:zero)   # [1, 0]
make_ket(:one)    # [0, 1]
make_ket(:plus)   # [1/√2, 1/√2]
make_ket(:minus)  # [1/√2, -1/√2]
```
"""
function make_ket(state::Symbol)
    if state == :zero
        return ComplexF64[1.0, 0.0]
    elseif state == :one
        return ComplexF64[0.0, 1.0]
    elseif state == :plus
        s = 1/sqrt(2)
        return ComplexF64[s, s]
    elseif state == :minus
        s = 1/sqrt(2)
        return ComplexF64[s, -s]
    else
        error("Unknown state: $state. Use :zero, :one, :plus, or :minus")
    end
end

"""
    make_rho(state::Symbol) -> Matrix{ComplexF64}

Create single-qubit density matrix |state⟩⟨state|.

# Arguments
- `state`: One of :zero, :one, :plus, :minus
"""
function make_rho(state::Symbol)
    ψ = make_ket(state)
    return ψ * ψ'
end

"""
    make_ket(state_str::String) -> Vector{ComplexF64}
    make_ket(state_str::String, N::Int) -> Vector{ComplexF64}

Create N-qubit product state from Dirac notation string.

# Two Usage Modes

**Mode 1: Uniform state (requires N)**
- `make_ket("|0>", N)` → All N qubits in |0⟩
- `make_ket("|+>", N)` → All N qubits in |+⟩

**Mode 2: Per-qubit specification (N optional/inferred)**
- `make_ket("|0+0-1+>")` → 6-qubit state: |0⟩⊗|+⟩⊗|0⟩⊗|-⟩⊗|1⟩⊗|+⟩
- `make_ket("|01+->")` → 4-qubit state: |0⟩⊗|1⟩⊗|+⟩⊗|-⟩

# Supported Characters
- `0` → |0⟩ (computational zero)
- `1` → |1⟩ (computational one)
- `+` → |+⟩ = (|0⟩+|1⟩)/√2
- `-` → |-⟩ = (|0⟩-|1⟩)/√2

# Examples
```julia
# Uniform states (all qubits same)
make_ket("|0>", 4)        # |0000⟩
make_ket("|+>", 3)        # |+++⟩

# Per-qubit specification
make_ket("|0+0-1+>")      # |0⟩⊗|+⟩⊗|0⟩⊗|-⟩⊗|1⟩⊗|+⟩ (6 qubits)
make_ket("|01>")          # |0⟩⊗|1⟩ = |01⟩
make_ket("|+-+-+>")       # Alternating |+⟩ and |-⟩

# GHZ state construction
psi_GHZ = make_ket("|0000>") + make_ket("|1111>")
normalize_state!(psi_GHZ)

# Bell state
psi_bell = make_ket("|00>") + make_ket("|11>")
normalize_state!(psi_bell)
```

# Notes
- When using per-qubit mode with explicit N, validates that string length matches N
- Dirac brackets (|, >, ⟩) are optional and stripped automatically
"""
function make_ket(state_str::String, N::Union{Int, Nothing}=nothing)
    # Clean up the state string - remove Dirac notation brackets
    state_clean = replace(strip(state_str), "|" => "", ">" => "", "⟩" => "")
    
    # Check if this is a uniform state (single character) or per-qubit specification
    if length(state_clean) == 1
        # Uniform state: all qubits in same state
        # N must be provided for uniform states
        if N === nothing
            error("For single-character states like \"|0>\", N must be provided. Use make_ket(\"|0>\", N)")
        end
        @assert N >= 1 "Need at least 1 qubit"
        
        state_sym = _char_to_state_symbol(state_clean[1])
        return make_product_ket(fill(state_sym, N))
    else
        # Per-qubit specification: each character is a qubit state
        n_qubits = length(state_clean)
        
        # If N provided, validate it matches string length
        if N !== nothing && N != n_qubits
            error("State string \"$state_str\" has $(n_qubits) qubits but N=$N was specified")
        end
        
        # Parse each character to a state symbol
        state_syms = [_char_to_state_symbol(c) for c in state_clean]
        
        return make_product_ket(state_syms)
    end
end

"""
    _char_to_state_symbol(c::Char) -> Symbol

Convert a single character to a state symbol. Internal helper function.

# Mapping
- '0' → :zero
- '1' → :one
- '+' → :plus
- '-' → :minus
"""
function _char_to_state_symbol(c::Char)
    if c == '0'
        return :zero
    elseif c == '1'
        return :one
    elseif c == '+'
        return :plus
    elseif c == '-'
        return :minus
    else
        error("Unknown state character: '$c'. Use '0', '1', '+', or '-'")
    end
end

"""
    make_rho(state_str::String) -> Matrix{ComplexF64}
    make_rho(state_str::String, N::Int) -> Matrix{ComplexF64}

Create N-qubit product density matrix from Dirac notation string.
Supports both uniform states and per-qubit specification (same as make_ket).

# Examples
```julia
# Uniform state
make_rho("|0>", 3)        # |000⟩⟨000|
make_rho("|+>", 2)        # |++⟩⟨++|

# Per-qubit specification  
make_rho("|0+0-1+>")      # 6-qubit product state density matrix
make_rho("|01>")          # |01⟩⟨01|
```

See also: [`make_ket`](@ref)
"""
function make_rho(state_str::String, N::Union{Int, Nothing}=nothing)
    ψ = make_ket(state_str, N)
    return ψ * ψ'
end

# ==============================================================================
# IN-PLACE NORMALIZATION
# ==============================================================================

"""
    normalize_state!(ψ::Vector{ComplexF64}) -> Vector{ComplexF64}

Normalize pure state vector in-place: ψ → ψ/||ψ||.

# Examples
```julia
psi = make_ket("|0>", 3) + make_ket("|1>", 3)
normalize_state!(psi)  # Now (|000⟩ + |111⟩)/√2

# Returns the normalized state for chaining
norm(normalize_state!(psi))  # ≈ 1.0
```
"""
function normalize_state!(ψ::Vector{ComplexF64})
    nrm = norm(ψ)
    if nrm < 1e-15
        error("Cannot normalize zero vector")
    end
    ψ ./= nrm
    return ψ
end

"""
    normalize_state!(ρ::Matrix{ComplexF64}) -> Matrix{ComplexF64}

Normalize density matrix in-place: ρ → ρ/Tr(ρ).

# Examples
```julia
rho = [2.0+0im 0; 0 2.0+0im]  # Trace = 4
normalize_state!(rho)               # Now Trace = 1
```
"""
function normalize_state!(ρ::Matrix{ComplexF64})
    trace_val = real(tr(ρ))
    if abs(trace_val) < 1e-15
        error("Cannot normalize matrix with zero trace")
    end
    ρ ./= trace_val
    return ρ
end

# ==============================================================================
# FAST NORM AND TRACE
# ==============================================================================
#
# OVERVIEW
# --------
# Custom implementations using @inbounds and @simd for maximum performance.
# These are consistent with the bitwise operations used throughout the codebase.
#
# For pure states: ||psi||_2 = sqrt(sum_i |psi_i|^2)
# For density matrices: Tr(rho) = sum_i rho[i,i]
#
# ==============================================================================

"""
    get_norm(psi::Vector{ComplexF64}) -> Float64

Compute the 2-norm of a pure state vector: ||psi||_2 = sqrt(sum_i |psi_i|^2).

Uses SIMD-optimized loop with @inbounds for maximum performance.
This is the same as LinearAlgebra.norm() but guaranteed to use our
fast bitwise-consistent implementation.

# Arguments
- `psi`: Pure state vector (length 2^N)

# Returns
- The 2-norm (Euclidean norm) of the state

# Examples
```julia
psi = make_ket("|0>", 3) + make_ket("|1>", 3)  # Unnormalized
get_norm(psi)  # sqrt(2)

psi_normalized = psi / get_norm(psi)
get_norm(psi_normalized)  # 1.0
```

# Complexity
O(2^N) - single pass over state vector.
"""
function get_norm(psi::Vector{ComplexF64})
    result = 0.0
    @inbounds @simd for i in 1:length(psi)
        result += abs2(psi[i])
    end
    return sqrt(result)
end

"""
    get_trace(rho::Matrix{ComplexF64}) -> ComplexF64
    get_trace(rho::Diagonal{ComplexF64}) -> ComplexF64

Compute the trace of a density matrix: Tr(rho) = sum_i rho[i,i].

Uses SIMD-optimized loop for dense matrices. For Diagonal matrices
(like from make_maximally_mixed), uses optimized diagonal sum.

# Arguments
- `rho`: Density matrix (2^N x 2^N) or Diagonal matrix

# Returns
- The trace (sum of diagonal elements)

# Examples
```julia
rho = make_rho("|0>", 3)
get_trace(rho)  # 1.0 (pure state)

rho_mixed = make_maximally_mixed(3)
get_trace(rho_mixed)  # 1.0 (normalized)
```

# Complexity
O(2^N) - single pass over diagonal.
"""
function get_trace(rho::Matrix{ComplexF64})
    dim = size(rho, 1)
    result = zero(ComplexF64)
    @inbounds @simd for i in 1:dim
        result += rho[i, i]
    end
    return result
end

# Optimized version for Diagonal matrices
function get_trace(rho::Diagonal{ComplexF64, Vector{ComplexF64}})
    result = zero(ComplexF64)
    diag_vals = rho.diag
    @inbounds @simd for i in 1:length(diag_vals)
        result += diag_vals[i]
    end
    return result
end

# ==============================================================================
# ENTANGLED STATE CONSTRUCTORS
# ==============================================================================
#
# OVERVIEW
# --------
# Construct maximally entangled states in various local bases.
#
# GHZ STATE
# ---------
# GHZ_N = (|0_B 0_B ... 0_B> + |1_B 1_B ... 1_B>) / sqrt(2)
#
# where B can be:
#   'z' - computational basis: |0_z>=|0>, |1_z>=|1>
#   'x' - X eigenbasis:        |0_x>=|+>, |1_x>=|->
#   'y' - Y eigenbasis:        |0_y>=|+i>, |1_y>=|-i>
#
# W STATE
# -------
# W_N = (|100...0> + |010...0> + ... + |000...1>) / sqrt(N)
# Single excitation equally distributed across all qubits.
#
# BELL STATES
# -----------
# |Phi+> = (|00> + |11>) / sqrt(2)
# |Phi-> = (|00> - |11>) / sqrt(2)
# |Psi+> = (|01> + |10>) / sqrt(2)
# |Psi-> = (|01> - |10>) / sqrt(2)
#
# ==============================================================================

"""
    _basis_eigenstates(axis::Char) -> (Vector{ComplexF64}, Vector{ComplexF64})

Get the |0_B> and |1_B> eigenstates for the given Pauli axis.

# Arguments
- `axis`: 'z', 'x', or 'y'

# Returns
- Tuple (|0_B>, |1_B>) where B is the specified basis

# Eigenstates
- Z: |0_z> = |0>,  |1_z> = |1>
- X: |0_x> = |+>,  |1_x> = |->
- Y: |0_y> = |+i>, |1_y> = |-i>  where |+i> = (|0> + i|1>)/sqrt(2)
"""
function _basis_eigenstates(axis::Char)
    s = 1 / sqrt(2)
    if axis == 'z' || axis == 'Z'
        return (ComplexF64[1.0, 0.0], ComplexF64[0.0, 1.0])
    elseif axis == 'x' || axis == 'X'
        return (ComplexF64[s, s], ComplexF64[s, -s])  # |+>, |->
    elseif axis == 'y' || axis == 'Y'
        return (ComplexF64[s, im*s], ComplexF64[s, -im*s])  # |+i>, |-i>
    else
        error("Unknown axis: '$axis'. Use 'x', 'y', or 'z'")
    end
end

"""
    make_ghz(N::Int, basis::String="z") -> Vector{ComplexF64}

Create N-qubit GHZ state in specified local basis.

GHZ = (|0_B 0_B ... 0_B> + |1_B 1_B ... 1_B>) / sqrt(2)

# Arguments
- `N`: Number of qubits
- `basis`: Basis specification (default "z")
  - Single char ("z", "x", "y"): uniform basis for all qubits
  - String of length N ("xyzzx"): per-qubit local basis

# Basis Meanings
- 'z': computational basis (|0>, |1>)
- 'x': X eigenbasis (|+>, |->)
- 'y': Y eigenbasis (|+i>, |-i>)

# Examples
```julia
# Standard GHZ in computational basis
ghz_z = make_ghz(4)        # (|0000> + |1111>) / sqrt(2)
ghz_z = make_ghz(4, "z")   # Same as above

# GHZ in X basis
ghz_x = make_ghz(4, "x")   # (|++++> + |---->) / sqrt(2)

# Per-qubit basis specification
ghz_mixed = make_ghz(5, "xyzzx")
# = (|0_x 0_y 0_z 0_z 0_x> + |1_x 1_y 1_z 1_z 1_x>) / sqrt(2)
# = (|+ +i 0 0 +> + |- -i 1 1 ->) / sqrt(2)
```

# Returns
- Normalized N-qubit pure state vector (length 2^N)

# Implementation
Uses direct bitwise matrix-free construction: O(2^N) single pass.
"""
function make_ghz(N::Int, basis::String="z")
    @assert N >= 2 "GHZ state requires at least 2 qubits"
    
    # Parse basis specification
    if length(basis) == 1
        # Uniform basis for all qubits
        axes = fill(basis[1], N)
    elseif length(basis) == N
        # Per-qubit basis
        axes = collect(basis)
    else
        error("Basis string must be length 1 (uniform) or length N=$N (per-qubit)")
    end
    
    # Precompute single-qubit states for each axis
    # For qubit k: |0_Bk> = [a0_k, a1_k], |1_Bk> = [b0_k, b1_k]
    a0 = Vector{ComplexF64}(undef, N)  # |0_Bk>[0] - amplitude when local bit=0
    a1 = Vector{ComplexF64}(undef, N)  # |0_Bk>[1] - amplitude when local bit=1
    b0 = Vector{ComplexF64}(undef, N)  # |1_Bk>[0]
    b1 = Vector{ComplexF64}(undef, N)  # |1_Bk>[1]
    
    for k in 1:N
        zero_k, one_k = _basis_eigenstates(axes[k])
        a0[k] = zero_k[1]
        a1[k] = zero_k[2]
        b0[k] = one_k[1]
        b1[k] = one_k[2]
    end
    
    # GHZ = (|0_B1 0_B2 ... 0_BN> + |1_B1 1_B2 ... 1_BN>) / sqrt(2)
    #
    # For each computational basis index i, the amplitude is:
    #   (1/sqrt(2)) * [product of a_{bit_k}[k] + product of b_{bit_k}[k]]
    #
    # where bit_k is the k-th bit of i (0 or 1), and:
    #   a_{0}[k] = a0[k], a_{1}[k] = a1[k]  (from |0_Bk>)
    #   b_{0}[k] = b0[k], b_{1}[k] = b1[k]  (from |1_Bk>)
    
    dim = 1 << N
    ghz = Vector{ComplexF64}(undef, dim)
    inv_sqrt2 = 1 / sqrt(2)
    
    @inbounds for i in 0:(dim-1)
        # Compute product for |all_zeros> term
        prod_zero = one(ComplexF64)
        prod_one = one(ComplexF64)
        
        for k in 0:(N-1)
            bit_k = (i >> k) & 1
            if bit_k == 0
                prod_zero *= a0[k+1]
                prod_one *= b0[k+1]
            else
                prod_zero *= a1[k+1]
                prod_one *= b1[k+1]
            end
        end
        
        ghz[i+1] = inv_sqrt2 * (prod_zero + prod_one)
    end
    
    return ghz
end

"""
    make_w(N::Int, basis::String="z") -> Vector{ComplexF64}

Create N-qubit W state (single excitation equally distributed) in specified local basis.

W_N = (|0_B...0_B 1_B 0_B...0_B> + permutations) / sqrt(N)

where exactly one qubit is in |1_B> and all others are in |0_B>.

# Arguments
- `N`: Number of qubits (must be >= 2)
- `basis`: Basis specification (default "z")
  - Single char ("z", "x", "y"): uniform basis for all qubits
  - String of length N ("xyzzx"): per-qubit local basis

# Properties
- Total of N terms, each with exactly one qubit in |1_B>
- Normalized: ||W||^2 = 1
- Robust to single-qubit loss

# Examples
```julia
# Standard W state
make_w(3)        # (|100> + |010> + |001>) / sqrt(3)
make_w(3, "z")   # Same

# W state in X basis
make_w(3, "x")   # (|--+> + |-+-> + |+-->) / sqrt(3)

# Per-qubit basis
make_w(3, "xyz") # W state with per-qubit local bases
```

# Implementation
Uses direct bitwise matrix-free construction: O(N * 2^N) single pass.
"""
function make_w(N::Int, basis::String="z")
    @assert N >= 2 "W state requires at least 2 qubits"
    
    # Parse basis specification
    if length(basis) == 1
        axes = fill(basis[1], N)
    elseif length(basis) == N
        axes = collect(basis)
    else
        error("Basis string must be length 1 (uniform) or length N=$N (per-qubit)")
    end
    
    # Precompute single-qubit states
    a0 = Vector{ComplexF64}(undef, N)  # |0_Bk>[0]
    a1 = Vector{ComplexF64}(undef, N)  # |0_Bk>[1]
    b0 = Vector{ComplexF64}(undef, N)  # |1_Bk>[0]
    b1 = Vector{ComplexF64}(undef, N)  # |1_Bk>[1]
    
    for k in 1:N
        zero_k, one_k = _basis_eigenstates(axes[k])
        a0[k] = zero_k[1]
        a1[k] = zero_k[2]
        b0[k] = one_k[1]
        b1[k] = one_k[2]
    end
    
    # W = sum_k |0_B1 ... 1_Bk ... 0_BN> / sqrt(N)
    # For each computational index i, sum contributions from all N terms
    
    dim = 1 << N
    w = zeros(ComplexF64, dim)
    inv_sqrtN = 1 / sqrt(N)
    
    # For each term k (where qubit k has the excitation)
    @inbounds for excite_qubit in 0:(N-1)
        # For each computational basis index i
        for i in 0:(dim-1)
            # Compute amplitude: product of appropriate local states
            # Qubit excite_qubit uses |1_B>, all others use |0_B>
            amp = one(ComplexF64)
            
            for k in 0:(N-1)
                bit_k = (i >> k) & 1
                if k == excite_qubit
                    # This qubit is in |1_B>
                    amp *= (bit_k == 0) ? b0[k+1] : b1[k+1]
                else
                    # This qubit is in |0_B>
                    amp *= (bit_k == 0) ? a0[k+1] : a1[k+1]
                end
            end
            
            w[i+1] += inv_sqrtN * amp
        end
    end
    
    return w
end

"""
    make_bell(type::Symbol, basis::String="z") -> Vector{ComplexF64}

Create one of the four Bell states in specified local basis.

# Arguments
- `type`: One of :phi_plus, :phi_minus, :psi_plus, :psi_minus
- `basis`: Basis specification (default "z")
  - Single char ("z", "x", "y"): uniform basis for both qubits
  - Two chars ("xy"): per-qubit local basis

# Bell States (in computational basis)
- :phi_plus  (|Phi+>): (|0_B 0_B> + |1_B 1_B>) / sqrt(2)
- :phi_minus (|Phi->): (|0_B 0_B> - |1_B 1_B>) / sqrt(2)
- :psi_plus  (|Psi+>): (|0_B 1_B> + |1_B 0_B>) / sqrt(2)
- :psi_minus (|Psi->): (|0_B 1_B> - |1_B 0_B>) / sqrt(2)

# Examples
```julia
# Standard Bell states
make_bell(:phi_plus)         # (|00> + |11>) / sqrt(2)
make_bell(:psi_minus)        # (|01> - |10>) / sqrt(2)

# Bell state in X basis
make_bell(:phi_plus, "x")    # (|++> + |-->) / sqrt(2)

# Per-qubit basis
make_bell(:phi_plus, "xy")   # (|0_x 0_y> + |1_x 1_y>) / sqrt(2)
```

# Implementation
Uses direct bitwise matrix-free construction: O(4) single pass.
"""
function make_bell(type::Symbol, basis::String="z")
    # Parse basis specification
    if length(basis) == 1
        axes = [basis[1], basis[1]]
    elseif length(basis) == 2
        axes = collect(basis)
    else
        error("Basis string must be length 1 (uniform) or length 2 (per-qubit)")
    end
    
    # Get local basis states
    a0 = Vector{ComplexF64}(undef, 2)
    a1 = Vector{ComplexF64}(undef, 2)
    b0 = Vector{ComplexF64}(undef, 2)
    b1 = Vector{ComplexF64}(undef, 2)
    
    for k in 1:2
        zero_k, one_k = _basis_eigenstates(axes[k])
        a0[k] = zero_k[1]
        a1[k] = zero_k[2]
        b0[k] = one_k[1]
        b1[k] = one_k[2]
    end
    
    # Determine which pairs to combine and with what sign
    # Phi states: (|0_B0_B> +/- |1_B1_B>)
    # Psi states: (|0_B1_B> +/- |1_B0_B>)
    
    if type == :phi_plus
        first_pair = (:zero, :zero)   # Both in |0_B>
        second_pair = (:one, :one)    # Both in |1_B>
        sign = 1
    elseif type == :phi_minus
        first_pair = (:zero, :zero)
        second_pair = (:one, :one)
        sign = -1
    elseif type == :psi_plus
        first_pair = (:zero, :one)    # First |0_B>, second |1_B>
        second_pair = (:one, :zero)   # First |1_B>, second |0_B>
        sign = 1
    elseif type == :psi_minus
        first_pair = (:zero, :one)
        second_pair = (:one, :zero)
        sign = -1
    else
        error("Unknown Bell state type: $type. Use :phi_plus, :phi_minus, :psi_plus, or :psi_minus")
    end
    
    # Build state using bitwise indexing
    bell = zeros(ComplexF64, 4)
    inv_sqrt2 = 1 / sqrt(2)
    
    @inbounds for i in 0:3
        bit_0 = i & 1
        bit_1 = (i >> 1) & 1
        
        # First term amplitude
        amp1 = one(ComplexF64)
        amp1 *= (first_pair[1] == :zero) ? 
                ((bit_0 == 0) ? a0[1] : a1[1]) : 
                ((bit_0 == 0) ? b0[1] : b1[1])
        amp1 *= (first_pair[2] == :zero) ? 
                ((bit_1 == 0) ? a0[2] : a1[2]) : 
                ((bit_1 == 0) ? b0[2] : b1[2])
        
        # Second term amplitude
        amp2 = one(ComplexF64)
        amp2 *= (second_pair[1] == :zero) ? 
                ((bit_0 == 0) ? a0[1] : a1[1]) : 
                ((bit_0 == 0) ? b0[1] : b1[1])
        amp2 *= (second_pair[2] == :zero) ? 
                ((bit_1 == 0) ? a0[2] : a1[2]) : 
                ((bit_1 == 0) ? b0[2] : b1[2])
        
        bell[i+1] = inv_sqrt2 * (amp1 + sign * amp2)
    end
    
    return bell
end

# ==============================================================================
# TENSOR PRODUCTS
# ==============================================================================
#
# OVERVIEW
# --------
# Tensor products combine quantum states from separate subsystems into a joint
# state. For pure states: |psi_AB> = |psi_A> (x) |psi_B>. For density matrices:
# rho_AB = rho_A (x) rho_B.
#
# The unified tensor() function dispatches on argument types and automatically
# infers qubit counts from state dimensions.
#
# BIT CONVENTION
# --------------
# LITTLE-ENDIAN: First argument occupies LOWER bits (qubits 1..N_A),
# second argument occupies UPPER bits (qubits N_A+1..N_A+N_B).
#
# Example: tensor(|01>, |10>) gives |0110> where:
#   - Qubits 1,2 contain |01> from first argument
#   - Qubits 3,4 contain |10> from second argument
#
# ==============================================================================

"""
    tensor(psi_A::Vector{ComplexF64}, psi_B::Vector{ComplexF64}) -> Vector{ComplexF64}

Compute tensor product of two pure states: |psi_A> (x) |psi_B>.

Automatically infers qubit counts from state dimensions (dim = 2^N).

# Arguments
- `psi_A`: First pure state vector (length 2^N_A)
- `psi_B`: Second pure state vector (length 2^N_B)

# Returns
- Combined state vector of length 2^(N_A + N_B)

# Bit Convention
- psi_A occupies lower bits (qubits 1..N_A)
- psi_B occupies upper bits (qubits N_A+1..N_A+N_B)

# Examples
```julia
# Bell state from product
psi_00 = tensor(make_ket(:zero), make_ket(:zero))  # |00>
psi_11 = tensor(make_ket(:one), make_ket(:one))    # |11>
psi_bell = psi_00 + psi_11
normalize_state!(psi_bell)  # (|00> + |11>)/sqrt(2)

# Multi-qubit tensor
psi_abc = tensor(tensor(psi_a, psi_b), psi_c)  # |a> (x) |b> (x) |c>
```

See also: [`tensor_product_ket`](@ref), [`tensor_product_rho`](@ref)
"""
function tensor(psi_A::Vector{ComplexF64}, psi_B::Vector{ComplexF64})
    dim_A = length(psi_A)
    dim_B = length(psi_B)
    
    # Infer qubit counts from dimensions (dim = 2^N)
    N_A = Int(log2(dim_A))
    N_B = Int(log2(dim_B))
    
    # Validate dimensions are powers of 2
    @assert (1 << N_A) == dim_A "psi_A dimension must be power of 2"
    @assert (1 << N_B) == dim_B "psi_B dimension must be power of 2"
    
    return tensor_product_ket(psi_A, psi_B, N_A, N_B)
end

"""
    tensor(rho_A::Matrix{ComplexF64}, rho_B::Matrix{ComplexF64}) -> Matrix{ComplexF64}

Compute tensor product of two density matrices: rho_A (x) rho_B.

Automatically infers qubit counts from matrix dimensions.

# Arguments
- `rho_A`: First density matrix (2^N_A x 2^N_A)
- `rho_B`: Second density matrix (2^N_B x 2^N_B)

# Returns
- Combined density matrix of dimension 2^(N_A + N_B) x 2^(N_A + N_B)

# Bit Convention
- rho_A occupies lower bits (qubits 1..N_A)
- rho_B occupies upper bits (qubits N_A+1..N_A+N_B)

# Examples
```julia
# Product state density matrix
rho_00 = tensor(make_rho(:zero), make_rho(:zero))  # |00><00|

# Mixed state tensor product
rho_mixed = tensor(rho_thermal_A, rho_thermal_B)
```

See also: [`tensor_product_rho`](@ref), [`tensor`](@ref) for pure states
"""
function tensor(rho_A::Matrix{ComplexF64}, rho_B::Matrix{ComplexF64})
    return tensor_product_rho(rho_A, rho_B)
end

# ==============================================================================
# MULTI-STATE TENSOR PRODUCT
# ==============================================================================
#
# OVERVIEW
# --------
# tensor_product([state1, state2, ...]) computes the tensor product of multiple
# quantum states in a single call. Handles mixed pure state / density matrix
# inputs intelligently:
#
# - All pure states -> returns pure state (Vector)
# - Any density matrix -> converts pure states to |psi><psi|, returns Matrix
#
# ALGORITHM
# ---------
# Uses left-fold reduction: ((A (x) B) (x) C) (x) D ...
# Each pairwise tensor uses our bitwise-optimized implementations.
#
# ==============================================================================

"""
    tensor_product(states::Vector) -> Vector{ComplexF64} or Matrix{ComplexF64}

Compute tensor product of multiple quantum states.

Handles mixed inputs of pure states (Vector) and density matrices (Matrix):
- If ALL inputs are pure states -> returns pure state tensor product
- If ANY input is a density matrix -> converts pure states to |psi><psi|
  and returns density matrix tensor product

# Arguments
- `states`: Vector of quantum states (Vector{ComplexF64} or Matrix{ComplexF64})

# Returns
- Pure state vector if all inputs are pure states
- Density matrix if any input is a density matrix

# Bit Convention
- States are tensored left-to-right: states[1] (x) states[2] (x) ...
- First state occupies lowest bits, last state occupies highest bits

# Examples
```julia
# Pure states only -> returns pure state
psi_abc = tensor_product([make_ket(:zero), make_ket(:one), make_ket(:plus)])
# Equivalent to: tensor(tensor(|0>, |1>), |+>)

# Density matrices only
rho_abc = tensor_product([make_rho(:zero), make_rho(:one), make_rho(:plus)])

# Mixed inputs -> pure states auto-converted to |psi><psi|
rho_mixed = tensor_product([make_rho(:zero), make_ket(:one), make_rho(:plus)])
# make_ket(:one) is converted to |1><1|, then all tensored as density matrices
```

# Performance
O(2^N_total) for pure states, O(4^N_total) for density matrices.
Uses bitwise-optimized pairwise tensor products internally.

See also: [`tensor`](@ref), [`tensor_product_ket`](@ref), [`tensor_product_rho`](@ref)
"""
function tensor_product(states::Vector)
    @assert length(states) >= 1 "Need at least one state"
    
    # Check if any state is a density matrix
    has_density_matrix = any(s -> s isa Matrix, states)
    
    if has_density_matrix
        # Convert all pure states to density matrices, then tensor
        return _tensor_product_as_rho(states)
    else
        # All pure states - tensor as vectors
        return _tensor_product_as_psi(states)
    end
end

"""
    _tensor_product_as_psi(states::Vector) -> Vector{ComplexF64}

Internal: Tensor product of pure states only. Uses left-fold reduction.
"""
function _tensor_product_as_psi(states::Vector)
    result = states[1]::Vector{ComplexF64}
    
    for i in 2:length(states)
        result = tensor(result, states[i]::Vector{ComplexF64})
    end
    
    return result
end

"""
    _tensor_product_as_rho(states::Vector) -> Matrix{ComplexF64}

Internal: Tensor product with density matrix output.
Pure states are converted to |psi><psi| before tensoring.
"""
function _tensor_product_as_rho(states::Vector)
    # Convert first state to density matrix if needed
    result = _to_rho(states[1])
    
    for i in 2:length(states)
        rho_i = _to_rho(states[i])
        result = tensor(result, rho_i)
    end
    
    return result
end

"""
    _to_rho(state) -> Matrix{ComplexF64}

Internal: Convert state to density matrix. If already a matrix, return as-is.
If a pure state vector, return |psi><psi|.
"""
function _to_rho(state)
    if state isa Matrix{ComplexF64}
        return state
    elseif state isa Vector{ComplexF64}
        return state * state'  # |psi><psi|
    else
        error("Unknown state type: $(typeof(state))")
    end
end

# ==============================================================================
# MAXIMALLY MIXED STATE
# ==============================================================================
#
# OVERVIEW
# --------
# The maximally mixed state rho = I/d (where d = 2^N) represents complete
# uncertainty about the quantum state. It has maximum von Neumann entropy.
#
# IMPLEMENTATION
# --------------
# For memory efficiency, we return a Diagonal matrix which stores only O(2^N)
# elements instead of O(4^N) for a full matrix. Julia's LinearAlgebra handles
# Diagonal matrices efficiently in operations like tr(), *, and eigenvalues.
#
# For applications requiring a full Matrix, use Matrix(make_maximally_mixed(N)).
#
# ==============================================================================

"""
    make_maximally_mixed(N::Int) -> Diagonal{ComplexF64}

Create the maximally mixed state rho = I/2^N for an N-qubit system.

# Properties
- Trace = 1 (properly normalized)
- Purity = 1/2^N (minimum possible)
- von Neumann entropy = N * log(2) (maximum possible)
- All eigenvalues equal to 1/2^N

# Implementation
Returns a Diagonal matrix for O(2^N) memory efficiency instead of O(4^N).
Most LinearAlgebra operations (tr, *, eigen) work efficiently with Diagonal.

# Arguments
- `N::Int`: Number of qubits

# Returns
- `Diagonal{ComplexF64}`: Maximally mixed density matrix

# Examples
```julia
rho_mixed = make_maximally_mixed(3)  # 3-qubit maximally mixed state
tr(rho_mixed)       # 1.0
size(rho_mixed)     # (8, 8)

# Purity check
real(tr(rho_mixed * rho_mixed))  # 0.125 = 1/8 = 1/2^3

# Convert to full matrix if needed
rho_full = Matrix(rho_mixed)
```

# Use Cases
- Initial state for open system dynamics
- Reference state for entropy calculations
- Depolarizing channel fixed point
- Trace normalization target
"""
function make_maximally_mixed(N::Int)
    @assert N >= 1 "Need at least 1 qubit"
    
    dim = 1 << N  # 2^N
    
    # Return Diagonal matrix for O(2^N) memory efficiency
    # Each diagonal element is 1/dim for Tr(rho) = 1
    return Diagonal(fill(ComplexF64(1.0 / dim), dim))
end

"""
    tensor_product_ket(ψ_A, ψ_B, N_A, N_B) -> Vector{ComplexF64}

Compute ψ_A ⊗ ψ_B (tensor product of pure states) using bitwise indexing.
Result is 2^(N_A+N_B) pure state vector.

BIT CONVENTION: A occupies lower bits (qubits 1..N_A), B occupies upper bits (qubits N_A+1..N_A+N_B).
"""
function tensor_product_ket(ψ_A::Vector{ComplexF64}, ψ_B::Vector{ComplexF64}, N_A::Int, N_B::Int)
    dim_A = 1 << N_A
    dim_B = 1 << N_B
    dim_total = dim_A * dim_B
    
    ψ_total = zeros(ComplexF64, dim_total)
    
    @inbounds for i_A in 0:(dim_A-1)
        for i_B in 0:(dim_B-1)
            i_total = i_A + (i_B << N_A)
            ψ_total[i_total+1] = ψ_A[i_A+1] * ψ_B[i_B+1]
        end
    end
    
    return ψ_total
end

"""
    tensor_product_ket!(ψ_out, ψ_A, ψ_B, N_A, N_B) -> nothing

IN-PLACE version: Compute ψ_A ⊗ ψ_B, writing result to pre-allocated ψ_out.
Avoids allocation - use in hot loops.

# Arguments
- `ψ_out`: Pre-allocated output buffer of length 2^(N_A+N_B)
- `ψ_A`, `ψ_B`: Input state vectors
- `N_A`, `N_B`: Number of qubits in each subsystem
"""
function tensor_product_ket!(ψ_out::Vector{ComplexF64}, 
                             ψ_A::Vector{ComplexF64}, ψ_B::Vector{ComplexF64}, 
                             N_A::Int, N_B::Int)
    dim_A = 1 << N_A
    dim_B = 1 << N_B
    
    @inbounds for i_A in 0:(dim_A-1)
        for i_B in 0:(dim_B-1)
            i_total = i_A + (i_B << N_A)
            ψ_out[i_total+1] = ψ_A[i_A+1] * ψ_B[i_B+1]
        end
    end
    return nothing
end


"""
    tensor_product_rho(rho_a, rho_b) -> Matrix{ComplexF64}

Compute tensor product of two density matrices: rho_a ⊗ rho_b.
Uses LITTLE-ENDIAN convention: A occupies lower bits, B occupies upper bits.
"""
function tensor_product_rho(rho_a::Matrix{ComplexF64}, rho_b::Matrix{ComplexF64})
    # kron(A,B) puts A in upper bits, B in lower bits
    # Our convention: A in lower bits, B in upper bits
    # So we need kron(B, A) to match tensor_product_ket convention
    return kron(rho_b, rho_a)
end

# ==============================================================================
# PRODUCT STATE CONSTRUCTION
# ==============================================================================

"""
    make_product_ket(states::Vector{Symbol}) -> Vector{ComplexF64}

Create N-qubit product state |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ... ⊗ |ψₙ⟩.

# Arguments
- `states`: Vector of state symbols (:zero, :one, :plus, :minus) for each qubit

# Example
```julia
make_product_ket([:zero, :plus, :one])  # |0⟩ ⊗ |+⟩ ⊗ |1⟩
```
"""
function make_product_ket(states::Vector{Symbol})
    N = length(states)
    @assert N >= 1 "Need at least one qubit"
    
    ψ = make_ket(states[1])
    for k in 2:N
        ψ_k = make_ket(states[k])
        ψ = tensor_product_ket(ψ, ψ_k, k-1, 1)
    end
    return ψ
end

"""
    make_product_rho(states::Vector{Symbol}) -> Matrix{ComplexF64}

Create N-qubit product density matrix ρ₁ ⊗ ρ₂ ⊗ ... ⊗ ρₙ.
"""
function make_product_rho(states::Vector{Symbol})
    ψ = make_product_ket(states)
    return ψ * ψ'
end

"""
    make_initial_psi(N, init_bits::Vector{Int}) -> Vector{ComplexF64}

Create computational basis state |init_bits[1], init_bits[2], ..., init_bits[N]⟩.
init_bits[k] ∈ {0, 1} for each qubit.
Little-endian: qubit 1 = LSB.
"""
function make_initial_psi(N::Int, init_bits::Vector{Int}=Int[])
    dim = 1 << N
    ψ = zeros(ComplexF64, dim)
    
    if isempty(init_bits)
        # Default: all qubits in |0⟩ state
        ψ[1] = 1.0
    else
        @assert length(init_bits) == N "init_bits must have length N"
        idx = 0
        for k in 1:N
            if init_bits[k] == 1
                idx |= (1 << (k - 1))
            end
        end
        ψ[idx + 1] = 1.0
    end
    
    return ψ
end

"""
    make_initial_rho(N, init_bits::Vector{Int}) -> Matrix{ComplexF64}

Create density matrix |init_bits⟩⟨init_bits|.
"""
function make_initial_rho(N::Int, init_bits::Vector{Int}=Int[])
    ψ = make_initial_psi(N, init_bits)
    return ψ * ψ'
end

"""
    make_product_state(N, angles::Vector{Float64}) -> Vector{ComplexF64}

Create product state where each qubit is cos(θ)|0⟩ + sin(θ)|1⟩.
angles[k] = angle θ for qubit k (in radians).
"""
function make_product_state(N::Int, angles::Vector{Float64})
    @assert length(angles) == N "angles must have length N"
    
    # Build product state via tensor products
    θ1 = angles[1]
    psi_current = ComplexF64[cos(θ1), sin(θ1)]
    
    for i in 2:N
        θ = angles[i]
        psi_qubit = ComplexF64[cos(θ), sin(θ)]
        psi_current = tensor_product_ket(psi_current, psi_qubit, i-1, 1)
    end
    
    return psi_current
end

# ==============================================================================
# QUBIT RESET OPERATIONS
# ==============================================================================

"""
    reset_qubit_psi(ψ::Vector{ComplexF64}, k::Int, state::Symbol, N::Int) -> Vector{ComplexF64}

Reset qubit k to |state⟩ by tracing out qubit k and tensoring with fresh state.

# Arguments
- `ψ`: Input pure state (2^N dimensional)
- `k`: Qubit index to reset (1-indexed)
- `state`: Target state (:zero, :one, :plus, :minus)
- `N`: Total number of qubits

# Returns
New state vector with qubit k reset to |state⟩.
"""
function reset_qubit_psi(ψ::Vector{ComplexF64}, k::Int, state::Symbol, N::Int)
    @assert 1 <= k <= N "Qubit index k must be in 1..N"
    
    # Trace out qubit k to get reduced density matrix of remaining qubits
    trace_qubits = [k]
    ρ_rest = partial_trace(ψ, trace_qubits, N)
    
    # Get eigenstate with largest eigenvalue (for mixed state) or just proceed with projection
    # For pure state after trace, we collapse to most probable outcome
    probs = real.(diag(ρ_rest))
    probs ./= sum(probs)
    
    # Sample outcome
    r = rand()
    cumsum_p = 0.0
    outcome = length(probs)
    for i in 1:length(probs)
        cumsum_p += probs[i]
        if r <= cumsum_p
            outcome = i
            break
        end
    end
    
    # Create pure state for remaining qubits (collapsed to outcome)
    dim_rest = 1 << (N-1)
    ψ_rest = zeros(ComplexF64, dim_rest)
    ψ_rest[outcome] = 1.0
    
    # Create fresh qubit state
    ψ_fresh = make_ket(state)
    
    # Tensor in correct position
    if k == 1
        # Fresh qubit goes in lower bits
        return tensor_product_ket(ψ_fresh, ψ_rest, 1, N-1)
    elseif k == N
        # Fresh qubit goes in upper bits
        return tensor_product_ket(ψ_rest, ψ_fresh, N-1, 1)
    else
        # Qubit in middle: split into lower (1:k-1), fresh (k), upper (k+1:N)
        # The ψ_rest has qubits [1..k-1, k+1..N] mapped to [1..N-1]
        # We need to insert fresh qubit at position k
        dim_new = 1 << N
        n_lower = k - 1
        n_upper = N - k
        dim_lower = 1 << n_lower
        dim_upper = 1 << n_upper
        
        ψ_new = zeros(ComplexF64, dim_new)
        fresh_0 = ψ_fresh[1]  # Amplitude for |0⟩
        fresh_1 = ψ_fresh[2]  # Amplitude for |1⟩
        
        @inbounds for i_rest in 0:(length(ψ_rest)-1)
            amp = ψ_rest[i_rest + 1]
            if amp == 0.0 + 0.0im
                continue
            end
            
            # Split i_rest into lower and upper parts
            # Lower k-1 qubits come from bits 0:k-2 of i_rest
            # Upper N-k qubits come from bits k-1:N-2 of i_rest
            lower_bits = i_rest & ((1 << n_lower) - 1)
            upper_bits = i_rest >> n_lower
            
            # New index with fresh qubit = 0 at position k
            idx_0 = lower_bits | (0 << n_lower) | (upper_bits << k)
            # New index with fresh qubit = 1 at position k
            idx_1 = lower_bits | (1 << n_lower) | (upper_bits << k)
            
            ψ_new[idx_0 + 1] += amp * fresh_0
            ψ_new[idx_1 + 1] += amp * fresh_1
        end
        
        return ψ_new
    end
end

"""
    reset_qubit_rho(ρ::Matrix{ComplexF64}, k::Int, state::Symbol, N::Int) -> Matrix{ComplexF64}

Reset qubit k in density matrix to |state⟩⟨state|.
"""
function reset_qubit_rho(ρ::Matrix{ComplexF64}, k::Int, state::Symbol, N::Int)
    @assert 1 <= k <= N "Qubit index k must be in 1..N"
    
    # Trace out qubit k
    trace_qubits = [k]
    ρ_rest = partial_trace(ρ, trace_qubits, N)
    
    # Fresh qubit DM
    ρ_fresh = make_rho(state)
    
    # Tensor in correct position
    if k == 1
        return tensor_product_rho(ρ_fresh, ρ_rest)
    elseif k == N
        return tensor_product_rho(ρ_rest, ρ_fresh)
    else
        error("reset_qubit_rho with k not at boundary not yet implemented.")
    end
end

"""
    reset_qubits_psi(ψ::Vector{ComplexF64}, qubits::Vector{Int}, states::Vector{Symbol}, N::Int) -> Vector{ComplexF64}

Reset multiple qubits to specified states.

# Arguments
- `ψ`: Input pure state
- `qubits`: Qubit indices to reset (must be contiguous from 1 or to N)
- `states`: Target states for each qubit
- `N`: Total number of qubits

For QRC: Typically reset qubits 1:L (input rail) to encoded states.
"""
function reset_qubits_psi(ψ::Vector{ComplexF64}, qubits::Vector{Int}, states::Vector{Symbol}, N::Int)
    @assert length(qubits) == length(states) "qubits and states must have same length"
    L = length(qubits)
    
    # Check if qubits are contiguous from start
    if sort(qubits) == collect(1:L)
        # Trace out qubits 1:L, keep L+1:N
        trace_qubits = collect(1:L)
        ρ_rest = partial_trace(ψ, trace_qubits, N)
        
        # Collapse to outcome
        probs = real.(diag(ρ_rest))
        probs ./= sum(probs)
        r = rand()
        cumsum_p = 0.0
        outcome = length(probs)
        for i in 1:length(probs)
            cumsum_p += probs[i]
            if r <= cumsum_p
                outcome = i
                break
            end
        end
        
        ψ_rest = zeros(ComplexF64, 1 << (N-L))
        ψ_rest[outcome] = 1.0
        
        # Create fresh qubits product state
        ψ_fresh = make_product_ket(states)
        
        return tensor_product_ket(ψ_fresh, ψ_rest, L, N-L)
        
    elseif sort(qubits) == collect((N-L+1):N)
        # Trace out qubits at end
        trace_qubits = collect((N-L+1):N)
        ρ_rest = partial_trace(ψ, trace_qubits, N)
        
        probs = real.(diag(ρ_rest))
        probs ./= sum(probs)
        r = rand()
        cumsum_p = 0.0
        outcome = length(probs)
        for i in 1:length(probs)
            cumsum_p += probs[i]
            if r <= cumsum_p
                outcome = i
                break
            end
        end
        
        ψ_rest = zeros(ComplexF64, 1 << (N-L))
        ψ_rest[outcome] = 1.0
        
        ψ_fresh = make_product_ket(states)
        
        return tensor_product_ket(ψ_rest, ψ_fresh, N-L, L)
    else
        error("reset_qubits_psi only supports contiguous qubit ranges from start or end")
    end
end

"""
    reset_qubits_rho(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, states::Vector{Symbol}, N::Int) -> Matrix{ComplexF64}

Reset multiple qubits in density matrix to specified states.
"""
function reset_qubits_rho(ρ::Matrix{ComplexF64}, qubits::Vector{Int}, states::Vector{Symbol}, N::Int)
    @assert length(qubits) == length(states) "qubits and states must have same length"
    L = length(qubits)
    
    if sort(qubits) == collect(1:L)
        # Trace out qubits 1:L
        ρ_rest = partial_trace(ρ, collect(1:L), N)
        ρ_fresh = make_product_rho(states)
        return tensor_product_rho(ρ_fresh, ρ_rest)
        
    elseif sort(qubits) == collect((N-L+1):N)
        # Trace out qubits at end
        ρ_rest = partial_trace(ρ, collect((N-L+1):N), N)
        ρ_fresh = make_product_rho(states)
        return tensor_product_rho(ρ_rest, ρ_fresh)
    else
        error("reset_qubits_rho only supports contiguous qubit ranges from start or end")
    end
end

# ==============================================================================
# UNIFIED RESET INTERFACE
# ==============================================================================
#
# OVERVIEW
# --------
# The reset() function provides a unified interface for qubit reset operations.
# It dispatches on state type (Vector for psi, Matrix for rho) and accepts
# either single qubit or list of qubits.
#
# OPERATION
# ---------
# Reset traces out specified qubit(s), then tensors fresh state(s) back:
#   rho_new = Tr_k(rho) (x) |fresh><fresh|
#   psi_new = collapse(Tr_k(|psi><psi|)) (x) |fresh>
#
# ==============================================================================

"""
    reset(psi::Vector{ComplexF64}, qubits, states, N::Int) -> Vector{ComplexF64}
    reset(rho::Matrix{ComplexF64}, qubits, states, N::Int) -> Matrix{ComplexF64}

Unified qubit reset: trace out specified qubit(s) and tensor in fresh state(s).

Automatically dispatches on state type (psi/rho) and handles single or multiple qubits.

# Arguments
- `psi` or `rho`: Input quantum state (pure or density matrix)
- `qubits`: Int or Vector{Int} of qubit indices to reset (1-indexed)
- `states`: Symbol or Vector{Symbol} of target states (:zero, :one, :plus, :minus)
- `N`: Total number of qubits

# Result
Replaces specified qubits with fresh state(s) via:
  Tr_{qubits}(state) (x) |fresh><fresh| (or |fresh> for pure states)

# Examples
```julia
# Single qubit reset
psi_new = reset(psi, 2, :zero, 4)      # Reset qubit 2 to |0>
rho_new = reset(rho, 3, :plus, 4)      # Reset qubit 3 to |+><+|

# Multiple qubit reset (must be contiguous from start or end)
psi_new = reset(psi, [1,2], [:zero, :one], 4)  # Reset qubits 1,2 to |0>,|1>
rho_new = reset(rho, [1,2], [:zero, :one], 4)  # Same for density matrix

# QRC usage: reset input rail each timestep
psi_t = reset(psi_t, 1:L, input_bits, N)
```

See also: [`reset_qubit_psi`](@ref), [`reset_qubits_rho`](@ref)
"""
function reset(psi::Vector{ComplexF64}, qubit::Int, state::Symbol, N::Int)
    return reset_qubit_psi(psi, qubit, state, N)
end

function reset(psi::Vector{ComplexF64}, qubits::Vector{Int}, states::Vector{Symbol}, N::Int)
    return reset_qubits_psi(psi, qubits, states, N)
end

function reset(psi::Vector{ComplexF64}, qubits::UnitRange{Int}, states::Vector{Symbol}, N::Int)
    return reset_qubits_psi(psi, collect(qubits), states, N)
end

function reset(rho::Matrix{ComplexF64}, qubit::Int, state::Symbol, N::Int)
    return reset_qubit_rho(rho, qubit, state, N)
end

function reset(rho::Matrix{ComplexF64}, qubits::Vector{Int}, states::Vector{Symbol}, N::Int)
    return reset_qubits_rho(rho, qubits, states, N)
end

function reset(rho::Matrix{ComplexF64}, qubits::UnitRange{Int}, states::Vector{Symbol}, N::Int)
    return reset_qubits_rho(rho, collect(qubits), states, N)
end

end # module


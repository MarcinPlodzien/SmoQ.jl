#!/usr/bin/env julia
#=
================================================================================
    demo_quantum_autoencoder.jl - Quantum Autoencoder for State Compression
================================================================================

DESCRIPTION:
    Demonstrates variational quantum autoencoder (QAE) training for compressing
    quantum states. The autoencoder learns to encode an N-qubit input state into
    a (N-k)-qubit latent representation by projecting k "trash" qubits to |0⟩,
    then decodes back to the full Hilbert space with high fidelity.
    
QUANTUM AUTOENCODER ARCHITECTURE:
    |ψ_in⟩ → [Encoder U(θ)] → [Measure+Reset k qubits] → [Decoder U†(θ)] → |ψ_out⟩
    
    - Encoder: Parameterized variational circuit U(θ)
    - Compression: Trace out (DM) or measure+reset (MCWF) the k trash qubits
    - Decoder: Inverse circuit U†(θ) applied to the reduced + reset state
    - Goal: Maximize fidelity F = ⟨ψ_in|ρ_out|ψ_in⟩ → 1

LOSS FUNCTION:
    L = w_fid*(1-F) + w_svn*S_vN + w_ham*H
    
    - F: Fidelity between input and reconstructed state
    - S_vN: von Neumann entropy of trash subsystem (encourages pure trash)
    - H: Hamming distance of measurement outcomes (MCWF only)

PARAMETER SWEEPS:
    - N: Number of qubits [4, 6, 8, ...]
    - L: Number of variational layers [2, 4, 6, ...]
    - k: Number of trash qubits [1, 2, N/2]
    - Ansatz: Circuit topology [:brick, :chain, :ring]
    - Representation: [:dm, :mcwf] (deterministic vs stochastic)

ANSATZ TYPES:
    :brick     - Alternating pair entanglement (good for 1D chains)
    :chain     - All nearest-neighbor CZ gates every layer
    :ring      - Chain + periodic boundary (CZ between qubit N and 1)
    :full      - All-to-all CZ connectivity (for long-range correlations)
    :butterfly - Logarithmic depth FFT-like pattern (strides 1,2,4,...)
    :hea       - Hardware Efficient Ansatz (standard VQE style)
    :xxz       - Heisenberg XXZ-inspired (CZ-RZ-RZ-CZ sandwich)

INPUT STATE:
    Ground state of Heisenberg XXZ model with transverse field:
    H = Jx*Σ XᵢXⱼ + Jy*Σ YᵢYⱼ + Jz*Σ ZᵢZⱼ + hx*Σ Xᵢ

OPTIMIZER:
    Adam + SPSA hybrid (Adam momentum with SPSA gradient estimates)

OUTPUT:
    - Fidelity training curves for each configuration
    - Summary tables comparing final fidelity across sweeps
    - PNG plots saved on-the-fly to demo_quantum_autoencoder/ folder

THEORY - SCHMIDT RANK AND COMPRESSIBILITY:
    
    Schmidt Decomposition:
    ----------------------
    Any pure bipartite state |ψ⟩_AB can be written as:
        |ψ⟩ = Σᵢ λᵢ |uᵢ⟩_A ⊗ |vᵢ⟩_B
    
    where {|uᵢ⟩} and {|vᵢ⟩} are orthonormal bases for subsystems A and B,
    and λᵢ ≥ 0 are the Schmidt coefficients (Σᵢ λᵢ² = 1).
    
    Schmidt Rank:
    -------------
    The number of non-zero Schmidt coefficients = Schmidt rank r.
    - r = 1: Product state (no entanglement)
    - r > 1: Entangled state
    - r = min(dim_A, dim_B): Maximally entangled in terms of rank
    
    Compressibility via Schmidt Rank:
    ---------------------------------
    For a quantum autoencoder with k trash qubits:
    - Latent space dimension: 2^(N-k)
    - Perfect compression (F=1) possible iff Schmidt rank ≤ 2^(N-k)
    
    Examples:
    - GHZ = (|00...0⟩ + |11...1⟩)/√2
      Schmidt rank = 2 (only 2 terms!) → compressible to k=N-1 (2D latent)
      Maximally entangled in CORRELATIONS, but minimal in SUPPORT.
    
    - Dicke(N,k) = Symmetric superposition of all states with k ones
      Schmidt rank = min(2^(floor(N/2)), binomial(N,k))
      For N=6, k=3: rank = min(8, 20) = 8 → needs at least 3 latent qubits
      High entanglement ENTROPY due to large support.
    
    - W state = (|10...0⟩ + |01...0⟩ + ... + |00...1⟩)/√N
      Schmidt rank = 2 for any bipartition (one term has 1, others have 0)
      Compressible, but entanglement more robust to qubit loss than GHZ.
    
    Key Insight:
    ------------
    "Maximally entangled" ≠ "hardest to compress"!
    - GHZ has maximal correlations but lives in a 2D subspace
    - Dicke states have lower per-pair entanglement but span many dimensions
    - Compressibility depends on EFFECTIVE DIMENSION, not correlation strength

================================================================================
=#

using LinearAlgebra
using Random
using Printf
using Plots
using Statistics
using Optimisers

const OUTPUT_DIR = joinpath(@__DIR__, "demo_quantum_autoencoder")
mkpath(OUTPUT_DIR)

const UTILS_CPU = joinpath(@__DIR__, "..", "utils", "cpu")
include(joinpath(UTILS_CPU, "cpuQuantumChannelGates.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateMeasurements.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateLanczos.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateCharacteristic.jl"))
include(joinpath(UTILS_CPU, "cpuVariationalQuantumCircuitCostFunctions.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateTomography.jl"))

using .CPUQuantumChannelGates
using .CPUQuantumStatePartialTrace: partial_trace
using .CPUQuantumStatePreparation: normalize_state!, make_product_rho
using .CPUQuantumStateMeasurements: projective_measurement!
using .CPUQuantumStateLanczos: ground_state_xxz
using .CPUQuantumStateCharacteristic: von_neumann_entropy
using .CPUVariationalQuantumCircuitCostFunctions: fidelity
using .CPUQuantumStateTomography: reconstruct_density_matrix

println("=" ^ 70)
println("  QUANTUM AUTOENCODER - DM vs MCWF Comparison")
println("=" ^ 70)

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
#
# This section contains all user-configurable parameters for the autoencoder.
# Modify these values to customize the experiment.
#
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM SIZE & TRAINING
# ─────────────────────────────────────────────────────────────────────────────
# N_VALUES: List of total qubit counts to simulate.
#   - N qubits → Hilbert space dimension 2^N
#   - Larger N = exponentially more expensive (DM: O(4^N), MCWF: O(2^N))
#   - Recommended: start with N=4-6 for testing, N=8-10 for production
const N_VALUES = [    
    6,
]

# L_VALUES: Number of variational ansatz layers.
#   - More layers = more expressive circuit = better fidelity (usually)
#   - More layers = more parameters = slower training & potential overfitting
#   - Rule of thumb: L ≈ N gives sufficient expressibility for most states
const L_VALUES = [
    1,
    2,
    3,
]

# N_EPOCHS: Total training epochs per configuration.
#   - Each epoch = one gradient step (SPSA uses 2 circuit evaluations)
#   - 500-1000 epochs typical for convergence with SPSA
const N_EPOCHS = 500

# LR: Learning rate for Adam optimizer.
#   - 0.01-0.05 works well for most cases
#   - Decrease if training is unstable (oscillating fidelity)
const LR = 0.01

# ─────────────────────────────────────────────────────────────────────────────
# COMPRESSION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# K_VALUES: Number of "trash" qubits to discard during compression.
#   - k trash qubits → latent dimension = 2^(N-k)
#   - :auto computes [1, 2, N÷2, N-3, N-1] automatically
#   - Larger k = more compression = harder to achieve high fidelity
#   - k = 1: mild compression (should always work well)
#   - k = N÷2: 50% compression (challenging for entangled states)
#   - k = N-1: extreme compression to 2D (only works for special states)
const K_VALUES = :auto_1_half_n3  # [1, N÷2, N-3] computed per N

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION REPRESENTATIONS
# ─────────────────────────────────────────────────────────────────────────────
# REPRESENTATIONS: How to simulate the quantum channel.
#   :dm   - Density matrix: deterministic, exact, O(4^N) memory
#           Trace out trash qubits analytically (partial trace)
#   :mcwf - Monte Carlo wave function: stochastic, O(2^N) memory
#           Project trash qubits via measurement + post-selection
const REPRESENTATIONS = [
    :dm,
    :mcwf,
]

# MCWF_TRAJ_VALUES: Number of trajectories for MCWF averaging.
#   - More trajectories = lower variance = slower simulation
#   - 100-1000 typical; use 1000+ for publication-quality results
const MCWF_TRAJ_VALUES = [
    #20,
    # 50,
    1000
]

# ─────────────────────────────────────────────────────────────────────────────
# VARIATIONAL ANSATZ TOPOLOGY
# ─────────────────────────────────────────────────────────────────────────────
# ANSATZ_TYPES: Circuit topology for variational layers.
#   Each layer applies: [single-qubit rotations] + [entangling pattern]
#
#   :brick     - Alternating odd/even pairs: (1-2)(3-4)... then (2-3)(4-5)...
#                Good for 1D nearest-neighbor architectures
#   :chain     - Sequential pairs: (1-2), (2-3), (3-4), ...
#                Simple, hardware-friendly
#   :ring      - Chain + periodic boundary: adds (N-1, N) entangler
#                Better for translation-invariant states
#   :full      - All-to-all CZ connectivity
#                Most expressive, but O(N²) gates per layer
#   :butterfly - Logarithmic depth FFT-like: strides 1, 2, 4, 8, ...
#                Efficient long-range correlations in O(log N) depth
#   :hea       - Hardware Efficient Ansatz (Ry-Rz per qubit + CZ ladder)
#                Standard VQE-style parameterization
#   :xxz       - Heisenberg-inspired: CZ-RZ-RZ-CZ sandwich
#                Good for Heisenberg ground states
const ANSATZ_TYPES = [
    :full,      # All-to-all connectivity
    :hea,        # Hardware Efficient Ansatz
    #:brick,
    #:chain,
    #:ring,
    #:butterfly,  # Log-depth FFT-like pattern
    #:xxz,        # Heisenberg XXZ-inspired
]

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-QUBIT ROTATION GATES
# ─────────────────────────────────────────────────────────────────────────────
# ROTATION_GATES: Which rotation gates to apply per qubit per layer.
#   - [:ry, :rz]       - Minimal (2 params/qubit): sufficient for universal control
#   - [:rx, :ry, :rz]  - Full SU(2) (3 params/qubit): redundant but sometimes faster
#
# Theory: Ry-Rz is sufficient because:
#   - Ry rotates around Y axis (θ → any latitude on Bloch sphere)
#   - Rz rotates around Z axis (φ → any longitude)
#   - Together they can reach any point on Bloch sphere
#   - The extra Rx in full SU(2) only adds a global phase (irrelevant)
#
# Practical: 2 rotations = 33% fewer parameters = faster training
const ROTATION_GATES = [:ry, :rz]  # Minimal universal set (recommended)
# const ROTATION_GATES = [:rx, :ry, :rz]  # Full SU(2) (more parameters)

# ─────────────────────────────────────────────────────────────────────────────
# ENTANGLING GATES
# ─────────────────────────────────────────────────────────────────────────────
# ENTANGLER_GATES: List of two-qubit gates to loop over for comparison.
#   :cz  - Controlled-Z: symmetric, diagonal, no swap of states
#          |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|10⟩, |11⟩→-|11⟩
#          Native on superconducting (Google, IBM) and ion trap devices
#   :cx  - Controlled-X (CNOT): asymmetric control→target
#          |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
#          Flips target when control is |1⟩
#
# Both are universal for 2-qubit operations. CZ is often preferred because:
#   - Symmetric (no control/target distinction)
#   - Easier to compile on some hardware
const ENTANGLER_GATES = [
    :cz,
    #:cx,
]

# ─────────────────────────────────────────────────────────────────────────────
# HAMILTONIAN CONFIGURATIONS (for ground state preparation)
# ─────────────────────────────────────────────────────────────────────────────
# HAMILTONIAN_CONFIGS: List of Hamiltonians to loop over.
# Each entry: (name, Jxx, Jyy, Jzz, hx, hy, hz)
#
# General Hamiltonian: H = Jxx∑XX + Jyy∑YY + Jzz∑ZZ + hx∑X + hy∑Y + hz∑Z
# NOTE: Signs are explicit in parameters. Use negative J for antiferromagnetic.
#
# Common models:
#   ("XXZ",   -1, -1, -Δ, h, 0, 0)  - Heisenberg XXZ antiferromagnetic
#   ("TFIM",   0,  0, -1, h, 0, 0)  - Transverse Field Ising (antiferromagnetic)
#   ("XY",    -1, -1,  0, h, 0, 0)  - XY model antiferromagnetic
const HAMILTONIAN_CONFIGS = [
    # (name,    Jxx,  Jyy,  Jzz,  hx,  hy,  hz)
    ("XXZ",   -1.0, -1.0, -0.5, -1.0, 0.0, 0.0),  # XXZ antiferromagnetic + transverse field
    ("TFIM",   0.0,  0.0, -1.0, -1.0, 0.0, 0.0),  # Transverse Field Ising
    #("XXX",  -1.0, -1.0, -1.0,  0.0, 0.0, 0.0),  # Isotropic Heisenberg AFM
    #("XY",   -1.0, -1.0,  0.0, -0.5, 0.0, 0.0),  # XY + field
]

# ─────────────────────────────────────────────────────────────────────────────
# INPUT STATES TO COMPRESS
# ─────────────────────────────────────────────────────────────────────────────
# INPUT_STATES: Quantum states to test autoencoder compression.
#
#   :GHZ           - GHZ state: (|00...0⟩ + |11...1⟩)/√2
#                    Maximally entangled, 1 ebit across any bipartition
#                    Should compress well with k=1 (only 2D essential subspace)
#
#   :W             - W state: (|10...0⟩ + |01...0⟩ + ... + |00...1⟩)/√N
#                    Symmetric, robust entanglement (survives qubit loss)
#                    Harder than GHZ; requires more latent dimensions
#
#   :Dicke         - Dicke state with N/2 excitations
#                    Symmetric superposition of all states with k ones
#                    High entanglement, challenging compression
#
#   :Hamiltonian_GS - Ground state of Hamiltonian from HAMILTONIAN_CONFIGS
#                     Loops over all entries in HAMILTONIAN_CONFIGS automatically
#                     Replaces the old :XXZ_GS and :TFIM_GS options
#
#   :Product       - Product state |+⟩⊗N = (|0⟩+|1⟩)⊗N / √(2^N)
#                    Separable baseline; should compress perfectly with k=N-1
#
#   :Custom        - User-defined state (set CUSTOM_STATE below)
const INPUT_STATES = [
    :Hamiltonian_GS,   # Ground states from HAMILTONIAN_CONFIGS (loops over all)
    #:GHZ,              # GHZ state: (|000...⟩ + |111...⟩)/√2
    :W,                # W state: (|100...⟩ + |010...⟩ + ...)/√N
    #:Dicke,            # Dicke state with N/2 excitations    
    #:Product,          # Product state |+⟩⊗N (separable, baseline)
    #:Custom,          # User-defined custom state (see CUSTOM_STATE below)
]

# Custom state - define your own state vector here (must match dim = 2^N)
# Example for N=4: CUSTOM_STATE = normalize(randn(ComplexF64, 16))
CUSTOM_STATE = nothing  # Set to a Vector{ComplexF64} to use :Custom
CUSTOM_STATE_NAME = "MyState"  # Name for titles/filenames

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER_TYPES: Parameter update strategies to compare.
#   :SPSA      - Pure SPSA with built-in step size scheduling (a_k, c_k decay)
#                θ ← θ - a_k * g where a_k = a/(A+k)^α (Spall 1998)
#   :SPSA_Adam - SPSA for gradient estimation + Adam for parameter updates
#                Best of both: noisy gradient estimation + adaptive momentum
const OPTIMIZER_TYPES = [
    :SPSA_Adam,  # SPSA gradient + Adam updates
]

# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT_CONFIGS: Tuples of (w_fid, w_svn, w_ham) defining loss composition.
#   Loss = w_fid × (1-F) + w_svn × S_vN + w_ham × D_H
#
#   w_fid: Weight on infidelity (1-F). Primary objective.
#   w_svn: Weight on von Neumann entropy of trash subsystem S_vN.
#          Encourages disentangling trash from latent qubits.
#   w_ham: Weight on Hamming distance of trash measurements D_H.
#          Encourages trash to project onto |0...0⟩ (MCWF-friendly).
#
# Different configs explore: fidelity-only, Hamming-only, or balanced losses.
const WEIGHT_CONFIGS = [
        (1.0, 0.0, 0.0),             # Fidelity only - primary metric
        (0.45, 0.1, 0.45),           #  
        (0.45, 0.45, 0.1),           # 
]

# Global weights (will be updated in loop)
W_FID, W_SVN, W_HAM = 1.0, 0.0, 0.0

println("\nConfig: N=$N_VALUES, L=$L_VALUES, $N_EPOCHS epochs")
println("k values: per-N [1, 2, N÷2]")
println("Input states: $INPUT_STATES")
println("Gradient: $(OPTIMIZER_TYPES[1]) | Representations: $REPRESENTATIONS (MCWF traj: $MCWF_TRAJ_VALUES)")
println("Hamiltonians: $(length(HAMILTONIAN_CONFIGS)) configs: $([h[1] for h in HAMILTONIAN_CONFIGS])")
println("Loss weights: w_fid=$W_FID, w_svn=$W_SVN, w_ham=$W_HAM")

# ==============================================================================
# ANSATZ BUILDERS
# ==============================================================================
#
# Three ansatz types for the encoder circuit:
#
# 1. :brick (Brick-layer / Alternating pairs)
#    Layer structure: [RX-RY-RZ on all qubits] → [CZ on odd pairs: (1,2),(3,4),...] 
#                                              → [CZ on even pairs: (2,3),(4,5),...]
#    Alternates entanglement pattern each layer. Good for 1D chain systems.
#
# 2. :chain (Linear nearest-neighbor)
#    Layer structure: [RX-RY-RZ on all qubits] → [CZ on ALL adjacent pairs: (1,2),(2,3),(3,4),...]
#    Every layer has same entanglement pattern. Maximum local connectivity.
#
# 3. :ring (Periodic boundary / Ring topology)  
#    Layer structure: [RX-RY-RZ on all qubits] → [CZ on ALL adjacent pairs + (N,1) closing the ring]
#    Like chain but with periodic boundary conditions. Good for systems with PBC.
#
# All ansatzes use: RX(θ), RY(θ), RZ(θ) for full SU(2) expressibility
#                   CZ for entanglement (symmetric, no parameter)
#
# ==============================================================================

function build_encoder_spec(N::Int, n_layers::Int, ansatz::Symbol, entangler_gate::Symbol=:cz)
    gates = Tuple{Symbol, Any, Int}[]
    pidx = 0
    
    for layer in 1:n_layers
        # Single-qubit rotations: use configurable ROTATION_GATES
        for q in 1:N
            for rot in ROTATION_GATES
                pidx += 1
                push!(gates, (rot, q, pidx))
            end
        end
        
        # Entanglement pattern depends on ansatz type
        if ansatz == :brick
            # Alternating: odd layers → pairs (1,2),(3,4),...
            #              even layers → pairs (2,3),(4,5),...
            if layer % 2 == 1
                for q in 1:2:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
            else
                for q in 2:2:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
            end
        elseif ansatz == :chain
            # All nearest-neighbor pairs every layer
            for q in 1:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
        elseif ansatz == :ring
            # All nearest-neighbor pairs + periodic (N,1)
            for q in 1:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
            push!(gates, (entangler_gate, (N, 1), 0))  # Close the ring
        elseif ansatz == :full
            # All-to-all connectivity - good for long-range correlations
            for i in 1:N, j in (i+1):N
                push!(gates, (entangler_gate, (i, j), 0))
            end
        elseif ansatz == :butterfly
            # Butterfly/FFT pattern - logarithmic depth connectivity
            # Strides: 1, 2, 4, ... up to N/2
            stride = 1 << ((layer - 1) % Int(ceil(log2(N))))  # Cycle through strides
            for i in 1:N
                j = i + stride
                if j <= N; push!(gates, (entangler_gate, (i, j), 0)); end
            end
        elseif ansatz == :hea
            # Hardware Efficient Ansatz (standard VQE style)
            # RY-RZ already added above, just add linear entangler
            for q in 1:(N-1); push!(gates, (entangler_gate, (q, q+1), 0)); end
        elseif ansatz == :xxz
            # Heisenberg XXZ-inspired: mimic exp(-iθ ZZ) with entangler sandwich
            # Pattern: CZ - RZ(θ) - RZ(θ) - CZ approximates exp(-iθ ZᵢZⱼ)
            for q in 1:(N-1)
                push!(gates, (entangler_gate, (q, q+1), 0))
                pidx += 1; push!(gates, (:rz, q, pidx))
                pidx += 1; push!(gates, (:rz, q+1, pidx))
                push!(gates, (entangler_gate, (q, q+1), 0))
            end
        else
            error("Unknown ansatz: $ansatz. Use :brick, :chain, :ring, :full, :butterfly, :hea, :xxz")
        end
    end
    
    # Final rotation layer
    for q in 1:N
        pidx += 1; push!(gates, (:ry, q, pidx))
    end
    return pidx, gates
end

# ==============================================================================
# PURE STATE (MCWF) OPERATIONS
# ==============================================================================

function apply_encoder_psi!(ψ, θ, gates, N)
    for (g, t, p) in gates
        g == :rx && apply_rx_psi!(ψ, t, θ[p], N)
        g == :ry && apply_ry_psi!(ψ, t, θ[p], N)
        g == :rz && apply_rz_psi!(ψ, t, θ[p], N)
        g == :cz && apply_cz_psi!(ψ, t[1], t[2], N)
    end
end

function apply_decoder_psi!(ψ, θ, gates, N)
    for i in length(gates):-1:1
        g, t, p = gates[i]
        g == :rx && apply_rx_psi!(ψ, t, -θ[p], N)
        g == :ry && apply_ry_psi!(ψ, t, -θ[p], N)
        g == :rz && apply_rz_psi!(ψ, t, -θ[p], N)
        g == :cz && apply_cz_psi!(ψ, t[1], t[2], N)
    end
end

# ==============================================================================
# MCWF AUTOENCODER: MEASURE + RESET
# ==============================================================================
#
# MCWF (pure state) implementation of quantum autoencoder reset:
#
#   1. Projective measurement on trash qubits → get bitstring outcomes
#   2. Apply X gate to any qubit measured as |1⟩ → force to |0⟩
#   3. Decode from |latent⟩ ⊗ |00...0⟩
#
# Properties:
#   • PHYSICAL: Valid quantum operation (measurement + unitary)
#   • STOCHASTIC: Different outcomes per trajectory → need averaging
#   • Gives HAMMING DISTANCE: sum(outcomes) = how many 1s measured
#   • Converges to DM as n_traj → ∞
#
function autoencoder_mcwf!(ψ, θ, gates, trash, N)
    # 1. Encode: compress N-qubit state into (N-k) latent + k trash qubits
    apply_encoder_psi!(ψ, θ, gates, N)
    
    # 2. Projective measurement on trash qubits in Z-basis
    #    Returns bitstring outcomes[i] ∈ {0,1} for each trash qubit
    #    Hamming distance = sum(outcomes) = popcount of measured bitstring
    outcomes, _ = projective_measurement!(ψ, trash, :z, N)
    
    # 3. Reset trash qubits to |0⟩ by applying X gate if measured |1⟩
    for (i, q) in enumerate(trash)
        outcomes[i] == 1 && apply_pauli_x_psi!(ψ, q, N)
    end
    
    # 4. Decode: reconstruct from latent representation
    apply_decoder_psi!(ψ, θ, gates, N)
    return outcomes  # Return measured bitstring for Hamming distance
end

# ==============================================================================
# DENSITY MATRIX OPERATIONS
# ==============================================================================

function apply_encoder_rho!(ρ, θ, gates, N)
    for (g, t, p) in gates
        g == :rx && apply_rx_rho!(ρ, t, θ[p], N)
        g == :ry && apply_ry_rho!(ρ, t, θ[p], N)
        g == :rz && apply_rz_rho!(ρ, t, θ[p], N)
        g == :cz && apply_cz_rho!(ρ, t[1], t[2], N)
    end
end

function apply_decoder_rho!(ρ, θ, gates, N)
    for i in length(gates):-1:1
        g, t, p = gates[i]
        g == :rx && apply_rx_rho!(ρ, t, -θ[p], N)
        g == :ry && apply_ry_rho!(ρ, t, -θ[p], N)
        g == :rz && apply_rz_rho!(ρ, t, -θ[p], N)
        g == :cz && apply_cz_rho!(ρ, t[1], t[2], N)
    end
end

function autoencoder_dm!(ρ, θ, gates, trash, N)
    # Encoder
    apply_encoder_rho!(ρ, θ, gates, N)
    
    # Trace out trash qubits (measure + reset = partial trace + tensor |0⟩⟨0|)
    latent = setdiff(1:N, trash)
    ρ_latent = partial_trace(ρ, trash, N)
    
    # Tensor product with |0...0⟩⟨0...0| for trash qubits
    k = length(trash)
    dim_latent = 1 << length(latent)
    dim_trash = 1 << k
    dim_total = 1 << N
    
    # Rebuild full density matrix: ρ_latent ⊗ |0⟩⟨0|_trash
    ρ .= zero(ComplexF64)
    
    # Map: latent qubits stay, trash qubits are |0⟩
    # trash qubits are at positions (N-k+1):N, so their contribution is 0 for |0⟩
    for i_lat in 0:(dim_latent-1)
        for j_lat in 0:(dim_latent-1)
            # Expand latent indices to full system (trash bits = 0)
            i_full = 0
            j_full = 0
            lat_bit = 0
            for q in 1:N
                if q in latent
                    i_full |= ((i_lat >> lat_bit) & 1) << (q-1)
                    j_full |= ((j_lat >> lat_bit) & 1) << (q-1)
                    lat_bit += 1
                end
                # trash bits remain 0
            end
            ρ[i_full+1, j_full+1] = ρ_latent[i_lat+1, j_lat+1]
        end
    end
    
    # Decoder
    apply_decoder_rho!(ρ, θ, gates, N)
end

# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

# Fidelity between density matrices (matrix-free for Enzyme compatibility)
# Computes tr(ρ1 * ρ2) without BLAS gemm
function dm_fidelity(ρ1::Matrix{ComplexF64}, ρ2::Matrix{ComplexF64})
    dim = size(ρ1, 1)
    fid = 0.0
    # tr(ρ1 * ρ2) = Σᵢⱼ ρ1[i,j] * ρ2[j,i]
    @inbounds for i in 1:dim
        for j in 1:dim
            fid += real(ρ1[i, j] * ρ2[j, i])
        end
    end
    return fid
end

# S_vN on LATENT subsystem (trace out trash) - measures entanglement between trash and latent
# For pure state: S(latent) = S(trash) by Schmidt decomposition
# Goal: minimize this → trash disentangled from latent → good compression
function latent_entropy_psi(ψ, trash, N)
    ρ_latent = partial_trace(ψ, trash, N)  # Trace out trash, keep latent
    return von_neumann_entropy(ρ_latent)
end

function latent_entropy_rho(ρ, trash, N)
    ρ_latent = partial_trace(ρ, trash, N)  # Trace out trash, keep latent
    return von_neumann_entropy(ρ_latent)
end

# Expected Hamming distance from trash density matrix
# E[H] = Σᵢ pᵢ × popcount(i) where pᵢ = ⟨i|ρ_trash|i⟩
function expected_hamming_dm(ρ, trash, N)
    latent = setdiff(1:N, trash)
    ρ_trash = partial_trace(ρ, latent, N)
    k = length(trash)
    H = 0.0
    for i in 0:(2^k - 1)
        p_i = real(ρ_trash[i+1, i+1])  # Probability of outcome i
        H += p_i * count_ones(i)       # popcount = Hamming distance from |0...0⟩
    end
    return H / k  # Normalize by number of trash qubits
end

function evaluate_dm(ρ_input, θ, gates, trash, N)
    # Full autoencoder pass for fidelity: encode → reset → decode
    ρ_out = copy(ρ_input)
    autoencoder_dm!(ρ_out, θ, gates, trash, N)
    F = dm_fidelity(ρ_input, ρ_out)
    
    # Only compute S and H if their weights are non-zero (avoids partial_trace for Enzyme)
    if W_SVN > 0.0 || W_HAM > 0.0
        ρ_enc = copy(ρ_input)
        apply_encoder_rho!(ρ_enc, θ, gates, N)
        S = W_SVN > 0.0 ? latent_entropy_rho(ρ_enc, trash, N) : 0.0
        H = W_HAM > 0.0 ? expected_hamming_dm(ρ_enc, trash, N) : 0.0
    else
        S = 0.0
        H = 0.0
    end
    
    # Loss = w_fid*(1-F) + w_svn*S + w_ham*H
    L = W_FID*(1-F) + W_SVN*S + W_HAM*H
    return L, F, S, H
end

function evaluate_mcwf(ψ_input, θ, gates, trash, N)
    # Encode: compress N-qubit state
    ψ_enc = copy(ψ_input)
    apply_encoder_psi!(ψ_enc, θ, gates, N)
    
    # S_vN on ENCODED state's LATENT subsystem (before reset/decode)
    # This measures how well encoder disentangles trash from latent
    S = latent_entropy_psi(ψ_enc, trash, N)
    
    # Full autoencoder pass for fidelity and Hamming
    ψ = copy(ψ_input)
    outcomes = autoencoder_mcwf!(ψ, θ, gates, trash, N)
    F = abs2(dot(ψ_input, ψ))
    H = sum(outcomes) / length(outcomes)  # Hamming distance
    
    # Loss = w_fid*(1-F) + w_svn*S + w_ham*H
    L = W_FID*(1-F) + W_SVN*S + W_HAM*H
    return L, F, S, H
end

function avg_metrics_mcwf(ψ_input, θ, gates, trash, N; n_traj=10)
    Fs, Hs, Ss = Float64[], Float64[], Float64[]
    sL = 0.0
    
    # Time only the MCWF simulation
    sim_time = @elapsed for _ in 1:n_traj
        L, F, S, H = evaluate_mcwf(ψ_input, θ, gates, trash, N)
        sL += L; push!(Fs, F); push!(Ss, S); push!(Hs, H)
    end
    
    # S_vN is computed on ENCODED pure state (same for all trajectories)
    # So just use mean (they should all be identical)
    # Return: mean_loss, mean_F, mean_S, mean_H, std_F, std_H, sim_time
    return sL/n_traj, mean(Fs), mean(Ss), mean(Hs), std(Fs), std(Hs), sim_time
end

# ==============================================================================
# SPSA OPTIMIZER
# ==============================================================================

# Full SPSA with Spall (1998) scheduling: a_k = a/(A+k)^α, c_k = c/k^γ
mutable struct SPSA
    a::Float64   # Learning rate scale
    c::Float64   # Perturbation scale  
    A::Float64   # Stability constant (usually 10% of max iterations)
    α::Float64   # LR decay exponent (standard: 0.602)
    γ::Float64   # Perturbation decay exponent (standard: 0.101)
    k::Int       # Current iteration
end
SPSA() = SPSA(0.1, 0.15, 10.0, 0.602, 0.101, 0)

# SPSA gradient estimation (used by both :SPSA and :SPSA_Adam)
function spsa_grad!(s::SPSA, cost_fn, θ)
    s.k += 1
    c_k = s.c / s.k^s.γ
    Δ = 2.0 .* (rand(length(θ)) .> 0.5) .- 1.0
    return (cost_fn(θ .+ c_k.*Δ) - cost_fn(θ .- c_k.*Δ)) ./ (2*c_k.*Δ)
end

# Pure SPSA step with built-in learning rate scheduling
function spsa_step!(s::SPSA, cost_fn, θ)
    g = spsa_grad!(s, cost_fn, θ)
    a_k = s.a / (s.A + s.k)^s.α
    θ .-= a_k .* g
    return θ
end

# ==============================================================================
# TRAINING
# ==============================================================================

function train_dm(ρ_input, N, k, n_layers, ansatz, entangler_gate, optimizer_type; n_epochs=300, lr=0.03)
    trash = collect((N-k+1):N)
    n_params, gates = build_encoder_spec(N, n_layers, ansatz, entangler_gate)
    θ = 0.1 * randn(n_params)
    spsa = SPSA()
    
    # Only setup Adam if using :SPSA_Adam
    opt = optimizer_type == :SPSA_Adam ? Optimisers.setup(Adam(lr), θ) : nothing
    
    fids, ents, hams = Float64[], Float64[], Float64[]
    
    total_time = @elapsed for ep in 1:n_epochs
        cost = θ_t -> evaluate_dm(ρ_input, θ_t, gates, trash, N)[1]
        
        if optimizer_type == :SPSA
            θ = spsa_step!(spsa, cost, θ)  # Pure SPSA with built-in LR scheduling
        else  # :SPSA_Adam
            g = spsa_grad!(spsa, cost, θ)
            opt, θ = Optimisers.update(opt, θ, g)
        end
        
        _, F, S, H = evaluate_dm(ρ_input, θ, gates, trash, N)
        push!(fids, F); push!(ents, S); push!(hams, H)
        
        ep % 50 == 0 && @printf("    [%s|DM|%s] k=%d ep=%d: F = %.4f, SvN = %.4f, D_H = %.4f\n", ansatz, optimizer_type, k, ep, F, S, H)
    end
    @printf("    [%s|DM|%s] k=%d Training time: %.2fs (%.2f ms/epoch)\n", ansatz, optimizer_type, k, total_time, 1000*total_time/n_epochs)
    return fids, ents, hams, total_time
end

function train_mcwf(ψ_input, N, k, n_layers, ansatz, entangler_gate, n_traj, optimizer_type; n_epochs=300, lr=0.03)
    trash = collect((N-k+1):N)
    n_params, gates = build_encoder_spec(N, n_layers, ansatz, entangler_gate)
    θ = 0.1 * randn(n_params)
    spsa = SPSA()
    
    # Only setup Adam if using :SPSA_Adam
    opt = optimizer_type == :SPSA_Adam ? Optimisers.setup(Adam(lr), θ) : nothing
    
    fids, ents, hams = Float64[], Float64[], Float64[]
    fids_std, hams_std = Float64[], Float64[]
    total_sim_time = 0.0  # Accumulate MCWF simulation time (excludes tomography)
    
    for ep in 1:n_epochs
        cost = θ_t -> avg_metrics_mcwf(ψ_input, θ_t, gates, trash, N; n_traj=5)[1]
        
        if optimizer_type == :SPSA
            θ = spsa_step!(spsa, cost, θ)  # Pure SPSA with built-in LR scheduling
        else  # :SPSA_Adam
            g = spsa_grad!(spsa, cost, θ)
            opt, θ = Optimisers.update(opt, θ, g)
        end
        
        _, F, S, H, σF, σH, sim_t = avg_metrics_mcwf(ψ_input, θ, gates, trash, N; n_traj=n_traj)
        total_sim_time += sim_t
        push!(fids, F); push!(ents, S); push!(hams, H)
        push!(fids_std, σF); push!(hams_std, σH)
        
        ep % 50 == 0 && @printf("    [%s|MCWF-%d|%s] k=%d ep=%d: F = %.4f±%.4f, SvN = %.4f, D_H = %.4f±%.4f\n", ansatz, n_traj, optimizer_type, k, ep, F, σF, S, H, σH)
    end
    @printf("    [%s|MCWF-%d|%s] k=%d Sim time: %.2fs (%.2f ms/epoch, excludes tomography)\n", ansatz, n_traj, optimizer_type, k, total_sim_time, 1000*total_sim_time/n_epochs)
    return fids, ents, hams, fids_std, hams_std, total_sim_time
end

# ==============================================================================
# INPUT STATE PREPARATION
# ==============================================================================

"""
Prepare input state and return (ψ, ρ, state_name, state_str_for_file)
"""
function prepare_input_state(state_type::Symbol, N::Int; ham_config=nothing)
    dim = 2^N
    
    if state_type == :Hamiltonian_GS
        # Ground state from HAMILTONIAN_CONFIGS entry
        if ham_config === nothing
            error("Hamiltonian_GS requires ham_config=(name, Jxx, Jyy, Jzz, hx, hy, hz)")
        end
        name, Jxx, Jyy, Jzz, hx, hy, hz = ham_config
        # Use ground_state_xxz (supports full XXZ + field)
        E_gs, ψ = ground_state_xxz(N, Jxx, Jyy, Jzz, hx)  # Note: hy, hz not yet used in ground_state_xxz
        param_str = "Jxx=$Jxx Jyy=$Jyy Jzz=$Jzz hx=$hx"
        fname_str = "$(lowercase(name))_Jxx$(Jxx)_Jyy$(Jyy)_Jzz$(Jzz)_hx$(hx)"
        return ψ, ψ * ψ', "$name GS ($param_str)", fname_str
        
    elseif state_type == :GHZ
        ψ = zeros(ComplexF64, dim)
        ψ[1] = 1/√2           # |000...0⟩
        ψ[end] = 1/√2         # |111...1⟩
        return ψ, ψ * ψ', "GHZ", "ghz"
        
    elseif state_type == :W
        ψ = zeros(ComplexF64, dim)
        for i in 0:(N-1)
            idx = 1 << i + 1  # |100...⟩, |010...⟩, etc.
            ψ[idx] = 1/√N
        end
        return ψ, ψ * ψ', "W", "w"
        
    elseif state_type == :Dicke
        # Dicke state with k = N÷2 excitations
        k_excit = div(N, 2)
        ψ = zeros(ComplexF64, dim)
        count = 0
        for i in 0:(dim-1)
            if count_ones(i) == k_excit
                ψ[i+1] = 1.0
                count += 1
            end
        end
        ψ ./= √count
        return ψ, ψ * ψ', "Dicke_k$(k_excit)", "dicke_k$(k_excit)"
        
    elseif state_type == :Product
        # |+⟩⊗N = (|0⟩ + |1⟩)⊗N / √(2^N)
        ψ = ones(ComplexF64, dim) / √dim
        return ψ, ψ * ψ', "Product_plus", "product"
        
    elseif state_type == :Custom
        # User-defined custom state
        if CUSTOM_STATE === nothing
            error("CUSTOM_STATE is not defined. Set CUSTOM_STATE to a Vector{ComplexF64} before using :Custom")
        end
        if length(CUSTOM_STATE) != dim
            error("CUSTOM_STATE has wrong dimension: got $(length(CUSTOM_STATE)), expected $dim for N=$N")
        end
        ψ = normalize(CUSTOM_STATE)
        return ψ, ψ * ψ', CUSTOM_STATE_NAME, lowercase(replace(CUSTOM_STATE_NAME, " " => "_"))
        
    else
        error("Unknown state type: $state_type")
    end
end

# ==============================================================================
# MAIN - GRID SEARCH: optimizer → state → ansatz → N → L → weights → rep → k
# ==============================================================================

colors = [:blue, :red, :green, :purple, :orange]
n_weights = length(WEIGHT_CONFIGS)

# Build state configs: [(input_state_type, ham_config), ...]
state_configs = Tuple{Symbol, Any}[]
for input_state_type in INPUT_STATES
    if input_state_type == :Hamiltonian_GS
        for ham_config in HAMILTONIAN_CONFIGS
            push!(state_configs, (input_state_type, ham_config))
        end
    else
        push!(state_configs, (input_state_type, nothing))
    end
end

# Build representation configs: [(:dm, nothing), (:mcwf, n_traj1), ...]
rep_configs = Tuple{Symbol, Union{Nothing, Int}}[(:dm, nothing)]
for n_traj in MCWF_TRAJ_VALUES
    push!(rep_configs, (:mcwf, n_traj))
end
rep_keys_local = ["DM", ["MCWF-$n" for n in MCWF_TRAJ_VALUES]...]

# OUTERMOST LOOP: Optimizer
for opt_type in OPTIMIZER_TYPES
    println("\n" * "="^70)
    println("  OPTIMIZER: $opt_type")
    println("="^70)

# Loop: state → ansatz → N → L
for (input_state_type, ham_config) in state_configs
    state_label = ham_config === nothing ? string(input_state_type) : "$(input_state_type) ($(ham_config[1]))"
    println("\n  STATE: $state_label")

for ansatz in ANSATZ_TYPES
    println("\n    ANSATZ: $ansatz")

for entangler_gate in ENTANGLER_GATES
    println("      Entangler: $entangler_gate")

for N in N_VALUES
    println("\n      N = $N QUBITS")
    
    # Prepare input state
    ψ_input, ρ_input, state_name, state_str = prepare_input_state(input_state_type, N; ham_config=ham_config)
    println("        State: $state_name")
    
    # k values for this N
    if K_VALUES == :auto
        k_values = sort(unique([N - 3, N - 1]))
    elseif K_VALUES == :auto_1_half_n3
        k_values = sort(unique(filter(k -> k >= 1 && k <= N-1, [1, N ÷ 2, N - 3])))
    else
        k_values = K_VALUES
    end

for n_layers in L_VALUES
    println("        L = $n_layers")
    
    # Storage for all weight configs
    results_w = Dict{Int, Dict{String, Dict{Int, NamedTuple}}}()
    
    # Filenames
    k_str = "k=$(join(k_values, ","))"
    opt_str = String(opt_type)
    k_fname = "k$(join(k_values, "-"))"
    rot_str = join([uppercase(string(r)[2]) for r in ROTATION_GATES], "")
    ent_str = uppercase(string(entangler_gate))
    base_fname = "qae_$(state_str)_$(ansatz)_N$(N)_L$(n_layers)_R$(rot_str)_$(ent_str)_$(k_fname)_$(opt_str)"
    
    # Train for each weight config → rep → k (innermost)
    for (w_idx, (w_fid, w_svn, w_ham)) in enumerate(WEIGHT_CONFIGS)
        global W_FID, W_SVN, W_HAM = w_fid, w_svn, w_ham
        println("          Weights: F=$w_fid S=$w_svn H=$w_ham")
        
        results_w[w_idx] = Dict{String, Dict{Int, NamedTuple}}()
        
        # Loop over representations and k values (innermost)
        for (rep_type, n_traj) in rep_configs
            key = rep_type == :dm ? "DM" : "MCWF-$n_traj"
            results_w[w_idx][key] = Dict{Int, NamedTuple}()
            
            for k in k_values  # k is innermost (fastest)
                seed = 42 + k + n_layers*100 + hash(ansatz) + N*1000 + w_idx*10000 + hash(entangler_gate) + hash(opt_type)
                Random.seed!(seed)
                
                if rep_type == :dm
                    f, s, h, t = train_dm(ρ_input, N, k, n_layers, ansatz, entangler_gate, opt_type; n_epochs=N_EPOCHS, lr=LR)
                    results_w[w_idx][key][k] = (fids=f, ents=s, hams=h, time=t)
                else  # :mcwf
                    f, s, h, f_std, h_std, t = train_mcwf(ψ_input, N, k, n_layers, ansatz, entangler_gate, n_traj, opt_type; n_epochs=N_EPOCHS, lr=LR)
                    results_w[w_idx][key][k] = (fids=f, ents=s, hams=h, fids_std=f_std, hams_std=h_std, time=t)
                end
            end
        end
                
                # INCREMENTAL PLOT: Create/update and save after each weight config
                # Layout: rows = metrics (1-F, SvN, Dh), columns = weight configs
                completed_weights = w_idx
                plt = plot(layout=(3, n_weights), size=(500*n_weights, 1100), 
                           plot_title="$state_name | $ansatz R:$rot_str $ent_str | N=$N L=$n_layers $k_str | $opt_str",
                           margin=5Plots.mm, left_margin=12Plots.mm, bottom_margin=6Plots.mm,
                           titlefontsize=16, guidefontsize=12, tickfontsize=10, legendfontsize=10,
                           link=:x)
                
                # Plot all completed weight configs
                for (col_idx, (pw_fid, pw_svn, pw_ham)) in enumerate(WEIGHT_CONFIGS[1:completed_weights])
                    w_label = "w=($pw_fid,$pw_svn,$pw_ham)"
                    is_last_col = (col_idx == n_weights)
                    is_first_col = (col_idx == 1)
                    xlab = "Epoch"
                    
                    # Row 1: Infidelity (1-F)
                    subplot_1f = (1-1)*n_weights + col_idx
                    for (i, k) in enumerate(k_values)
                        for (j, rep) in enumerate(rep_keys_local)
                            ls = j == 1 ? :solid : :dash
                            data = results_w[col_idx][rep][k]
                            time_str = @sprintf("%.1fs", data.time)
                            lbl = is_first_col ? "k=$k $rep ($time_str)" : ""
                            infid = 1.0 .- data.fids
                            if startswith(rep, "MCWF") && hasproperty(data, :fids_std)
                                plot!(plt[subplot_1f], infid, ribbon=data.fids_std, fillalpha=0.2,
                                      lw=2, color=colors[i], ls=ls, label=lbl,
                                      xlabel="", ylabel=is_first_col ? "1-F" : "",
                                      title=w_label,
                                      legend=is_first_col ? :topright : false)
                            else
                                plot!(plt[subplot_1f], infid, lw=2, color=colors[i], ls=ls, 
                                      label=lbl,
                                      xlabel="", ylabel=is_first_col ? "1-F" : "",
                                      title=w_label,
                                      legend=is_first_col ? :topright : false)
                            end
                        end
                    end
                    hline!(plt[subplot_1f], [0.0], lw=1, color=:black, ls=:dot, label="")
                    
                    # Row 2: Entropy S_vN
                    subplot_svn = (2-1)*n_weights + col_idx
                    for (i, k) in enumerate(k_values)
                        for (j, rep) in enumerate(rep_keys_local)
                            ls = j == 1 ? :solid : :dash
                            data = results_w[col_idx][rep][k]
                            plot!(plt[subplot_svn], data.ents, lw=2, color=colors[i], ls=ls, 
                                  label="", xlabel="", ylabel=is_first_col ? "Sᵥₙ" : "", legend=false)
                        end
                    end
                    hline!(plt[subplot_svn], [0.0], lw=1, color=:black, ls=:dot, label="")
                    
                    # Row 3: Hamming distance
                    subplot_dh = (3-1)*n_weights + col_idx
                    for (i, k) in enumerate(k_values)
                        for (j, rep) in enumerate(rep_keys_local)
                            ls = j == 1 ? :solid : :dash
                            data = results_w[col_idx][rep][k]
                            if startswith(rep, "MCWF") && hasproperty(data, :hams_std)
                                plot!(plt[subplot_dh], data.hams, ribbon=data.hams_std, fillalpha=0.2,
                                      lw=2, color=colors[i], ls=ls, label="",
                                      xlabel=xlab, ylabel=is_first_col ? "Dₕ" : "", legend=false)
                            else
                                plot!(plt[subplot_dh], data.hams, lw=2, color=colors[i], ls=ls, 
                                      label="", xlabel=xlab, ylabel=is_first_col ? "Dₕ" : "", legend=false)
                            end
                        end
                    end
                    hline!(plt[subplot_dh], [0.0], lw=1, color=:black, ls=:dot, label="")
                end
                
                # Save incrementally (overwrites previous version)
                savefig(plt, joinpath(OUTPUT_DIR, "fig_$base_fname.png"))
                println("        → Saved: fig_$base_fname.png (weights $w_idx/$n_weights)")
                
                # === ADDITIONAL LOG-SCALE FIGURE ===
                # Same layout but with log scale, filtering to positive values only
                # First, compute row-wise min/max from ALL completed weight configs
                all_infid, all_svn, all_dh = Float64[], Float64[], Float64[]
                for c_idx in 1:w_idx
                    for k in k_values
                        for rep in rep_keys_local
                            d = results_w[c_idx][rep][k]
                            append!(all_infid, filter(x -> x > 0, 1.0 .- d.fids))
                            append!(all_svn, filter(x -> x > 0, d.ents))
                            append!(all_dh, filter(x -> x > 0, d.hams))
                        end
                    end
                end
                # Compute y-limits with 10x margin below min
                ymin_infid = isempty(all_infid) ? 1e-4 : max(minimum(all_infid) / 10, 1e-8)
                ymin_svn = isempty(all_svn) ? 1e-4 : max(minimum(all_svn) / 10, 1e-8)
                ymin_dh = isempty(all_dh) ? 1e-4 : max(minimum(all_dh) / 10, 1e-8)
                
                plt_log = plot(layout=(3, n_weights), size=(500*n_weights, 1100), 
                           plot_title="LOG | $state_name | $ansatz R:$rot_str $ent_str | N=$N L=$n_layers $k_str | $opt_str",
                           margin=5Plots.mm, left_margin=12Plots.mm, bottom_margin=6Plots.mm,
                           titlefontsize=16, guidefontsize=12, tickfontsize=10, legendfontsize=10,
                           link=:x)  # Share x-axis; will set ylims per row
                
                for (col_idx, (pw_fid, pw_svn, pw_ham)) in enumerate(WEIGHT_CONFIGS[1:w_idx])
                    w_label = "w=($pw_fid,$pw_svn,$pw_ham)"
                    is_first_col = (col_idx == 1)
                    xlab = "Epoch"
                    
                    # Row 1: Infidelity (1-F) log scale - positive only
                    subplot_1f = (1-1)*n_weights + col_idx
                    log_ticks = [10.0^(-i) for i in 0:2:6]  # 1, 0.01, 0.0001, 10^-6
                    log_labels = ["10⁰", "10⁻²", "10⁻⁴", "10⁻⁶"]
                    for (i, k) in enumerate(k_values)
                        for (j, rep) in enumerate(rep_keys_local)
                            ls = j == 1 ? :solid : :dash
                            data = results_w[col_idx][rep][k]
                            time_str = @sprintf("%.1fs", data.time)
                            lbl = is_first_col ? "k=$k $rep ($time_str)" : ""
                            infid = 1.0 .- data.fids
                            eps = collect(1:length(infid))
                            mask = infid .> 0
                            if any(mask)
                                plot!(plt_log[subplot_1f], eps[mask], infid[mask], 
                                      lw=2, color=colors[i], ls=ls, label=lbl,
                                      xlabel="", ylabel=is_first_col ? "1-F" : "",
                                      title=w_label, ylims=(ymin_infid, 1.0),
                                      legend=is_first_col ? :topright : false, yscale=:log10)
                            end
                        end
                    end
                    
                    # Row 2: Entropy S_vN log scale - positive only
                    subplot_svn = (2-1)*n_weights + col_idx
                    for (i, k) in enumerate(k_values)
                        for (j, rep) in enumerate(rep_keys_local)
                            ls = j == 1 ? :solid : :dash
                            data = results_w[col_idx][rep][k]
                            eps = collect(1:length(data.ents))
                            mask = data.ents .> 0
                            if any(mask)
                                plot!(plt_log[subplot_svn], eps[mask], data.ents[mask], 
                                      lw=2, color=colors[i], ls=ls, label="", 
                                      xlabel="", ylabel=is_first_col ? "Sᵥₙ" : "", 
                                      ylims=(ymin_svn, 1.0),
                                      legend=false, yscale=:log10)
                            end
                        end
                    end
                    
                    # Row 3: Hamming distance log scale - positive only
                    subplot_dh = (3-1)*n_weights + col_idx
                    dh_ticks = [10.0^(-i) for i in 0:2:6]  # 1, 0.01, 0.0001, 10^-6
                    dh_labels = ["10⁰", "10⁻²", "10⁻⁴", "10⁻⁶"]
                    for (i, k) in enumerate(k_values)
                        for (j, rep) in enumerate(rep_keys_local)
                            ls = j == 1 ? :solid : :dash
                            data = results_w[col_idx][rep][k]
                            eps = collect(1:length(data.hams))
                            mask = data.hams .> 0
                            if any(mask)
                                plot!(plt_log[subplot_dh], eps[mask], data.hams[mask], 
                                      lw=2, color=colors[i], ls=ls, label="", 
                                      xlabel=xlab, ylabel=is_first_col ? "Dₕ" : "", 
                                      ylims=(ymin_dh, 1.0),
                                      legend=false, yscale=:log10)
                            end
                        end
                    end
                end
                
                try
                    savefig(plt_log, joinpath(OUTPUT_DIR, "fig_log_$base_fname.png"))
                    println("        → Saved: fig_log_$base_fname.png")
                catch e
                    println("        ⚠ Log plot failed (data may have zeros): $e")
                end
            end # for w_idx
            
            # Save data to data/ subfolder
            data_dir = joinpath(OUTPUT_DIR, "data")
            mkpath(data_dir)
            
            # For first weight config only (main results)
            w_idx = 1
            for k in k_values
                # DM data
                dm_data = results_w[w_idx]["DM"][k]
                dm_fname = joinpath(data_dir, "dm_$(base_fname)_k$(k).txt")
                open(dm_fname, "w") do f
                    println(f, "# DM results: $(ansatz), N=$N, L=$n_layers, k=$k")
                    println(f, "epoch,fidelity,svn,hamming")
                    for ep in 1:length(dm_data.fids)
                        @printf(f, "%d,%.8f,%.8f,%.8f\n", ep, dm_data.fids[ep], dm_data.ents[ep], dm_data.hams[ep])
                    end
                end
                
                # MCWF data (no svn_std since S_vN computed from single reconstructed ρ)
                for n_traj in MCWF_TRAJ_VALUES
                    key = "MCWF-$n_traj"
                    mcwf_data = results_w[w_idx][key][k]
                    mcwf_fname = joinpath(data_dir, "mcwf$(n_traj)_$(base_fname)_k$(k).txt")
                    open(mcwf_fname, "w") do f
                        println(f, "# MCWF results: $(ansatz), N=$N, L=$n_layers, k=$k, n_traj=$n_traj")
                        println(f, "epoch,fidelity_mean,fidelity_std,svn,hamming_mean,hamming_std")
                        for ep in 1:length(mcwf_data.fids)
                            @printf(f, "%d,%.8f,%.8f,%.8f,%.8f,%.8f\n", 
                                ep, mcwf_data.fids[ep], mcwf_data.fids_std[ep],
                                mcwf_data.ents[ep],
                                mcwf_data.hams[ep], mcwf_data.hams_std[ep])
                        end
                    end
                end
            end
            println("      → Data saved to data/ subfolder")
end  # n_layers
end  # N
end  # entangler_gate
end  # ansatz
end  # state_configs
end  # opt_type (OPTIMIZER_TYPES)

rep_keys = ["DM", ["MCWF-$n" for n in MCWF_TRAJ_VALUES]...]

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

println("\n" * "="^70)
println("  SUMMARY: Final Fidelity by Ansatz, L, k, Rep")
println("="^70)

for ansatz in ANSATZ_TYPES
    println("\n  ANSATZ: $ansatz")
    for n_layers in L_VALUES
        println("    L = $n_layers:")
        print("      | k |")
        for rep in rep_keys; @printf(" %-8s |", rep); end
        println()
        print("      |---|")
        for _ in rep_keys; print("----------|"); end
        println()
        
        # Get k values from results (handles :auto case)
        k_vals_used = sort(collect(keys(results[ansatz][n_layers][first(rep_keys)])))
        for k in k_vals_used
            @printf("      | %d |", k)
            for rep in rep_keys
                @printf(" %.4f   |", results[ansatz][n_layers][rep][k].fids[end])
            end
            println()
        end
    end
end

# Save log
open(joinpath(OUTPUT_DIR, "output.txt"), "w") do f
    println(f, "QUANTUM AUTOENCODER SWEEP")
    println(f, "="^60)
    println(f, "Ansatzes: $ANSATZ_TYPES")
    println(f, "Input: $input_state")
    println(f, "N=$N_QUBITS, L=$L_VALUES, Epochs=$N_EPOCHS")
    println(f, "Heisenberg: Jx=$Jx, Jy=$Jy, Jz=$Jz, hx=$hx_field, E_GS=$E_gs")
    println(f, "Weights: w_fid=$W_FID, w_svn=$W_SVN, w_ham=$W_HAM")
    println(f, "\nFinal Fidelity:")
    for ansatz in ANSATZ_TYPES
        println(f, "\n$ansatz:")
        for n_layers in L_VALUES
            println(f, "  L=$n_layers:")
            # Get k values from results
            k_vals_used = sort(collect(keys(results[ansatz][n_layers][first(rep_keys)])))
            for k in k_vals_used
                @printf(f, "    k=%d: ", k)
                for rep in rep_keys
                    @printf(f, "%s=%.4f ", rep, results[ansatz][n_layers][rep][k].fids[end])
                end
                println(f)
            end
        end
    end
end
println("\n  Saved: output.txt")

println("\nDone!")

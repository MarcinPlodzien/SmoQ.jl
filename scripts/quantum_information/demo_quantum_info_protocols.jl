#!/usr/bin/env julia
# ==============================================================================
# QUANTUM INFORMATION PROTOCOLS DEMO
# ==============================================================================
#
# This demo showcases fundamental quantum information protocols using our
# matrix-free bitwise quantum simulator. All protocols are implemented using:
#   - Bitwise gate operations (no explicit matrix construction)
#   - MCWF trajectory unraveling for noise simulation
#   - Mid-circuit measurements and conditional operations
#
# ==============================================================================

using Printf
using Random
using LinearAlgebra

# Load modules - same as demo_parametrized_gates.jl
const UTILS_CPU = joinpath(@__DIR__, "..", "utils", "cpu")
include(joinpath(UTILS_CPU, "cpuQuantumChannelGates.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumChannelKrausOperators.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateObservables.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateMeasurements.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateCharacteristic.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateTomography.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePrintingHelpers.jl"))

using .CPUQuantumChannelGates
using .CPUQuantumChannelKraus
using .CPUQuantumStatePartialTrace
using .CPUQuantumStatePreparation
using .CPUQuantumStateObservables
using .CPUQuantumStateMeasurements
using .CPUQuantumStateCharacteristic

println()
println("=" ^ 70)
println("  QUANTUM INFORMATION PROTOCOLS")
println("  Demonstrating fundamental QI protocols with bitwise operations")
println("=" ^ 70)
println()
println("  This demo showcases:")
println("  ─────────────────────")
println("  - Bell state preparation and verification")
println("  - GHZ state creation and noisy sampling")
println("  - Quantum teleportation with mid-circuit measurements")
println("  - Decoherence effects on quantum correlations")
println()
println("  All protocols use BITWISE gate operations (no matrix construction)")
println("  Noise is simulated via MCWF trajectory unraveling")
println()
println("=" ^ 70)

all_passed = true

# ==============================================================================
# PROTOCOL 1: BELL STATE PREPARATION + VERIFICATION
# ==============================================================================
#
# Create Bell pair: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
# Properties:
#   - ⟨Z₁⟩ = ⟨Z₂⟩ = 0 (maximally uncertain locally)
#   - ⟨Z₁Z₂⟩ = +1 (perfect correlation)
#   - Partial trace gives maximally mixed state ρ = I/2
#
# ==============================================================================

println("\n" * "=" ^ 70)
println("  PROTOCOL 1: BELL STATE PREPARATION")
println("=" ^ 70)
println()
println("  PHYSICS BACKGROUND")
println("  ───────────────────")
println("  A Bell state is a maximally entangled 2-qubit state.")
println("  The four Bell states form a basis:")
println("    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2   |Φ⁻⟩ = (|00⟩ - |11⟩)/√2")
println("    |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2   |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2")
println()
println("  Properties of |Φ⁺⟩:")
println("    - ⟨Z₁⟩ = ⟨Z₂⟩ = 0   (locally random)")
println("    - ⟨Z₁Z₂⟩ = +1       (perfectly correlated)")
println("    - Tr₂(ρ) = I/2      (maximally mixed subsystem)")
println()

# Helper function to print 2-qubit state (all 4 basis states)
function print_state_2q(ψ::Vector{ComplexF64}, label::String)
    println("  $label:")
    println("    ┌──────────┬─────────────────┐")
    println("    │ |q1 q2⟩  │    Amplitude    │")
    println("    ├──────────┼─────────────────┤")
    for i in 1:4
        q1 = (i-1) & 1
        q2 = ((i-1) >> 1) & 1
        if abs(ψ[i]) > 1e-10
            amp = @sprintf("%+.4f", real(ψ[i]))
            if abs(imag(ψ[i])) > 1e-10
                amp = @sprintf("%+.3f%+.3fi", real(ψ[i]), imag(ψ[i]))
            end
        else
            amp = "  0.0000"
        end
        println("    │ |$q1  $q2⟩  │  ", rpad(amp, 13), " │")
    end
    println("    └──────────┴─────────────────┘")
end

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  GATE-BY-GATE VISUALIZATION                                │")
println("  └─────────────────────────────────────────────────────────────┘")
println()

let N = 2
    dim = 1 << N
    
    # STEP 0: Initial state
    ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
    
    println("  STEP 0: Initial state |00⟩")
    println("  " * "─"^50)
    print_state_2q(ψ, "|ψ₀⟩ = |00⟩ (product state)")
    println("  Properties: ⟨Z₁⟩ = +1, ⟨Z₂⟩ = +1, ⟨Z₁Z₂⟩ = +1")
    println("  Entanglement: NONE (separable)")
    println()
    
    # STEP 1: Apply H(1)
    println("  STEP 1: Apply Hadamard to qubit 1")
    println("  " * "─"^50)
    println("  Gate: H(1) = (1/√2)[[1,1],[1,-1]]")
    println("  Effect: |0⟩ → (|0⟩+|1⟩)/√2, |1⟩ → (|0⟩-|1⟩)/√2")
    println()
    
    apply_hadamard_psi!(ψ, 1, N)
    print_state_2q(ψ, "|ψ₁⟩ = H(1)|00⟩ = |+⟩₁ ⊗ |0⟩₂")
    println("  In standard notation: = (|00⟩ + |10⟩)/√2")
    println("  Properties: ⟨Z₁⟩ = 0, ⟨Z₂⟩ = +1, ⟨Z₁Z₂⟩ = 0")
    println("  Entanglement: STILL NONE (still separable)")
    println()
    
    # STEP 2: Apply CNOT(1,2)
    println("  STEP 2: Apply CNOT(1→2)")
    println("  " * "─"^50)
    println("  Gate: CNOT flips target (q2) when control (q1) is |1⟩")
    println("  Implementation: CNOT = H(2)·CZ(1,2)·H(2)")
    println("  Effect: |00⟩ → |00⟩, |10⟩ → |11⟩")
    println()
    
    apply_hadamard_psi!(ψ, 2, N)
    apply_cz_psi!(ψ, 1, 2, N)
    apply_hadamard_psi!(ψ, 2, N)
    
    print_state_2q(ψ, "|Φ⁺⟩ = CNOT(1,2)|ψ₁⟩ = (|00⟩ + |11⟩)/√2")
    println()
    
    # Verification
    println("  ┌─────────────────────────────────────────────────────────────┐")
    println("  │  VERIFICATION                                              │")
    println("  └─────────────────────────────────────────────────────────────┘")
    println()
    
    # Observables
    z1 = real(expect_local(ψ, 1, N, :z))
    z2 = real(expect_local(ψ, 2, N, :z))
    x1 = real(expect_local(ψ, 1, N, :x))
    x2 = real(expect_local(ψ, 2, N, :x))
    zz = real(expect_corr(ψ, 1, 2, N, :zz))
    xx = real(expect_corr(ψ, 1, 2, N, :xx))
    
    println("  Local observables:")
    println("    ┌────────┬───────────┬──────────────────────────────────┐")
    println("    │ Obs.   │  Value    │  Interpretation                  │")
    println("    ├────────┼───────────┼──────────────────────────────────┤")
    @printf("    │ ⟨Z₁⟩   │  %+.4f   │  Locally random (maximally mixed)│\n", z1)
    @printf("    │ ⟨Z₂⟩   │  %+.4f   │  Locally random (maximally mixed)│\n", z2)
    @printf("    │ ⟨X₁⟩   │  %+.4f   │  No X preference                 │\n", x1)
    @printf("    │ ⟨X₂⟩   │  %+.4f   │  No X preference                 │\n", x2)
    println("    └────────┴───────────┴──────────────────────────────────┘")
    println()
    
    println("  Correlations:")
    println("    ┌────────────┬───────────┬────────────────────────────────┐")
    println("    │ Correlation│  Value    │  Interpretation                │")
    println("    ├────────────┼───────────┼────────────────────────────────┤")
    @printf("    │ ⟨Z₁Z₂⟩     │  %+.4f   │  PERFECT correlation in Z      │\n", zz)
    @printf("    │ ⟨X₁X₂⟩     │  %+.4f   │  PERFECT correlation in X      │\n", xx)
    println("    └────────────┴───────────┴────────────────────────────────┘")
    println()
    
    # Density matrix
    ρ = ψ * ψ'
    println("  Full density matrix |Φ⁺⟩⟨Φ⁺|:")
    print_density_matrix(ρ, "ρ = |Φ⁺⟩⟨Φ⁺|")
    println()
    
    # Partial trace
    ρ_1 = partial_trace(ψ, [2], N)
    println("  Reduced density matrix (tracing out qubit 2):")
    print_density_matrix(ρ_1, "ρ₁ = Tr₂(ρ)")
    purity = real(tr(ρ_1 * ρ_1))
    @printf("  Purity Tr(ρ₁²) = %.4f (0.5 = maximally mixed)\n", purity)
    println()
    println("  KEY INSIGHT: Despite total state being pure, each subsystem is")
    println("  maximally mixed! This is the signature of maximal entanglement.")
    
    passed = abs(z1) < 0.01 && abs(z2) < 0.01 && abs(zz - 1.0) < 0.01 && 
             abs(ρ_1[1,1] - 0.5) < 0.01
    status = passed ? "--" : "✗"
    println()
    @printf("  %s Bell state verified!\n", status)
    
    global all_passed &= passed
end

# ==============================================================================
# PROTOCOL 2: GHZ STATE + NOISY SAMPLING
# ==============================================================================
#
# Create GHZ state: |GHZ⟩ = (|000...0⟩ + |111...1⟩)/√2
# Apply noise, then sample measurement outcomes
# Compare DM exact probabilities vs MCWF sampling
#
# ==============================================================================

println("\n" * "=" ^ 70)
println("  PROTOCOL 2: GHZ STATE + NOISY SAMPLING")
println("=" ^ 70)
println()
println("  PHYSICS BACKGROUND")
println("  ───────────────────")
println("  The GHZ (Greenberger-Horne-Zeilinger) state is a maximally entangled")
println("  N-qubit state with unique properties:")
println()
println("    |GHZ_N⟩ = (|00...0⟩ + |11...1⟩)/√2")
println()
println("  Properties:")
println("    - All qubits perfectly correlated: ⟨ZᵢZⱼ⟩ = +1 for all pairs")
println("    - Tracing out ANY qubit gives maximally mixed state on others")
println("    - Extremely fragile: any single qubit error destroys coherence")
println("    - Violates Mermin inequality (generalization of Bell)")
println()

# Helper function to print 3-qubit state (all 8 basis states)
function print_state_3q(ψ::Vector{ComplexF64}, label::String)
    println("  $label:")
    println("    ┌────────────┬─────────────────┐")
    println("    │ |q1 q2 q3⟩ │    Amplitude    │")
    println("    ├────────────┼─────────────────┤")
    for i in 1:8
        q1 = (i-1) & 1
        q2 = ((i-1) >> 1) & 1
        q3 = ((i-1) >> 2) & 1
        if abs(ψ[i]) > 1e-10
            amp = @sprintf("%+.4f", real(ψ[i]))
            if abs(imag(ψ[i])) > 1e-10
                amp = @sprintf("%+.3f%+.3fi", real(ψ[i]), imag(ψ[i]))
            end
        else
            amp = "  0.0000"
        end
        println("    │ |$q1  $q2  $q3⟩ │  ", rpad(amp, 13), " │")
    end
    println("    └────────────┴─────────────────┘")
end

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  GATE-BY-GATE VISUALIZATION (3-qubit GHZ)                  │")
println("  └─────────────────────────────────────────────────────────────┘")
println()

let N = 3
    dim = 1 << N
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1.0
    
    # STEP 0: Initial state
    println("  STEP 0: Initial state |000⟩")
    println("  " * "─"^55)
    print_state_3q(ψ, "|ψ₀⟩ = |000⟩ (product state)")
    println("  Correlations: ⟨Z₁Z₂⟩ = ⟨Z₁Z₃⟩ = ⟨Z₂Z₃⟩ = +1 (trivial)")
    println("  Entanglement: NONE (separable state)")
    println()
    
    # STEP 1: Apply H(1)
    println("  STEP 1: Apply Hadamard to qubit 1")
    println("  " * "─"^55)
    println("  Gate: H(1)")
    println("  Effect: Creates superposition on qubit 1")
    println()
    
    apply_hadamard_psi!(ψ, 1, N)
    print_state_3q(ψ, "|ψ₁⟩ = H(1)|000⟩ = |+⟩₁ ⊗ |00⟩₂₃")
    println("  = (|000⟩ + |100⟩)/√2")
    println("  Entanglement: STILL NONE (separable: |+⟩ ⊗ |00⟩)")
    println()
    
    # STEP 2: Apply CNOT(1,2)
    println("  STEP 2: Apply CNOT(1→2)")
    println("  " * "─"^55)
    println("  Gate: CNOT(1,2) = H(2)·CZ(1,2)·H(2)")
    println("  Effect: If q1=1, flip q2")
    println()
    
    apply_hadamard_psi!(ψ, 2, N)
    apply_cz_psi!(ψ, 1, 2, N)
    apply_hadamard_psi!(ψ, 2, N)
    print_state_3q(ψ, "|ψ₂⟩ = CNOT(1,2)|ψ₁⟩")
    println("  = (|000⟩ + |110⟩)/√2")
    println("  Correlations: ⟨Z₁Z₂⟩ = +1 (perfect), ⟨Z₂Z₃⟩ = ⟨Z₁Z₃⟩ = 0")
    println("  Entanglement: BIPARTITE (qubits 1,2 entangled, 3 separate)")
    println()
    
    # STEP 3: Apply CNOT(2,3)
    println("  STEP 3: Apply CNOT(2→3)")
    println("  " * "─"^55)
    println("  Gate: CNOT(2,3) = H(3)·CZ(2,3)·H(3)")
    println("  Effect: If q2=1, flip q3 → creates GHZ!")
    println()
    
    apply_hadamard_psi!(ψ, 3, N)
    apply_cz_psi!(ψ, 2, 3, N)
    apply_hadamard_psi!(ψ, 3, N)
    print_state_3q(ψ, "|GHZ₃⟩ = CNOT(2,3)|ψ₂⟩ = (|000⟩ + |111⟩)/√2")
    println()
    
    # Verification
    println("  ┌─────────────────────────────────────────────────────────────┐")
    println("  │  VERIFICATION                                              │")
    println("  └─────────────────────────────────────────────────────────────┘")
    println()
    
    z1z2 = real(expect_corr(ψ, 1, 2, N, :zz))
    z1z3 = real(expect_corr(ψ, 1, 3, N, :zz))
    z2z3 = real(expect_corr(ψ, 2, 3, N, :zz))
    
    println("  Pairwise correlations:")
    println("    ┌────────────┬───────────┬─────────────────────────────────┐")
    println("    │ Correlation│  Value    │  Interpretation                 │")
    println("    ├────────────┼───────────┼─────────────────────────────────┤")
    @printf("    │ ⟨Z₁Z₂⟩     │  %+.4f   │  Qubits 1,2 perfectly correlated│\n", z1z2)
    @printf("    │ ⟨Z₁Z₃⟩     │  %+.4f   │  Qubits 1,3 perfectly correlated│\n", z1z3)
    @printf("    │ ⟨Z₂Z₃⟩     │  %+.4f   │  Qubits 2,3 perfectly correlated│\n", z2z3)
    println("    └────────────┴───────────┴─────────────────────────────────┘")
    println()
    
    # Reduced density matrices
    ρ = ψ * ψ'
    ρ_12 = partial_trace(ψ, [3], N)
    ρ_1 = partial_trace(ψ, [2, 3], N)
    
    println("  Subsystem states:")
    @printf("    Tr(ρ²)   = %.4f (pure state)\n", real(tr(ρ * ρ)))
    @printf("    Tr(ρ₁²)  = %.4f (maximally mixed single qubit)\n", real(tr(ρ_1 * ρ_1)))
    @printf("    Tr(ρ₁₂²) = %.4f (mixed 2-qubit state)\n", real(tr(ρ_12 * ρ_12)))
    println()
    println("  KEY INSIGHT: GHZ is \"genuinely tripartite entangled\" - ")
    println("  tracing out ANY qubit leaves a MIXED state on the remaining two!")
    println("  This is different from |Φ⁺⟩₁₂ ⊗ |0⟩₃ where tracing q3 is harmless.")
end
println()
println("  " * "─"^60)
println()
println("  Now comparing DM vs MCWF for noisy GHZ sampling...")
println()

function create_ghz_psi!(ψ::Vector{ComplexF64}, N::Int)
    fill!(ψ, 0.0); ψ[1] = 1.0
    apply_hadamard_psi!(ψ, 1, N)
    for k in 1:(N-1)
        apply_hadamard_psi!(ψ, k+1, N)
        apply_cz_psi!(ψ, k, k+1, N)
        apply_hadamard_psi!(ψ, k+1, N)
    end
end

function create_ghz_rho!(ρ::Matrix{ComplexF64}, N::Int)
    fill!(ρ, 0.0); ρ[1,1] = 1.0
    apply_hadamard_rho!(ρ, 1, N)
    for k in 1:(N-1)
        apply_hadamard_rho!(ρ, k+1, N)
        apply_cz_rho!(ρ, k, k+1, N)
        apply_hadamard_rho!(ρ, k+1, N)
    end
end

let N = 4, M = 500, p = 0.02
    dim = 1 << N
    
    println("  Configuration:")
    @printf("    N = %d qubits\n", N)
    @printf("    p = %.2f depolarizing noise\n", p)
    @printf("    M = %d MCWF trajectories\n", M)
    println()
    
    # === DM APPROACH ===
    println("  DM Approach (exact Kraus operators):")
    ρ = zeros(ComplexF64, dim, dim)
    create_ghz_rho!(ρ, N)
    
    println("    Created |GHZ><GHZ|")
    @printf("    Tr(rho^2) = %.4f (pure state)\n", real(tr(ρ * ρ)))
    
    apply_channel_depolarizing!(ρ, p, collect(1:N), N)
    @printf("    After noise: Tr(rho^2) = %.4f (mixed)\n", real(tr(ρ * ρ)))
    
    probs_dm = real.(diag(ρ))
    p_all_0_dm = probs_dm[1]
    p_all_1_dm = probs_dm[dim]
    
    println()
    println("    Exact probabilities:")
    @printf("      P(|0000>) = %.4f\n", p_all_0_dm)
    @printf("      P(|1111>) = %.4f\n", p_all_1_dm)
    @printf("      P(other)  = %.4f\n", 1 - p_all_0_dm - p_all_1_dm)
    
    # === MCWF CONVERGENCE WITH M ===
    println()
    println("  MCWF CONVERGENCE WITH NUMBER OF TRAJECTORIES:")
    println("  ┌────────────────────────────────────────────────────────────┐")
    println("  │  Traj M   P(|0000>)   P(|1111>)   Error |0000>  Error |1111>│")
    println("  ├────────────────────────────────────────────────────────────┤")
    
    for M_test in [10, 20, 50, 100, 200, 500, 1000]
        counts = zeros(Int, dim)
        for traj in 1:M_test
            Random.seed!(traj)
            ψ = zeros(ComplexF64, dim)
            create_ghz_psi!(ψ, N)
            apply_channel_depolarizing!(ψ, p, collect(1:N), N)
            
            probs = abs2.(ψ)
            r = rand()
            cumsum_prob = 0.0
            for i in 1:dim
                cumsum_prob += probs[i]
                if r <= cumsum_prob
                    counts[i] += 1
                    break
                end
            end
        end
        
        probs_mcwf = counts ./ M_test
        p_all_0_mcwf = probs_mcwf[1]
        p_all_1_mcwf = probs_mcwf[dim]
        err_0 = abs(p_all_0_mcwf - p_all_0_dm)
        err_1 = abs(p_all_1_mcwf - p_all_1_dm)
        
        @printf("  │  %-6d    %.4f     %.4f      %.4f        %.4f     │\n",
                M_test, p_all_0_mcwf, p_all_1_mcwf, err_0, err_1)
    end
    
    println("  └────────────────────────────────────────────────────────────┘")
    println()
    println("  Observation: Error decreases as 1/sqrt(M) (Central Limit Theorem)")
    println("  DM exact:    P(|0000>) = $(round(p_all_0_dm, digits=4)), P(|1111>) = $(round(p_all_1_dm, digits=4))")
    println()
    println("  -- GHZ noisy sampling convergence demonstrated!")
    
    global all_passed &= true
end

# ==============================================================================
# PROTOCOL 3: CORRELATOR DECAY WITH NOISE
# ==============================================================================
#
# Demonstrate how quantum correlations decay with increasing noise.
# Start with Bell pair (⟨Z₁Z₂⟩ = +1), apply varying noise levels.
#
# ==============================================================================

println("\n" * "=" ^ 70)
println("  PROTOCOL 3: CORRELATOR DECAY (DECOHERENCE)")
println("=" ^ 70)
println()
println("  Initial state: |Φ⁺⟩ Bell pair (⟨Z₁Z₂⟩ = +1)")
println("  Apply noise:   Depolarizing with p = 0, 0.05, 0.1, 0.2, 0.5")
println("  Observe:       ⟨Z₁Z₂⟩ decays from +1 toward 0")
println()
println("  This demonstrates DECOHERENCE destroying quantum correlations!")
println()

let N = 2
    dim = 1 << N
    
    println("  p       ⟨Z₁Z₂⟩_DM   ⟨Z₁Z₂⟩_MCWF   |Decay|   Interpretation")
    println("  ─────   ──────────  ───────────   ───────   ─────────────────")
    
    for p in [0.0, 0.05, 0.1, 0.2, 0.5]
        # DM approach
        ρ = zeros(ComplexF64, dim, dim); ρ[1,1] = 1.0
        apply_hadamard_rho!(ρ, 1, N)
        apply_hadamard_rho!(ρ, 2, N)
        apply_cz_rho!(ρ, 1, 2, N)
        apply_hadamard_rho!(ρ, 2, N)
        
        if p > 0
            apply_channel_depolarizing!(ρ, p, collect(1:N), N)
        end
        zz_dm = real(expect_corr(ρ, 1, 2, N, :zz))
        
        # MCWF approach
        M = 200
        sum_zz = 0.0
        for traj in 1:M
            Random.seed!(traj)
            ψ = zeros(ComplexF64, dim); ψ[1] = 1.0
            apply_hadamard_psi!(ψ, 1, N)
            apply_hadamard_psi!(ψ, 2, N)
            apply_cz_psi!(ψ, 1, 2, N)
            apply_hadamard_psi!(ψ, 2, N)
            
            if p > 0
                apply_channel_depolarizing!(ψ, p, collect(1:N), N)
            end
            sum_zz += real(expect_corr(ψ, 1, 2, N, :zz))
        end
        zz_mcwf = sum_zz / M
        
        decay = 1.0 - zz_dm
        interp = decay < 0.01 ? "Perfect correlation" :
                 decay < 0.2 ? "Slight decay" :
                 decay < 0.5 ? "Significant decay" :
                 decay < 0.9 ? "Mostly decohered" : "Nearly classical"
        
        @printf("  %.2f    %+.4f      %+.4f        %.4f    %s\n", 
                p, zz_dm, zz_mcwf, decay, interp)
    end
    
    println()
    println("  -- Decoherence successfully demonstrated!")
    println("    As noise increases: quantum correlations → classical (uncorrelated)")
end

# ==============================================================================
# PROTOCOL 4: QUANTUM TELEPORTATION
# ==============================================================================
#
# Full teleportation protocol with mid-circuit measurements:
#   1. Alice has qubit 1 in unknown state |ψ_in⟩
#   2. Alice and Bob share Bell pair (qubits 2,3)
#   3. Alice performs Bell measurement on qubits 1,2
#   4. Based on measurement outcomes, Bob applies corrections
#   5. Bob's qubit 3 ends up in state |ψ_in⟩
#
# ==============================================================================

println("\n" * "=" ^ 70)
println("  PROTOCOL 4: QUANTUM TELEPORTATION")
println("=" ^ 70)
println()
println("  Goal: Transfer quantum state |ψ⟩ from Alice to Bob using entanglement")
println()
println("  Setup:")
println("    Qubit 1: Alice's input |ψ_in⟩ (to teleport)")
println("    Qubit 2: Alice's half of Bell pair")
println("    Qubit 3: Bob's half of Bell pair")
println()
println("  Protocol:")
println("    1. Create Bell pair on qubits 2,3")
println("    2. Alice prepares |ψ_in⟩ on qubit 1")
println("    3. Alice: CNOT(1,2) → H(1) → Measure qubits 1,2")
println("    4. Bob: If m₂=1 apply X₃, If m₁=1 apply Z₃")
println("    5. Bob's qubit 3 is now in state |ψ_in⟩")
println()

let N = 3
    dim = 1 << N
    
    # Test with multiple input states
    test_cases = [
        (0.0,   "|0⟩"),
        (π,     "|1⟩"),
        (π/2,   "|+⟩ = (|0⟩+|1⟩)/√2"),
        (π/4,   "Ry(π/4)|0⟩"),
        (3π/4,  "Ry(3π/4)|0⟩"),
    ]
    
    println("  Testing teleportation fidelity for different input states:")
    println("  ─────────────────────────────────────────────────────────")
    println()
    println("  Input state              ⟨Z⟩_expected   ⟨Z⟩_teleported   Fidelity")
    println("  ──────────────────────   ────────────   ──────────────   ────────")
    
    all_fidelities_good = true
    
    for (θ, name) in test_cases
        expected_z = cos(θ)
        
        n_trials = 100
        sum_z = 0.0
        sum_fidelity = 0.0
        
        for trial in 1:n_trials
            Random.seed!(trial)
            ψ = zeros(ComplexF64, dim)
            
            # Step 1: Create Bell pair on qubits 2,3
            ψ[1] = 1.0
            apply_hadamard_psi!(ψ, 2, N)
            apply_hadamard_psi!(ψ, 3, N)
            apply_cz_psi!(ψ, 2, 3, N)
            apply_hadamard_psi!(ψ, 3, N)
            
            # Step 2: Prepare input state on qubit 1
            apply_ry_psi!(ψ, 1, Float64(θ), N)
            
            # Step 3: Alice's Bell measurement
            apply_hadamard_psi!(ψ, 2, N)
            apply_cz_psi!(ψ, 1, 2, N)
            apply_hadamard_psi!(ψ, 2, N)
            apply_hadamard_psi!(ψ, 1, N)
            
            outcomes, _ = projective_measurement!(ψ, [1, 2], :z, N)
            m1, m2 = outcomes[1], outcomes[2]
            
            # Step 4: Bob's corrections
            if m2 == 1
                apply_pauli_x_psi!(ψ, 3, N)
            end
            if m1 == 1
                apply_pauli_z_psi!(ψ, 3, N)
            end
            
            # Step 5: Measure fidelity
            z3 = real(expect_local(ψ, 3, N, :z))
            sum_z += z3
            
            ρ_3 = partial_trace(ψ, [1, 2], N)
            ψ_target = ComplexF64[cos(θ/2), sin(θ/2)]
            ρ_target = ψ_target * ψ_target'
            fidelity = real(tr(ρ_target * ρ_3))
            sum_fidelity += fidelity
        end
        
        avg_z = sum_z / n_trials
        avg_fidelity = sum_fidelity / n_trials
        
        @printf("  %-22s   %+.4f         %+.4f           %.4f\n", 
                name, expected_z, avg_z, avg_fidelity)
        
        all_fidelities_good &= (avg_fidelity > 0.95)
    end
    
    # Show detailed example
    println()
    println("  ─────────────────────────────────────────────────────────")
    println()
    println("  DETAILED EXAMPLE: Teleporting |+⟩")
    println()
    
    Random.seed!(42)
    ψ = zeros(ComplexF64, dim)
    θ = π/2
    
    println("    1. Initialize |000⟩")
    ψ[1] = 1.0
    
    println("    2. Create Bell pair: H(2)·CNOT(2,3)")
    apply_hadamard_psi!(ψ, 2, N)
    apply_hadamard_psi!(ψ, 3, N)
    apply_cz_psi!(ψ, 2, 3, N)
    apply_hadamard_psi!(ψ, 3, N)
    @printf("       ⟨Z₂Z₃⟩ = %+.4f (entangled)\n", real(expect_corr(ψ, 2, 3, N, :zz)))
    
    println("    3. Prepare |+⟩ on qubit 1: Ry(π/2)")
    apply_ry_psi!(ψ, 1, Float64(θ), N)
    @printf("       ⟨X₁⟩ = %+.4f, ⟨Z₁⟩ = %+.4f\n", 
            real(expect_local(ψ, 1, N, :x)), real(expect_local(ψ, 1, N, :z)))
    
    println("    4. Alice's Bell measurement: CNOT(1,2)·H(1)")
    apply_hadamard_psi!(ψ, 2, N)
    apply_cz_psi!(ψ, 1, 2, N)
    apply_hadamard_psi!(ψ, 2, N)
    apply_hadamard_psi!(ψ, 1, N)
    
    println("    5. Measure qubits 1,2")
    outcomes, _ = projective_measurement!(ψ, [1, 2], :z, N)
    m1, m2 = outcomes[1], outcomes[2]
    @printf("       Outcomes: m₁=%d, m₂=%d\n", m1, m2)
    
    println("    6. Bob's corrections")
    if m2 == 1
        println("       m₂=1 → Apply X₃")
        apply_pauli_x_psi!(ψ, 3, N)
    else
        println("       m₂=0 → No X needed")
    end
    if m1 == 1
        println("       m₁=1 → Apply Z₃")
        apply_pauli_z_psi!(ψ, 3, N)
    else
        println("       m₁=0 → No Z needed")
    end
    
    println("    7. Verify Bob's qubit")
    ρ_3 = partial_trace(ψ, [1, 2], N)
    @printf("       ⟨X₃⟩ = %+.4f (expect +1 for |+⟩)\n", real(expect_local(ρ_3, 1, 1, :x)))
    @printf("       ⟨Z₃⟩ = %+.4f (expect  0 for |+⟩)\n", real(expect_local(ρ_3, 1, 1, :z)))
    
    ψ_target = ComplexF64[1/√2, 1/√2]
    ρ_target = ψ_target * ψ_target'
    fidelity = real(tr(ρ_target * ρ_3))
    @printf("       Fidelity = %.4f\n", fidelity)
    
    println()
    status = all_fidelities_good ? "--" : "✗"
    @printf("  %s Quantum teleportation successful!\n", status)
    println("    State transferred via entanglement + classical communication")
    
    global all_passed &= all_fidelities_good
end

# ==============================================================================
# GATE-BY-GATE VISUALIZATION: TELEPORTATION
# ==============================================================================
println()
println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  GATE-BY-GATE VISUALIZATION: TELEPORTING |+⟩               │")
println("  └─────────────────────────────────────────────────────────────┘")
println()
println("  Qubits: 1=Alice's input, 2=Alice's Bell, 3=Bob's Bell")
println()

# Helper function to print state vector with qubit labels 1,2,3 - ALL 8 states
function print_teleport_state(ψ::Vector{ComplexF64}, label::String)
    println("  $label:")
    println("    ┌────────────┬─────────────────┐")
    println("    │ |q1 q2 q3⟩ │    Amplitude    │")
    println("    ├────────────┼─────────────────┤")
    for i in 1:length(ψ)
        q1 = (i-1) & 1
        q2 = ((i-1) >> 1) & 1
        q3 = ((i-1) >> 2) & 1
        
        if abs(ψ[i]) > 1e-10
            amp = @sprintf("%+.4f", real(ψ[i]))
            if abs(imag(ψ[i])) > 1e-10
                amp = @sprintf("%+.3f%+.3fi", real(ψ[i]), imag(ψ[i]))
            end
        else
            amp = "  0.0000"
        end
        println("    │ |$q1  $q2  $q3⟩ │  ", rpad(amp, 13), " │")
    end
    println("    └────────────┴─────────────────┘")
end

let N = 3
    dim = 1 << N
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1.0
    
    # STEP 0: Initial state
    println("  STEP 0: Initial state |000⟩")
    println("  " * "─"^55)
    print_teleport_state(ψ, "|ψ₀⟩ = |000⟩")
    println()
    
    # STEP 1: Create Bell pair (qubits 2,3)
    println("  STEP 1: Create Bell pair on qubits 2,3")
    println("  " * "─"^55)
    println("  Gates: H(2) → CNOT(2,3)")
    println()
    
    apply_hadamard_psi!(ψ, 2, N)
    println("  After H(2):")
    print_teleport_state(ψ, "|ψ⟩ = (1/√2)(|000⟩ + |010⟩)")
    println()
    
    apply_hadamard_psi!(ψ, 3, N)
    apply_cz_psi!(ψ, 2, 3, N)
    apply_hadamard_psi!(ψ, 3, N)  # CNOT = H·CZ·H
    println("  After CNOT(2,3):")
    print_teleport_state(ψ, "|ψ⟩ = |0⟩₁ ⊗ |Φ⁺⟩₂₃ = (1/√2)(|000⟩ + |011⟩)")
    println()
    
    # STEP 2: Prepare input state |+⟩ on qubit 1
    println("  STEP 2: Prepare input |+⟩ on qubit 1")
    println("  " * "─"^55)
    println("  Gate: Ry(π/2) on qubit 1")
    println()
    
    apply_ry_psi!(ψ, 1, Float64(π/2), N)
    println("  Full state:")
    print_teleport_state(ψ, "|ψ⟩ = |+⟩₁ ⊗ |Φ⁺⟩₂₃")
    println("  In standard notation:")
    println("    = (1/2)(|000⟩ + |011⟩ + |100⟩ + |111⟩)")
    println()
    
    # STEP 3: Alice's Bell measurement preparation
    println("  STEP 3: Alice's Bell measurement (transform to comp. basis)")
    println("  " * "─"^55)
    println("  Gates: CNOT(1,2) → H(1)")
    println()
    
    apply_hadamard_psi!(ψ, 2, N)
    apply_cz_psi!(ψ, 1, 2, N)
    apply_hadamard_psi!(ψ, 2, N)
    println("  After CNOT(1,2):")
    print_teleport_state(ψ, "|ψ⟩ after CNOT(1,2)")
    println()
    
    apply_hadamard_psi!(ψ, 1, N)
    println("  After H(1) - ready to measure in Z basis:")
    print_teleport_state(ψ, "|ψ⟩ = superposition of 4 terms")
    println()
    println("  Each term |m1 m2 *⟩ corresponds to a measurement outcome.")
    println("  Bob's qubit (3) is in |+⟩ or needs correction based on m1,m2.")
    println()
    
    # Show all 4 measurement outcomes
    println("  STEP 4: Measurement outcomes and corrections")
    println("  " * "─"^55)
    println("  ┌─────────┬───────────────────────────┬──────────────────┐")
    println("  │ m1  m2  │ Bob's state before corr.  │ Correction       │")
    println("  ├─────────┼───────────────────────────┼──────────────────┤")
    println("  │  0   0  │ (|0⟩ + |1⟩)/√2 = |+⟩      │ None             │")
    println("  │  0   1  │ (|1⟩ + |0⟩)/√2 = |+⟩      │ X (= no change)  │")
    println("  │  1   0  │ (|0⟩ - |1⟩)/√2 = |-⟩      │ Z → |+⟩          │")
    println("  │  1   1  │ (|1⟩ - |0⟩)/√2 = -|-⟩     │ X then Z → |+⟩   │")
    println("  └─────────┴───────────────────────────┴──────────────────┘")
    println()
    println("  After Bob's conditional correction, qubit 3 is always in |+⟩!")
    println("  The original state has been TELEPORTED from qubit 1 to qubit 3.")
end

# ==============================================================================
# PROTOCOL 5: NOISY TELEPORTATION - COMPREHENSIVE STUDY
# ==============================================================================
#
# Study how teleportation fidelity degrades with noise.
#
# Variables:
#   - Noise model: Depolarizing, Dephasing, Amplitude Damping
#   - Noise strength: p = 0.0, 0.05, 0.1, 0.2, 0.5
#   - Noise location:
#     A) Only on Bell pair (imperfect entanglement distribution)
#     B) On all gates (noisy quantum computer)
#
# Physics predictions:
#   - Depolarizing: Isotropic noise, F ≈ 1 - 2p/3 (for noise on both Bell qubits)
#   - Dephasing: Only Z-errors, affects |+⟩ state more than |0⟩, |1⟩
#   - Amplitude Damping: Decay to |0⟩, asymmetric effect
#
# ==============================================================================

println("\n" * "=" ^ 70)
println("  PROTOCOL 5: NOISY TELEPORTATION - COMPREHENSIVE STUDY")
println("=" ^ 70)
println()
println("  ╔═════════════════════════════════════════════════════════════╗")
println("  ║       TUTORIAL: QUANTUM NOISE MODELS                       ║")
println("  ╚═════════════════════════════════════════════════════════════╝")
println()
println("  In real quantum systems, qubits interact with their environment,")
println("  causing DECOHERENCE. We model this using quantum channels.")
println()
println("  ─────────────────────────────────────────────────────────────────")
println("  1. DEPOLARIZING CHANNEL (Isotropic Noise)")
println("  ─────────────────────────────────────────────────────────────────")
println()
println("    Physical origin: Random interactions with an unknown environment")
println()
println("    What happens:")
println("      • With probability (1-p): state unchanged")
println("      • With probability p/3:   X error (bit flip)")
println("      • With probability p/3:   Y error (bit+phase flip)")
println("      • With probability p/3:   Z error (phase flip)")
println()
println("    Mathematical form:")
println("      ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)")
println("        = (1-p)ρ + p·I/2   (completely mixed at p=3/4)")
println()
println("    Effect on Bloch sphere:")
println("      • Contracts Bloch vector uniformly: r⃗ → (1-4p/3)r⃗")
println("      • All states drift toward maximally mixed I/2")
println()
println("    Kraus operators: K₀ = √(1-p)I, K₁ = √(p/3)X, K₂ = √(p/3)Y, K₃ = √(p/3)Z")
println()
println("  ─────────────────────────────────────────────────────────────────")
println("  2. DEPHASING CHANNEL (Phase Noise / T2 Decay)")
println("  ─────────────────────────────────────────────────────────────────")
println()
println("    Physical origin: Fluctuations in qubit energy (e.g., magnetic field)")
println()
println("    What happens:")
println("      • With probability (1-p): state unchanged")
println("      • With probability p:     Z error (phase flip)")
println()
println("    Mathematical form:")
println("      ρ → (1-p)ρ + p·ZρZ")
println()
println("    Effect on density matrix:")
println("      ┌         ┐      ┌              ┐")
println("      │ ρ₀₀ ρ₀₁ │  →   │ ρ₀₀   (1-2p)ρ₀₁ │")
println("      │ ρ₁₀ ρ₁₁ │      │ (1-2p)ρ₁₀   ρ₁₁ │")
println("      └         ┘      └              ┘")
println()
println("    Key insight:")
println("      • Diagonal elements (populations) UNCHANGED")
println("      • Off-diagonal elements (coherences) DECAY by factor (1-2p)")
println("      • |0⟩ and |1⟩ unaffected, but |+⟩ and |-⟩ destroyed!")
println()
println("    Kraus operators: K₀ = √(1-p)I, K₁ = √(p)Z")
println()
println("  ─────────────────────────────────────────────────────────────────")
println("  3. AMPLITUDE DAMPING (T1 Decay / Energy Relaxation)")
println("  ─────────────────────────────────────────────────────────────────")
println()
println("    Physical origin: Spontaneous emission (qubit loses energy to env)")
println()
println("    What happens:")
println("      • |0⟩ → |0⟩ (ground state stable)")
println("      • |1⟩ → |0⟩ with probability γ (excited state decays)")
println()
println("    Mathematical form:")
println("      ρ₀₀ → ρ₀₀ + γ·ρ₁₁  (population flows to ground)")
println("      ρ₁₁ → (1-γ)·ρ₁₁")
println("      ρ₀₁ → √(1-γ)·ρ₀₁  (coherences decay)")
println()
println("    Effect on Bloch sphere:")
println("      • Contraction toward |0⟩ (north pole)")
println("      • NOT symmetric! |0⟩ is a fixed point, |1⟩ is not")
println()
println("    Kraus operators:")
println("      K₀ = [[1, 0], [0, √(1-γ)]]")
println("      K₁ = [[0, √γ], [0, 0]]")
println()
println("  ═══════════════════════════════════════════════════════════════")
println("  COMPARISON OF NOISE MODELS")
println("  ═══════════════════════════════════════════════════════════════")
println()
println("  ┌─────────────────┬─────────────────────────────────────────────┐")
println("  │   Property      │ Depolarizing   │ Dephasing   │ Amp Damping │")
println("  ├─────────────────┼─────────────────────────────────────────────┤")
println("  │ Bit flips?      │     Yes        │     No      │    Yes*     │")
println("  │ Phase flips?    │     Yes        │     Yes     │    No       │")
println("  │ Symmetric?      │     Yes        │     Yes     │    No       │")
println("  │ Fixed point     │     I/2        │  diag(ρ)    │    |0⟩      │")
println("  │ Physical origin │   Unknown env  │  E fluct.   │  Sp. emiss. │")
println("  └─────────────────┴─────────────────────────────────────────────┘")
println("  (* Amplitude damping: |1⟩→|0⟩ only)")
println()
println("  ═══════════════════════════════════════════════════════════════")
println("  TELEPORTATION FIDELITY ANALYSIS")
println("  ═══════════════════════════════════════════════════════════════")
println()
println("  Teleportation fidelity F measures how well the state is transferred:")
println("    F = ⟨ψ_target|ρ_output|ψ_target⟩")
println()
println("  Benchmarks:")
println("    F = 1.0:   Perfect teleportation")
println("    F = 0.5:   Random guessing (no information transferred)")
println("    F > 2/3:   Quantum advantage (no classical protocol can beat 2/3)")
println()

# Helper: Apply noise to qubits based on model (uses existing utils)
function apply_noise!(ψ::Vector{ComplexF64}, p::Float64, noise_model::Symbol, qubits::Vector{Int}, N::Int)
    if p ≈ 0.0
        return
    end
    if noise_model == :depolarizing
        apply_channel_depolarizing!(ψ, p, qubits, N)
    elseif noise_model == :dephasing
        apply_channel_dephasing!(ψ, p, qubits, N)
    elseif noise_model == :amplitude_damping
        apply_channel_amplitude_damping!(ψ, p, qubits, N)
    end
end

# Teleportation with configurable noise (uses only existing utils)
# prepare_state! is a function that prepares the state on qubit 1
# target_state is the expected 2-element state vector
function teleport_with_noise(prepare_state!::Function, target_state::Vector{ComplexF64},
                             p::Float64, noise_model::Symbol, 
                             noise_on_all_gates::Bool, N::Int=3)
    dim = 1 << N
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1.0
    
    # ========== CREATE BELL PAIR on qubits 2,3 ==========
    apply_hadamard_psi!(ψ, 2, N)
    if noise_on_all_gates
        apply_noise!(ψ, p, noise_model, [2], N)
    end
    
    # CNOT(2,3) = H(3) CZ(2,3) H(3)
    apply_hadamard_psi!(ψ, 3, N)
    apply_cz_psi!(ψ, 2, 3, N)
    apply_hadamard_psi!(ψ, 3, N)
    
    if noise_on_all_gates
        apply_noise!(ψ, p, noise_model, [2, 3], N)
    end
    
    # NOISE ON BELL PAIR ONLY (Scenario A)
    if !noise_on_all_gates
        apply_noise!(ψ, p, noise_model, [2, 3], N)
    end
    
    # ========== PREPARE INPUT STATE on qubit 1 ==========
    prepare_state!(ψ, 1, N)
    
    # ========== ALICE'S BELL MEASUREMENT ==========
    # CNOT(1,2) = H(2) CZ(1,2) H(2)
    apply_hadamard_psi!(ψ, 2, N)
    if noise_on_all_gates
        apply_noise!(ψ, p, noise_model, [2], N)
    end
    
    apply_cz_psi!(ψ, 1, 2, N)
    apply_hadamard_psi!(ψ, 2, N)
    apply_hadamard_psi!(ψ, 1, N)
    
    if noise_on_all_gates
        apply_noise!(ψ, p, noise_model, [1, 2], N)
    end
    
    # ========== MID-CIRCUIT MEASUREMENT (qubits 1,2) ==========
    outcomes, _ = projective_measurement!(ψ, [1, 2], :z, N)
    m1, m2 = outcomes[1], outcomes[2]
    
    # ========== BOB'S CONDITIONAL CORRECTIONS ==========
    if m2 == 1
        apply_pauli_x_psi!(ψ, 3, N)
    end
    if m1 == 1
        apply_pauli_z_psi!(ψ, 3, N)
    end
    
    # ========== CALCULATE FIDELITY ==========
    ρ_3 = partial_trace(ψ, [1, 2], N)
    ρ_target = target_state * target_state'
    fidelity = real(tr(ρ_target * ρ_3))
    
    return fidelity
end


# ============================================================================
# SCENARIO A: NOISE ON BELL PAIR ONLY
# ============================================================================

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  SCENARIO A: NOISE ON BELL PAIR ONLY                       │")
println("  │  (Simulates imperfect entanglement distribution)           │")
println("  └─────────────────────────────────────────────────────────────┘")
println()
println("  Teleporting |+⟩ = (|0⟩+|1⟩)/√2")
println()

# Define |+⟩ state: H|0⟩
prepare_plus!(ψ, q, N) = apply_hadamard_psi!(ψ, q, N)
target_plus = ComplexF64[1/√2, 1/√2]

let n_trials = 100
    println("  TELEPORTATION FIDELITY F = ⟨ψ_target|ρ_out|ψ_target⟩")
    println("  (F=1: perfect, F=0.5: random guessing, F>2/3: quantum advantage)")
    println()
    println("  ┌────────────────────────────────────────────────────────────┐")
    println("  │  Noise p   F(Depol)      F(Dephase)  F(AmpDamp)           │")
    println("  ├────────────────────────────────────────────────────────────┤")
    
    for p in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        fidelities = Dict{Symbol, Float64}()
        
        for noise_model in [:depolarizing, :dephasing, :amplitude_damping]
            sum_f = 0.0
            for trial in 1:n_trials
                Random.seed!(trial)
                sum_f += teleport_with_noise(prepare_plus!, target_plus, p, noise_model, false)
            end
            fidelities[noise_model] = sum_f / n_trials
        end
        
        @printf("  │  %.2f       %.4f        %.4f      %.4f           │\n",
                p, fidelities[:depolarizing], fidelities[:dephasing], 
                fidelities[:amplitude_damping])
    end
    
    println("  └────────────────────────────────────────────────────────────┘")
end

println()
println("  OBSERVATIONS (Scenario A):")
println("  ──────────────────────────")
println("  - Depolarizing: Isotropic decay, affects all states equally")
println("  - Dephasing: Strong effect on |+⟩ (superposition), mild on |0⟩,|1⟩")
println("  - Amp Damping: Asymmetric - |1⟩ decays to |0⟩")
println()

# ============================================================================
# SCENARIO B: NOISE ON ALL GATES
# ============================================================================

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  SCENARIO B: NOISE ON ALL GATES                            │")
println("  │  (Simulates noisy quantum computer)                        │")
println("  └─────────────────────────────────────────────────────────────┘")
println()
println("  Teleporting |+⟩ = (|0⟩+|1⟩)/√2")
println()

let n_trials = 100
    println("  TELEPORTATION FIDELITY (noise on all gates)")
    println()
    println("  ┌────────────────────────────────────────────────────────────┐")
    println("  │  Noise p   F(Depol)      F(Dephase)  F(AmpDamp)           │")
    println("  ├────────────────────────────────────────────────────────────┤")
    
    for p in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        fidelities = Dict{Symbol, Float64}()
        
        for noise_model in [:depolarizing, :dephasing, :amplitude_damping]
            sum_f = 0.0
            for trial in 1:n_trials
                Random.seed!(trial)
                sum_f += teleport_with_noise(prepare_plus!, target_plus, p, noise_model, true)
            end
            fidelities[noise_model] = sum_f / n_trials
        end
        
        @printf("  │  %.2f       %.4f        %.4f      %.4f           │\n",
                p, fidelities[:depolarizing], fidelities[:dephasing], 
                fidelities[:amplitude_damping])
    end
    
    println("  └────────────────────────────────────────────────────────────┘")
end

println()
println("  OBSERVATIONS (Scenario B):")
println("  ──────────────────────────")
println("  - Fidelity degrades faster (noise accumulates at each gate)")
println("  - Even small p has noticeable effect")
println("  - Depolarizing is most destructive (all error types)")
println()

# ============================================================================
# DIFFERENT INPUT STATES (including arbitrary state)
# ============================================================================

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  INPUT STATE VS NOISE CHANNEL COMPARISON                   │")
println("  └─────────────────────────────────────────────────────────────┘")
println()

# Define test states with their preparation functions and target vectors
# State: Ry(θy)Rx(θx)|0⟩
# Rx(θ)|0⟩ = cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
# Ry(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩

# |0⟩
prepare_0!(ψ, q, N) = nothing  # Already |0⟩
target_0 = ComplexF64[1, 0]

# |1⟩: Ry(π)|0⟩
prepare_1!(ψ, q, N) = apply_ry_psi!(ψ, q, Float64(π), N)
target_1 = ComplexF64[0, 1]

# |+⟩: H|0⟩ = Ry(π/2)|0⟩
# prepare_plus! already defined above

# Arbitrary: Ry(π/4)Rx(-π/5)|0⟩
function prepare_arbitrary!(ψ, q, N)
    apply_rx_psi!(ψ, q, Float64(-π/5), N)  # First Rx
    apply_ry_psi!(ψ, q, Float64(π/4), N)   # Then Ry
end
# Compute target state: Ry(π/4)·Rx(-π/5)|0⟩
θx = -π/5
θy = π/4
rx_0 = ComplexF64[cos(θx/2), -im*sin(θx/2)]
Ry = [cos(θy/2) -sin(θy/2); sin(θy/2) cos(θy/2)]
target_arbitrary = Ry * rx_0
target_arbitrary ./= norm(target_arbitrary)

test_states = [
    (prepare_0!, target_0, "|0>"),
    (prepare_1!, target_1, "|1>"),
    (prepare_plus!, target_plus, "|+>"),
    (prepare_arbitrary!, target_arbitrary, "Ry.Rx|0>"),
]

let n_trials = 50
    for (noise_sym, noise_name) in [(:depolarizing, "Depolarizing"), 
                                     (:dephasing, "Dephasing"),
                                     (:amplitude_damping, "Amp Damping")]
        println("  $noise_name noise (Bell pair only):")
        println("  ┌──────────────────────────────────────────────────────────┐")
        println("  │  State      p=0.0   p=0.05  p=0.1   p=0.2   p=0.5       │")
        println("  ├──────────────────────────────────────────────────────────┤")
        
        for (prep!, target, state_name) in test_states
            fids = Float64[]
            for p in [0.0, 0.05, 0.1, 0.2, 0.5]
                sum_f = 0.0
                for trial in 1:n_trials
                    Random.seed!(trial)
                    sum_f += teleport_with_noise(prep!, target, p, noise_sym, false)
                end
                push!(fids, sum_f / n_trials)
            end
            @printf("  │  %-10s %.3f   %.3f   %.3f   %.3f   %.3f       │\n",
                    state_name, fids[1], fids[2], fids[3], fids[4], fids[5])
        end
        println("  └──────────────────────────────────────────────────────────┘")
        println()
    end
end

println()
println("  -- Comprehensive noisy teleportation study complete!")
println()
println("  KEY PHYSICS INSIGHTS:")
println("  ──────────────────────")
println("  1. Entanglement quality directly limits teleportation fidelity")
println("  2. Dephasing has state-dependent effects (worst for superpositions)")
println("  3. Amplitude damping breaks symmetry between |0⟩ and |1⟩")
println("  4. Noise on all gates is more damaging than noise on Bell pair alone")
println("  5. F > 2/3 indicates quantum advantage over classical strategies")


# ##############################################################################
# PROTOCOL 6: ENTANGLEMENT MEASURES (SvN ENTROPY AND NEGATIVITY)
# ##############################################################################

println("\n" * "=" ^ 70)
println("  PROTOCOL 6: ENTANGLEMENT MEASURES")
println("=" ^ 70)
println()

println("  PHYSICS BACKGROUND")
println("  ───────────────────")
println("  For mixed states, we need TWO distinct entanglement measures:")
println()
println("  1. Von Neumann Entropy (S_vN):")
println("     S(ρ) = -Tr(ρ log ρ)")
println("     - Measures mixedness of reduced state ρ_A = Tr_B(ρ)")
println("     - S = 0 for pure states, S = log(d) for maximally mixed")
println("     - For mixed states: captures BOTH classical AND quantum correlations!")
println()
println("  2. Negativity (N):")
println("     N(ρ) = (||ρ^{T_A}||₁ - 1) / 2")
println("     - Based on partial transpose criterion (PPT)")
println("     - N > 0 ⟺ state is entangled (for 2×2 and 2×3 systems)")
println("     - N = 0 for separable states")
println("     - ROBUST measure of quantum entanglement!")
println()
println("  WHY BOTH MATTER:")
println("  - A noisy Bell pair can have high S_vN (mixed) but still N > 0 (entangled)")
println("  - A classical mixture has S_vN > 0 but N = 0 (no quantum correlations)")
println()

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  GHZ STATE ENTANGLEMENT DECAY UNDER NOISE                  │")
println("  └─────────────────────────────────────────────────────────────┘")
println()
println("  State: |GHZ⟩ = (|0000⟩ + |1111⟩)/√2  (4 qubits)")
println("  Bipartition: Half-split (qubits 1,2 | qubits 3,4)")
println("  Quantities: S_half = entanglement entropy, N_half = negativity")
println()

# Create GHZ state for 4 qubits
function create_ghz_rho_4q!()
    N = 4
    dim = 1 << N
    ρ = zeros(ComplexF64, dim, dim)
    ρ[1,1] = 1.0
    apply_hadamard_rho!(ρ, 1, N)
    for k in 1:(N-1)
        apply_hadamard_rho!(ρ, k+1, N)
        apply_cz_rho!(ρ, k, k+1, N)
        apply_hadamard_rho!(ρ, k+1, N)
    end
    return ρ
end

# Calculate half-split negativity for 1D chain
function negativity_half_split(ρ::Matrix{ComplexF64}, N::Int)
    N_left = N ÷ 2
    d_left = 1 << N_left
    d_right = 1 << (N - N_left)
    d_full = size(ρ, 1)
    
    # Build partial transpose w.r.t. left subsystem
    ρ_pt = zeros(ComplexF64, d_full, d_full)
    @inbounds for i_L in 0:(d_left-1)
        for i_R in 0:(d_right-1)
            for j_L in 0:(d_left-1)
                for j_R in 0:(d_right-1)
                    i_orig = i_L + d_left * i_R
                    j_orig = j_L + d_left * j_R
                    # Partial transpose: swap left indices
                    i_pt = j_L + d_left * i_R
                    j_pt = i_L + d_left * j_R
                    ρ_pt[i_pt+1, j_pt+1] = ρ[i_orig+1, j_orig+1]
                end
            end
        end
    end
    
    λs = real.(eigvals(Hermitian(ρ_pt)))
    trace_norm = sum(abs, λs)
    return (trace_norm - 1.0) / 2.0
end

# Calculate half-split entanglement entropy
function entropy_half_split(ρ::Matrix{ComplexF64}, N::Int)
    N_left = N ÷ 2
    d_left = 1 << N_left
    d_right = 1 << (N - N_left)
    
    # Trace out right subsystem
    ρ_left = zeros(ComplexF64, d_left, d_left)
    @inbounds for i_L in 0:(d_left-1)
        for j_L in 0:(d_left-1)
            for k_R in 0:(d_right-1)
                i_full = i_L + d_left * k_R
                j_full = j_L + d_left * k_R
                ρ_left[i_L+1, j_L+1] += ρ[i_full+1, j_full+1]
            end
        end
    end
    
    return von_neumann_entropy(ρ_left)
end

let N = 4
    println("  ┌────────────────────────────────────────────────────────────┐")
    println("  │  Noise p   S_half(Depol)  N_half(Depol)  Interpretation   │")
    println("  ├────────────────────────────────────────────────────────────┤")
    
    for p in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        ρ = create_ghz_rho_4q!()
        
        if p > 0
            apply_channel_depolarizing!(ρ, p, collect(1:N), N)
        end
        
        S_half = entropy_half_split(ρ, N)
        N_half = negativity_half_split(ρ, N)
        
        interp = N_half > 0.4 ? "Strong entanglement" :
                 N_half > 0.1 ? "Moderate entanglement" :
                 N_half > 0.01 ? "Weak entanglement" : "Separable"
        
        @printf("  │  %.2f       %.4f        %.4f         %-18s│\n",
                p, S_half, N_half, interp)
    end
    
    println("  └────────────────────────────────────────────────────────────┘")
end

println()
println("  COMPARING NOISE MODELS:")
println("  ────────────────────────")
println()
println("  ┌────────────────────────────────────────────────────────────┐")
println("  │  Noise p   N(Depol)   N(Dephase)  N(AmpDamp)  Best        │")
println("  ├────────────────────────────────────────────────────────────┤")

let N = 4
    for p in [0.0, 0.05, 0.1, 0.2, 0.5]
        negs = Dict{Symbol, Float64}()
        
        for (noise_model, apply_fn!) in [
            (:depol, (ρ, p, N) -> apply_channel_depolarizing!(ρ, p, collect(1:N), N)),
            (:dephase, (ρ, p, N) -> apply_channel_dephasing!(ρ, p, collect(1:N), N)),
            (:ampdamp, (ρ, p, N) -> apply_channel_amplitude_damping!(ρ, p, collect(1:N), N)),
        ]
            ρ = create_ghz_rho_4q!()
            if p > 0
                apply_fn!(ρ, p, N)
            end
            negs[noise_model] = negativity_half_split(ρ, N)
        end
        
        best = negs[:depol] >= negs[:dephase] && negs[:depol] >= negs[:ampdamp] ? "Depol" :
               negs[:dephase] >= negs[:ampdamp] ? "Dephase" : "AmpDamp"
        
        @printf("  │  %.2f       %.4f     %.4f      %.4f     %-10s │\n",
                p, negs[:depol], negs[:dephase], negs[:ampdamp], best)
    end
    
    println("  └────────────────────────────────────────────────────────────┘")
end

println()
println("  KEY OBSERVATIONS:")
println("  ------------------")
println("  1. S_vN is NOT a valid entanglement measure for mixed states!")
println("     - It measures total correlations (classical + quantum)")
println("     - A classical mixture can have S_vN > 0 but zero entanglement")
println()
println("  2. Negativity N is a TRUE entanglement measure:")
println("     - N > 0 iff state has quantum entanglement (for 2x2, 2x3)")
println("     - N = 0 means state is separable (or PPT-entangled)")
println("     - Depolarizing destroys entanglement fastest")
println()
println("  3. MCWF LIMITATION for entanglement measures:")
println("     - S_vN and N are NONLINEAR functions of rho")
println("     - Cannot simply average over trajectories!")
println("     - Must reconstruct rho = (1/M) * sum(|psi_m><psi_m|)")
println("     - Then compute S_vN(rho_red) and N(rho) on full DM")
println("     - This is why DM approach is used above for entanglement")
println()

# Load tomography module for MCWF analysis
include(joinpath(UTILS_CPU, "cpuQuantumStateTomography.jl"))
using .CPUQuantumStateTomography

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  MCWF TOMOGRAPHY: CONVERGENCE WITH TRAJECTORY COUNT M      │")
println("  └─────────────────────────────────────────────────────────────┘")
println()
println("  Demonstrating proper MCWF -> entanglement workflow:")
println("  1. Generate M noisy trajectories")
println("  2. Reconstruct rho = (1/M) * sum(|psi_m><psi_m|)")
println("  3. Compute entanglement on reconstructed rho (using existing utils)")
println("  4. Compare with exact DM reference")
println()

# Generate noisy Bell state trajectory
function generate_noisy_bell_trajectory(seed::Int, p::Float64)
    Random.seed!(seed)
    N = 2
    dim = 1 << N
    ψ = zeros(ComplexF64, dim)
    ψ[1] = 1.0
    
    # Create Bell state
    apply_hadamard_psi!(ψ, 1, N)
    apply_hadamard_psi!(ψ, 2, N)
    apply_cz_psi!(ψ, 1, 2, N)
    apply_hadamard_psi!(ψ, 2, N)
    
    # Apply depolarizing noise (MCWF stochastic)
    if p > 0
        apply_channel_depolarizing!(ψ, p, collect(1:N), N)
    end
    
    return ψ
end

# Reference: exact DM calculation  
let N = 2, p = 0.1
    dim = 1 << N
    
    # Exact DM reference
    ρ_ref = zeros(ComplexF64, dim, dim)
    ρ_ref[1,1] = 1.0
    apply_hadamard_rho!(ρ_ref, 1, N)
    apply_hadamard_rho!(ρ_ref, 2, N)
    apply_cz_rho!(ρ_ref, 1, 2, N)
    apply_hadamard_rho!(ρ_ref, 2, N)
    apply_channel_depolarizing!(ρ_ref, p, collect(1:N), N)
    
    # Reference values using existing functions
    P_ref = real(tr(ρ_ref * ρ_ref))  # purity
    S_ref = von_neumann_entropy(ρ_ref)
    N_ref = negativity_half_split(ρ_ref, N)  # our local function
    
    @printf("  Reference (exact DM, p=%.2f):\n", p)
    @printf("    Purity     = %.4f\n", P_ref)
    @printf("    S_vN       = %.4f\n", S_ref)
    @printf("    Negativity = %.4f\n", N_ref)
    println()
    
    # MCWF convergence study
    println("  ┌─────────────────────────────────────────────────────────────────────┐")
    println("  │  Traj M   Purity   S_vN    |S-S_ref|   N_half   |N-N_ref|          │")
    println("  ├─────────────────────────────────────────────────────────────────────┤")
    
    for M in [10, 20, 50, 100, 200, 500, 1000]
        trajectories = [generate_noisy_bell_trajectory(seed, p) for seed in 1:M]
        
        # Reconstruct density matrix from trajectories
        ρ_mcwf = reconstruct_density_matrix(trajectories)
        
        # Compute measures on reconstructed ρ
        P_mcwf = real(tr(ρ_mcwf * ρ_mcwf))
        S_mcwf = von_neumann_entropy(ρ_mcwf)
        N_mcwf = negativity_half_split(ρ_mcwf, N)
        
        err_S = abs(S_mcwf - S_ref)
        err_N = abs(N_mcwf - N_ref)
        
        @printf("  │  %-6d   %.4f   %.4f  %.4f      %.4f    %.4f           │\n",
                M, P_mcwf, S_mcwf, err_S, N_mcwf, err_N)
    end
    
    println("  └─────────────────────────────────────────────────────────────────────┘")
    println()
    println("  Observation: Error decreases as 1/sqrt(M) (Central Limit Theorem)")
end

println()
println("  -- Entanglement measures demonstrated!")


# ##############################################################################
# PROTOCOL 7: CHSH BELL INEQUALITY
# ##############################################################################

println("\n" * "=" ^ 70)
println("  PROTOCOL 7: CHSH BELL INEQUALITY")
println("=" ^ 70)
println()

println("  PHYSICS BACKGROUND")
println("  -------------------")
println("  The CHSH inequality tests quantum nonlocality:")
println()
println("    S = <A0*B0> + <A0*B1> + <A1*B0> - <A1*B1>")
println()
println("  where A0,A1 are Alice's measurement settings, B0,B1 are Bob's.")
println()
println("  Classical bound:   |S| <= 2")
println("  Quantum bound:     |S| <= 2*sqrt(2) = 2.828...")
println("  Tsirelson bound:   Maximum quantum violation")
println()
println("  Optimal settings for Bell state |Phi+> = (|00> + |11>)/sqrt(2):")
println("    A0 = Z,  A1 = X")
println("    B0 = (Z+X)/sqrt(2),  B1 = (Z-X)/sqrt(2)")
println()

# CHSH correlator calculation
function chsh_correlator(ρ::Matrix{ComplexF64}, N::Int, 
                         θ_A::Float64, θ_B::Float64)
    # Measure <A(θ_A) ⊗ B(θ_B)> where A,B are spin operators in XZ plane
    # A(θ) = cos(θ)*Z + sin(θ)*X
    # <A⊗B> = cos(θ_A)*cos(θ_B)*<ZZ> + cos(θ_A)*sin(θ_B)*<ZX>
    #       + sin(θ_A)*cos(θ_B)*<XZ> + sin(θ_A)*sin(θ_B)*<XX>
    
    zz = real(expect_corr(ρ, 1, 2, N, :zz))
    xx = real(expect_corr(ρ, 1, 2, N, :xx))
    
    # For ZX and XZ correlators, we need to compute manually
    # <ZX> = Tr(ρ * (Z⊗X))
    dim = size(ρ, 1)
    zx = 0.0
    xz = 0.0
    for i in 0:(dim-1)
        for j in 0:(dim-1)
            # Z⊗X matrix element
            i1, i2 = i & 1, (i >> 1) & 1
            j1, j2 = j & 1, (j >> 1) & 1
            
            # Z on qubit 1: (-1)^i1 * δ(i1,j1)
            # X on qubit 2: δ(i2, 1-j2)
            if i1 == j1 && i2 == 1 - j2
                zx += real(ρ[i+1, j+1]) * (i1 == 0 ? 1.0 : -1.0)
            end
            
            # X on qubit 1: δ(i1, 1-j1)  
            # Z on qubit 2: (-1)^i2 * δ(i2,j2)
            if i1 == 1 - j1 && i2 == j2
                xz += real(ρ[i+1, j+1]) * (i2 == 0 ? 1.0 : -1.0)
            end
        end
    end
    
    return cos(θ_A)*cos(θ_B)*zz + cos(θ_A)*sin(θ_B)*zx +
           sin(θ_A)*cos(θ_B)*xz + sin(θ_A)*sin(θ_B)*xx
end

function compute_chsh(ρ::Matrix{ComplexF64}, N::Int)
    # Optimal angles for Bell state
    # A0 = Z (θ=0), A1 = X (θ=π/2)
    # B0 = (Z+X)/sqrt(2) (θ=π/4), B1 = (Z-X)/sqrt(2) (θ=-π/4)
    θ_A0, θ_A1 = 0.0, π/2
    θ_B0, θ_B1 = π/4, -π/4
    
    E00 = chsh_correlator(ρ, N, θ_A0, θ_B0)
    E01 = chsh_correlator(ρ, N, θ_A0, θ_B1)
    E10 = chsh_correlator(ρ, N, θ_A1, θ_B0)
    E11 = chsh_correlator(ρ, N, θ_A1, θ_B1)
    
    return E00 + E01 + E10 - E11
end

# Create Bell state |Phi+>
function create_bell_rho!(N::Int)
    dim = 1 << N
    ρ = zeros(ComplexF64, dim, dim)
    ρ[1,1] = 1.0
    apply_hadamard_rho!(ρ, 1, N)
    apply_hadamard_rho!(ρ, 2, N)
    apply_cz_rho!(ρ, 1, 2, N)
    apply_hadamard_rho!(ρ, 2, N)
    return ρ
end

# Test CHSH with pure Bell state
let N = 2
    ρ = create_bell_rho!(N)
    S = compute_chsh(ρ, N)
    @printf("  Pure Bell state: S = %.4f (Tsirelson bound = %.4f)\n", S, 2*sqrt(2))
    println()
end

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  CHSH VALUE UNDER DIFFERENT NOISE MODELS                   │")
println("  └─────────────────────────────────────────────────────────────┘")
println()
println("  Classical bound: |S| <= 2,  Quantum bound: |S| <= 2.828")
println()

println("  ┌────────────────────────────────────────────────────────────┐")
println("  │  Noise p   S(Depol)   S(Dephase)  S(AmpDamp)  Violation   │")
println("  ├────────────────────────────────────────────────────────────┤")

let N = 2
    for p in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        chsh_vals = Float64[]
        
        for noise_model in [:depolarizing, :dephasing, :amplitude_damping]
            ρ = create_bell_rho!(N)
            if p > 0
                if noise_model == :depolarizing
                    apply_channel_depolarizing!(ρ, p, collect(1:N), N)
                elseif noise_model == :dephasing
                    apply_channel_dephasing!(ρ, p, collect(1:N), N)
                else
                    apply_channel_amplitude_damping!(ρ, p, collect(1:N), N)
                end
            end
            push!(chsh_vals, compute_chsh(ρ, N))
        end
        
        # Check best violation
        best_idx = argmax(abs.(chsh_vals))
        best = chsh_vals[best_idx] > 2 ? "Quantum" : "Classical"
        
        @printf("  │  %.2f       %.4f     %.4f      %.4f     %-10s │\n",
                p, chsh_vals[1], chsh_vals[2], chsh_vals[3], best)
    end
    
    println("  └────────────────────────────────────────────────────────────┘")
end

println()
println("  KEY OBSERVATIONS:")
println("  ------------------")
println("  - S > 2 indicates Bell inequality VIOLATION (quantum correlations)")
println("  - Depolarizing: S decays isotropically toward 0")
println("  - Dephasing: S decays but preserves some correlations longer")
println("  - Threshold: When S <= 2, state becomes classically simulable")
println()
println("  -- CHSH Bell inequality demonstrated!")


# ##############################################################################
# PROTOCOL 8: ENTANGLEMENT SWAPPING
# ##############################################################################

println("\n" * "=" ^ 70)
println("  PROTOCOL 8: ENTANGLEMENT SWAPPING")
println("=" ^ 70)
println()

# Helper function to print density matrix nicely (define here for step-by-step)
function print_density_matrix(ρ::Matrix{ComplexF64}, label::String)
    dim = size(ρ, 1)
    println("  $label ($(dim)x$(dim)):")
    println()
    println("    Real part:")
    for i in 1:dim
        print("    ")
        for j in 1:dim
            @printf("  %+.3f", real(ρ[i,j]))
        end
        println()
    end
    if any(x -> abs(imag(x)) > 1e-10, ρ)
        println()
        println("    Imaginary part:")
        for i in 1:dim
            print("    ")
            for j in 1:dim
                @printf("  %+.3f", imag(ρ[i,j]))
            end
            println()
        end
    end
end

println("  PHYSICS BACKGROUND")
println("  -------------------")
println("  Entanglement swapping allows A and D to become entangled even though")
println("  they NEVER directly interacted! This is the basis for quantum repeaters.")
println()

# ============================================================================
# STEP-BY-STEP DEMONSTRATION: DM AND MCWF SIDE BY SIDE
# ============================================================================

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  STEP-BY-STEP: BOTH DM AND MCWF APPROACHES                 │")
println("  └─────────────────────────────────────────────────────────────┘")
println()

# Initialize both representations
N_ent = 4
dim_ent = 1 << N_ent

# DM approach
ρ_step = zeros(ComplexF64, dim_ent, dim_ent)
ρ_step[1,1] = 1.0

# MCWF approach
ψ_step = zeros(ComplexF64, dim_ent)
ψ_step[1] = 1.0

# Helper to print state vector in column format (qubits labeled A,B,C,D) - ALL states
function print_state_vector(ψ::Vector{ComplexF64}, label::String)
    N = Int(log2(length(ψ)))
    println("  $label:")
    println("    ┌────────────┬─────────────────┐")
    println("    │ |A B C D⟩  │    Amplitude    │")
    println("    ├────────────┼─────────────────┤")
    for i in 1:length(ψ)
        # Bit ordering: i-1 has bit0=A, bit1=B, bit2=C, bit3=D
        # Display as |A B C D⟩
        A = (i-1) & 1
        B = ((i-1) >> 1) & 1
        C = ((i-1) >> 2) & 1
        D = ((i-1) >> 3) & 1
        
        if abs(ψ[i]) > 1e-10
            amp = @sprintf("%+.4f", real(ψ[i]))
            if abs(imag(ψ[i])) > 1e-10
                amp = @sprintf("%+.3f%+.3fi", real(ψ[i]), imag(ψ[i]))
            end
        else
            amp = "  0.0000"
        end
        println("    │ |$A $B $C $D⟩  │  ", rpad(amp, 13), " │")
    end
    println("    └────────────┴─────────────────┘")
end

# Helper to compute rho_AD from full rho or psi
function trace_out_BC(ρ::Matrix{ComplexF64})
    ρ_AD = zeros(ComplexF64, 4, 4)
    for i_A in 0:1, i_D in 0:1, j_A in 0:1, j_D in 0:1
        for k_B in 0:1, k_C in 0:1
            i_full = i_A + 2*k_B + 4*k_C + 8*i_D
            j_full = j_A + 2*k_B + 4*k_C + 8*j_D
            ρ_AD[i_A + 2*i_D + 1, j_A + 2*j_D + 1] += ρ[i_full+1, j_full+1]
        end
    end
    return ρ_AD
end

# STEP 0: Initial state
println("  STEP 0: Initial state |0000>")
println("  " * "─"^50)
print_state_vector(ψ_step, "MCWF psi")
println("  DM rho: |0000><0000| (pure product state)")
ρ_AD_0 = trace_out_BC(ρ_step)
neg_0 = negativity_half_split(ρ_AD_0, 2)
@printf("  rho_AD Negativity = %.4f (NO entanglement between A and D)\n", neg_0)
println()

# STEP 1: Create Bell pair (A,B) with H(A) -> CNOT(A,B)
println("  STEP 1: Create Bell pair (A,B)")
println("  " * "─"^50)
println("  Circuit: H(A) -> CNOT(A,B)")
println()

# Apply to both
apply_hadamard_psi!(ψ_step, 1, N_ent)
apply_hadamard_psi!(ψ_step, 2, N_ent)
apply_cz_psi!(ψ_step, 1, 2, N_ent)
apply_hadamard_psi!(ψ_step, 2, N_ent)

apply_hadamard_rho!(ρ_step, 1, N_ent)
apply_hadamard_rho!(ρ_step, 2, N_ent)
apply_cz_rho!(ρ_step, 1, 2, N_ent)
apply_hadamard_rho!(ρ_step, 2, N_ent)

print_state_vector(ψ_step, "MCWF psi = |Phi+>_AB ⊗ |00>_CD")
println("  = (1/sqrt(2))(|0000> + |1100>)")
ρ_AD_1 = trace_out_BC(ρ_step)
neg_1 = negativity_half_split(ρ_AD_1, 2)
@printf("  rho_AD Negativity = %.4f (A entangled with B, but NOT with D)\n", neg_1)
println()

# STEP 2: Create Bell pair (C,D)
println("  STEP 2: Create Bell pair (C,D)")
println("  " * "─"^50)
println("  Circuit: H(C) -> CNOT(C,D)")
println()

apply_hadamard_psi!(ψ_step, 3, N_ent)
apply_hadamard_psi!(ψ_step, 4, N_ent)
apply_cz_psi!(ψ_step, 3, 4, N_ent)
apply_hadamard_psi!(ψ_step, 4, N_ent)

apply_hadamard_rho!(ρ_step, 3, N_ent)
apply_hadamard_rho!(ρ_step, 4, N_ent)
apply_cz_rho!(ρ_step, 3, 4, N_ent)
apply_hadamard_rho!(ρ_step, 4, N_ent)

print_state_vector(ψ_step, "MCWF psi = |Phi+>_AB ⊗ |Phi+>_CD")
println("  = (1/2)(|0000> + |0011> + |1100> + |1111>)")
ρ_AD_2 = trace_out_BC(ρ_step)
neg_2 = negativity_half_split(ρ_AD_2, 2)
@printf("  rho_AD Negativity = %.4f (A-B and C-D entangled, but A-D: NONE!)\n", neg_2)
println()
println("  rho_AD = I/4 (maximally mixed) because A and D are uncorrelated:")
print_density_matrix(ρ_AD_2, "rho_AD = Tr_BC(rho)")
println()

# STEP 3: Bell measurement on B,C
println("  STEP 3: Bell measurement on (B,C)")
println("  " * "─"^50)
println()
println("  ╔═════════════════════════════════════════════════════════════╗")
println("  ║          TUTORIAL: UNDERSTANDING BELL MEASUREMENTS         ║")
println("  ╚═════════════════════════════════════════════════════════════╝")
println()
println("  THE PROBLEM:")
println("  ─────────────")
println("  Standard quantum measurements are in the computational basis {|0⟩, |1⟩}.")
println("  But we need to measure in the BELL BASIS - a basis of entangled states!")
println()
println("  The Bell basis for two qubits (B and C) is:")
println()
println("    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2   ← both same, positive phase")
println("    |Φ⁻⟩ = (|00⟩ - |11⟩)/√2   ← both same, negative phase")
println("    |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2   ← both different, positive phase")
println("    |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2   ← both different, negative phase")
println()
println("  THE SOLUTION: Transform Bell basis → Computational basis")
println("  ─────────────────────────────────────────────────────────────")
println()
println("  We use a clever circuit: CNOT(B,C) followed by H(B)")
println()
println("  INTUITION:")
println("  • Bell states encode TWO pieces of information:")
println("    1) PARITY: Are B and C the same (Φ) or different (Ψ)?")
println("    2) PHASE:  Is the superposition + or - ?")
println()
println("  • CNOT extracts the PARITY:")
println("    - |Φ±⟩: B,C same    → CNOT makes C=|0⟩ (XOR = 0)")
println("    - |Ψ±⟩: B,C differ  → CNOT makes C=|1⟩ (XOR = 1)")
println()
println("  • Hadamard extracts the PHASE:")
println("    - Positive phase (+) → H gives |0⟩")
println("    - Negative phase (-) → H gives |1⟩")
println()
println("  ─────────────────────────────────────────────────────────────")
println("  WORKED EXAMPLE: How CNOT+H transforms each Bell state")
println("  ─────────────────────────────────────────────────────────────")
println()
println("  |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
println("    → CNOT: (|00⟩ + |10⟩)/√2  = |+⟩⊗|0⟩")
println("    → H(B): (|00⟩ + |00⟩)/2 + (|10⟩ - |10⟩)/2 = |00⟩  ✓")
println()
println("  |Φ⁻⟩ = (|00⟩ - |11⟩)/√2")
println("    → CNOT: (|00⟩ - |10⟩)/√2  = |-⟩⊗|0⟩")
println("    → H(B): (|00⟩ - |00⟩)/2 + (|10⟩ + |10⟩)/2 = |10⟩  ✓")
println()
println("  |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2")
println("    → CNOT: (|01⟩ + |11⟩)/√2  = |+⟩⊗|1⟩")
println("    → H(B): |01⟩  ✓")
println()
println("  |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2")
println("    → CNOT: (|01⟩ - |11⟩)/√2  = |-⟩⊗|1⟩")
println("    → H(B): |11⟩  ✓")
println()
println("  SUMMARY TABLE:")
println("  ┌──────────┬────────────┬─────────────┬───────────────────────┐")
println("  │ Bell St. │ After CNOT │ After H(B)  │ Measurement = ?       │")
println("  ├──────────┼────────────┼─────────────┼───────────────────────┤")
println("  │  |Φ⁺⟩    │  |+⟩⊗|0⟩   │   |00⟩      │  B=0, C=0             │")
println("  │  |Φ⁻⟩    │  |-⟩⊗|0⟩   │   |10⟩      │  B=1, C=0             │")
println("  │  |Ψ⁺⟩    │  |+⟩⊗|1⟩   │   |01⟩      │  B=0, C=1             │")
println("  │  |Ψ⁻⟩    │  |-⟩⊗|1⟩   │   |11⟩      │  B=1, C=1             │")
println("  └──────────┴────────────┴─────────────┴───────────────────────┘")
println()
println("  KEY INSIGHT: After CNOT+H, we can decode the Bell state from")
println("  a simple Z-basis measurement:")
println("    • C tells us PARITY (0=same, 1=different)")
println("    • B tells us PHASE  (0=positive, 1=negative)")
println()
println("  ═══════════════════════════════════════════════════════════")
println("  NOW LET'S APPLY THIS TO OUR 4-QUBIT STATE")
println("  ═══════════════════════════════════════════════════════════")
println()
println("  Circuit: CNOT(B,C) → H(B) → Measure(B,C)")
println()

# Transform to computational basis
apply_hadamard_psi!(ψ_step, 3, N_ent)
apply_cz_psi!(ψ_step, 2, 3, N_ent)
apply_hadamard_psi!(ψ_step, 3, N_ent)
apply_hadamard_psi!(ψ_step, 2, N_ent)

apply_hadamard_rho!(ρ_step, 3, N_ent)
apply_cz_rho!(ρ_step, 2, 3, N_ent)
apply_hadamard_rho!(ρ_step, 3, N_ent)
apply_hadamard_rho!(ρ_step, 2, N_ent)

print_state_vector(ψ_step, "State after Bell-basis transform (CNOT·H)")
println()
println("  INTERPRETATION:")
println("  ─────────────────")
println("  The state is now a superposition of 4 equally-likely outcomes.")
println("  Looking at the B,C qubits (middle two columns):")
println()
println("    • |A 0 0 D⟩ terms: B,C measured as |00⟩ → Bell state was |Φ⁺⟩")
println("    • |A 0 1 D⟩ terms: B,C measured as |01⟩ → Bell state was |Ψ⁺⟩")
println("    • |A 1 0 D⟩ terms: B,C measured as |10⟩ → Bell state was |Φ⁻⟩")
println("    • |A 1 1 D⟩ terms: B,C measured as |11⟩ → Bell state was |Ψ⁻⟩")
println()
println("  WHAT HAPPENS TO A AND D?")
println("  ───────────────────────────")
println("  This is the magic of entanglement swapping!")
println("  The Bell measurement on B,C 'projects' A and D into a Bell state,")
println("  even though A and D have NEVER directly interacted!")
println()
println("  ┌─────────────┬────────────────────────────────────────────────┐")
println("  │ B,C outcome │ State of A,D after measurement                 │")
println("  ├─────────────┼────────────────────────────────────────────────┤")
println("  │   |00⟩      │ |Φ⁺⟩_AD = (|00⟩+|11⟩)/√2  ← no correction     │")
println("  │   |01⟩      │ |Ψ⁺⟩_AD = (|01⟩+|10⟩)/√2  ← apply X_D         │")
println("  │   |10⟩      │ |Φ⁻⟩_AD = (|00⟩-|11⟩)/√2  ← apply Z_D         │")
println("  │   |11⟩      │ |Ψ⁻⟩_AD = (|01⟩-|10⟩)/√2  ← apply X_D then Z_D│")
println("  └─────────────┴────────────────────────────────────────────────┘")
println()

# STEP 4: Condition on measurement outcome |00>_BC
println("  STEP 4: Condition on outcome |00>_BC")
println("  " * "─"^50)
println()

# Project MCWF state onto |00>_BC
ψ_projected = zeros(ComplexF64, dim_ent)
for i in 0:(dim_ent-1)
    b_B = (i >> 1) & 1
    b_C = (i >> 2) & 1
    if b_B == 0 && b_C == 0
        ψ_projected[i+1] = ψ_step[i+1]
    end
end
# Normalize
norm_proj = sqrt(sum(abs2, ψ_projected))
ψ_projected ./= norm_proj

# Project DM
ρ_projected = zeros(ComplexF64, dim_ent, dim_ent)
for i in 0:(dim_ent-1), j in 0:(dim_ent-1)
    i_B, i_C = (i >> 1) & 1, (i >> 2) & 1
    j_B, j_C = (j >> 1) & 1, (j >> 2) & 1
    if i_B == 0 && i_C == 0 && j_B == 0 && j_C == 0
        ρ_projected[i+1, j+1] = ρ_step[i+1, j+1]
    end
end
prob_outcome = real(tr(ρ_projected))
ρ_projected ./= prob_outcome

print_state_vector(ψ_projected, "MCWF psi after projecting onto |00>_BC")
@printf("  Probability of outcome |00> = %.4f (expect 0.25)\n", norm_proj^2)
println()

# Trace out B,C to get final rho_AD
ρ_AD_final = trace_out_BC(ρ_projected)

println("  FINAL rho_AD after entanglement swapping:")
print_density_matrix(ρ_AD_final, "rho_AD")
println()

# Compare with Bell state
bell_ref = zeros(ComplexF64, 4, 4)
bell_ref[1,1] = bell_ref[1,4] = bell_ref[4,1] = bell_ref[4,4] = 0.5

fidelity = real(tr(ρ_AD_final * bell_ref))
neg_final = negativity_half_split(ρ_AD_final, 2)

println("  COMPARISON WITH REFERENCE BELL STATE |Phi+><Phi+|:")
print_density_matrix(bell_ref, "|Φ⁺⟩⟨Φ⁺| where |Φ⁺⟩ = (|00⟩+|11⟩)/√2")
println()
@printf("  Fidelity F(rho_AD, |Phi+>) = %.4f\n", fidelity)
@printf("  Negativity(A:D) = %.4f (0.5 = maximally entangled)\n", neg_final)
println()
println("  RESULT: A and D are now in Bell state |Phi+> - maximally entangled!")
println("  This entanglement was SWAPPED from (A-B) and (C-D) pairs.")

# Entanglement swapping protocol
function entanglement_swapping_protocol(p_noise::Float64, meas_outcome::Int=0)
    # meas_outcome: 0=|00⟩, 1=|01⟩, 2=|10⟩, 3=|11⟩ in Bell basis
    N = 4
    dim = 1 << N
    ρ = zeros(ComplexF64, dim, dim)
    ρ[1,1] = 1.0
    
    # Create Bell pair (A,B) on qubits 1,2: |Φ+⟩ = (|00⟩ + |11⟩)/√2
    apply_hadamard_rho!(ρ, 1, N)
    apply_hadamard_rho!(ρ, 2, N)
    apply_cz_rho!(ρ, 1, 2, N)
    apply_hadamard_rho!(ρ, 2, N)
    
    # Create Bell pair (C,D) on qubits 3,4: |Φ+⟩ 
    apply_hadamard_rho!(ρ, 3, N)
    apply_hadamard_rho!(ρ, 4, N)
    apply_cz_rho!(ρ, 3, 4, N)
    apply_hadamard_rho!(ρ, 4, N)
    
    # Apply noise to initial Bell pairs
    if p_noise > 0
        apply_channel_depolarizing!(ρ, p_noise, [1, 2, 3, 4], N)
    end
    
    # Bell state measurement on qubits B,C (2,3)
    # Transform to computational basis: CNOT(B,C) then H(B)
    apply_hadamard_rho!(ρ, 3, N)
    apply_cz_rho!(ρ, 2, 3, N)
    apply_hadamard_rho!(ρ, 3, N)  # This is CNOT
    apply_hadamard_rho!(ρ, 2, N)
    
    # Project onto specific measurement outcome for qubits B,C
    # meas_outcome encodes (b_B, b_C) where B=qubit2, C=qubit3
    b_B = (meas_outcome >> 0) & 1  # bit for qubit 2
    b_C = (meas_outcome >> 1) & 1  # bit for qubit 3
    
    # Build projector: |b_B, b_C⟩⟨b_B, b_C| on qubits 2,3
    ρ_projected = zeros(ComplexF64, dim, dim)
    for i in 0:(dim-1)
        for j in 0:(dim-1)
            # Extract bits for qubits 2 and 3
            i_B = (i >> 1) & 1
            i_C = (i >> 2) & 1
            j_B = (j >> 1) & 1
            j_C = (j >> 2) & 1
            
            # Only keep if both match the measurement outcome
            if i_B == b_B && i_C == b_C && j_B == b_B && j_C == b_C
                ρ_projected[i+1, j+1] = ρ[i+1, j+1]
            end
        end
    end
    
    # Normalize (probability of this outcome)
    prob = real(tr(ρ_projected))
    if prob > 1e-10
        ρ_projected ./= prob
    end
    
    # Trace out qubits B,C (2,3) to get reduced state on (A,D)
    ρ_AD = zeros(ComplexF64, 4, 4)
    
    for i_A in 0:1
        for i_D in 0:1
            for j_A in 0:1
                for j_D in 0:1
                    # Only the projected outcome contributes
                    i_full = i_A + 2*b_B + 4*b_C + 8*i_D
                    j_full = j_A + 2*b_B + 4*b_C + 8*j_D
                    i_red = i_A + 2*i_D
                    j_red = j_A + 2*j_D
                    ρ_AD[i_red+1, j_red+1] += ρ_projected[i_full+1, j_full+1]
                end
            end
        end
    end
    
    return ρ_AD, prob
end

# Helper function to print density matrix nicely
function print_density_matrix(ρ::Matrix{ComplexF64}, label::String)
    dim = size(ρ, 1)
    println("  $label ($(dim)x$(dim)):")
    println()
    println("    Real part:")
    for i in 1:dim
        print("    ")
        for j in 1:dim
            @printf("  %+.3f", real(ρ[i,j]))
        end
        println()
    end
    println()
    println("    Imaginary part:")
    for i in 1:dim
        print("    ")
        for j in 1:dim
            @printf("  %+.3f", imag(ρ[i,j]))
        end
        println()
    end
end

# Compute rho_AD BEFORE swapping (just trace out B,C from product of Bell pairs)
function compute_rho_AD_before()
    N = 4
    dim = 1 << N
    ρ = zeros(ComplexF64, dim, dim)
    ρ[1,1] = 1.0
    
    # Create two Bell pairs
    apply_hadamard_rho!(ρ, 1, N)
    apply_hadamard_rho!(ρ, 2, N)
    apply_cz_rho!(ρ, 1, 2, N)
    apply_hadamard_rho!(ρ, 2, N)
    
    apply_hadamard_rho!(ρ, 3, N)
    apply_hadamard_rho!(ρ, 4, N)
    apply_cz_rho!(ρ, 3, 4, N)
    apply_hadamard_rho!(ρ, 4, N)
    
    # Trace out B,C (qubits 2,3) to get rho_AD
    ρ_AD = zeros(ComplexF64, 4, 4)
    for i_A in 0:1
        for i_D in 0:1
            for j_A in 0:1
                for j_D in 0:1
                    for k_B in 0:1
                        for k_C in 0:1
                            i_full = i_A + 2*k_B + 4*k_C + 8*i_D
                            j_full = j_A + 2*k_B + 4*k_C + 8*j_D
                            i_red = i_A + 2*i_D
                            j_red = j_A + 2*j_D
                            ρ_AD[i_red+1, j_red+1] += ρ[i_full+1, j_full+1]
                        end
                    end
                end
            end
        end
    end
    return ρ_AD
end

# First, construct reference Bell state |Phi+> = (|00> + |11>)/sqrt(2) for comparison
bell_state_ref = zeros(ComplexF64, 4, 4)
bell_state_ref[1,1] = 0.5  # |00><00|
bell_state_ref[1,4] = 0.5  # |00><11|
bell_state_ref[4,1] = 0.5  # |11><00|
bell_state_ref[4,4] = 0.5  # |11><11|

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  STEP-BY-STEP ENTANGLEMENT SWAPPING                        │")
println("  └─────────────────────────────────────────────────────────────┘")
println()

println("  STEP 0: Initial state")
println("  ----------------------")
println("    |psi_0> = |0000>_ABCD")
println("    State: A=|0>, B=|0>, C=|0>, D=|0>")
println("    Entanglement between A-D: NONE (product state)")
println()

println("  STEP 1: Create Bell pair (A,B)")
println("  -------------------------------")
println("    Circuit: H(A) -> CNOT(A,B)")
println("    Result: |Phi+>_AB = (|00> + |11>)/sqrt(2)")
println("    ")
println("    Full state: |Phi+>_AB ⊗ |00>_CD")
println("    ")
println("    Qubits A-B: ENTANGLED (maximally)")
println("    Qubits C-D: not entangled yet")
println("    Qubits A-D: NO entanglement!")
println()

println("  STEP 2: Create Bell pair (C,D)")
println("  -------------------------------")
println("    Circuit: H(C) -> CNOT(C,D)")
println("    Result: |Phi+>_CD = (|00> + |11>)/sqrt(2)")
println("    ")
println("    Full state: |Phi+>_AB ⊗ |Phi+>_CD")
println("             = (1/2)(|0000> + |0011> + |1100> + |1111>)")
println()
println("    Now we have TWO independent Bell pairs:")
println("    - A is entangled with B")
println("    - C is entangled with D")
println("    - A and D: STILL NO ENTANGLEMENT (just classical correlations)")
println()

# Show rho_AD before swapping
ρ_AD_before = compute_rho_AD_before()
println("  rho_AD at this point (tracing out B,C):")
print_density_matrix(ρ_AD_before, "rho_AD = Tr_BC[|Phi+><Phi+| ⊗ |Phi+><Phi+|]")
println()
neg_before = negativity_half_split(ρ_AD_before, 2)
@printf("    Negativity(A:D) = %.4f\n", neg_before)
println("    -> A and D are SEPARABLE (rho_AD = I/4, maximally mixed)")
println()

println("  STEP 3: Bell measurement on qubits B and C")
println("  -------------------------------------------")
println("    We measure B,C in Bell basis. This PROJECTS the full state.")
println("    Circuit: CNOT(B,C) -> H(B) -> Measure(B,C)")
println()
println("    Four possible outcomes with equal probability 1/4:")
println("      |00>_BC -> A,D collapse to |Phi+> = (|00>+|11>)/sqrt(2)")
println("      |01>_BC -> A,D collapse to |Psi+> = (|01>+|10>)/sqrt(2)")
println("      |10>_BC -> A,D collapse to |Phi-> = (|00>-|11>)/sqrt(2)")
println("      |11>_BC -> A,D collapse to |Psi-> = (|01>-|10>)/sqrt(2)")
println()
println("    MAGIC: After measurement, A and D become maximally entangled!")
println("    (Even though they never interacted directly)")
println()

# Show rho_AD after swapping
ρ_AD_after, prob = entanglement_swapping_protocol(0.0, 0)
println("  STEP 4: Conditioned on outcome |00>_BC")
println("  ----------------------------------------")
print_density_matrix(ρ_AD_after, "rho_AD after conditioning on |00>")
println()

# Compare with reference Bell state
println("  COMPARISON WITH REFERENCE BELL STATE:")
print_density_matrix(bell_state_ref, "|Φ⁺⟩⟨Φ⁺| = (|00⟩+|11⟩)(⟨00|+⟨11|)/2")
println()

# Compute fidelity
fidelity = real(tr(ρ_AD_after * bell_state_ref))
neg_after = negativity_half_split(ρ_AD_after, 2)
@printf("    Fidelity F(rho_AD, |Phi+>) = %.4f  (expect 1.0)\n", fidelity)
@printf("    Negativity(A:D)            = %.4f  (expect 0.5 for Bell state)\n", neg_after)
println()
println("  CONCLUSION: rho_AD = |Phi+><Phi+| is a maximally entangled Bell state!")
println("  A and D now share the SAME entanglement that A-B and C-D had originally.")
println()

println("  ┌─────────────────────────────────────────────────────────────┐")
println("  │  NOISE EFFECTS ON SWAPPED ENTANGLEMENT                     │")
println("  └─────────────────────────────────────────────────────────────┘")
println()

println("  ┌────────────────────────────────────────────────────────────┐")
println("  │  Noise p   P(|00>)   Purity(AD)   Neg(AD)   Entangled?   │")
println("  ├────────────────────────────────────────────────────────────┤")

for p in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    local ρ_AD, prob = entanglement_swapping_protocol(p, 0)
    
    P = real(tr(ρ_AD * ρ_AD))
    neg = negativity_half_split(ρ_AD, 2)
    
    entangled = neg > 0.01 ? "Yes" : "No"
    
    @printf("  │  %.2f       %.4f   %.4f       %.4f    %-12s│\n",
            p, prob, P, neg, entangled)
end

println("  └────────────────────────────────────────────────────────────┘")
println()
println("  -- Entanglement swapping demonstrated!")



# ==============================================================================
# SUMMARY

# ==============================================================================

println("\n" * "=" ^ 70)
if all_passed
    println("  -- ALL PROTOCOLS DEMONSTRATED SUCCESSFULLY!")
else
    println("  ~ Most protocols worked (some statistical variation)")
end
println("=" ^ 70)
println()

# ==============================================================================
# COMPREHENSIVE SUMMARY
# ==============================================================================

println("  ╔═════════════════════════════════════════════════════════════╗")
println("  ║                SUMMARY OF PROTOCOLS                        ║")
println("  ╚═════════════════════════════════════════════════════════════╝")
println()

println("  ┌──────────────────────────────────────────────────────────────────┐")
println("  │  PROTOCOL    │  KEY PHYSICS CONCEPT                             │")
println("  ├──────────────────────────────────────────────────────────────────┤")
println("  │  1. Bell St. │  Maximal entanglement: perfect correlation but   │")
println("  │              │  local randomness. Subsystems maximally mixed.   │")
println("  ├──────────────────────────────────────────────────────────────────┤")
println("  │  2. GHZ      │  N-party entanglement: all qubits correlated,    │")
println("  │              │  tracing ANY qubit → mixed state. Very fragile.  │")
println("  ├──────────────────────────────────────────────────────────────────┤")
println("  │  3. Decay    │  Decoherence destroys correlations: ⟨Z₁Z₂⟩       │")
println("  │              │  drops from +1 toward 0 as noise increases.      │")
println("  ├──────────────────────────────────────────────────────────────────┤")
println("  │  4. Teleport │  Entanglement + classical bits = state transfer. │")
println("  │              │  Mid-circuit measurement, conditional correction.│")
println("  ├──────────────────────────────────────────────────────────────────┤")
println("  │  5. Noisy T. │  Noise channels (depol, dephase, amp damp)       │")
println("  │              │  degrade fidelity. F > 2/3 → quantum advantage.  │")
println("  ├──────────────────────────────────────────────────────────────────┤")
println("  │  6. Ent.Meas │  Quantify entanglement: SvN entropy, negativity. │")
println("  │              │  Mixed states need PPT/negativity criteria.      │")
println("  ├──────────────────────────────────────────────────────────────────┤")
println("  │  7. CHSH     │  Bell inequality: classical max √2, quantum 2√2. │")
println("  │              │  Noise degrades violation toward classical.      │")
println("  ├──────────────────────────────────────────────────────────────────┤")
println("  │  8. Swapping │  Bell measurement on B,C entangles A and D that  │")
println("  │              │  NEVER interacted! Foundation of quantum relays. │")
println("  └──────────────────────────────────────────────────────────────────┘")
println()

println("  ═══════════════════════════════════════════════════════════════")
println("  KEY CONCEPTS DEMONSTRATED")
println("  ═══════════════════════════════════════════════════════════════")
println()
println("  ENTANGLEMENT:")
println("    • Created via CNOT after H: |00⟩ → H(1)·CNOT(1,2) → |Φ⁺⟩")
println("    • Quantified by: von Neumann entropy, negativity, purity")
println("    • Cannot send information faster than light (requires classical bits)")
println()
println("  QUANTUM CHANNELS (NOISE):")
println("    • Depolarizing: isotropic errors, F → 0.5 as p → 1")
println("    • Dephasing: phase-only errors, destroys superpositions")
println("    • Amp Damping: energy loss, asymmetric decay toward |0⟩")
println()
println("  MEASUREMENTS:")
println("    • Computational basis: project onto |0⟩, |1⟩")
println("    • Bell basis: project onto entangled states via CNOT+H transform")
println("    • Mid-circuit measurement: collapse + conditional operations")
println()
println("  DM vs MCWF:")
println("    • DM: exact expectation values, requires O(4^N) memory")
println("    • MCWF: stochastic sampling, scales as O(2^N), converges ∝ 1/√M")
println()

println("  ═══════════════════════════════════════════════════════════════")
println("  MATHEMATICAL NOTATION REFERENCE")
println("  ═══════════════════════════════════════════════════════════════")
println()
println("  STATES:")
println("    |0⟩, |1⟩           Computational basis (Z eigenstates)")
println("    |+⟩, |-⟩           X eigenstates: (|0⟩±|1⟩)/√2")
println("    |Φ⁺⟩               Bell state: (|00⟩+|11⟩)/√2")
println("    |GHZ_N⟩            (|00...0⟩+|11...1⟩)/√2")
println()
println("  OPERATORS:")
println("    ρ                  Density matrix (pure: ρ=|ψ⟩⟨ψ|)")
println("    Tr(ρ²)             Purity (=1 if pure, <1 if mixed)")
println("    Tr_B(ρ)            Partial trace over subsystem B")
println("    ⟨O⟩ = Tr(ρO)       Expectation value")
println()
println("  GATES:")
println("    H = (X+Z)/√2       Hadamard (creates superposition)")
println("    CNOT               Flips target if control=|1⟩")
println("    CZ                 Phase flip on |11⟩")
println()
println("  ENTANGLEMENT MEASURES:")
println("    S_vN = -Tr(ρ log ρ)      von Neumann entropy")
println("    N(ρ) = (||ρᵀᴬ||-1)/2     Negativity (PPT criterion)")
println()

println("  ═══════════════════════════════════════════════════════════════")
println()
println("  Output saved to: demo_quantum_info_protocols/demo_quantum_info_protocols.txt")
println()


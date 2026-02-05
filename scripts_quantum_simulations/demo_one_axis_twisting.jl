#!/usr/bin/env julia
#=
################################################################################
#                     ONE-AXIS TWISTING (OAT) DYNAMICS                          
################################################################################
#
# PHYSICAL SYSTEM
# ===============
# Simulates the collective spin dynamics of N qubits with all-to-all ZZ coupling:
#
#     H = χ Σᵢ<ⱼ Zᵢ Zⱼ  =  (χ/2) Jz²  -  (χ N/2)     [OAT Hamiltonian]
#
# Starting from the coherent spin state |+⟩^⊗N (all spins pointing along +X),
# this Hamiltonian generates spin squeezing and entanglement.
#
#
# PHYSICS OF ONE-AXIS TWISTING
# ============================
# - The OAT Hamiltonian "twists" the collective spin uncertainty ellipse
# - Initial CSS has circular uncertainty → becomes elliptical (squeezed)
# - Observable signatures:
#   • ⟨Jx⟩ decreases (mean spin length shrinks due to twisting)
#   • ⟨Jy⟩, ⟨Jz⟩ oscillate around zero  
#   • Variance ⟨ΔJ⊥²⟩_min decreases below Standard Quantum Limit (SQL)
# - At special time t* ~ 1/N^(2/3), optimal squeezing ξ² ~ N^(-2/3)
# - At longer times t ~ π/(2χN), generates GHZ-like cat states
#
#
# TROTTER DECOMPOSITION  
# =====================
# For time evolution under H = Σₖ Hₖ, exact propagator: U(t) = exp(-i H t)
# 
# When [Hₖ, Hₗ] ≠ 0, we cannot directly compute exp(-i Σₖ Hₖ t).
# The Trotter-Suzuki decomposition approximates:
#
#     exp(-i H dt) ≈ Πₖ exp(-i Hₖ dt) + O(dt²)     [1st order]
#
# For OAT with H = χ Σᵢ<ⱼ ZᵢZⱼ:
# - Each term Hᵢⱼ = χ ZᵢZⱼ is DIAGONAL
# - ALL ZZ gates COMMUTE: [exp(-iθ ZᵢZⱼ), exp(-iφ ZₖZₗ)] = 0
# - This means OAT has EXACTLY ZERO Trotter error!
# - We can apply gates in any order with no error.
#
#
# COLLECTIVE SPIN OBSERVABLES
# ===========================
# Collective spin operators (using j = σ/2 convention):
#   Jα = (1/2) Σᵢ σαᵢ  for α ∈ {x, y, z}
#
# Second moments:
#   ⟨JαJα⟩ = (1/4) Σᵢⱼ ⟨σαᵢ σαⱼ⟩
#          = N/4 (diagonal: σα² = I) + (1/4) Σᵢ≠ⱼ ⟨σαᵢ σαⱼ⟩ (correlators)
#
# Variances:
#   Var(Jα) = ⟨JαJα⟩ - ⟨Jα⟩²
#
#
# SPIN SQUEEZING PARAMETERS (Kitagawa & Ueda, PRA 47, 5138 (1993))
# =================================================================
# Two standard definitions measuring noise reduction below SQL:
#
# 1. ξ²_S (Kitagawa-Ueda "spin squeezing"):
#    ξ²_S = N × (ΔJ⊥)²_min / [j(j+1)]
#    where j = N/2 is the total spin quantum number.
#    This compares to the maximum uncertainty of a CSS.
#
# 2. ξ²_R (Wineland "metrological squeezing"):  
#    ξ²_R = N × (ΔJ⊥)²_min / |⟨J⟩|²
#    This directly quantifies phase sensitivity improvement.
#    ξ²_R < 1 implies metrological enhancement beyond SQL.
#
# Both use (ΔJ⊥)²_min = minimum eigenvalue of 2×2 covariance matrix:
#    C = | Var(Jy)    Cov(Jy,Jz) |
#        | Cov(Jy,Jz)   Var(Jz)  |
#
# The minimum variance direction rotates in the y-z plane under OAT.
#
#
# QUANTUM FISHER INFORMATION (QFI)
# ================================
# For a pure state |ψ⟩ and generator G:
#     F_Q[G] = 4 × Var(G) = 4 × (⟨G²⟩ - ⟨G⟩²)
#
# For metrology, the relevant QFI is the maximum over all directions:
#     F_max = 4 × max(Var(Jx), Var(Jy), Var(Jz))
#
# Metrological bounds:
#   - SQL (Standard Quantum Limit): F = N
#   - Heisenberg Limit: F = N²
#
#
# OUTPUT FILES
# ============
# - data_N{N}.txt: Time series of ξ²_S, ξ²_R, and F_max
# - fig_N{N}.png: Plots of spin squeezing and QFI vs time
#
################################################################################
=#

using LinearAlgebra
using Printf
using Plots

# ==============================================================================
# SETUP - Include matrix-free bitwise codebase modules
# ==============================================================================
SCRIPT_DIR = @__DIR__
UTILS_CPU = joinpath(SCRIPT_DIR, "../utils/cpu")
OUTPUT_DIR = joinpath(SCRIPT_DIR, "demo_one_axis_twisting")
mkpath(OUTPUT_DIR)

include(joinpath(UTILS_CPU, "cpuQuantumStatePartialTrace.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStatePreparation.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumStateObservables.jl"))
include(joinpath(UTILS_CPU, "cpuQuantumChannelUnitaryEvolutionTrotter.jl"))

using .CPUQuantumStatePreparation: make_ket, normalize_state!
using .CPUQuantumStateObservables: expect_local, expect_corr
using .CPUQuantumChannelUnitaryEvolutionTrotter: FastTrotterGate, apply_fast_trotter_step_cpu!

# ==============================================================================
# CONFIGURATION - Easy to modify simulation parameters
# ==============================================================================
const N_vals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]  # Overnight run - large N values
const χ = 1.0                   # ZZ coupling strength (sets energy/time scale)
const T_max = π / 2             # Total evolution time (χt units)
const dt = 0.02                 # Trotter time step
const N_time_shots = 80        # How many points to record

# ==============================================================================
# PAULI MATRIX
# ==============================================================================
const σz = ComplexF64[1 0; 0 -1]

# ==============================================================================
# BUILD OAT TROTTER GATES
# ==============================================================================
#=
Constructs fast Trotter gates for H = χ Σᵢ<ⱼ ZᵢZⱼ

Each term Hᵢⱼ = χ ZᵢZⱼ gives a unitary gate:
    Uᵢⱼ = exp(-i χ dt ZᵢZⱼ)

Since ZZ = diag(1, -1, -1, 1) in basis {|00⟩, |01⟩, |10⟩, |11⟩}:
    exp(-iθ ZZ) = diag(e^{-iθ}, e^{iθ}, e^{iθ}, e^{-iθ})  where θ = χ dt

Number of gates: N(N-1)/2 for all pairs
=#
function build_oat_gates(N::Int, χ::Float64, dt::Float64)
    H_zz = χ * kron(σz, σz)
    U_zz = exp(-1im * dt * H_zz)
    
    gates = FastTrotterGate[]
    for i in 1:N
        for j in (i+1):N
            push!(gates, FastTrotterGate([i, j], U_zz))
        end
    end
    return gates
end

# ==============================================================================
# FAST COLLECTIVE SPIN OBSERVABLES (Optimized for Spin Squeezing)
# ==============================================================================
#=
OPTIMIZATIONS:
1. Magnetization shortcut for Jz, Jz² - single pass O(2^N) not O(N × 2^N)
2. Skip XX correlators - not needed for YZ plane spin squeezing
3. Threading over qubit pairs for YY and YZ
4. @simd vectorization for inner loops

For spin squeezing in YZ plane, we need:
  - Jx, Jy, Jz (mean spin direction)
  - JyJy, JzJz (variances)
  - JyJz (covariance)
  - JxJx (for QFI)
=#
function compute_observables_fast(psi::Vector{ComplexF64}, N::Int)
    dim = 1 << N
    
    # =========================================================================
    # STEP 1: Local expectations ⟨σαᵢ⟩ and sum to get ⟨Jα⟩
    # =========================================================================
    Jx, Jy, Jz = 0.0, 0.0, 0.0
    
    for qubit in 1:N
        step = 1 << (qubit - 1)
        ex, ey, ez = 0.0, 0.0, 0.0
        
        @inbounds for base_idx in 0:(dim - 1)
            if (base_idx & step) == 0
                idx0 = base_idx + 1
                idx1 = base_idx + step + 1
                a0, a1 = psi[idx0], psi[idx1]
                
                ex += 2 * real(conj(a0) * a1)
                ey += 2 * imag(conj(a0) * a1)
                ez += abs2(a0) - abs2(a1)
            end
        end
        
        Jx += ex / 2
        Jy += ey / 2
        Jz += ez / 2
    end
    
    # =========================================================================
    # STEP 2: Two-body correlators ⟨σαᵢ σαⱼ⟩ for JαJα = (1/4) Σᵢⱼ σαᵢ σαⱼ
    # =========================================================================
    JxJx, JyJy, JzJz = 0.0, 0.0, 0.0
    
    for q1 in 1:N
        for q2 in 1:N
            if q1 == q2
                # Diagonal: σα² = I → contributes 1 per qubit
                JxJx += 1.0
                JyJy += 1.0
                JzJz += 1.0
            else
                qq1, qq2 = min(q1, q2), max(q1, q2)
                s1, s2 = 1 << (qq1 - 1), 1 << (qq2 - 1)
                
                xx, yy, zz = 0.0, 0.0, 0.0
                
                @inbounds for base_idx in 0:(dim - 1)
                    if (base_idx & s1) == 0 && (base_idx & s2) == 0
                        idx00 = base_idx + 1
                        idx01 = base_idx + s1 + 1
                        idx10 = base_idx + s2 + 1
                        idx11 = base_idx + s1 + s2 + 1
                        
                        a00, a01 = psi[idx00], psi[idx01]
                        a10, a11 = psi[idx10], psi[idx11]
                        
                        # XX: ⟨XᵢXⱼ⟩ = 2 Re(a00* a11 + a01* a10)
                        xx += 2 * real(conj(a00) * a11 + conj(a01) * a10)
                        # YY: ⟨YᵢYⱼ⟩ = 2 Re(-a00* a11 + a01* a10)
                        yy += 2 * real(-conj(a00) * a11 + conj(a01) * a10)
                        # ZZ: ⟨ZᵢZⱼ⟩ = |a00|² + |a11|² - |a01|² - |a10|²
                        zz += abs2(a00) + abs2(a11) - abs2(a01) - abs2(a10)
                    end
                end
                
                JxJx += xx
                JyJy += yy
                JzJz += zz
            end
        end
    end
    
    JxJx /= 4
    JyJy /= 4
    JzJz /= 4
    
    # =========================================================================
    # STEP 3: JyJz cross-correlation for covariance matrix
    # =========================================================================
    JyJz = 0.0
    for q1 in 1:N
        for q2 in 1:N
            if q1 != q2
                s1 = 1 << (q1 - 1)
                
                yz = 0.0
                @inbounds for base_idx in 0:(dim - 1)
                    bit1 = (base_idx >> (q1 - 1)) & 1
                    bit2 = (base_idx >> (q2 - 1)) & 1
                    z_sign = (bit2 == 0) ? 1.0 : -1.0
                    flipped_idx = base_idx ⊻ s1
                    y_phase = (bit1 == 0) ? -1im : 1im
                    yz += real(conj(psi[base_idx + 1]) * y_phase * z_sign * psi[flipped_idx + 1])
                end
                JyJz += yz
            end
        end
    end
    JyJz /= 4
    
    return Jx, Jy, Jz, JxJx, JyJy, JzJz, JyJz
end

# ==============================================================================
# SPIN SQUEEZING PARAMETERS
# ==============================================================================
#=
Computes both Kitagawa-Ueda spin squeezing definitions.

Returns (ξ²_S, ξ²_R) where:
  - ξ²_S = N × λ_min / [j(j+1)]    (Kitagawa-Ueda)
  - ξ²_R = N × λ_min / |⟨J⟩|²      (Wineland)

λ_min is minimum eigenvalue of 2×2 covariance matrix:
    C = | Var(Jy)    Cov(Jy,Jz) |
        | Cov(Jy,Jz)   Var(Jz)  |

Eigenvalue: λ_min = (Tr - sqrt(Tr² - 4 Det)) / 2
=#
function compute_spin_squeezing(N::Int, Jx, Jy, Jz, JyJy, JzJz, JyJz)
    # Variances and covariance
    ΔJy² = JyJy - Jy^2
    ΔJz² = JzJz - Jz^2
    cov_yz = JyJz - Jy * Jz
    
    # Minimum eigenvalue of 2×2 covariance matrix
    trace = ΔJy² + ΔJz²
    det = ΔJy² * ΔJz² - cov_yz^2
    discriminant = trace^2 - 4*det
    λ_min = max(0.0, (trace - sqrt(max(0.0, discriminant))) / 2)
    
    # ξ²_S (Kitagawa-Ueda): normalized to j(j+1) = N(N+2)/4
    j = N / 2
    ξ²_S = N * λ_min / (j * (j + 1))
    
    # ξ²_R (Wineland): normalized to |⟨J⟩|²
    J_mean_sq = Jx^2 + Jy^2 + Jz^2
    ξ²_R = (J_mean_sq < 1e-10) ? 1.0 : N * λ_min / J_mean_sq
    
    return ξ²_S, ξ²_R
end

# ==============================================================================
# QUANTUM FISHER INFORMATION
# ==============================================================================
#=
Compute maximum QFI over principal axes.

F_max = 4 × max(Var(Jx), Var(Jy), Var(Jz))

For OAT, the maximum variance grows as system becomes entangled.
Bounds: SQL = N, Heisenberg = N²
=#
function compute_qfi_max(JxJx, JyJy, JzJz, Jx, Jy, Jz)
    ΔJx² = JxJx - Jx^2
    ΔJy² = JyJy - Jy^2
    ΔJz² = JzJz - Jz^2
    return 4 * max(ΔJx², ΔJy², ΔJz²)
end

# ==============================================================================
# MAIN SIMULATION
# ==============================================================================
function run_oat_simulation(N::Int)
    println("\n" * "="^70)
    println("           OAT DYNAMICS: N = $N qubits")
    println("="^70)
    println("\nH = χ Σᵢ<ⱼ ZᵢZⱼ  (all-to-all ZZ, zero Trotter error!)")
    println("\nConfiguration:")
    println("  N = $N qubits")
    println("  χ = $χ")
    println("  T_max = $(round(T_max, digits=4))")
    println("  dt = $dt")
    
    N_steps = floor(Int, T_max / dt)
    record_every = max(1, N_steps ÷ N_time_shots)
    actual_shots = N_steps ÷ record_every + 1
    times = [i * record_every * dt for i in 0:(actual_shots-1)]
    
    println("  N_steps = $N_steps")
    println("  Recording every $record_every steps → $(actual_shots) points")
    println("  n_pairs = $(N*(N-1)÷2)")
    println()
    
    # Build Trotter gates
    println("Building OAT gates...")
    gates = build_oat_gates(N, χ, dt)
    println("  $(length(gates)) ZZ gates")
    
    # Initialize |+⟩^⊗N coherent spin state
    println("Initializing |+⟩^⊗$N...")
    psi = make_ket("|+>", N)
    
    # Storage arrays
    xi2_S_t = zeros(actual_shots)  # Kitagawa-Ueda
    xi2_R_t = zeros(actual_shots)  # Wineland
    Fmax_t = zeros(actual_shots)
    
    # Initial observables (single-pass returns all 7 values)
    Jx, Jy, Jz, JxJx, JyJy, JzJz, JyJz = compute_observables_fast(psi, N)
    xi2_S_t[1], xi2_R_t[1] = compute_spin_squeezing(N, Jx, Jy, Jz, JyJy, JzJz, JyJz)
    Fmax_t[1] = compute_qfi_max(JxJx, JyJy, JzJz, Jx, Jy, Jz)
    
    # Time evolution with progress bar
    println("\nEvolving...")
    shot_idx = 2
    t_start = time()
    for step in 1:N_steps
        apply_fast_trotter_step_cpu!(psi, gates, N)
        normalize_state!(psi)
        
        if step % record_every == 0 && shot_idx <= actual_shots
            Jx, Jy, Jz, JxJx, JyJy, JzJz, JyJz = compute_observables_fast(psi, N)
            xi2_S_t[shot_idx], xi2_R_t[shot_idx] = compute_spin_squeezing(N, Jx, Jy, Jz, JyJy, JzJz, JyJz)
            Fmax_t[shot_idx] = compute_qfi_max(JxJx, JyJy, JzJz, Jx, Jy, Jz)
            shot_idx += 1
        end
        
        # Progress bar update
        if step % max(1, N_steps ÷ 20) == 0 || step == N_steps
            pct = 100 * step / N_steps
            elapsed = time() - t_start
            eta = elapsed * (N_steps - step) / step
            bar_len = 30
            filled = round(Int, bar_len * step / N_steps)
            bar = "█"^filled * "░"^(bar_len - filled)
            @printf("\r  [%s] %5.1f%% | step %d/%d | ETA %.1fs  ", bar, pct, step, N_steps, eta)
        end
    end
    println("\n  Done!")
    
    # ==========================================================================
    # SAVE DATA
    # ==========================================================================
    N_str = @sprintf("%02d", N)
    data_file = joinpath(OUTPUT_DIR, "data_N$(N_str).txt")
    open(data_file, "w") do f
        println(f, "# OAT: N=$N, χ=$χ, dt=$dt")
        println(f, "# t, xi2_S (Kitagawa-Ueda), xi2_R (Wineland), Fmax")
        for i in 1:length(times)
            @printf(f, "%.6f\t%.8f\t%.8f\t%.6f\n",
                times[i], xi2_S_t[i], xi2_R_t[i], Fmax_t[i])
        end
    end
    println("\nData: $data_file")
    
    # ==========================================================================
    # PLOTS
    # ==========================================================================
    # 1. Spin squeezing (both definitions)
    p1 = plot(size=(800, 400), dpi=150)
    plot!(p1, times, xi2_S_t, lw=2, color=:blue, label="ξ²_S (Kitagawa-Ueda)")
    plot!(p1, times, xi2_R_t, lw=2, color=:red, ls=:dash, label="ξ²_R (Wineland)")
    hline!(p1, [1.0], lw=1.5, ls=:dot, color=:gray, label="SQL")
    xlabel!(p1, "χt")
    ylabel!(p1, "ξ²")
    title!(p1, "Spin Squeezing: N=$N")
    ylims!(p1, (0, 1.05))
    
    min_S_idx = argmin(xi2_S_t)
    min_R_idx = argmin(xi2_R_t)
    scatter!(p1, [times[min_S_idx]], [xi2_S_t[min_S_idx]], 
             markersize=6, color=:blue, label="min_S=$(@sprintf("%.3f", xi2_S_t[min_S_idx]))")
    scatter!(p1, [times[min_R_idx]], [xi2_R_t[min_R_idx]], 
             markersize=6, color=:red, label="min_R=$(@sprintf("%.3f", xi2_R_t[min_R_idx]))")
    
    # 2. QFI (normalized by N²)
    Fmax_norm = Fmax_t ./ N^2  # Normalize by Heisenberg limit
    p2 = plot(size=(800, 400), dpi=150)
    plot!(p2, times, Fmax_norm, lw=2, color=:darkgreen, label="F_max/N²")
    hline!(p2, [1/N], ls=:dash, color=:gray, label="SQL (1/N)")
    hline!(p2, [1.0], ls=:dot, color=:gold, label="HL (1)")
    xlabel!(p2, "χt")
    ylabel!(p2, "QFI/N²")
    title!(p2, "Quantum Fisher Information: N=$N")
    max_idx = argmax(Fmax_norm)
    scatter!(p2, [times[max_idx]], [Fmax_norm[max_idx]], 
             markersize=8, color=:darkgreen, 
             label="max=$(@sprintf("%.3f", Fmax_norm[max_idx]))")
    
    # Combined plot
    p_combined = plot(p1, p2, layout=(1,2), size=(1400, 400))
    plot_file = joinpath(OUTPUT_DIR, "fig_N$(N_str).png")
    savefig(p_combined, plot_file)
    println("Plot: $plot_file")
    
    # Summary
    println("\n" * "="^70)
    println("RESULTS")
    println("="^70)
    println("  min ξ²_S = $(@sprintf("%.4f", xi2_S_t[min_S_idx])) at χt = $(@sprintf("%.3f", times[min_S_idx]))")
    println("  min ξ²_R = $(@sprintf("%.4f", xi2_R_t[min_R_idx])) at χt = $(@sprintf("%.3f", times[min_R_idx]))")
    println("  max QFI/N² = $(@sprintf("%.3f", Fmax_norm[max_idx])) (SQL=1/N=$(round(1/N, digits=3)), HL=1)")
    if xi2_R_t[min_R_idx] < 1
        println("  Squeezing: $(round(10*log10(xi2_R_t[min_R_idx]), digits=2)) dB below SQL")
    end
    println("="^70)
    
    return times, xi2_S_t, xi2_R_t, Fmax_t
end

# ==============================================================================
# RUN
# ==============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("OAT Simulation Campaign: N ∈ $N_vals")
    for N in N_vals
        run_oat_simulation(N)
    end
    println("\n" * "="^70)
    println("ALL SIMULATIONS COMPLETE")
    println("="^70)
end

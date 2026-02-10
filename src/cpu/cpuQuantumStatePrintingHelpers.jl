# ==============================================================================
# Quantum State Printing Helpers
# ==============================================================================
#
# Utility functions for displaying quantum states and density matrices
# in a human-readable format for educational demos.
#
# Functions:
#   - print_density_matrix: Display density matrix with real/imag parts
#   - print_state_vector: Display state vector with basis labels
#   - print_reduced_dm: Display reduced density matrix after partial trace
# ==============================================================================

using Printf, LinearAlgebra

export print_density_matrix, print_state_vector, print_state_summary

"""
    print_density_matrix(ρ::Matrix{ComplexF64}, label::String; indent::Int=2)

Print a density matrix with real and imaginary parts separated.
Only prints imaginary part if there are non-zero imaginary components.

# Arguments
- `ρ`: Density matrix to display
- `label`: Label describing the state
- `indent`: Number of spaces for indentation (default 2)
"""
function print_density_matrix(ρ::Matrix{ComplexF64}, label::String; indent::Int=2)
    dim = size(ρ, 1)
    pad = " "^indent

    println("$(pad)$label ($(dim)x$(dim)):")
    println()
    println("$(pad)  Real part:")

    for i in 1:dim
        print("$(pad)  ")
        for j in 1:dim
            @printf("  %+.3f", real(ρ[i,j]))
        end
        println()
    end

    # Only print imaginary part if there are non-zero values
    if any(x -> abs(imag(x)) > 1e-10, ρ)
        println()
        println("$(pad)  Imaginary part:")
        for i in 1:dim
            print("$(pad)  ")
            for j in 1:dim
                @printf("  %+.3f", imag(ρ[i,j]))
            end
            println()
        end
    end
end

"""
    print_state_vector(ψ::Vector{ComplexF64}, N::Int, label::String;
                       threshold::Float64=1e-10, indent::Int=2)

Print a state vector in column format with basis state labels.
Shows each non-zero component on its own line for clarity.

# Arguments
- `ψ`: State vector
- `N`: Number of qubits
- `label`: Label for the state
- `threshold`: Minimum amplitude to display (default 1e-10)
- `indent`: Indentation spaces (default 2)
"""
function print_state_vector(ψ::Vector{ComplexF64}, N::Int, label::String;
                            threshold::Float64=1e-10, indent::Int=2)
    pad = " "^indent
    println(pad, label, ":")
    println(pad, "  ┌────────────┬─────────────────────┐")
    println(pad, "  │   Basis    │      Amplitude      │")
    println(pad, "  ├────────────┼─────────────────────┤")

    for i in 1:length(ψ)
        if abs(ψ[i]) > threshold
            # Show basis in ABCD order (reverse bit order for display)
            bits_raw = string(i-1, base=2, pad=N)
            bits_abcd = reverse(bits_raw)  # Reverse so qubit 1 is leftmost

            re, im = real(ψ[i]), imag(ψ[i])

            if abs(im) < threshold
                amp_str = @sprintf("%+.4f", re)
            elseif abs(re) < threshold
                amp_str = @sprintf("%+.4fi", im)
            else
                amp_str = @sprintf("%+.4f %+.4fi", re, im)
            end

            println(pad, "  │  |", bits_abcd, "⟩   │  ", rpad(amp_str, 17), " │")
        end
    end
    println(pad, "  └────────────┴─────────────────────┘")
end

"""
    print_state_vector_compact(ψ::Vector{ComplexF64}, N::Int; threshold::Float64=1e-10)

Print state vector in compact inline format.
"""
function print_state_vector_compact(ψ::Vector{ComplexF64}, N::Int; threshold::Float64=1e-10)
    print("  |ψ⟩ = ")
    first = true
    for i in 1:length(ψ)
        if abs(ψ[i]) > threshold
            if !first
                print(" + ")
            end
            bits_raw = string(i-1, base=2, pad=N)
            bits_abcd = reverse(bits_raw)
            re = real(ψ[i])
            @printf("%.3f|%s⟩", re, bits_abcd)
            first = false
        end
    end
    println()
end

"""
    print_state_summary(ρ::Matrix{ComplexF64}, label::String; indent::Int=2)

Print a concise summary of a quantum state: purity, von Neumann entropy, and trace.

# Arguments
- `ρ`: Density matrix
- `label`: Label for the state
- `indent`: Indentation spaces (default 2)
"""
function print_state_summary(ρ::Matrix{ComplexF64}, label::String; indent::Int=2)
    pad = " "^indent

    # Compute properties
    trace_val = real(tr(ρ))
    purity = real(tr(ρ * ρ))

    # von Neumann entropy
    eigvals_ρ = eigvals(Hermitian(ρ))
    eigvals_ρ = max.(eigvals_ρ, 0)  # Numerical stability
    S_vN = -sum(λ -> λ > 1e-15 ? λ * log2(λ) : 0.0, eigvals_ρ)

    println(pad, label, ":")
    print(pad, "  Tr(ρ)  = ")
    @printf("%.4f (should be 1.0)\n", trace_val)
    print(pad, "  Tr(ρ²) = ")
    @printf("%.4f (1=pure, 1/d=maximally mixed)\n", purity)
    print(pad, "  S_vN   = ")
    @printf("%.4f bits\n", S_vN)
end

"""
    print_qubit_labels(N::Int; indent::Int=2)

Print qubit labeling convention for N qubits.
"""
function print_qubit_labels(N::Int; indent::Int=2)
    pad = " "^indent
    println("$(pad)Qubit labeling: |q$(N)...q2 q1⟩ (q1 is rightmost bit)")
    print("$(pad)Basis states: ")
    for i in 0:min(7, 2^N-1)
        bits = string(i, base=2, pad=N)
        @printf("|%s⟩=%d  ", bits, i)
    end
    if 2^N > 8
        print("...")
    end
    println()
end

"""
    print_state_comparison(ψ1, ψ2, N, label1, label2; show_sum=true, show_dm=false)

Print two state vectors side-by-side with Fock basis and optionally their sum.

# Arguments
- `ψ1`, `ψ2`: State vectors to compare
- `N`: Number of qubits
- `label1`, `label2`: Labels for the states
- `show_sum`: Also show ψ1 + ψ2 column (default true)
- `show_dm`: Also show density matrix of sum (default false)
"""
function print_state_comparison(ψ1::Vector{ComplexF64}, ψ2::Vector{ComplexF64},
                                 N::Int, label1::String, label2::String;
                                 show_sum::Bool=true, show_dm::Bool=false, indent::Int=2)
    pad = " "^indent
    dim = length(ψ1)
    ψ_sum = ψ1 .+ ψ2

    # Header
    println(pad, "┌──────────┬─────────────────┬─────────────────",
            show_sum ? "┬─────────────────┐" : "┐")
    print(pad, "│  Basis   │  ", rpad(label1, 14), " │  ", rpad(label2, 14), " │")
    if show_sum
        println("  Sum             │")
    else
        println()
    end
    println(pad, "├──────────┼─────────────────┼─────────────────",
            show_sum ? "┼─────────────────┤" : "┤")

    # Format amplitude as string
    function fmt_amp(z::ComplexF64)
        re, im = real(z), imag(z)
        if abs(z) < 1e-10
            return "     0.000      "
        elseif abs(im) < 1e-10
            return @sprintf("  %+.3f        ", re)
        elseif abs(re) < 1e-10
            return @sprintf("       %+.3fi   ", im)
        else
            return @sprintf("  %+.3f%+.3fi", re, im)
        end
    end

    # Data rows
    for i in 1:dim
        if abs(ψ1[i]) > 1e-10 || abs(ψ2[i]) > 1e-10 || (show_sum && abs(ψ_sum[i]) > 1e-10)
            # Format basis in qubit order
            bits = ""
            for q in 0:(N-1)
                bits *= string((i-1 >> q) & 1)
            end

            print(pad, "│ |", bits, "⟩ │ ", fmt_amp(ψ1[i]), "│ ", fmt_amp(ψ2[i]), "│")
            if show_sum
                println(fmt_amp(ψ_sum[i]), "│")
            else
                println()
            end
        end
    end

    println(pad, "└──────────┴─────────────────┴─────────────────",
            show_sum ? "┴─────────────────┘" : "┘")

    # Optionally show density matrix of sum
    if show_dm && show_sum
        # Normalize sum
        norm_sum = sqrt(sum(abs2, ψ_sum))
        if norm_sum > 1e-10
            ψ_sum_norm = ψ_sum ./ norm_sum
            ρ_sum = ψ_sum_norm * ψ_sum_norm'
            println()
            println(pad, "Density matrix of normalized sum (|ψ1⟩ + |ψ2⟩)/||...||:")
            print_density_matrix(ρ_sum, "ρ = |ψ_sum⟩⟨ψ_sum|"; indent=indent+2)
        end
    end
end

"""
    print_state_operation(ψ_before, ψ_after, N, op_name; indent=2)

Print state before and after an operation, showing the transformation.
"""
function print_state_operation(ψ_before::Vector{ComplexF64}, ψ_after::Vector{ComplexF64},
                                N::Int, op_name::String; indent::Int=2)
    pad = " "^indent
    println(pad, "Operation: ", op_name)
    print_state_comparison(ψ_before, ψ_after, N, "Before", "After";
                          show_sum=false, show_dm=false, indent=indent)
end

"""
    print_gate_action(gate_fn!, ψ, qubits, N, gate_name; indent=2)

Visualize a gate action on a quantum state:
1. Shows input state in Fock basis
2. Shows the gate and target qubits
3. Applies the gate
4. Shows output state in Fock basis

Returns the modified state vector.

# Arguments
- `gate_fn!`: Gate function that modifies ψ in-place, signature: `gate_fn!(ψ, qubit, N)`
- `ψ`: State vector (will be copied before modification)
- `qubits`: Single qubit index or tuple of qubits (1-indexed)
- `N`: Total number of qubits
- `gate_name`: Display name for the gate (e.g., "H", "CNOT", "CZ")
- `indent`: Indentation (default 2)

# Example
```julia
ψ = zeros(ComplexF64, 4); ψ[1] = 1.0
ψ_out = print_gate_action(apply_hadamard_psi!, ψ, 1, 2, "H")
```
"""
function print_gate_action(gate_fn!::Function, ψ::Vector{ComplexF64},
                           qubits, N::Int, gate_name::String; indent::Int=2)
    pad = " "^indent
    ψ_before = copy(ψ)
    ψ_after = copy(ψ)

    # Format qubit string
    if qubits isa Tuple
        qubit_str = join(["q$q" for q in qubits], ",")
    else
        qubit_str = "q$qubits"
    end

    # Print gate header
    println(pad, "┌", "─"^50, "┐")
    println(pad, "│  Gate: ", rpad("$gate_name($qubit_str)", 40), " │")
    println(pad, "└", "─"^50, "┘")
    println()

    # Print input state
    println(pad, "INPUT STATE |ψ_in⟩:")
    println(pad, "  ┌────────────┬─────────────────┐")
    println(pad, "  │   Basis    │    Amplitude    │")
    println(pad, "  ├────────────┼─────────────────┤")

    for i in 1:length(ψ_before)
        if abs(ψ_before[i]) > 1e-10
            bits = ""
            for q in 0:(N-1)
                bits *= string(((i-1) >> q) & 1)
            end
            amp_str = @sprintf("%+.4f", real(ψ_before[i]))
            if abs(imag(ψ_before[i])) > 1e-10
                amp_str = @sprintf("%+.3f%+.3fi", real(ψ_before[i]), imag(ψ_before[i]))
            end
            println(pad, "  │  |", bits, "⟩   │  ", rpad(amp_str, 13), " │")
        end
    end
    println(pad, "  └────────────┴─────────────────┘")
    println()

    # Apply gate
    if qubits isa Tuple
        gate_fn!(ψ_after, qubits..., N)
    else
        gate_fn!(ψ_after, qubits, N)
    end

    # Print arrow with gate symbol
    println(pad, "         │")
    println(pad, "         ▼  ", gate_name, "(", qubit_str, ")")
    println(pad, "         │")
    println()

    # Print output state
    println(pad, "OUTPUT STATE |ψ_out⟩:")
    println(pad, "  ┌────────────┬─────────────────┐")
    println(pad, "  │   Basis    │    Amplitude    │")
    println(pad, "  ├────────────┼─────────────────┤")

    for i in 1:length(ψ_after)
        if abs(ψ_after[i]) > 1e-10
            bits = ""
            for q in 0:(N-1)
                bits *= string(((i-1) >> q) & 1)
            end
            amp_str = @sprintf("%+.4f", real(ψ_after[i]))
            if abs(imag(ψ_after[i])) > 1e-10
                amp_str = @sprintf("%+.3f%+.3fi", real(ψ_after[i]), imag(ψ_after[i]))
            end
            println(pad, "  │  |", bits, "⟩   │  ", rpad(amp_str, 13), " │")
        end
    end
    println(pad, "  └────────────┴─────────────────┘")
    println()

    return ψ_after
end

"""
    print_gate_sequence(gates, ψ_initial, N; indent=2)

Visualize a sequence of gates applied to a state.

# Arguments
- `gates`: Vector of tuples (gate_fn!, qubits, gate_name)
- `ψ_initial`: Initial state vector
- `N`: Number of qubits
"""
function print_gate_sequence(gates::Vector, ψ_initial::Vector{ComplexF64},
                              N::Int; indent::Int=2)
    pad = " "^indent
    ψ = copy(ψ_initial)

    println(pad, "GATE SEQUENCE:")
    println(pad, "═"^60)

    for (step, (gate_fn!, qubits, gate_name)) in enumerate(gates)
        println()
        println(pad, "Step $step:")
        ψ = print_gate_action(gate_fn!, ψ, qubits, N, gate_name; indent=indent+2)
    end

    return ψ
end

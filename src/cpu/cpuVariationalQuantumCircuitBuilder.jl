# Date: 2026
#
#=
================================================================================
    cpuVariationalQuantumCircuitBuilder.jl - Noisy Variational Quantum Circuit
================================================================================

PURPOSE:
--------
Define variational quantum circuits with support for:
- Parameterized gates (Rx, Ry, Rz, Rxx, Ryy, Rzz)
- Fixed gates (H, X, Y, Z, CZ, CNOT)
- Noise channels (depolarizing, amplitude_damping, dephasing, bit_flip)
- Mid-circuit measurements (Z, X, Y basis)
- Qubit resets (force to |0⟩, |1⟩, |+⟩, |-⟩)
- Lindbladian analog evolution blocks

All operations use matrix-free bitwise implementations for maximum performance.

OPERATION TYPES:
----------------
GATES:
  :rx, :ry, :rz           - Single-qubit rotations (parameterized)
  :h, :x, :y, :z, :s, :t  - Single-qubit fixed gates
  :cz, :cnot, :cx         - Two-qubit fixed gates
  :rxx, :ryy, :rzz        - Two-qubit rotations (parameterized)

NOISE:
  :depolarizing           - Depolarizing channel (p)
  :amplitude_damping      - Amplitude damping (γ)
  :dephasing              - Dephasing/phase flip (p)
  :bit_flip               - Bit flip (p)

MEASUREMENT:
  :measure_z, :measure_x, :measure_y  - Projective measurement

RESET:
  :reset                  - Force qubit to target state

ANALOG:
  :lindblad_step          - Lindbladian evolution block

USAGE:
------
```julia
using .CPUVariationalQuantumCircuitBuilder

# Define circuit with gates, noise, and measurements
ops = [
    CircuitOperation(:ry, [1], param_idx=1),
    CircuitOperation(:cz, [1, 2]),
    CircuitOperation(:depolarizing, [1, 2], p=0.01),
    CircuitOperation(:measure_z, [1]),
    CircuitOperation(:reset, [1], target_state=:zero),
    CircuitOperation(:ry, [1], param_idx=2),
]

circuit = ParameterizedCircuit(4, ops)
```

================================================================================
=#

module CPUVariationalQuantumCircuitBuilder

using LinearAlgebra

# Import gate functions from existing module
include("cpuQuantumChannelGates.jl")
using .CPUQuantumChannelGates

export CircuitOperation, CircuitLayer, ParameterizedCircuit, CircuitResult
export gate, noise, measure, reset, trace_out, lindblad_step  # Constructors
export hardware_efficient_ansatz, strong_entangling_ansatz
export single_qubit_rotation_layer, entangling_layer, noise_layer
export get_parameter_count, describe_circuit, count_operations

# ==============================================================================
# OPERATION CATEGORIES
# ==============================================================================

const GATE_TYPES = Set([
    :rx, :ry, :rz, :h, :x, :y, :z, :s, :t,
    :cz, :cnot, :cx, :rxx, :ryy, :rzz
])

const NOISE_TYPES = Set([
    :depolarizing, :amplitude_damping, :dephasing, :bit_flip,
    :phase_damping, :thermal_relaxation
])

const MEASURE_TYPES = Set([:measure_z, :measure_x, :measure_y, :measure])
const RESET_TYPES = Set([:reset])
const TRACE_TYPES = Set([:trace_out])
const ANALOG_TYPES = Set([:lindblad_step, :trotter_step])

# ==============================================================================
# CIRCUIT OPERATION STRUCT
# ==============================================================================

"""
    CircuitOperation

A single operation in a variational quantum circuit. Can be a gate, noise channel,
measurement, reset, trace out, or Lindbladian block.

# Fields
- `type::Symbol`: Operation type (:rx, :depolarizing, :measure_z, :reset, :trace_out, etc.)
- `qubits::Vector{Int}`: Affected qubit indices (1-based)
- `param_idx::Int`: Parameter index (0=fixed, >0=use θ[param_idx])
- `fixed_angle::Float64`: Fixed angle for non-variational gates
- `p::Float64`: Noise probability or rate
- `target_state::Symbol`: Target state for reset (:zero, :one, :plus, :minus)
- `basis::Symbol`: Measurement basis (:z, :x, :y)
- `if_outcome::Int`: Classical conditioning (0=always, >0=apply only if outcomes[if_outcome]==1)
- `H::Any`: Hamiltonian for Lindbladian
- `L_ops::Vector`: Jump operators for Lindbladian
- `dt::Float64`: Time step for analog evolution
"""
struct CircuitOperation
    type::Symbol
    qubits::Vector{Int}
    param_idx::Int
    fixed_angle::Float64
    p::Float64
    target_state::Symbol
    basis::Symbol
    if_outcome::Int           # NEW: 0=always, >0=apply if outcomes[if_outcome]==1
    H::Any
    L_ops::Vector
    dt::Float64
end

"""
    CircuitResult

Result of circuit execution containing final state and measurement outcomes.

# Fields
- `state::Union{Vector{ComplexF64}, Matrix{ComplexF64}}`: Final quantum state
- `outcomes::Vector{Int}`: Measurement outcomes in order of execution (0 or 1 each)

# Example
```julia
result = apply_circuit!(ψ, circuit, θ)
println("Measurement outcomes: ", result.outcomes)
println("Final state norm: ", norm(result.state))
```
"""
struct CircuitResult
    state::Union{Vector{ComplexF64}, Matrix{ComplexF64}}
    outcomes::Vector{Int}
end

# ==============================================================================
# CONVENIENT CONSTRUCTORS
# ==============================================================================

"""
    gate(type, qubits; param_idx=0, angle=0.0, if_outcome=0)

Create a gate operation.

# Examples
```julia
gate(:ry, [1], param_idx=1)           # Variational Ry
gate(:cz, [1, 2])                      # Fixed CZ
gate(:rz, [3], angle=π/4)              # Fixed Rz(π/4)
gate(:x, [2], if_outcome=1)            # X on qubit 2 only if outcomes[1]==1
```
"""
function gate(type::Symbol, qubits::Vector{Int};
              param_idx::Int=0, angle::Float64=0.0, if_outcome::Int=0)
    @assert type in GATE_TYPES "Unknown gate type: $type"
    return CircuitOperation(type, qubits, param_idx, angle, 0.0, :none, :z, if_outcome, nothing, [], 0.0)
end

# Single qubit convenience
gate(type::Symbol, qubit::Int; kwargs...) = gate(type, [qubit]; kwargs...)

"""
    noise(type, qubits; p=0.0)

Create a noise channel operation.

# Examples
```julia
noise(:depolarizing, [1, 2], p=0.01)
noise(:amplitude_damping, [1], p=0.1)
```
"""
function noise(type::Symbol, qubits::Vector{Int}; p::Float64=0.0, if_outcome::Int=0)
    @assert type in NOISE_TYPES "Unknown noise type: $type"
    return CircuitOperation(type, qubits, 0, 0.0, p, :none, :z, if_outcome, nothing, [], 0.0)
end

noise(type::Symbol, qubit::Int; kwargs...) = noise(type, [qubit]; kwargs...)

"""
    measure(qubits; basis=:z)

Create a measurement operation.

# Examples
```julia
measure([1], basis=:z)    # Z-basis measurement
measure([1, 2], basis=:x) # X-basis measurement on both
```
"""
function measure(qubits::Vector{Int}; basis::Symbol=:z)
    type = Symbol("measure_", basis)
    return CircuitOperation(type, qubits, 0, 0.0, 0.0, :none, basis, 0, nothing, [], 0.0)
end

measure(qubit::Int; kwargs...) = measure([qubit]; kwargs...)

"""
    reset(qubits; target_state=:zero, if_outcome=0)

Create a reset operation to force qubit(s) to target state.

# Examples
```julia
reset([1], target_state=:zero)  # Reset to |0⟩
reset([2], target_state=:plus)  # Reset to |+⟩
reset([1], target_state=:zero, if_outcome=1)  # Conditional reset
```
"""
function reset(qubits::Vector{Int}; target_state::Symbol=:zero, if_outcome::Int=0)
    return CircuitOperation(:reset, qubits, 0, 0.0, 0.0, target_state, :z, if_outcome, nothing, [], 0.0)
end

reset(qubit::Int; kwargs...) = reset([qubit]; kwargs...)

"""
    lindblad_step(qubits; H=nothing, L_ops=[], dt=0.0)

Create a Lindbladian evolution block.

# Examples
```julia
lindblad_step([1, 2], H=H_local, L_ops=[L1, L2], dt=0.01)
```
"""
function lindblad_step(qubits::Vector{Int}; H=nothing, L_ops::Vector=[], dt::Float64=0.0)
    return CircuitOperation(:lindblad_step, qubits, 0, 0.0, 0.0, :none, :z, 0, H, L_ops, dt)
end

"""
    trace_out(qubits)

Create a trace-out operation to remove qubits from the system.
Returns reduced density matrix of remaining qubits.

# Examples
```julia
trace_out([3])        # Trace out qubit 3
trace_out([1, 5, 8])  # Trace out non-neighboring qubits
```
"""
function trace_out(qubits::Vector{Int})
    return CircuitOperation(:trace_out, qubits, 0, 0.0, 0.0, :none, :z, 0, nothing, [], 0.0)
end

trace_out(qubit::Int) = trace_out([qubit])

# ==============================================================================
# CIRCUIT LAYER
# ==============================================================================

"""
    CircuitLayer

A group of operations that can be applied in parallel (disjoint qubits).
"""
struct CircuitLayer
    operations::Vector{CircuitOperation}
end

# ==============================================================================
# PARAMETERIZED CIRCUIT
# ==============================================================================

"""
    ParameterizedCircuit

A complete variational quantum circuit.

# Fields
- `N::Int`: Number of qubits
- `operations::Vector{CircuitOperation}`: Flat list of operations (in order)
- `layers::Vector{CircuitLayer}`: Optional grouped layers
- `n_params::Int`: Total variational parameters
"""
struct ParameterizedCircuit
    N::Int
    operations::Vector{CircuitOperation}
    layers::Vector{CircuitLayer}
    n_params::Int
end

# Constructor from flat list
function ParameterizedCircuit(N::Int, operations::Vector{CircuitOperation})
    n_params = 0
    for op in operations
        if op.param_idx > 0
            n_params = max(n_params, op.param_idx)
        end
    end
    return ParameterizedCircuit(N, operations, CircuitLayer[], n_params)
end

# Constructor from layers
function ParameterizedCircuit(N::Int, layers::Vector{CircuitLayer})
    operations = CircuitOperation[]
    n_params = 0
    for layer in layers
        for op in layer.operations
            push!(operations, op)
            if op.param_idx > 0
                n_params = max(n_params, op.param_idx)
            end
        end
    end
    return ParameterizedCircuit(N, operations, layers, n_params)
end

# ==============================================================================
# LAYER BUILDERS
# ==============================================================================

"""
    single_qubit_rotation_layer(N, param_offset; gates=[:ry])

Create a layer of single-qubit rotations on all qubits.
"""
function single_qubit_rotation_layer(N::Int, param_offset::Int; gates::Vector{Symbol}=[:ry])
    ops = CircuitOperation[]
    idx = param_offset

    for gate_type in gates
        for q in 1:N
            push!(ops, gate(gate_type, [q], param_idx=idx))
            idx += 1
        end
    end

    return CircuitLayer(ops), idx
end

"""
    entangling_layer(N; topology=:chain, gate_type=:cz)

Create a layer of entangling gates.
"""
function entangling_layer(N::Int; topology::Symbol=:chain, gate_type::Symbol=:cz)
    ops = CircuitOperation[]

    if topology == :chain
        for q in 1:(N-1)
            push!(ops, gate(gate_type, [q, q+1]))
        end
    elseif topology == :ring
        for q in 1:(N-1)
            push!(ops, gate(gate_type, [q, q+1]))
        end
        N > 2 && push!(ops, gate(gate_type, [N, 1]))
    elseif topology == :all
        for i in 1:N, j in (i+1):N
            push!(ops, gate(gate_type, [i, j]))
        end
    else
        error("Unknown topology: $topology")
    end

    return CircuitLayer(ops)
end

"""
    noise_layer(N, noise_type; p=0.0)

Create a layer of noise on all qubits.
"""
function noise_layer(N::Int, noise_type::Symbol; p::Float64=0.0)
    ops = [noise(noise_type, [q], p=p) for q in 1:N]
    return CircuitLayer(ops)
end

# ==============================================================================
# PREDEFINED ANSÄTZE
# ==============================================================================

"""
    hardware_efficient_ansatz(N, n_layers; kwargs...)

Hardware-efficient ansatz: rotation layers + entangling layers.
"""
function hardware_efficient_ansatz(N::Int, n_layers::Int;
                                    entangler::Symbol=:cz,
                                    topology::Symbol=:chain,
                                    rotations::Vector{Symbol}=[:ry],
                                    noise_type::Union{Nothing,Symbol}=nothing,
                                    noise_p::Float64=0.0)
    layers = CircuitLayer[]
    param_idx = 1

    for _ in 1:n_layers
        # Rotation layer
        rot_layer, param_idx = single_qubit_rotation_layer(N, param_idx; gates=rotations)
        push!(layers, rot_layer)

        # Optional noise after rotations
        if !isnothing(noise_type)
            push!(layers, noise_layer(N, noise_type, p=noise_p))
        end

        # Entangling layer
        if N > 1
            push!(layers, entangling_layer(N; topology=topology, gate_type=entangler))
        end
    end

    # Final rotation layer
    rot_layer, param_idx = single_qubit_rotation_layer(N, param_idx; gates=rotations)
    push!(layers, rot_layer)

    return ParameterizedCircuit(N, layers)
end

"""
    strong_entangling_ansatz(N, n_layers; kwargs...)

Strong entangling ansatz with full Rx-Ry-Rz rotations.
"""
function strong_entangling_ansatz(N::Int, n_layers::Int;
                                   topology::Symbol=:ring,
                                   noise_type::Union{Nothing,Symbol}=nothing,
                                   noise_p::Float64=0.0)
    return hardware_efficient_ansatz(N, n_layers;
                                     entangler=:cz,
                                     topology=topology,
                                     rotations=[:rx, :ry, :rz],
                                     noise_type=noise_type,
                                     noise_p=noise_p)
end

# ==============================================================================
# UTILITIES
# ==============================================================================

export get_parameter_count, get_parameter_info, initialize_parameters, extract_gate_angles

get_parameter_count(circuit::ParameterizedCircuit) = circuit.n_params

"""
    get_parameter_info(circuit) -> Vector{NamedTuple}

Get information about each variational parameter.

# Returns
Vector of (param_idx, gate_type, qubits, op_index) for each parameter.

# Example
```julia
info = get_parameter_info(circuit)
for p in info
    println("θ[\$(p.param_idx)] → \$(p.gate_type) on qubits \$(p.qubits)")
end
```
"""
function get_parameter_info(circuit::ParameterizedCircuit)
    params = NamedTuple{(:param_idx, :gate_type, :qubits, :op_index), Tuple{Int, Symbol, Vector{Int}, Int}}[]

    for (i, op) in enumerate(circuit.operations)
        if op.param_idx > 0
            push!(params, (param_idx=op.param_idx, gate_type=op.type, qubits=op.qubits, op_index=i))
        end
    end

    sort!(params, by=p -> p.param_idx)
    return params
end

"""
    initialize_parameters(circuit; init=:zeros) -> Vector{Float64}

Create initial parameter vector for circuit.

# Options
- `:zeros` - All zeros
- `:random` - Uniform [-π, π]
- `:small_random` - Uniform [-0.1, 0.1]
- Number - All set to that value
"""
function initialize_parameters(circuit::ParameterizedCircuit; init::Union{Symbol, Number}=:zeros)
    n = circuit.n_params

    if init == :zeros
        return zeros(Float64, n)
    elseif init == :random
        return (2π * rand(n)) .- π
    elseif init == :small_random
        return 0.2 * rand(n) .- 0.1
    elseif init isa Number
        return fill(Float64(init), n)
    else
        error("Unknown init: $init")
    end
end

"""
    extract_gate_angles(circuit, θ) -> Dict{Int, Float64}

Map parameter vector θ to gate angles by operation index.

# Example
```julia
angles = extract_gate_angles(circuit, θ)
# angles[op_index] = angle for that gate
```
"""
function extract_gate_angles(circuit::ParameterizedCircuit, θ::Vector{Float64})
    angles = Dict{Int, Float64}()

    for (i, op) in enumerate(circuit.operations)
        if op.param_idx > 0
            angles[i] = θ[op.param_idx]
        elseif op.fixed_angle != 0.0
            angles[i] = op.fixed_angle
        end
    end

    return angles
end

"""
    describe_parameters(circuit)

Print human-readable description of all variational parameters.
"""
function describe_parameters(circuit::ParameterizedCircuit)
    info = get_parameter_info(circuit)

    println("Circuit has $(circuit.n_params) variational parameters:")
    println("-" ^ 50)

    for p in info
        println("  θ[$(p.param_idx)] → $(p.gate_type) on qubit(s) $(p.qubits)")
    end
end

function count_operations(circuit::ParameterizedCircuit)
    counts = Dict{Symbol, Int}()
    for op in circuit.operations
        counts[op.type] = get(counts, op.type, 0) + 1
    end
    return counts
end

function describe_circuit(circuit::ParameterizedCircuit)
    println("ParameterizedCircuit: $(circuit.N) qubits, $(length(circuit.operations)) operations, $(circuit.n_params) parameters")
    println("-" ^ 60)

    counts = count_operations(circuit)

    gates = filter(kv -> kv[1] in GATE_TYPES, counts)
    noise = filter(kv -> kv[1] in NOISE_TYPES, counts)
    meas = filter(kv -> kv[1] in MEASURE_TYPES || kv[1] in RESET_TYPES, counts)

    !isempty(gates) && println("  Gates: ", join(["$k×$v" for (k,v) in gates], ", "))
    !isempty(noise) && println("  Noise: ", join(["$k×$v" for (k,v) in noise], ", "))
    !isempty(meas) && println("  Meas/Reset: ", join(["$k×$v" for (k,v) in meas], ", "))
end

# ==============================================================================
# LAYER REPETITION
# ==============================================================================

export repeat_layer, repeat_operations

"""
    repeat_layer(layer_fn, L, N; kwargs...) -> (operations, final_param_idx)

Repeat a layer-building function L times, incrementing param_idx automatically.

# Arguments
- `layer_fn`: Function (N, param_offset) -> (CircuitLayer, next_param_idx)
- `L`: Number of repetitions
- `N`: Number of qubits
- `param_offset`: Starting parameter index (default 1)

# Example
```julia
# Repeat hardware-efficient layer 5 times
function make_hea_layer(N, offset)
    rot, next = single_qubit_rotation_layer(N, offset; gates=[:ry, :rz])
    ent = entangling_layer(N)
    return vcat(rot.operations, ent.operations), next
end

ops, final_idx = repeat_layer(make_hea_layer, 5, N)
circuit = ParameterizedCircuit(N, ops)
```
"""
function repeat_layer(layer_fn::Function, L::Int, N::Int; param_offset::Int=1)
    all_ops = CircuitOperation[]
    param_idx = param_offset

    for _ in 1:L
        ops, param_idx = layer_fn(N, param_idx)
        append!(all_ops, ops)
    end

    return all_ops, param_idx
end

"""
    repeat_operations(ops::Vector{CircuitOperation}, L; renumber_params=true) -> Vector{CircuitOperation}

Repeat a list of operations L times, optionally renumbering parameters.

# Example
```julia
# Define one layer
layer_ops = [
    gate(:ry, [1], param_idx=1),
    gate(:ry, [2], param_idx=2),
    gate(:cz, [1, 2]),
    noise(:depolarizing, [1, 2], p=0.01)
]

# Repeat 3 times with fresh parameters each repetition
all_ops = repeat_operations(layer_ops, 3)
# Now has param_idx 1-6 (2 params × 3 layers)
```
"""
function repeat_operations(ops::Vector{CircuitOperation}, L::Int; renumber_params::Bool=true)
    if !renumber_params
        # Simple concatenation
        result = CircuitOperation[]
        for _ in 1:L
            append!(result, ops)
        end
        return result
    end

    # Find max param index in one layer
    max_param = 0
    for op in ops
        max_param = max(max_param, op.param_idx)
    end

    result = CircuitOperation[]
    offset = 0

    for _ in 1:L
        for op in ops
            if op.param_idx > 0
                # Create new operation with shifted param_idx
                new_op = CircuitOperation(
                    op.type, op.qubits, op.param_idx + offset, op.fixed_angle,
                    op.p, op.target_state, op.basis, op.if_outcome,
                    op.H, op.L_ops, op.dt
                )
                push!(result, new_op)
            else
                push!(result, op)
            end
        end
        offset += max_param
    end

    return result
end

end # module CPUVariationalQuantumCircuitBuilder

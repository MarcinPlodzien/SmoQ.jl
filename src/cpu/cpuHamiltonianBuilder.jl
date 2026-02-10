# Date: 2026
#
#=
================================================================================
    HamiltonianBuilderHelper.jl - Unified Hamiltonian Construction
================================================================================

OVERVIEW
--------
Self-contained module for building QRC Hamiltonians. Includes:
- Core types: CouplingTerm, FieldTerm, HamiltonianParams
- Predefined Hamiltonian types (Heisenberg, TFIM, etc.)
- Geometry utilities for 2D lattice
- Sparse Hamiltonian construction (bitwise, no bitwise dependency)

USAGE
-----
    include("utils/helpers/HamiltonianBuilderHelper.jl")
    using .HamiltonianBuilderHelper

================================================================================
=#

module CPUHamiltonianBuilder

using LinearAlgebra
using SparseArrays

# ==============================================================================
# STRUCT DEFINITIONS
# ==============================================================================

"""
    CouplingTerm

Two-qubit coupling: H = Jxx·(σx⊗σx) + Jyy·(σy⊗σy) + Jzz·(σz⊗σz)

Note: Jxx/Jyy/Jzz are PAULI operator couplings.
The SPATIAL direction (x-axis vs y-axis) is determined by which
collection (x_bonds or y_bonds) the term belongs to.
"""
struct CouplingTerm
    i::Int          # First qubit index (1-indexed)
    j::Int          # Second qubit index (1-indexed)
    Jxx::Float64    # XX coupling (σx⊗σx)
    Jyy::Float64    # YY coupling (σy⊗σy)
    Jzz::Float64    # ZZ coupling (σz⊗σz)
end

"""Single-qubit field: H = hx·σx + hy·σy + hz·σz"""
struct FieldTerm
    idx::Int        # Qubit index (1-indexed)
    hx::Float64     # X field (σx coefficient)
    hy::Float64     # Y field (σy coefficient)
    hz::Float64     # Z field (σz coefficient)
end

"""
    HamiltonianParams

Full 2D grid Hamiltonian with anisotropic couplings.

# Fields
- `N`: Total qubits (Nx × Ny)
- `x_bonds`: Couplings along x-axis (horizontal) - each has Jxx, Jyy, Jzz
- `y_bonds`: Couplings along y-axis (vertical) - each has Jxx, Jyy, Jzz
- `fields`: Local magnetic fields
"""
struct HamiltonianParams
    N::Int
    x_bonds::Vector{CouplingTerm}   # Horizontal bonds (x-direction)
    y_bonds::Vector{CouplingTerm}   # Vertical bonds (y-direction)
    fields::Vector{FieldTerm}
end

export CouplingTerm, FieldTerm, HamiltonianParams
export get_hamiltonian_coupling, build_hamiltonian_parameters, build_hamiltonian_parameters_simple
export parse_config_couplings
export get_site_index, get_rail_position, get_total_qubits
export construct_sparse_hamiltonian
export HEISENBERG_XXX_X, TFIM_ZZ_X, XX_X

# ==============================================================================
# HAMILTONIAN TYPE DEFINITIONS
# ==============================================================================

"""Heisenberg XXX model with transverse X field: H = J(σxσx + σyσy + σzσz) + h·σx"""
const HEISENBERG_XXX_X = (type=:heisenberg_X, J=1.0, h=1.0)

"""Transverse-field Ising model: H = J·σzσz + h·σx"""
const TFIM_ZZ_X = (type=:TFIM_ZZ_X, J=1.0, h=1.0)

"""XY model with transverse X field: H = J(σxσx + σyσy) + h·σx"""
const XX_X = (type=:XX_X, J=1.0, h=1.0)

"""
    get_hamiltonian_coupling(ham_type; J=1.0, h=1.0)

Get standard coupling configuration for Hamiltonian type.

# Supported Types
| Symbol          | Coupling          | Field     |
|-----------------|-------------------|-----------|
| `:heisenberg_X` | Jxx=Jyy=Jzz=J     | hx=h      |
| `:TFIM_ZZ_X`    | Jzz=J (Jxx=Jyy=0) | hx=h      |
| `:XX_X`         | Jxx=Jyy=J, Jzz=0  | hx=h      |
| `:XXX_0`        | Jxx=Jyy=Jzz=J     | h=0       |

# Returns
- `(J_x_direction, J_y_direction, h_field)` tuples
"""
function get_hamiltonian_coupling(ham_type::Symbol; J=1.0, h=1.0)
    if ham_type == :heisenberg || ham_type == :heisenberg_X
        J_x_direction = (J, J, J)
        J_y_direction = (J, J, J)
        h_field = (h, 0.0, 0.0)
    elseif ham_type == :TFIM_ZZ_X
        J_x_direction = (0.0, 0.0, J)
        J_y_direction = (0.0, 0.0, J)
        h_field = (h, 0.0, 0.0)
    elseif ham_type == :heisenberg_no_field || ham_type == :XXX_0
        J_x_direction = (J, J, J)
        J_y_direction = (J, J, J)
        h_field = (0.0, 0.0, 0.0)
    elseif ham_type == :XX_X
        J_x_direction = (J, J, 0.0)
        J_y_direction = (J, J, 0.0)
        h_field = (h, 0.0, 0.0)
    else
        error("Unknown Hamiltonian type: $ham_type. Supported: :heisenberg_X, :TFIM_ZZ_X, :XXX_0, :XX_X")
    end
    return (J_x_direction, J_y_direction, h_field)
end

"""
    parse_config_couplings(hamiltonian_couplings, h_field_dict)

Parse CONFIG Dict structure into tuple format.
"""
function parse_config_couplings(hamiltonian_couplings::Dict, h_field_dict::Dict)
    J_x = hamiltonian_couplings["x_direction"]
    J_y = hamiltonian_couplings["y_direction"]

    J_x_direction = (J_x["Jxx"], J_x["Jyy"], J_x["Jzz"])
    J_y_direction = (J_y["Jxx"], J_y["Jyy"], J_y["Jzz"])
    h_field = (h_field_dict["hx"], h_field_dict["hy"], h_field_dict["hz"])

    return (J_x_direction, J_y_direction, h_field)
end

# ==============================================================================
# HAMILTONIAN PARAMS BUILDER
# ==============================================================================

"""
    build_hamiltonian_parameters(Nx, Ny, ham_type; J=1.0, h=1.0)
    build_hamiltonian_parameters(Nx, Ny; J_x_direction, J_y_direction, h_field)

Build HamiltonianParams for Nx × Ny 2D lattice.
"""
function build_hamiltonian_parameters(Nx::Int, Ny::Int, ham_type::Symbol; J=1.0, h=1.0)
    J_x_direction, J_y_direction, h_field = get_hamiltonian_coupling(ham_type; J=J, h=h)
    return build_hamiltonian_parameters(Nx, Ny; J_x_direction=J_x_direction, J_y_direction=J_y_direction, h_field=h_field)
end

function build_hamiltonian_parameters(Nx::Int, Ny::Int;
                                   J_x_direction::Tuple,
                                   J_y_direction::Tuple,
                                   h_field::Tuple)
    N = Nx * Ny

    # X-direction bonds: horizontal, within each row
    x_bonds = [CouplingTerm((r-1)*Nx + i, (r-1)*Nx + i + 1, J_x_direction...)
               for r in 1:Ny for i in 1:(Nx-1)]

    # Y-direction bonds: vertical, between adjacent rows
    y_bonds = CouplingTerm[]
    if Ny >= 2
        for i in 1:Nx
            push!(y_bonds, CouplingTerm(i, Nx + i, J_y_direction...))
        end
        for r in 2:(Ny-1)
            for i in 1:Nx
                push!(y_bonds, CouplingTerm((r-1)*Nx + i, r*Nx + i, J_y_direction...))
            end
        end
    end

    # Field terms: on all sites
    fields = [FieldTerm(i, h_field...) for i in 1:N]

    return HamiltonianParams(N, x_bonds, y_bonds, fields)
end

"""Simplified version that returns component arrays instead of HamiltonianParams."""
function build_hamiltonian_parameters_simple(L::Int, n_rails::Int, ham_type::Symbol; J=1.0, h=1.0)
    N = L * n_rails
    J_intra, J_inter, field = get_hamiltonian_coupling(ham_type; J=J, h=h)
    intra_pairs = [(r-1)*L + i => (r-1)*L + i + 1 for r in 1:n_rails for i in 1:(L-1)]
    inter_pairs = [i => L + i for i in 1:L]
    field_sites = collect(1:N)
    return (N=N, intra_pairs=intra_pairs, inter_pairs=inter_pairs,
            field_sites=field_sites, J_intra=J_intra, J_inter=J_inter, field=field)
end

# ==============================================================================
# SPARSE HAMILTONIAN CONSTRUCTION
# ==============================================================================

"""
    construct_sparse_hamiltonian(params) -> SparseMatrixCSC

Constructs a sparse Heisenberg Hamiltonian using bitwise indexing.
No bitwise operations dependency - builds COO format from scratch.
"""
function construct_sparse_hamiltonian(params::HamiltonianParams)
    N = params.N
    dim = 1 << N  # 2^N

    # COO format storage
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]

    sizehint!(rows, 4 * N * dim)
    sizehint!(cols, 4 * N * dim)
    sizehint!(vals, 4 * N * dim)

    # Process all couplings
    all_couplings = vcat(params.x_bonds, params.y_bonds)

    for coup in all_couplings
        i, j = coup.i, coup.j
        Jx, Jy, Jz = coup.Jxx, coup.Jyy, coup.Jzz

        stride_i = 1 << (i - 1)
        stride_j = 1 << (j - 1)
        mask = stride_i | stride_j

        for k in 0:(dim - 1)
            k_flipped = xor(k, mask)
            bit_i = (k & stride_i) != 0
            bit_j = (k & stride_j) != 0

            # XX term
            if abs(Jx) > 1e-14
                push!(rows, k + 1)
                push!(cols, k_flipped + 1)
                push!(vals, Complex(Jx))
            end

            # YY term
            if abs(Jy) > 1e-14
                yy_phase = (bit_i == bit_j) ? -1.0 : 1.0
                push!(rows, k + 1)
                push!(cols, k_flipped + 1)
                push!(vals, Complex(Jy * yy_phase))
            end

            # ZZ term (diagonal)
            if abs(Jz) > 1e-14
                zz_sign = (bit_i ? -1.0 : 1.0) * (bit_j ? -1.0 : 1.0)
                push!(rows, k + 1)
                push!(cols, k + 1)
                push!(vals, Complex(Jz * zz_sign))
            end
        end
    end

    # Process local fields
    for field in params.fields
        idx = field.idx
        hx, hy, hz = field.hx, field.hy, field.hz
        stride = 1 << (idx - 1)

        for k in 0:(dim - 1)
            bit_k = (k & stride) != 0
            k_flipped = xor(k, stride)

            # Z field: diagonal
            if abs(hz) > 1e-14
                z_sign = bit_k ? -1.0 : 1.0
                push!(rows, k + 1)
                push!(cols, k + 1)
                push!(vals, Complex(hz * z_sign))
            end

            # X field: off-diagonal
            if abs(hx) > 1e-14
                push!(rows, k + 1)
                push!(cols, k_flipped + 1)
                push!(vals, Complex(hx))
            end

            # Y field: off-diagonal with phase
            if abs(hy) > 1e-14
                y_phase = bit_k ? 1im : -1im
                push!(rows, k + 1)
                push!(cols, k_flipped + 1)
                push!(vals, hy * y_phase)
            end
        end
    end

    return sparse(rows, cols, vals, dim, dim)
end

# ==============================================================================
# GEOMETRY UTILITIES
# ==============================================================================

"""Convert (rail, position) to linear site index. 1-indexed."""
get_site_index(rail::Int, position::Int, L::Int) = (rail - 1) * L + position

"""Convert linear site index to (rail, position)."""
function get_rail_position(site::Int, L::Int)
    rail = div(site - 1, L) + 1
    position = mod(site - 1, L) + 1
    return (rail, position)
end

"""Get total number of qubits in ladder geometry."""
get_total_qubits(L::Int, n_rails::Int) = L * n_rails

end # module CPUHamiltonianBuilder

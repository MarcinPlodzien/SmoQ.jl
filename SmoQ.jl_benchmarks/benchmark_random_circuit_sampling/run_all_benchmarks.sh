#!/bin/bash
# run_all_benchmarks.sh - Run all benchmarks with 4 threads for fair comparison
#
# Runs 7 benchmarks:
#   1. SmoQ Float64              — baseline (ComplexF64 + @simd)
#   2. SmoQ Float64-Re_Im        — separated re/im layout + @simd
#   3. SmoQ Float32-AVX-Re_Im    — separated re/im + Float32 + @turbo
#   4. Qiskit Aer                — C++ backend (qasm_simulator)
#   5. Cirq                      — Google's framework
#   6. PennyLane Lightning       — Xanadu's C++ backend
#   7. QiliSDK                   — QiliSim backend
#
# Usage: ./run_all_benchmarks.sh

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "  Running ALL benchmarks with 4 threads"
echo "========================================================================"
echo ""
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

# 1. SmoQ Float64 (baseline)
echo "========================================================================"
echo "  1. SmoQ.jl Float64 (Julia) — baseline"
echo "========================================================================"
julia --threads=auto run_benchmark_smoq_Float64.jl

# 2. SmoQ Float64-Re_Im (separated layout)
echo ""
echo "========================================================================"
echo "  2. SmoQ.jl Float64-Re_Im (Julia) — separated layout"
echo "========================================================================"
julia --threads=auto run_benchmark_smoq_Float64_Re_Im.jl

# 3. SmoQ Float32-AVX-Re_Im (fully optimized)
echo ""
echo "========================================================================"
echo "  3. SmoQ.jl Float32-AVX-Re_Im (Julia) — fully optimized"
echo "========================================================================"
julia --threads=auto run_benchmark_smoq_Float32_AVX_Re_Im.jl

# 4. Qiskit
echo ""
echo "========================================================================"
echo "  4. Qiskit Aer (Python)"
echo "========================================================================"
python run_benchmark_qiskit.py

# 5. Cirq
echo ""
echo "========================================================================"
echo "  5. Cirq (Python)"
echo "========================================================================"
python run_benchmark_cirq.py

# 6. PennyLane
echo ""
echo "========================================================================"
echo "  6. PennyLane Lightning (Python)"
echo "========================================================================"
python run_benchmark_pennylane.py

# 7. QiliSDK
echo ""
echo "========================================================================"
echo "  7. QiliSDK (Python)"
echo "========================================================================"
python run_benchmark_qilisdk.py

echo ""
echo "========================================================================"
echo "  ALL BENCHMARKS COMPLETE!"
echo "========================================================================"
echo "  Saved timing files to parent directory."

# Random Circuit Sampling Benchmarks

Benchmark scripts for comparing quantum circuit simulation performance across different frameworks.

## Configuration
- **Gates per circuit**: 1000 (70% single-qubit, 30% two-qubit)
- **Shots**: 100
- **Qubit range**: 2, 4, 6, ..., 20
- **Threads**: 4 (set via `OMP_NUM_THREADS`; Julia uses `--threads=auto`)

## Quick Start

Run all 7 benchmarks in sequence:
```bash
./run_all_benchmarks.sh
```

## Scripts

### SmoQ.jl (Three-Tier Optimization Study)

| Script | Tier | Description |
|--------|------|-------------|
| `run_benchmark_smoq_Float64.jl` | Float64 (baseline) | ComplexF64 + gate fusion + `@simd` |
| `run_benchmark_smoq_Float64_Re_Im.jl` | Float64-Re_Im | Separated re/im layout + `@simd` |
| `run_benchmark_smoq_Float32_AVX_Re_Im.jl` | Float32-AVX-Re_Im | Float32 + separated re/im + `@turbo` (AVX) |

### External Frameworks

| Script | Framework | Backend |
|--------|-----------|---------|
| `run_benchmark_qiskit.py` | Qiskit Aer | C++ (qasm_simulator) |
| `run_benchmark_cirq.py` | Cirq | Python |
| `run_benchmark_pennylane.py` | PennyLane Lightning | C++ |
| `run_benchmark_qilisdk.py` | QiliSDK | QiliSim + QuTiP |

## Requirements

### Julia
```bash
# SmoQ.jl and dependencies are loaded from the local project
julia --threads=auto run_benchmark_smoq_Float64.jl
```

### Python
```bash
pip install qiskit qiskit-aer cirq pennylane pennylane-lightning qilisdk
```

## Output

Each script saves timing results to `timings_<framework>.txt`:

| File | Framework |
|------|-----------|
| `timings_smoq_Float64.txt` | SmoQ Float64 baseline |
| `timings_smoq_Float64_Re_Im.txt` | SmoQ Float64-Re_Im |
| `timings_smoq_Float32_AVX_Re_Im.txt` | SmoQ Float32-AVX-Re_Im |
| `timings_qiskit_aer.txt` | Qiskit Aer |
| `timings_cirq.txt` | Cirq |
| `timings_pennylane.txt` | PennyLane Lightning |
| `timings_qilisdk_qilisim.txt` | QiliSDK (QiliSim) |
| `timings_qilisdk_qutip.txt` | QiliSDK (QuTiP, N ≤ 10) |

## Figures

Generate figures from the parent directory:
```bash
julia get_benchmark_figure.jl       # log-scale timing comparison
julia get_speedup_figure.jl         # three-row speedup bar chart (all N)
julia get_speedup_figure_large_N.jl # three-row speedup bar chart (N ≥ 10)
```

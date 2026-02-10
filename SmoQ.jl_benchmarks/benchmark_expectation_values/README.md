# Expectation Value Benchmark

Timing benchmark for state preparation and observable computation across quantum simulation frameworks.

## What is Benchmarked

Each framework performs the **exact same physics task**:
1. **State preparation** — apply a deterministic 4-layer brickwork circuit to |00…0⟩
2. **Observable computation** — compute exact ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for all N qubits + ⟨XX⟩, ⟨YY⟩, ⟨ZZ⟩ for all nearest-neighbour pairs

All expectation values are computed **exactly** from the full statevector (Born rule), with **no sampling** involved.

### Circuit Structure (per layer × 4 layers)

| Step | Operation |
|------|-----------|
| 1 | H on even qubits (1-indexed) |
| 2 | Ry(θ) on all qubits |
| 3 | CNOT on odd bonds |
| 4 | S on odd qubits |
| 5 | Rx(θ) on all qubits |
| 6 | CZ on even bonds |

Rotation angles are fixed across all frameworks to ensure identical results.

## How to Compare with the Random Circuit Sampling Benchmark

This benchmark and the [`benchmark/`](../../../benchmark/) directory measure **fundamentally different workloads**:

| | This Benchmark (Expectation Values) | Random Circuit Sampling (`benchmark/`) |
|---|---|---|
| **Circuit** | Fixed 4-layer ansatz (~24N gates) | 1000 random gates |
| **Task** | 1× statevector + exact observables | 100 shots × (init + apply + measure) |
| **SmoQ advantage** | Matrix-free bitwise kernels + zero Python overhead → **6–10× faster** | Qiskit/PennyLane C++ backends amortize per-gate cost over 100 shots → **competitive** |

**Why SmoQ wins here but not in sampling:**
- The sampling benchmark multiplies 100 shots × 1000 gates. Qiskit compiles the circuit to C++ once and runs all shots internally. PennyLane Lightning does the same. SmoQ re-initializes per shot from Julia.
- Here, the workload is a single statevector simulation + bitwise observable computation. SmoQ's matrix-free kernels and Julia's native SIMD compilation dominate.

## Results (N = 2 to 24)

```
  N    │  SmoQ F64  │  SmoQ AVX  │    Qiskit  │      Cirq  │  PennyLane │   QiliSDK
  ─────┼────────────┼────────────┼────────────┼────────────┼────────────┼──────────
   2   │   0.000006 │   0.000006 │   0.002183 │   0.005757 │   0.008442 │  0.002205
   4   │   0.000010 │   0.000009 │   0.002994 │   0.008114 │   0.011204 │  0.002169
   8   │   0.000057 │   0.000048 │   0.006506 │   0.017431 │   0.048398 │  0.005169
  12   │   0.000826 │   0.000832 │   0.012757 │   0.031745 │   0.029037 │  0.034394
  16   │   0.019295 │   0.015298 │   0.235381 │   0.432251 │   0.049268 │  0.885648
  20   │   0.489213 │   0.314780 │   3.323554 │   5.147435 │   0.832997 │ 25.967246
  24   │  12.489273 │   8.861429 │  85.688816 │  69.511367 │  28.166562 │       OOM
```

> **Note:** QiliSDK only performs Born rule computation (state preparation). Its digital API does not support exact expectation value calculation — only sampling.

## Quick Start

Run all benchmarks and generate the plot:
```bash
julia bench_smoq.jl
julia bench_smoq_avx.jl
python3 bench_qiskit.py
python3 bench_cirq.py
python3 bench_pennylane.py
python3 bench_qilisdk.py
julia plot_timing.jl
```

## Scripts

| Script | Framework | Observable Method |
|--------|-----------|-------------------|
| `bench_smoq.jl` | SmoQ.jl (ComplexF64) | Bitwise matrix-free |
| `bench_smoq_avx.jl` | SmoQ.jl (Float32 AVX) | Bitwise matrix-free (SIMD) |
| `bench_qiskit.py` | Qiskit | `SparsePauliOp` + `Statevector.expectation_value()` |
| `bench_cirq.py` | Cirq | `PauliString.expectation_from_state_vector()` |
| `bench_pennylane.py` | PennyLane Lightning | `qml.expval(PauliX/Y/Z)` |
| `bench_qilisdk.py` | QiliSDK (QiliSim) | N/A (Born rule only) |
| `plot_timing.jl` | — | Collects data, prints table, saves CSV + PNG |

## Output

- `timing_<framework>.txt` — per-framework timing data
- `timing_all.csv` — combined CSV for all frameworks
- `fig_observables_calculation_evaluation_vs_time.png` — log-scale plot of execution time vs N
- `fig_observables_calculation_speedup.png` — speedup ratio bar chart (SmoQ.jl vs others)

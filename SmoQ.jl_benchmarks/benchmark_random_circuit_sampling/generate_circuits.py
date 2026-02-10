#!/usr/bin/env python3
"""
Generate shared random circuits for benchmarking.

All frameworks load the SAME circuits from these files,
ensuring a perfectly fair comparison.

Gate format (one gate per line):
  GATE_NAME QUBIT1 [QUBIT2] [THETA]

Qubit indices are 0-indexed.
Single-qubit gates: H 2  or  RX 3 1.2345
Two-qubit gates:    CNOT 0 3  or  CZ 1 2

Usage:
  python generate_circuits.py
"""

import os
import random

# ==============================================================================
# CONFIGURATION — Must match all benchmark scripts
# ==============================================================================
SEED = 42
N_GATES = 1000
N_RANGE = range(2, 22, 2)  # 2, 4, 6, ..., 20
SINGLE_GATE_RATIO = 0.7

SINGLE_GATES = ["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"]
TWO_GATES = ["CNOT", "CZ"]
ROTATION_GATES = {"RX", "RY", "RZ"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CIRCUIT_DIR = os.path.join(SCRIPT_DIR, "circuits")

# ==============================================================================
# GENERATION
# ==============================================================================

def generate_circuit(n_qubits, n_gates, rng):
    """Generate a random circuit as a list of gate descriptions."""
    gates = []
    for _ in range(n_gates):
        if rng.random() < SINGLE_GATE_RATIO or n_qubits == 1:
            gate = rng.choice(SINGLE_GATES)
            q = rng.randint(0, n_qubits - 1)
            if gate in ROTATION_GATES:
                theta = rng.uniform(0, 2 * 3.141592653589793)
                gates.append(f"{gate} {q} {theta:.15f}")
            else:
                gates.append(f"{gate} {q}")
        else:
            gate = rng.choice(TWO_GATES)
            q1 = rng.randint(0, n_qubits - 1)
            q2 = rng.randint(0, n_qubits - 2)
            if q2 >= q1:
                q2 += 1
            gates.append(f"{gate} {q1} {q2}")
    return gates


def main():
    os.makedirs(CIRCUIT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Generating shared benchmark circuits")
    print("=" * 60)
    print(f"  Seed: {SEED}")
    print(f"  Gates per circuit: {N_GATES}")
    print(f"  Single/two-qubit ratio: {SINGLE_GATE_RATIO:.0%} / {1-SINGLE_GATE_RATIO:.0%}")
    print()

    rng = random.Random(SEED)

    for n in N_RANGE:
        filename = f"circuit_N{n:02d}.txt"
        filepath = os.path.join(CIRCUIT_DIR, filename)

        gates = generate_circuit(n, N_GATES, rng)

        with open(filepath, "w") as f:
            f.write(f"# N={n} n_gates={N_GATES} seed={SEED} single_ratio={SINGLE_GATE_RATIO}\n")
            for g in gates:
                f.write(g + "\n")

        print(f"  ✓ {filename}  ({n:2d} qubits, {N_GATES} gates)")

    print(f"\n  Saved to: circuits/")


if __name__ == "__main__":
    main()

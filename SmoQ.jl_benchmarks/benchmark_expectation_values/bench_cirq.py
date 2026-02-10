#!/usr/bin/env python3
"""bench_cirq.py - Cirq timing sweep over N"""
import numpy as np, time, cirq, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
L = 4
ANGLES_RY = [np.pi/4, np.pi/3, np.pi/6, np.pi/5, np.pi/7, np.pi/8,
             np.pi/9, np.pi/10, np.pi/11, np.pi/12, np.pi/13, np.pi/14,
             np.pi/15, np.pi/16, np.pi/17, np.pi/18, np.pi/19, np.pi/20,
             np.pi/21, np.pi/22, np.pi/23, np.pi/24, np.pi/25, np.pi/26]
ANGLES_RX = [np.pi/5, np.pi/7, np.pi/4, np.pi/9, np.pi/3, np.pi/11,
             np.pi/6, np.pi/13, np.pi/8, np.pi/10, np.pi/12, np.pi/14,
             np.pi/15, np.pi/17, np.pi/16, np.pi/19, np.pi/18, np.pi/20,
             np.pi/21, np.pi/22, np.pi/23, np.pi/24, np.pi/25, np.pi/26]
N_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

def build_circuit(N):
    qubits = cirq.LineQubit.range(N); circuit = cirq.Circuit()
    for layer in range(L):
        for q in range(1, N, 2):     circuit.append(cirq.H(qubits[q]))
        for q in range(N):           circuit.append(cirq.ry(ANGLES_RY[q])(qubits[q]))
        for q in range(0, N-1, 2):   circuit.append(cirq.CNOT(qubits[q], qubits[q+1]))
        for q in range(0, N, 2):     circuit.append(cirq.S(qubits[q]))
        for q in range(N):           circuit.append(cirq.rx(ANGLES_RX[q])(qubits[q]))
        for q in range(1, N-1, 2):   circuit.append(cirq.CZ(qubits[q], qubits[q+1]))
    return circuit, qubits

def compute_observables(result, qubits, N):
    psi = result.final_state_vector
    qm = {qb: i for i, qb in enumerate(qubits)}
    for q in range(N):
        cirq.X(qubits[q]).expectation_from_state_vector(psi, qubit_map=qm)
        cirq.Y(qubits[q]).expectation_from_state_vector(psi, qubit_map=qm)
        cirq.Z(qubits[q]).expectation_from_state_vector(psi, qubit_map=qm)
    for b in range(N-1):
        q1, q2 = qubits[b], qubits[b+1]
        (cirq.X(q1)*cirq.X(q2)).expectation_from_state_vector(psi, qubit_map=qm)
        (cirq.Y(q1)*cirq.Y(q2)).expectation_from_state_vector(psi, qubit_map=qm)
        (cirq.Z(q1)*cirq.Z(q2)).expectation_from_state_vector(psi, qubit_map=qm)

print("=" * 60); print("  Cirq — Timing Sweep"); print("=" * 60)
output_file = os.path.join(SCRIPT_DIR, "timing_cirq.txt")
with open(output_file, "w") as f:
    f.write("# Cirq timing\n# N  time_s\n")
    for N_val in N_VALUES:
        t = time.perf_counter()
        circuit, qubits = build_circuit(N_val)
        result = cirq.Simulator(dtype=np.complex128).simulate(circuit)
        compute_observables(result, qubits, N_val)
        dt = time.perf_counter() - t
        f.write(f"{N_val}  {dt:.6f}\n"); print(f"  N = {N_val:2d} : {dt:.6f} s")
print(f"  Saved: {os.path.basename(output_file)}")

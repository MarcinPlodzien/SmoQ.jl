#!/usr/bin/env python3
"""bench_qiskit.py - Qiskit timing sweep over N"""
import numpy as np, time, os
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

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
    qc = QuantumCircuit(N)
    for layer in range(L):
        for q in range(1, N, 2):     qc.h(q)
        for q in range(N):           qc.ry(ANGLES_RY[q], q)
        for q in range(0, N-1, 2):   qc.cx(q, q+1)
        for q in range(0, N, 2):     qc.s(q)
        for q in range(N):           qc.rx(ANGLES_RX[q], q)
        for q in range(1, N-1, 2):   qc.cz(q, q+1)
    return qc

def compute_observables(sv, N):
    for q in range(N):
        for pauli in ['X', 'Y', 'Z']:
            label = 'I'*(N-1-q) + pauli + 'I'*q
            sv.expectation_value(SparsePauliOp(label)).real
    for bond in range(N-1):
        for pauli in ['XX', 'YY', 'ZZ']:
            lbl = ['I']*N; lbl[bond] = pauli[0]; lbl[bond+1] = pauli[1]
            sv.expectation_value(SparsePauliOp(''.join(reversed(lbl)))).real

print("=" * 60); print("  Qiskit — Timing Sweep"); print("=" * 60)
output_file = os.path.join(SCRIPT_DIR, "timing_qiskit.txt")
with open(output_file, "w") as f:
    f.write("# Qiskit timing\n# N  time_s\n")
    for N_val in N_VALUES:
        t = time.perf_counter()
        qc = build_circuit(N_val)
        sv = Statevector.from_instruction(qc)
        compute_observables(sv, N_val)
        dt = time.perf_counter() - t
        f.write(f"{N_val}  {dt:.6f}\n"); print(f"  N = {N_val:2d} : {dt:.6f} s")
print(f"  Saved: {os.path.basename(output_file)}")

#!/usr/bin/env python3
"""bench_pennylane.py - PennyLane Lightning timing sweep over N"""
import numpy as np, time, pennylane as qml, os

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

def apply_circuit(N):
    for layer in range(L):
        for q in range(1, N, 2):     qml.Hadamard(wires=q)
        for q in range(N):           qml.RY(ANGLES_RY[q], wires=q)
        for q in range(0, N-1, 2):   qml.CNOT(wires=[q, q+1])
        for q in range(0, N, 2):     qml.S(wires=q)
        for q in range(N):           qml.RX(ANGLES_RX[q], wires=q)
        for q in range(1, N-1, 2):   qml.CZ(wires=[q, q+1])

def run_benchmark(N_val):
    dev = qml.device("lightning.qubit", wires=N_val)
    @qml.qnode(dev)
    def get_state():
        apply_circuit(N_val); return qml.state()
    @qml.qnode(dev)
    def get_single_q():
        apply_circuit(N_val)
        return ([qml.expval(qml.PauliX(q)) for q in range(N_val)] +
                [qml.expval(qml.PauliY(q)) for q in range(N_val)] +
                [qml.expval(qml.PauliZ(q)) for q in range(N_val)])
    @qml.qnode(dev)
    def get_nn_corr():
        apply_circuit(N_val)
        return ([qml.expval(qml.PauliX(b) @ qml.PauliX(b+1)) for b in range(N_val-1)] +
                [qml.expval(qml.PauliY(b) @ qml.PauliY(b+1)) for b in range(N_val-1)] +
                [qml.expval(qml.PauliZ(b) @ qml.PauliZ(b+1)) for b in range(N_val-1)])
    t = time.perf_counter()
    get_state(); get_single_q(); get_nn_corr()
    return time.perf_counter() - t

print("=" * 60); print("  PennyLane Lightning — Timing Sweep"); print("=" * 60)
output_file = os.path.join(SCRIPT_DIR, "timing_pennylane.txt")
with open(output_file, "w") as f:
    f.write("# PennyLane Lightning timing\n# N  time_s\n")
    for N_val in N_VALUES:
        dt = run_benchmark(N_val)
        f.write(f"{N_val}  {dt:.6f}\n"); print(f"  N = {N_val:2d} : {dt:.6f} s")
print(f"  Saved: {os.path.basename(output_file)}")

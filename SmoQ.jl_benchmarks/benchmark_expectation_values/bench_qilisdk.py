#!/usr/bin/env python3
"""bench_qilisdk.py - QiliSDK (QiliSim) timing sweep — Born rule only"""
import numpy as np, time, os
from qilisdk.digital import Circuit, H, S, RX, RY, CNOT, CZ, M
from qilisdk.backends.qilisim import QiliSim
from qilisdk.functionals import Sampling

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
N_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

def build_and_run(N):
    circuit = Circuit(N)
    for layer in range(L):
        for q in range(1, N, 2):     circuit.add(H(q))
        for q in range(N):           circuit.add(RY(q, theta=ANGLES_RY[q]))
        for q in range(0, N-1, 2):   circuit.add(CNOT(q, q+1))
        for q in range(0, N, 2):     circuit.add(S(q))
        for q in range(N):           circuit.add(RX(q, theta=ANGLES_RX[q]))
        for q in range(1, N-1, 2):   circuit.add(CZ(q, q+1))
    for q in range(N): circuit.add(M(q))
    return QiliSim().execute(Sampling(circuit, 1))

print("=" * 60); print("  QiliSDK (QiliSim) — Timing Sweep (Born rule)"); print("=" * 60)
output_file = os.path.join(SCRIPT_DIR, "timing_qilisdk.txt")
with open(output_file, "w") as f:
    f.write("# QiliSDK (QiliSim) timing — Born rule\n# N  time_s\n")
    for N_val in N_VALUES:
        t = time.perf_counter()
        build_and_run(N_val)
        dt = time.perf_counter() - t
        f.write(f"{N_val}  {dt:.6f}\n"); f.flush()
        print(f"  N = {N_val:2d} : {dt:.6f} s")
print(f"  Saved: {os.path.basename(output_file)}")

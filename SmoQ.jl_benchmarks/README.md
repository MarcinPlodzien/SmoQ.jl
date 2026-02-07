# SmoQ.jl Benchmarks

**[SmoQ.jl](https://github.com/MarcinPlodzien/SmoQ.jl)** — CPU-Based Matrix-Free Quantum Simulator in Julia

---

## Philosophy: Matrix-Free Quantum Simulation

Traditional quantum simulators represent quantum gates as full unitary matrices and apply them to state vectors via matrix-vector multiplication. For an $N$-qubit system, the state vector lives in a Hilbert space of dimension $d = 2^N$, so each gate naively requires constructing and storing a $2^N \times 2^N$ matrix. This is exponentially expensive: for $N = 20$ qubits, a single gate matrix contains $\sim 10^{12}$ entries, requiring terabytes of memory. Even for moderate $N$, the matrix construction and multiplication dominate the computational cost.

**SmoQ.jl** takes a different approach. It recognizes that quantum gates — even when written as large matrices — have a highly structured action: a single-qubit gate really only performs a $2 \times 2$ rotation, and a two-qubit gate performs a $4 \times 4$ one. The remaining structure of the full $2^N \times 2^N$ matrix is just identity (acting trivially on all other qubits). SmoQ.jl exploits this by **never constructing gate matrices at all**. Instead, it identifies which amplitudes are coupled by a given gate using **bitwise operations on the state vector indices**, and updates only those amplitudes in-place.

This matrix-free strategy reduces the cost of applying a gate from $O(4^N)$ (matrix-vector multiply) to $O(2^N)$ (a single pass over the state vector), with **zero additional memory** for gate storage. The same bitwise philosophy extends to computing expectation values and partial traces.

| Traditional Approach | SmoQ.jl Approach |
|---------------------|------------------|
| Construct $2^N \times 2^N$ gate matrices | **No matrices** — gates act via bitwise index manipulation |
| $O(4^N)$ matrix-vector products per gate | **$O(2^N)$** direct amplitude updates per gate |
| $O(4^N)$ memory for a single gate | **$O(1)$** auxiliary memory (just the $2 \times 2$ or $4 \times 4$ parameters) |
| Full matrix construction for observables | **Bitwise reduction** over amplitudes |

---

## Bitwise Tricks for Gates

### State Vector Representation

A quantum state of $N$ qubits is stored as a vector of $2^N$ complex amplitudes $\psi_i$, where each index $i \in \{0, 1, \ldots, 2^N - 1\}$ labels a computational basis state. The binary representation of the index directly encodes the qubit configuration:

$$|i\rangle = |b_{N-1} \ldots b_1 b_0\rangle \quad \text{where } i = \sum_{k=0}^{N-1} b_k \cdot 2^k$$

Here, $b_k \in \{0, 1\}$ is the value of qubit $k$. The least significant bit ($b_0$) corresponds to qubit 0, and the most significant bit ($b_{N-1}$) to qubit $N-1$.

**Example with $N = 4$ qubits:** The index $i = 11 = 1011_2$ represents the state $|1011\rangle$, meaning qubit 0 is in $|1\rangle$, qubit 1 is in $|1\rangle$, qubit 2 is in $|0\rangle$, and qubit 3 is in $|1\rangle$. The full state vector has $2^4 = 16$ complex entries.

This binary encoding is the key that makes bitwise operations so natural for quantum simulation: the structure of the Hilbert space is already encoded in the bit patterns of array indices.

### Single-Qubit Gates

A single-qubit gate on qubit $k$ is described by a $2 \times 2$ unitary matrix $U$. In the full $2^N$-dimensional Hilbert space, this gate acts as $I \otimes \cdots \otimes U \otimes \cdots \otimes I$ — identity on every qubit except $k$. This means the gate only affects bit $k$ of the index, coupling pairs of amplitudes that are identical in every bit except bit $k$.

**The XOR trick:** Given any index $i$, its "partner" under a gate on qubit $k$ is:

$$j = i \oplus 2^k$$

where $\oplus$ denotes bitwise XOR and $2^k$ is a single-bit mask. This operation flips exactly bit $k$ of $i$, leaving all other bits unchanged. If $i$ has bit $k = 0$, then $j$ has bit $k = 1$, and vice versa — exactly the two states that the $2 \times 2$ gate couples.

**Algorithm:** Loop over all indices $i$ where bit $k$ is 0 (there are $2^{N-1}$ such indices). For each:

1. Compute partner: $j = i \oplus 2^k$ (bit $k$ is now 1)
2. Read the amplitude pair: $(\psi_i, \psi_j)$ — these correspond to qubit $k$ being in $|0\rangle$ and $|1\rangle$
3. Apply the $2 \times 2$ gate matrix:

$$\psi'_i = u_{00}\,\psi_i + u_{01}\,\psi_j$$

$$\psi'_j = u_{10}\,\psi_i + u_{11}\,\psi_j$$

4. Write back the updated amplitudes

The condition "bit $k$ is 0" ensures each pair is processed exactly once. It is checked via the bitwise test `(i >> k) & 1 == 0`.

**Why this works:** The full $2^N \times 2^N$ unitary matrix $I \otimes \cdots \otimes U \otimes \cdots \otimes I$ is block-diagonal, consisting of $2^{N-1}$ identical $2 \times 2$ blocks. The bitwise approach directly applies each block without ever assembling the full matrix.

**Concrete single-qubit gate examples:**

| Gate | Matrix $(u_{00}, u_{01}, u_{10}, u_{11})$ | Description |
|------|------------------------------------------|-------------|
| $R_x(\theta)$ | $(\cos\frac{\theta}{2},\; -i\sin\frac{\theta}{2},\; -i\sin\frac{\theta}{2},\; \cos\frac{\theta}{2})$ | Rotation about $X$-axis. Mixes ∣0⟩ ↔ ∣1⟩ with imaginary coupling. |
| $R_y(\theta)$ | $(\cos\frac{\theta}{2},\; -\sin\frac{\theta}{2},\; \sin\frac{\theta}{2},\; \cos\frac{\theta}{2})$ | Rotation about $Y$-axis. Real-valued — no complex phases involved. |
| $R_z(\theta)$ | $(e^{-i\theta/2},\; 0,\; 0,\; e^{i\theta/2})$ | Rotation about $Z$-axis. Diagonal — no mixing between $\psi_i$ and $\psi_j$; only phases change. |
| $H$ | $\frac{1}{\sqrt{2}}(1,\; 1,\; 1,\; -1)$ | Hadamard gate. Maps ∣0⟩ → (∣0⟩ + ∣1⟩)/√2, ∣1⟩ → (∣0⟩ − ∣1⟩)/√2. |
| $X$ | $(0,\; 1,\; 1,\; 0)$ | Pauli-$X$ (NOT gate). Swaps $\psi_i \leftrightarrow \psi_j$, i.e., flips qubit $k$. |

Note that diagonal gates like $R_z$ are particularly efficient: since $u_{01} = u_{10} = 0$, each amplitude is simply multiplied by a phase factor without accessing its partner. SmoQ.jl can optimize this case to avoid unnecessary memory reads.

### Two-Qubit Gates

A two-qubit gate on qubits $k$ and $l$ acts non-trivially on two bits of the index, coupling **quartets** of amplitudes. The four coupled states correspond to the four combinations of bits $k$ and $l$: $(0,0)$, $(0,1)$, $(1,0)$, $(1,1)$.

**Quartet construction:** For each "base" index $i_{00}$ where both bit $k$ and bit $l$ are 0, the four coupled indices are:

| Index | Bit $k$ | Bit $l$ | Construction |
|-------|---------|---------|-------------|
| $i_{00}$ | 0 | 0 | Base index (unchanged) |
| $i_{01}$ | 0 | 1 | $i_{00} \oplus 2^l$ |
| $i_{10}$ | 1 | 0 | $i_{00} \oplus 2^k$ |
| $i_{11}$ | 1 | 1 | $i_{00} \oplus 2^k \oplus 2^l$ |

More compactly: $i_{ab} = i_{00} \oplus a \cdot 2^k \oplus b \cdot 2^l$.

There are $2^{N-2}$ independent quartets, and each undergoes a $4 \times 4$ transformation specified by the gate. Again, no $2^N \times 2^N$ matrix is constructed.

**Concrete two-qubit gate examples:**

**CZ (Controlled-Z):** This is a diagonal gate — it only modifies the phase of the $|11\rangle$ component:

$$\psi'_{i_{11}} = -\psi_{i_{11}}, \quad \text{all others unchanged}$$

Since only one amplitude per quartet changes (and it is just a sign flip), CZ is extremely cheap to apply.

**CNOT (Controlled-NOT), control $k$, target $l$:** If bit $k = 1$, the target bit $l$ is flipped. In practice, this swaps two amplitudes within each quartet:

$$\psi'_{i_{10}} \leftrightarrow \psi'_{i_{11}}$$

The $|00\rangle$ and $|01\rangle$ components (where the control bit is 0) are left unchanged.

**Explicit two-qubit rotation gates** (acting on the ordered basis $|00\rangle, |01\rangle, |10\rangle, |11\rangle$):

**$R_{zz}(\theta)$** — A diagonal gate that applies phases based on the parity of the two qubits. When both qubits agree ($|00\rangle$ or $|11\rangle$, even parity), the phase is $e^{-i\theta/2}$; when they disagree ($|01\rangle$ or $|10\rangle$, odd parity), it is $e^{+i\theta/2}$:

$$\psi'_{00} = e^{-i\theta/2}\,\psi_{00},\quad \psi'_{01} = e^{+i\theta/2}\,\psi_{01},\quad \psi'_{10} = e^{+i\theta/2}\,\psi_{10},\quad \psi'_{11} = e^{-i\theta/2}\,\psi_{11}$$

Being diagonal, $R_{zz}$ requires no amplitude mixing — just phase multiplications.

**$R_{xx}(\theta)$** — Couples states that differ in both bits: $|00\rangle \leftrightarrow |11\rangle$ and $|01\rangle \leftrightarrow |10\rangle$:

$$\psi'_{00} = c\,\psi_{00} - is\,\psi_{11},\quad \psi'_{11} = -is\,\psi_{00} + c\,\psi_{11}$$

$$\psi'_{01} = c\,\psi_{01} - is\,\psi_{10},\quad \psi'_{10} = -is\,\psi_{01} + c\,\psi_{10}$$

This gate generates entanglement by mixing computational basis states that differ in both qubits simultaneously.

**$R_{yy}(\theta)$** — Similar coupling structure but with opposite sign conventions on the $(00, 11)$ pair:

$$\psi'_{00} = c\,\psi_{00} + is\,\psi_{11},\quad \psi'_{11} = is\,\psi_{00} + c\,\psi_{11}$$

$$\psi'_{01} = c\,\psi_{01} - is\,\psi_{10},\quad \psi'_{10} = -is\,\psi_{01} + c\,\psi_{10}$$

where $c = \cos(\theta/2)$ and $s = \sin(\theta/2)$ in both cases.

### Complexity Advantage

The matrix-free approach provides a significant computational advantage that grows exponentially with the number of qubits:

| Method | Memory for gate | Operations per gate | Total for $G$-gate circuit |
|--------|----------------|--------------------|----|
| Full matrix construction | $O(4^N)$ | $O(4^N)$ | $O(G \cdot 4^N)$ |
| **Bitwise (SmoQ.jl)** | $O(1)$ | $O(2^N)$ | $O(G \cdot 2^N)$ |

The speedup factor is $2^N$ per gate. To put this in perspective:

| $N$ (qubits) | Speedup factor $2^N$ | Practical impact |
|---|---|---|
| 10 | $\sim 10^3$ | ~1000× faster |
| 16 | $\sim 6.5 \times 10^4$ | Matrix approach becomes impractical |
| 20 | $\sim 10^6$ | Bitwise simulates in seconds what matrix methods cannot store |
| 26 | $\sim 6.7 \times 10^7$ | SmoQ.jl handles this on a laptop; matrices would need exabytes |

Beyond speed, the memory savings are equally important: the matrix-free approach only stores the $2^N$-element state vector plus the gate parameters (a few floating-point numbers), while the matrix approach requires storing and filling a $2^N \times 2^N$ array.

---

## Bitwise Tricks for Observables

Computing expectation values $\langle \psi | O | \psi \rangle$ is a core operation in quantum simulation — it extracts physical predictions from the quantum state. SmoQ.jl computes these without constructing the observable matrix, using the same bitwise philosophy as for gates.

The key insight is that the Pauli operators $\{I, X, Y, Z\}$ — the building blocks of all observables — each have a simple, predictable action on computational basis indices:
- **$I$** (identity): does nothing, can be ignored
- **$Z$**: diagonal — contributes a sign $(-1)^{b_k}$ based on bit $k$, with no amplitude coupling
- **$X$**: flips bit $k$ — couples amplitude pairs $\psi_i \leftrightarrow \psi_{i \oplus 2^k}$
- **$Y$**: flips bit $k$ with an extra phase $\pm i$ — couples the same pairs as $X$ but with imaginary coefficients

### Diagonal Observables ($Z$-type)

**Single-qubit $\langle Z_k \rangle$:** Since $Z|0\rangle = +|0\rangle$ and $Z|1\rangle = -|1\rangle$, the operator $Z_k$ acting on qubit $k$ is diagonal with eigenvalue $(-1)^{b_k}$:

$$\langle Z_k \rangle = \sum_{i=0}^{2^N-1} (-1)^{b_k(i)} \lvert\psi_i\rvert^2$$

where $b_k(i)$ extracts bit $k$ from index $i$ via the bitwise operation `(i >> k) & 1` (right-shift by $k$ positions, then AND with 1).

**In plain language:** loop over all $2^N$ amplitudes, compute $\lvert\psi_i\rvert^2$, and add it with a $+$ sign if qubit $k$ is in $\vert 0\rangle$ (bit $k$ of $i$ is 0) or a $-$ sign if qubit $k$ is in $\vert 1\rangle$ (bit $k$ of $i$ is 1). The result is just a weighted sum of probabilities — cost $O(2^N)$, no matrix needed.

**Two-qubit $\langle Z_k Z_l \rangle$:** The tensor product $Z_k \otimes Z_l$ is also diagonal. Its eigenvalue is $(-1)^{b_k \oplus b_l}$, where $\oplus$ denotes XOR. The sign is $+1$ when both qubits have the same value (both $|0\rangle$ or both $|1\rangle$) and $-1$ when they differ:

$$\langle Z_k Z_l \rangle = \sum_{i=0}^{2^N-1} (-1)^{b_k(i) \oplus b_l(i)} \lvert\psi_i\rvert^2$$

This generalizes naturally: for any product of $Z$ operators on qubits $\{k_1, k_2, \ldots, k_m\}$, the sign is determined by the parity (XOR) of the corresponding bits.

### Non-Diagonal Observables ($X$- and $Y$-type)

**Single-qubit $\langle X_k \rangle$:** The Pauli-$X$ operator flips bit $k$, mapping $|b_k\rangle \to |1 - b_k\rangle$. This means $\langle i | X_k | j \rangle$ is nonzero only when $j = i \oplus 2^k$ (indices differing in exactly bit $k$). The expectation value involves products of amplitudes at paired indices:

$$\langle X_k \rangle = \sum_{i: b_k(i)=0} 2 \cdot \text{Re}\left(\psi_i^* \psi_{i \oplus 2^k}\right)$$

We restrict the sum to indices with bit $k = 0$ to count each pair once (since the pair $(i, j)$ with $j = i \oplus 2^k$ would otherwise be counted twice). The factor of 2 and $\text{Re}(\cdot)$ come from combining the two conjugate contributions: $\psi_i^* \psi_j + \psi_j^* \psi_i = 2\,\text{Re}(\psi_i^* \psi_j)$.

**Single-qubit $\langle Y_k \rangle$:** The Pauli-$Y$ operator also flips bit $k$ but introduces an imaginary phase: $Y = -iXZ$, so $Y|0\rangle = i|1\rangle$ and $Y|1\rangle = -i|0\rangle$. This changes the real part to an imaginary part:

$$\langle Y_k \rangle = \sum_{i: b_k(i)=0} 2 \cdot \text{Im}\left(\psi_i^* \psi_{i \oplus 2^k}\right)$$

The structure is identical to $\langle X_k \rangle$ — same pairs of indices, same loop — just taking the imaginary part instead of the real part.

### Arbitrary Pauli Strings

Any observable in quantum computing can be decomposed into a sum of **Pauli strings** — tensor products of single-qubit Pauli operators. For a general $N$-qubit Pauli string:

$$P = \sigma_1 \otimes \sigma_2 \otimes \cdots \otimes \sigma_N, \quad \sigma_k \in \{I, X, Y, Z\}$$

(for example, $P = X \otimes Z \otimes I \otimes Y$ on 4 qubits), we partition the qubits into four sets:

- $S_I$: qubits with identity $I$ — these contribute nothing and are ignored
- $S_Z$: qubits with $Z$ — diagonal, each contributes a sign $(-1)^{b_k}$
- $S_X$: qubits with $X$ — off-diagonal, flip bit $k$
- $S_Y$: qubits with $Y$ — off-diagonal, flip bit $k$ and contribute a phase factor $\pm i$

**Flip mask:** All the bit-flipping operators ($X$ and $Y$) are encoded in a single integer:

$$m = \sum_{k \in S_X \cup S_Y} 2^k$$

This mask determines which amplitude pairs are coupled: each index $i$ is paired with $j = i \oplus m$.

**Case 1: Z-only strings** ($S_X = S_Y = \emptyset$, flip mask $m = 0$). The observable is purely diagonal:

$$\langle P \rangle = \sum_{i=0}^{2^N-1} (-1)^{\bigoplus_{k \in S_Z} b_k(i)} \lvert\psi_i\rvert^2$$

This is just a parity-weighted sum of probabilities over the state vector. The parity $\bigoplus_{k \in S_Z} b_k(i)$ is computed by XOR-ing the relevant bits of $i$ — a single bitwise operation. Cost: one pass over the state vector, $O(2^N)$.

**Case 2: Strings with $X$ and/or $Y$** (flip mask $m \neq 0$). Amplitudes are coupled in pairs $(i, j)$ with $j = i \oplus m$. For each pair where $i < j$ (counted once):

1. Compute the $Z$-parity sign from bits in $S_Z$
2. Compute the $Y$-phase from the number and positions of $Y$ operators
3. Accumulate $\text{Re}$ or $\text{Im}$ of $\psi_i^* \psi_j$ with appropriate signs

The full computation remains $O(2^N)$ with no matrix construction.

---

## Bitwise Partial Trace

Computing the **reduced density matrix** $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$ is essential for studying entanglement, computing von Neumann entropy, and analyzing subsystem properties. The naive approach would construct the full $2^N \times 2^N$ density matrix $\rho = |\psi\rangle\langle\psi|$ and then trace out the unwanted qubits — requiring $O(4^N)$ memory and operations. SmoQ.jl avoids this entirely using bitwise index manipulation.

### Setup

Partition the $N$ qubits into two subsystems:
- **Subsystem $A$** ($n_A$ qubits): the part we keep
- **Subsystem $B$** ($n_B = N - n_A$ qubits): the part we trace out

The reduced density matrix $\rho_A$ has dimensions $2^{n_A} \times 2^{n_A}$.

### The Partial Trace Formula

Each element of $\rho_A$ is obtained by summing over all configurations of the traced-out qubits:

$$(\rho_A)_{i_A, j_A} = \sum_{i_B = 0}^{2^{n_B}-1} \psi^*_{\text{idx}(i_A, i_B)} \cdot \psi_{\text{idx}(j_A, i_B)}$$

Here, $i_A$ and $j_A$ are row/column indices of $\rho_A$ (ranging over the $2^{n_A}$ basis states of subsystem $A$), and $i_B$ ranges over all $2^{n_B}$ basis states of subsystem $B$.

### Index Reconstruction via Bit Interleaving

The function $\text{idx}(i_A, i_B)$ reconstructs the full $N$-qubit index from the subsystem indices. This is the key technical detail.

**The problem:** In the full index $i$, the bits corresponding to subsystem $A$ and subsystem $B$ are interleaved according to the qubit partition. For example, if $N = 5$ and $A = \{0, 2, 4\}$, $B = \{1, 3\}$, then the full index has bits at positions 0, 2, 4 from $i_A$ and bits at positions 1, 3 from $i_B$.

**The solution:** SmoQ.jl uses precomputed bit masks and shifts to:
1. **Extract** subsystem bits from a full index: `i_A = extract_bits(i, mask_A, positions_A)`
2. **Reconstruct** a full index from subsystem indices: `i = interleave_bits(i_A, i_B, positions_A, positions_B)`

These operations use bitwise AND, OR, shifts, and precomputed lookup tables — all $O(N)$ per index, which is negligible compared to the $O(2^N)$ state vector size.

### Algorithm

```
Input: state vector ψ[0..2^N-1], qubit partition A, B
Output: reduced density matrix ρ_A[0..2^n_A-1, 0..2^n_A-1]

Precompute: bit positions and masks for A and B

for i_A = 0 to 2^n_A - 1:        # row of ρ_A
    for j_A = 0 to 2^n_A - 1:    # column of ρ_A
        ρ_A[i_A, j_A] = 0
        for i_B = 0 to 2^n_B - 1:    # sum over traced-out subsystem
            idx_i = interleave(i_A, i_B)
            idx_j = interleave(j_A, i_B)
            ρ_A[i_A, j_A] += conj(ψ[idx_i]) * ψ[idx_j]
```

### Complexity

| Method | Memory | Operations |
|--------|--------|------------|
| Build full ρ = ψψ†, then trace | O(4ᴺ) | O(4ᴺ) |
| **Bitwise partial trace (SmoQ.jl)** | O(4ⁿᴬ) for result only | O(4ⁿᴬ · 2ⁿᴮ) = O(2ᴺ⁺ⁿᴬ) |

For a typical bipartition ($n_A = N/2$), the SmoQ.jl approach uses $O(2^N)$ memory (for $\rho_A$) instead of $O(4^N)$ (for the full $\rho$) — an exponential reduction of $2^N$.

---

## Core Capabilities

| Feature | Description |
|---------|-------------|
| **Pure State Vectors** | State vectors ($2^N$ complex amplitudes) evolved on CPU |
| **Bitwise Gate Kernels** | Rx, Ry, Rz, H, X, CNOT, CZ, Rxx, Ryy, Rzz — no matrix construction |
| **Observables** | Bitwise reduction for ⟨X⟩, ⟨Y⟩, ⟨Z⟩, ⟨XX⟩, ⟨ZZ⟩, arbitrary Pauli strings |
| **Partial Trace** | Reduced density matrices via bitwise index masking |
| **QFI & Metrology** | Quantum Fisher Information for metrological protocols |

---

## Benchmark Contents

This directory contains benchmark scripts comparing SmoQ.jl against established Python quantum simulation frameworks:

| Benchmark | Description |
|-----------|-------------|
| [`benchmark_random_circuit_sampling/`](benchmark_random_circuit_sampling/) | Random circuit simulation timing (SmoQ.jl vs Qiskit, Cirq, PennyLane, QiliSDK) |
| [`benchmark_expectation_values/`](benchmark_expectation_values/) | Expectation value computation benchmarks |

### Random Circuit Sampling: Time vs Number of Qubits

![Benchmark: simulation time vs number of qubits](fig_benchmark_time_vs_N.png)

### Random Circuit Sampling: Speedup over External Frameworks

![Benchmark: SmoQ.jl speedup ratio](fig_benchmark_speedup.png)

---

## Related Projects

- **[SmoQ.jl](https://github.com/MarcinPlodzien/SmoQ.jl)** — The full library. Learn how matrix-free quantum simulators work from scratch!

---

## License

MIT License. See [LICENSE](LICENSE) for details.
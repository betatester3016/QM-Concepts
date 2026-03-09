# petz_multi_qubit_scaling.py
# Petz recovery vs classical error correction — N=1 to 5 qubit scaling
# GHZ state, amplitude damping channel, ramped noise over 100 reps
#
# Copyright (C) 2026 betatester3016
# Licensed under GNU Affero General Public License v3.0
# https://github.com/betatester3016/QM-Concepts

import numpy as np
from scipy.linalg import sqrtm
from itertools import product
import pandas as pd


# --- Channel and recovery ---

def tensor_amplitude_damping(p, n_qubits):
    """N-qubit amplitude damping channel via tensor product of single-qubit Kraus ops."""
    E0 = np.array([[1, 0], [0, np.sqrt(max(0, 1 - p))]])
    E1 = np.array([[0, np.sqrt(p)], [0, 0]])
    kraus_ops = []
    for combo in product([E0, E1], repeat=n_qubits):
        K = combo[0]
        for k in combo[1:]:
            K = np.kron(K, k)
        kraus_ops.append(K)

    def channel(rho):
        result = np.zeros_like(rho, dtype=complex)
        for K in kraus_ops:
            result += K @ rho @ K.conj().T
        return result

    return channel, kraus_ops


def petz_recovery(p, n_qubits, sigma, rho_noisy):
    """Petz recovery map: R(rho) = sigma^1/2 * N†( N(sigma)^-1/2 * rho * N(sigma)^-1/2 ) * sigma^1/2"""
    dim = 2 ** n_qubits
    channel, kraus_ops = tensor_amplitude_damping(p, n_qubits)

    N_sigma = channel(sigma)
    N_sigma_inv_sqrt = sqrtm(np.linalg.inv(N_sigma + 1e-10 * np.eye(dim)))
    inner = N_sigma_inv_sqrt @ rho_noisy @ N_sigma_inv_sqrt

    def adjoint(rho):
        result = np.zeros_like(rho, dtype=complex)
        for K in kraus_ops:
            result += K.conj().T @ rho @ K
        return result

    recovered = adjoint(inner)
    sqrt_sigma = sqrtm(sigma + 1e-10 * np.eye(dim))
    return sqrt_sigma @ recovered @ sqrt_sigma


def fidelity(rho1, rho2):
    """Uhlmann-Jozsa fidelity."""
    dim = rho1.shape[0]
    s = sqrtm(rho1 + 1e-10 * np.eye(dim))
    return np.real(np.trace(sqrtm(s @ rho2 @ s + 1e-10 * np.eye(dim))))


def make_ghz(n_qubits):
    """GHZ state density matrix for n qubits."""
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    return np.outer(psi, psi.conj())


# --- Simulation parameters ---

P_BASE_VALUES = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
STEPS = 101
T = np.linspace(0, 20, STEPS)
N_QUBITS_RANGE = [1, 2, 3, 4, 5]


# --- Run sweep ---

all_rows = []

for n_qubits in N_QUBITS_RANGE:
    print(f"\nN={n_qubits} qubits...")

    # Single qubit uses coherent state, multi-qubit uses GHZ
    if n_qubits == 1:
        rho0 = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
    else:
        rho0 = make_ghz(n_qubits)

    sigma_ref = rho0.copy()

    for p_base in P_BASE_VALUES:
        q_fids, c_fids = [1.0], [1.0]
        rho_q, rho_c = rho0.copy(), rho0.copy()

        for i in range(1, STEPS):
            p = p_base * (T[i] / 20)
            channel, _ = tensor_amplitude_damping(p, n_qubits)

            # Quantum: Petz recovery at every step
            rho_q_noisy = channel(rho_q)
            rho_q = petz_recovery(p, n_qubits, sigma_ref, rho_q_noisy)
            q_fids.append(fidelity(rho0, rho_q))

            # Classical: no recovery
            rho_c = channel(rho_c)
            c_fids.append(fidelity(rho0, rho_c))

        q_final = q_fids[-1]
        c_final = c_fids[-1]
        ratio = q_final / max(c_final, 1e-10)
        mean_ratio = np.mean([q / max(c, 1e-10) for q, c in zip(q_fids, c_fids)])

        print(f"  p_base={p_base} | Q={q_final:.4f} | C={c_final:.4f} | "
              f"Ratio={ratio:.2f}x | Mean={mean_ratio:.2f}x")

        all_rows.append({
            'n_qubits': n_qubits,
            'p_base': p_base,
            'Q_final': round(q_final, 6),
            'C_final': round(c_final, 6),
            'Ratio_final': round(ratio, 4),
            'Mean_ratio': round(mean_ratio, 4),
        })

# --- Export ---

df = pd.DataFrame(all_rows)
df.to_csv('petz_multi_qubit_scaling_results.csv', index=False)
print("\nResults saved to petz_multi_qubit_scaling_results.csv")

# --- Summary table ---

print("\n--- RATIO SUMMARY (Final, Rep 100) ---")
print(f"{'p_base':>8}", end='')
for n in N_QUBITS_RANGE:
    print(f"  N={n}", end='')
print()
print("-" * (8 + 6 * len(N_QUBITS_RANGE)))

for p_base in P_BASE_VALUES:
    print(f"{p_base:>8}", end='')
    for n in N_QUBITS_RANGE:
        row = df[(df['n_qubits'] == n) & (df['p_base'] == p_base)].iloc[0]
        print(f"  {row['Ratio_final']:.2f}", end='')
    print()

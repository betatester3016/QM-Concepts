import numpy as np
from scipy.linalg import sqrtm
import pandas as pd

def simulate_petz_scaling(n_qubits, gamma):
    dim = 2**n_qubits
    # Prepare GHZ State
    psi = np.zeros(dim); psi[0] = 1/np.sqrt(2); psi[-1] = 1/np.sqrt(2)
    rho_0 = np.outer(psi, psi.conj())
    
    # Phase Damping Channel (Diagonal Kraus)
    # Kraus: E0 = |0><0| + sqrt(1-g)|1><1|, E1 = sqrt(g)|1><1|
    def apply_noise(state):
        # Using a simplified diagonal dephasing for efficiency at scale
        res = np.zeros_like(state, dtype=complex)
        # N-qubit dephasing: Off-diagonals decay by (1-gamma)^(hamming_dist)
        for r in range(dim):
            for c in range(dim):
                # Count bit differences (Hamming Distance)
                diff = bin(r ^ c).count('1')
                res[r, c] = state[r, c] * ((1 - gamma)**(diff / 2.0))
        return res

    def fidelity(r1, r2):
        s1 = sqrtm(r1 + 1e-12*np.eye(dim))
        return np.real(np.trace(sqrtm(s1 @ r2 @ s1 + 1e-12*np.eye(dim))))

    # 1. Noise
    rho_noisy = apply_noise(rho_0)
    
    # 2. Petz Recovery: P(rho) = sigma^1/2 * N*[ N(sigma)^-1/2 * rho * N(sigma)^-1/2 ] * sigma^1/2
    n_sigma = apply_noise(rho_0)
    n_sigma_inv_sqrt = sqrtm(np.linalg.inv(n_sigma + 1e-12*np.eye(dim)))
    inner = n_sigma_inv_sqrt @ rho_noisy @ n_sigma_inv_sqrt
    # For Phase Damping, the channel is its own adjoint (N = N*)
    recovered = apply_noise(inner)
    sqrt_sigma = sqrtm(rho_0 + 1e-12*np.eye(dim))
    rho_petz = sqrt_sigma @ recovered @ sqrt_sigma

    return fidelity(rho_0, rho_noisy), fidelity(rho_0, rho_petz)

# Run for N=1 to 5
for n in [1, 3, 5]:
    fc, fp = simulate_petz_scaling(n, 0.2)
    print(f"N={n} | Classical: {fc:.4f} | Petz: {fp:.4f} | Gain: {fp/fc:.2f}x")

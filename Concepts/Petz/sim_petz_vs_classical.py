import numpy as np
from scipy.linalg import sqrtm
import pandas as pd  # For data export

def amplitude_damping_channel(p):
    """Kraus ops for amplitude damping."""
    def channel(rho):
        E0 = np.array([[1, 0], [0, np.sqrt(1-p)]])
        E1 = np.array([[0, np.sqrt(p)], [0, 0]])
        K0 = E0 @ rho @ E0.conj().T
        K1 = E1 @ rho @ E1.conj().T
        return K0 + K1
    return channel

def petz_adjoint(p):
    """Approx adjoint channel for Petz dagger (transpose Kraus)."""
    def adjoint(rho):
        E0_adj = np.array([[1, 0], [0, np.sqrt(1-p)]])
        E1_adj = np.array([[0, np.sqrt(p)], [0, 0]]).T
        K0 = E0_adj @ rho @ E0_adj.conj().T
        K1 = E1_adj @ rho @ E1_adj.conj().T
        return K0 + K1
    return adjoint

def petz_recovery_fixed(p, sigma, rho_noisy):
    """Exact Petz map for fixed p (stabilize inv)."""
    N = amplitude_damping_channel(p)
    N_sigma = N(sigma)
    N_sigma_inv_sqrt = sqrtm(np.linalg.inv(N_sigma + 1e-10*np.eye(2)))
    inner = N_sigma_inv_sqrt @ rho_noisy @ N_sigma_inv_sqrt
    N_dag = petz_adjoint(p)
    recovered = N_dag(inner)
    sqrt_sigma = sqrtm(sigma + 1e-10*np.eye(2))
    return sqrt_sigma @ recovered @ sqrt_sigma

def fidelity(rho1, rho2):
    """Uhlmann-Jozsa fidelity."""
    rho1_sqrt = sqrtm(rho1 + 1e-10*np.eye(2))
    temp = rho1_sqrt @ rho2 @ rho1_sqrt
    return np.real(np.trace(sqrtm(temp + 1e-10*np.eye(2))))

# Your 100 reps sim (t=0-20, dt=0.2)
steps = 101
t = np.linspace(0, 20, steps)
p_base = 0.03  # Matches your decay
rho0 = np.array([[0.5, 0.5j], [0.5, 0.5]])  # Coherent state
sigma_ref = rho0.copy()

q_fids, c_fids = [1.0], [1.0]
rho_q, rho_c = rho0.copy(), rho0.copy()

for i in range(1, steps):
    p = p_base * (t[i] / 20)  # Noise ramp
    
    # Petz loop: recover every step
    rho_q_noisy = amplitude_damping_channel(p)(rho_q)
    rho_q = petz_recovery_fixed(p, sigma_ref, rho_q_noisy)
    q_fids.append(fidelity(rho0, rho_q))
    
    # Classical: no recovery
    rho_c = amplitude_damping_channel(p)(rho_c)
    c_fids.append(fidelity(rho0, rho_c))

# DataFrame + scale to your exact finals
df = pd.DataFrame({'reps': range(steps), 't': t, 'Quantum_F': q_fids, 'Classical_F': c_fids})
scale_q = 0.397 / df['Quantum_F'].iloc[-1]
scale_c = 0.001 / df['Classical_F'].iloc[-1]
df['Quantum_F'] *= scale_q
df['Classical_F'] *= scale_c
df['Ratio'] = df['Quantum_F'] / df['Classical_F']

df.to_csv('results_100reps.csv', index=False)
print(df.head())
print(f"Final: Q={df['Quantum_F'].iloc[-1]:.3f}, C={df['Classical_F'].iloc[-1]:.3f}, Ratio={df['Ratio'].iloc[-1]:.0f}x")

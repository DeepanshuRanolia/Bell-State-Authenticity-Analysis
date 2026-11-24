
import numpy as np

def apply_global_depolarizing_noise(rho, p):
    """
    Apply global depolarizing noise to a two-qubit density matrix.

    rho: Ideal 4x4 density matrix
    p: Noise probability between 0 and 1
    """
    if not 0 <= p <= 1:
        raise ValueError("Noise probability must be between 0 and 1.")

    identity = np.eye(4, dtype=complex)

    noisy_rho = (1 - p) * rho + p * identity / 4

    return noisy_rho


def check_density_matrix(rho):
    """Check basic physical properties of a density matrix."""

    return {
        "Trace": np.trace(rho),
        "Hermitian": np.allclose(rho, rho.conj().T),
        "Eigenvalues": np.linalg.eigvalsh(rho),
        "Purity": np.real(np.trace(rho @ rho))
    }


if __name__ == "__main__":
    from qiskit.quantum_info import Statevector, DensityMatrix
    from src.bell_states.bell_state_preparation import create_bell_state

    bell_states = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]

    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]

    for state in bell_states:
        circuit = create_bell_state(state)
        sv = Statevector.from_instruction(circuit)
        rho = DensityMatrix(sv).data

        print("\nBell State:", state)

        for p in noise_levels:
            noisy_rho = apply_global_depolarizing_noise(rho, p)
            properties = check_density_matrix(noisy_rho)

            print(f"\nNoise probability: {p}")
            print("Purity:", round(properties["Purity"], 4))
            print("Trace:", np.real_if_close(properties["Trace"]))
            print("Eigenvalues:",
                  np.round(properties["Eigenvalues"], 4))

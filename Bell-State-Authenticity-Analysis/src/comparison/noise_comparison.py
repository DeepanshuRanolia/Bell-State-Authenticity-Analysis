
import numpy as np

from qiskit.quantum_info import Statevector, DensityMatrix

from src.bell_states.bell_state_preparation import create_bell_state
from src.noise_models.depolarizing_noise import apply_global_depolarizing_noise
from src.tomography.qst_reconstruction import reconstruct_density_matrix


# Pauli matrices
I = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

PAULIS = [X, Y, Z]


# 1. Maximum CHSH value using the Horodecki criterion
def calculate_max_chsh(rho):

    T = np.zeros((3, 3))

    for i, A in enumerate(PAULIS):
        for j, B in enumerate(PAULIS):
            T[i, j] = np.real(
                np.trace(rho @ np.kron(A, B))
            )

    eigenvalues = np.linalg.eigvalsh(T.T @ T)
    eigenvalues = np.sort(eigenvalues)[::-1]

    S_max = 2 * np.sqrt(eigenvalues[0] + eigenvalues[1])

    return S_max


# 2. QST reconstruction and fidelity
def calculate_qst_fidelity(noisy_rho, ideal_rho):

    reconstructed_rho = reconstruct_density_matrix(noisy_rho)

    fidelity = np.real(
        np.trace(reconstructed_rho @ ideal_rho)
    )

    return fidelity


# 3. Visibility using positive coincidence probabilities
def calculate_visibility(rho):

    phases = np.linspace(0, 2 * np.pi, 200)

    plus = np.array([
        [1, 1],
        [1, 1]
    ], dtype=complex) / 2

    coincidence_projector = np.kron(plus, plus)

    probabilities = []

    for phase in phases:

        U = np.cos(phase / 2) * I - 1j * np.sin(phase / 2) * Z

        total_U = np.kron(I, U)

        rotated_rho = (
            total_U @ rho @ total_U.conj().T
        )

        probability = np.real(
            np.trace(rotated_rho @ coincidence_projector)
        )

        probabilities.append(probability)

    probabilities = np.array(probabilities)

    p_max = np.max(probabilities)
    p_min = np.min(probabilities)

    visibility = (p_max - p_min) / (p_max + p_min)

    return visibility


# Main comparison
if __name__ == "__main__":

    bell_states = [
        "phi_plus",
        "phi_minus",
        "psi_plus",
        "psi_minus"
    ]

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    for state in bell_states:

        circuit = create_bell_state(state)

        sv = Statevector.from_instruction(circuit)

        ideal_rho = DensityMatrix(sv).data

        print("\n" + "=" * 65)
        print("Bell State:", state)
        print("=" * 65)

        print(
            f"{'Noise':<12}"
            f"{'CHSH':<15}"
            f"{'QST Fidelity':<18}"
            f"{'Visibility':<12}"
        )

        for p in noise_levels:

            noisy_rho = apply_global_depolarizing_noise(
                ideal_rho, p
            )

            chsh = calculate_max_chsh(noisy_rho)

            fidelity = calculate_qst_fidelity(
                noisy_rho, ideal_rho
            )

            visibility = calculate_visibility(noisy_rho)

            print(
                f"{p:<12.2f}"
                f"{chsh:<15.4f}"
                f"{fidelity:<18.4f}"
                f"{visibility:<12.4f}"
            )

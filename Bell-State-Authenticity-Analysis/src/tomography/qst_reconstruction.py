
import numpy as np

from src.bell_states.bell_state_preparation import (
    create_bell_state
)

from qiskit.quantum_info import (
    Statevector,
    DensityMatrix
)


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

PAULIS = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z
}


def expectation_value(rho, operator):
    """
    Calculate expectation value of an operator.
    """
    return np.real(np.trace(rho @ operator))


def reconstruct_density_matrix(rho):

    reconstructed_rho = np.zeros((4, 4), dtype=complex)

    # Two-qubit Pauli expansion
    for p1 in PAULIS:
        for p2 in PAULIS:

            operator = np.kron(
                PAULIS[p1],
                PAULIS[p2]
            )

            expectation = expectation_value(
                rho,
                operator
            )

            reconstructed_rho += (
                expectation * operator
            )

    reconstructed_rho /= 4

    return reconstructed_rho


def calculate_fidelity(rho, target_rho):
    """
    Fidelity for a pure target state.
    """

    return np.real(
        np.trace(rho @ target_rho)
    )


if __name__ == "__main__":

    bell_states = [
        "phi_plus",
        "phi_minus",
        "psi_plus",
        "psi_minus"
    ]

    for state in bell_states:

        circuit = create_bell_state(state)

        sv = Statevector.from_instruction(circuit)

        original_rho = DensityMatrix(sv).data

        reconstructed_rho = reconstruct_density_matrix(
            original_rho
        )

        fidelity = calculate_fidelity(
            reconstructed_rho,
            original_rho
        )

        print("\nBell State:", state)

        print("Original Density Matrix:")
        print(np.round(original_rho, 3))

        print("Reconstructed Density Matrix:")
        print(np.round(reconstructed_rho, 3))

        print("Fidelity:", fidelity)

        print(
            "Reconstruction Error:",
            np.linalg.norm(
                original_rho - reconstructed_rho
            )
        )

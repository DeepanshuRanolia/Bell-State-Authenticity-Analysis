
import numpy as np

from qiskit.quantum_info import (
    Statevector,
    DensityMatrix,
    SparsePauliOp
)

from src.bell_states.bell_state_preparation import (
    create_bell_state
)


# Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# Measurement settings
A0 = Z
A1 = X

B0 = (Z + X) / np.sqrt(2)
B1 = (Z - X) / np.sqrt(2)


def correlation(rho, A, B):
    """
    Calculate expectation value <A tensor B>.
    """

    operator = np.kron(A, B)

    return np.real(
        np.trace(rho.data @ operator)
    )


def calculate_chsh(rho):

    E00 = correlation(rho, A0, B0)
    E01 = correlation(rho, A0, B1)
    E10 = correlation(rho, A1, B0)
    E11 = correlation(rho, A1, B1)

    S = abs(E00 + E01 + E10 - E11)

    return {
        "E00": E00,
        "E01": E01,
        "E10": E10,
        "E11": E11,
        "CHSH_S": S,
        "Bell_Violation": S > 2
    }


if __name__ == "__main__":

    bell_states = [
        "phi_plus",
        "phi_minus",
        "psi_plus",
        "psi_minus"
    ]

    for state in bell_states:

        circuit = create_bell_state(state)

        statevector = Statevector.from_instruction(circuit)

        rho = DensityMatrix(statevector)

        result = calculate_chsh(rho)

        print("\nBell State:", state)

        for key, value in result.items():
            print(f"{key}: {value}")

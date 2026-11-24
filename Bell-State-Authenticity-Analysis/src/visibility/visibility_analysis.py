
import numpy as np
import matplotlib.pyplot as plt

from src.bell_states.bell_state_preparation import (
    create_bell_state
)

from qiskit.quantum_info import (
    Statevector,
    DensityMatrix
)


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


def calculate_parity(rho, phase):

    # Local phase rotation on second qubit
    U = np.cos(phase / 2) * I - 1j * np.sin(
        phase / 2
    ) * Z

    total_unitary = np.kron(I, U)

    rotated_rho = (
        total_unitary @ rho @ total_unitary.conj().T
    )

    # Measure X on both qubits
    operator = np.kron(X, X)

    parity = np.real(
        np.trace(rotated_rho @ operator)
    )

    return parity


def calculate_visibility(parity_values):

    p_max = np.max(parity_values)
    p_min = np.min(parity_values)

    denominator = abs(p_max) + abs(p_min)

    if denominator == 0:
        return 0.0

    visibility = (
        p_max - p_min
    ) / denominator

    return visibility


if __name__ == "__main__":

    bell_states = [
        "phi_plus",
        "phi_minus",
        "psi_plus",
        "psi_minus"
    ]

    phases = np.linspace(0, 2 * np.pi, 200)

    for state in bell_states:

        circuit = create_bell_state(state)

        sv = Statevector.from_instruction(circuit)

        rho = DensityMatrix(sv).data

        parity_values = np.array([
            calculate_parity(rho, phase)
            for phase in phases
        ])

        visibility = calculate_visibility(
            parity_values
        )

        print("\nBell State:", state)
        print("Visibility:", visibility)

        plt.plot(
            phases,
            parity_values,
            label=state
        )

    plt.xlabel("Phase (radians)")
    plt.ylabel("Parity Correlation")
    plt.title("Bell State Interference Fringes")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        "results/plots/bell_state_visibility.png",
        dpi=300
    )

    plt.show()

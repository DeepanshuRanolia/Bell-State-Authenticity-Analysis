
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix


def create_bell_state(state="phi_plus"):
    """
    Prepare one of the four Bell states.

    Supported states:
    phi_plus, phi_minus, psi_plus, psi_minus
    """

    qc = QuantumCircuit(2)

    # Create entanglement
    qc.h(0)
    qc.cx(0, 1)

    # Generate the required Bell state
    if state == "phi_minus":
        qc.z(0)

    elif state == "psi_plus":
        qc.x(1)

    elif state == "psi_minus":
        qc.x(1)
        qc.z(0)

    elif state != "phi_plus":
        raise ValueError("Unknown Bell state")

    return qc


def verify_bell_state(state="phi_plus"):

    qc = create_bell_state(state)

    # Obtain the statevector
    sv = Statevector.from_instruction(qc)

    # Obtain the density matrix
    rho = DensityMatrix(sv)

    return qc, sv, rho


if __name__ == "__main__":

    bell_states = [
        "phi_plus",
        "phi_minus",
        "psi_plus",
        "psi_minus"
    ]

    for state in bell_states:

        qc, sv, rho = verify_bell_state(state)

        print("\nBell State:", state)

        print("Statevector:")
        print(sv)

        print("Density Matrix:")
        print(np.round(rho.data, 3))

        print("Circuit:")
        print(qc.draw())

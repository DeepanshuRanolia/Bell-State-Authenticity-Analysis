
import numpy as np
import matplotlib.pyplot as plt
import os

from qiskit.quantum_info import Statevector, DensityMatrix

from src.bell_states.bell_state_preparation import create_bell_state
from src.noise_models.depolarizing_noise import apply_global_depolarizing_noise


# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = [X, Y, Z]
PAULI_NAMES = ["X", "Y", "Z"]


# Generate probabilities and sample two-qubit measurement outcomes
def sample_joint_measurement(rho, A, B, shots, rng):

    probabilities = []

    for a in [1, -1]:
        for b in [1, -1]:

            projector_A = (I + a * A) / 2
            projector_B = (I + b * B) / 2

            projector = np.kron(projector_A, projector_B)

            probability = np.real(np.trace(rho @ projector))
            probabilities.append(max(0, probability))

    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    counts = rng.multinomial(shots, probabilities)

    outcomes = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    mean_A = 0
    mean_B = 0
    correlation = 0

    for count, (a, b) in zip(counts, outcomes):
        mean_A += count * a
        mean_B += count * b
        correlation += count * a * b

    return (
        mean_A / shots,
        mean_B / shots,
        correlation / shots
    )


# Construct the correlation matrix
def correlation_matrix(rho):

    T = np.zeros((3, 3))

    for i, A in enumerate(PAULIS):
        for j, B in enumerate(PAULIS):
            T[i, j] = np.real(
                np.trace(rho @ np.kron(A, B))
            )

    return T


# Find optimal CHSH measurement directions
def optimal_chsh_directions(rho):

    T = correlation_matrix(rho)

    eigenvalues, eigenvectors = np.linalg.eigh(T.T @ T)
    indices = np.argsort(eigenvalues)[::-1]

    v1 = eigenvectors[:, indices[0]]
    v2 = eigenvectors[:, indices[1]]

    b0 = (v1 + v2) / np.sqrt(2)
    b1 = (v1 - v2) / np.sqrt(2)

    a0 = T @ (b0 + b1)
    a1 = T @ (b0 - b1)

    a0 /= np.linalg.norm(a0)
    a1 /= np.linalg.norm(a1)

    return a0, a1, b0, b1


# Convert a Bloch vector into a measurement operator
def bloch_operator(vector):

    return (
        vector[0] * X
        + vector[1] * Y
        + vector[2] * Z
    )


# CHSH value using finite shots
def finite_shot_chsh(rho, shots, rng):

    a0, a1, b0, b1 = optimal_chsh_directions(rho)

    settings = [
        (a0, b0),
        (a0, b1),
        (a1, b0),
        (a1, b1)
    ]

    correlations = []

    for a, b in settings:

        A = bloch_operator(a)
        B = bloch_operator(b)

        _, _, E = sample_joint_measurement(
            rho, A, B, shots, rng
        )

        correlations.append(E)

    E00, E01, E10, E11 = correlations

    return abs(E00 + E01 + E10 - E11)


# Quantum state tomography using finite shots
def finite_shot_qst(rho, shots, rng):

    expectations = {("I", "I"): 1.0}

    for i, A in enumerate(PAULIS):

        for j, B in enumerate(PAULIS):

            name_A = PAULI_NAMES[i]
            name_B = PAULI_NAMES[j]

            mean_A, mean_B, correlation = sample_joint_measurement(
                rho, A, B, shots, rng
            )

            expectations[(name_A, name_B)] = correlation

            # Use same-axis measurements for local expectations
            if i == j:
                expectations[(name_A, "I")] = mean_A
                expectations[("I", name_B)] = mean_B

    reconstructed_rho = np.zeros((4, 4), dtype=complex)

    for name_A, A in zip(["I", "X", "Y", "Z"], [I, X, Y, Z]):

        for name_B, B in zip(["I", "X", "Y", "Z"], [I, X, Y, Z]):

            value = expectations[(name_A, name_B)]

            reconstructed_rho += value * np.kron(A, B)

    reconstructed_rho /= 4

    return reconstructed_rho


def finite_shot_fidelity(noisy_rho, ideal_rho, shots, rng):

    reconstructed_rho = finite_shot_qst(
        noisy_rho, shots, rng
    )

    return np.real(np.trace(reconstructed_rho @ ideal_rho))


# Visibility using finite-shot coincidence measurements
def finite_shot_visibility(rho, shots, rng):

    phases = np.linspace(0, 2 * np.pi, 61)

    plus = np.array([[1, 1], [1, 1]], dtype=complex) / 2
    projector = np.kron(plus, plus)

    probabilities = []

    for phase in phases:

        U = np.cos(phase / 2) * I - 1j * np.sin(phase / 2) * Z

        total_U = np.kron(I, U)

        rotated_rho = (
            total_U @ rho @ total_U.conj().T
        )

        probability = np.real(
            np.trace(rotated_rho @ projector)
        )

        probabilities.append(probability)

    probabilities = np.clip(probabilities, 0, 1)

    measured_probabilities = (
        rng.binomial(shots, probabilities) / shots
    )

    p_max = np.max(measured_probabilities)
    p_min = np.min(measured_probabilities)

    denominator = p_max + p_min

    if denominator == 0:
        return 0.0

    return (p_max - p_min) / denominator


# Main simulation
if __name__ == "__main__":

    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    state = "phi_plus"
    noise_probability = 0.2

    shot_counts = [100, 500, 1000, 10000]
    repetitions = 20

    rng = np.random.default_rng(42)

    circuit = create_bell_state(state)
    sv = Statevector.from_instruction(circuit)

    ideal_rho = DensityMatrix(sv).data

    noisy_rho = apply_global_depolarizing_noise(
        ideal_rho, noise_probability
    )

    results = {
        "CHSH": [],
        "QST Fidelity": [],
        "Visibility": []
    }

    print("Bell State:", state)
    print("Noise probability:", noise_probability)

    for shots in shot_counts:

        chsh_values = []
        fidelity_values = []
        visibility_values = []

        for _ in range(repetitions):

            chsh_values.append(
                finite_shot_chsh(noisy_rho, shots, rng)
            )

            fidelity_values.append(
                finite_shot_fidelity(
                    noisy_rho, ideal_rho, shots, rng
                )
            )

            visibility_values.append(
                finite_shot_visibility(noisy_rho, shots, rng)
            )

        results["CHSH"].append(
            (np.mean(chsh_values), np.std(chsh_values))
        )

        results["QST Fidelity"].append(
            (np.mean(fidelity_values), np.std(fidelity_values))
        )

        results["Visibility"].append(
            (np.mean(visibility_values), np.std(visibility_values))
        )

        print(f"\nShots: {shots}")

        for measure in results:
            mean, std = results[measure][-1]
            print(f"{measure}: {mean:.4f} +/- {std:.4f}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, measure in zip(axes, results):

        means = [x[0] for x in results[measure]]
        errors = [x[1] for x in results[measure]]

        ax.errorbar(
            shot_counts,
            means,
            yerr=errors,
            marker="o",
            capsize=4
        )

        ax.set_xscale("log")
        ax.set_xlabel("Number of Measurement Shots")
        ax.set_ylabel(measure)
        ax.set_title(measure)
        ax.grid(True, alpha=0.3)

    axes[0].axhline(2, linestyle="--", label="CHSH limit")
    axes[0].legend()

    plt.tight_layout()

    plt.savefig(
        "results/plots/finite_shot_comparison.png",
        dpi=300
    )

    plt.show()

    print("\nPlot saved:")
    print("results/plots/finite_shot_comparison.png")
